"""
run_all_parks.py
==================
Full run of the parks-and-cars measurement for every park in
parks_resolved.json, fully offline except for boundary lookups.
Reuses the metric functions from park_metrics.py, so every park is
measured exactly like Monsanto (car space by road category, pieces,
crossings, speed limits).

Phases
  1. Boundaries (Nominatim, ~1 request/second)
       OSM ID from parks_resolved.json -> exact polygon by ID
       no ID and no coordinates        -> name search
       no ID but coordinates           -> resolved offline in phase 3
  2. Regions: picks the smallest Geofabrik extract containing each park
     (from Geofabrik's index of extracts)
  3. Per region: downloads its extract once (cached in pbf/), then per park:
       coordinates only -> the park-like polygon around those coordinates
                           whose area best matches Wikipedia's
       cut the park's area out with osmium, measure, save

Every park's area is checked against Wikipedia's; outside 0.5x to 2x it is
flagged needs_review (likely the wrong polygon).

Outputs (in run/)
  parks_results.csv        one row per park, all metrics
  summaries/<slug>.json    per park (also makes the run resumable)
  geojson/<slug>.geojson   per park map layers, same format as Monsanto's
  boundaries/<slug>.geojson

overrides.json (optional), keyed by slug:
  {"<slug>": {"osm_id": "R123"}}          force a polygon
  {"<slug>": {"osm_ids": ["R1", "W2"]}}   a park made of several OSM polygons
  {"<slug>": {"lat": 38.7, "lon": -9.2}}  force coordinates
  {"<slug>": {"skip": true}}              leave a park out
  {"<slug>": {"verified": true}}          boundary checked by hand: never flag it
  {"<slug>": {"osm_ids": [...], "close_gaps_m": 25}}
                                          a park assembled from many pieces (woods, lakes...):
                                          fills gaps narrower than 2 x close_gaps_m (the
                                          internal roads) so they count as inside the park

Usage
  python run_all_parks.py --plan                 boundaries + download plan, no big downloads
  python run_all_parks.py --only slug1,slug2     test on a few parks
  python run_all_parks.py                        everything (resumable: re-run to continue)
  python run_all_parks.py --delete-pbf           delete each extract after its region is done
  python run_all_parks.py --refresh              re-measure parks already done
"""

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd
import requests
import shapely
from shapely.geometry import Point, box, shape
from shapely.ops import unary_union

import park_metrics as core             # shared, validated metric functions

BASE      = Path(__file__).resolve().parent
RESOLVED  = BASE / "parks_resolved.json"
OVERRIDES = BASE / "overrides.json"
PBF_DIR   = BASE / "pbf"
RUN       = BASE / "run"
BOUND_DIR = RUN / "boundaries"
WORK      = RUN / "work"
SUM_DIR   = RUN / "summaries"
GEO_DIR   = RUN / "geojson"
for d in (PBF_DIR, BOUND_DIR, WORK, SUM_DIR, GEO_DIR):
    d.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = RUN / "parks_results.csv"

INDEX_URL  = "https://download.geofabrik.de/index-v1.json"
INDEX_PATH = PBF_DIR / "geofabrik-index-v1.json"

MARGIN_M  = 200                    # context kept around each park
AREA_OK   = (0.45, 2.1)            # OSM area / Wikipedia area
AREA_FILTERS = ["nwr/leisure=park,nature_reserve", "nwr/boundary=protected_area,national_park",
                "nwr/landuse=forest,recreation_ground", "nwr/natural=wood"]

# User-Agent: osmnx default (generic) for Nominatim; curl/requests defaults for Geofabrik.
ox.settings.requests_timeout = 180
ox.settings.log_console = False
ox.settings.use_cache = True
ox.settings.cache_folder = str(BASE / "osmnx_http_cache")


# ── helpers ────────────────────────────────────────────────────────────

def load_parks():
    parks = json.loads(RESOLVED.read_text())
    overrides = json.loads(OVERRIDES.read_text()) if OVERRIDES.exists() else {}
    for p in parks:
        ov = overrides.get(p["slug"], {})
        ids = ov.get("osm_ids") or ([ov["osm_id"]] if ov.get("osm_id") else [])
        if ids:
            p["override_ids"] = ids
            p["osm_id"], p["osm_id_source"] = ids[0], "override"
        if ov.get("lat") is not None:
            p["lat"], p["lon"] = ov["lat"], ov["lon"]
        p["skip"] = bool(ov.get("skip"))
        p["verified"] = bool(ov.get("verified"))
        p["close_gaps_m"] = ov.get("close_gaps_m")
    return parks


def bounds_with_margin(geom_wgs, margin_m):
    minx, miny, maxx, maxy = geom_wgs.bounds
    dlat = margin_m / 111_000
    dlon = margin_m / (111_000 * max(math.cos(math.radians((miny + maxy) / 2)), 0.1))
    return (minx - dlon, miny - dlat, maxx + dlon, maxy + dlat)


def read(path, crs, cols=()):
    try:
        g = gpd.read_file(path) if Path(path).exists() else None
    except Exception:
        g = None
    if g is None or not len(g):
        return gpd.GeoDataFrame({c: [] for c in cols}, geometry=[], crs=crs)
    return g.to_crs(crs)


def nominatim(query, by_osmid=False):
    for attempt in range(3):
        try:
            gdf = ox.geocode_to_gdf(query, by_osmid=by_osmid)
            time.sleep(1.1)
            geom = gdf.geometry.iloc[0]
            return geom if geom.geom_type in ("Polygon", "MultiPolygon") else None
        except Exception as e:
            time.sleep(1.1)
            if type(e).__name__ in ("InsufficientResponseError", "TypeError", "ValueError"):
                return None
            print(f"      X Nominatim {type(e).__name__}, retrying")
            time.sleep(5 * (attempt + 1))
    return None


def override_method(p):
    ids = p["override_ids"]
    label = "+".join(ids) if len(ids) <= 5 else f"{ids[0]}+{len(ids) - 1} more"
    gaps = f", gaps closed {p['close_gaps_m']} m" if p.get("close_gaps_m") else ""
    return f"OSM {label} (override{gaps})"


def close_gaps(geom, p):
    """Morphological closing: grow by d, shrink by d. Fills gaps narrower than
    2d (roads between woods, lakes and lawns) without moving the outer edge."""
    d = p.get("close_gaps_m")
    if not d:
        return geom
    gs = gpd.GeoSeries([geom.buffer(0)], crs="EPSG:4326")
    utm = gs.estimate_utm_crs()
    closed = gs.to_crs(utm).iloc[0].buffer(d, join_style="round").buffer(-d, join_style="round")
    return gpd.GeoSeries([closed], crs=utm).to_crs("EPSG:4326").iloc[0]


def save_boundary(slug, geom, method):
    gpd.GeoDataFrame({"method": [method]}, geometry=[geom], crs="EPSG:4326").to_file(
        BOUND_DIR / f"{slug}.geojson", driver="GeoJSON")


def load_boundary(slug):
    path = BOUND_DIR / f"{slug}.geojson"
    if not path.exists():
        return None, None
    g = gpd.read_file(path)
    return g.geometry.iloc[0], g["method"].iloc[0]


# ── phase 1: boundaries ────────────────────────────────────────────────

def phase_boundaries(parks):
    def current(p):
        path = BOUND_DIR / f"{p['slug']}.geojson"
        if not path.exists():
            return False
        if not p.get("override_ids"):
            return True
        return str(load_boundary(p["slug"])[1] or "").startswith(override_method(p))
    todo = [p for p in parks if not p["skip"] and not current(p)]
    print(f"\n1/3 Boundaries: {len(todo)} to look up")
    for i, p in enumerate(todo, 1):
        geom, method = None, None
        if p.get("override_ids") and len(p["override_ids"]) > 5:
            print(f"  [{i}/{len(todo)}] {p['name']}: {len(p['override_ids'])} override IDs, "
                  f"will read them from the extract")
            continue
        if p.get("override_ids"):
            parts = [nominatim(i, by_osmid=True) for i in p["override_ids"]]
            parts = [g for g in parts if g is not None]
            if len(parts) == len(p["override_ids"]):
                geom = close_gaps(unary_union([g.buffer(0) for g in parts]), p)
                method = override_method(p)
            else:
                print(f"  [{i}/{len(todo)}] {p['name']}: override not found on Nominatim, "
                      f"will read it from the extract")
                continue
        elif p.get("osm_id"):
            geom = nominatim(p["osm_id"], by_osmid=True)
            method = f"OSM {p['osm_id']} ({p.get('osm_id_source')})"
        if geom is None and p.get("lat") is None:
            for q in (f"{p['name']}, {p['city']}, {p['country']}", f"{p['name']}, {p['city']}"):
                geom = nominatim(q)
                if geom is not None:
                    method = f"name search: {q}"
                    break
        if geom is not None:
            save_boundary(p["slug"], geom, method)
            print(f"  [{i}/{len(todo)}] {p['name']}: {method}")
        else:
            print(f"  [{i}/{len(todo)}] {p['name']}: "
                  f"{'will use coordinates' if p.get('lat') is not None else 'NOT FOUND (add to overrides.json)'}")


# ── phase 2: regions ───────────────────────────────────────────────────

def load_regions():
    if not INDEX_PATH.exists():
        print("  Downloading Geofabrik index...")
        r = requests.get(INDEX_URL, timeout=120)
        r.raise_for_status()
        INDEX_PATH.write_bytes(r.content)
    regions = []
    for f in json.loads(INDEX_PATH.read_text())["features"]:
        pbf = (f["properties"].get("urls") or {}).get("pbf")
        if pbf and f.get("geometry"):
            g = shape(f["geometry"])
            regions.append({"id": f["properties"]["id"], "name": f["properties"]["name"],
                            "pbf": pbf, "geom": g, "area": g.area})
    return sorted(regions, key=lambda r: r["area"])


def park_target(p):
    """The park's own shape (not its bounding box), or a circle of roughly its
    size around its coordinates when the polygon is found later, offline."""
    geom, _ = load_boundary(p["slug"])
    if geom is not None:
        return geom.buffer(0)
    if p.get("lat") is not None:
        r_deg = max(0.005, math.sqrt(p["size_ha"] * 1e4) / 2 / 111_000)
        return Point(p["lon"], p["lat"]).buffer(r_deg)
    return None


def covers(region_geom, target, share=0.98):
    """A region fits if it contains the park, or at least 98% of it: Geofabrik
    extracts reach slightly past their borders, so nothing meaningful is lost."""
    if not box(*region_geom.bounds).intersects(target):
        return False
    if region_geom.contains(target):
        return True
    try:
        return region_geom.intersection(target).area >= share * target.area
    except Exception:
        return False


def phase_regions(parks):
    print("\n2/3 Assigning Geofabrik extracts")
    regions = load_regions()
    plan = {}
    for p in parks:
        if p["skip"]:
            continue
        target = park_target(p)
        if target is None:
            continue
        r = next((r for r in regions if covers(r["geom"], target)), None)
        if r is None:
            print(f"  no extract contains {p['name']} ({p['city']})")
            continue
        p["region"] = r
        plan.setdefault(r["id"], {"region": r, "parks": []})["parks"].append(p)
    return plan


def pbf_path(r):
    legacy = PBF_DIR / r["pbf"].rsplit("/", 1)[-1]           # e.g. portugal-latest.osm.pbf
    return legacy if legacy.exists() else PBF_DIR / f"{r['id'].replace('/', '_')}-latest.osm.pbf"


def print_plan(plan):
    total = 0
    print(f"\n{'Extract':<42} {'Parks':>5} {'Size':>9}  Status")
    print("-" * 75)
    for rid, item in sorted(plan.items()):
        path = pbf_path(item["region"])
        size = None
        if path.exists():
            size, status = path.stat().st_size, "downloaded"
        else:
            status = "to download"
            try:
                h = requests.head(item["region"]["pbf"], allow_redirects=True, timeout=30)
                size = int(h.headers.get("Content-Length", 0)) or None
            except requests.RequestException:
                pass
        total += size or 0
        print(f"{rid:<42} {len(item['parks']):>5} {(f'{size/1e9:.2f} GB' if size else '?'):>9}  {status}")
    print("-" * 75)
    print(f"{'Total':<42} {sum(len(i['parks']) for i in plan.values()):>5} {total/1e9:>6.1f} GB")


def ensure_pbf(r):
    path = pbf_path(r)
    if not path.exists():
        print(f"  Downloading {r['id']} ...")
        tmp = path.with_suffix(".part")
        subprocess.run(["curl", "-L", "--retry", "3", "-f", "-o", str(tmp), r["pbf"]], check=True)
        tmp.rename(path)
        print(f"  Done: {path.stat().st_size/1e9:.2f} GB")
    return path


# ── phase 3: per park ──────────────────────────────────────────────────

def export_layers(pbf, bounds, workdir, layers):
    workdir.mkdir(parents=True, exist_ok=True)
    cut = workdir / "cut.osm.pbf"
    core.osmium("extract", "-b", ",".join(f"{v:.6f}" for v in bounds), "-s", "smart",
                pbf, "-o", cut, "--overwrite")
    for name, (filters, geom_type) in layers.items():
        filtered = workdir / f"{name}.osm.pbf"
        core.osmium("tags-filter", cut, *filters, "-o", filtered, "--overwrite")
        core.osmium("export", filtered, "-o", workdir / f"{name}.geojson", "--overwrite",
                    "-a", "type,id", f"--geometry-types={geom_type}")


def boundary_from_ids(p, pbf):
    """Polygon for the override IDs, read from the region's extract with osmium.
    Tries, in order: the IDs' own polygons; the union of their members' polygons
    (for type=collection relations, e.g. Forêt de Fontainebleau); a polygon
    assembled from their member ways (for boundaries with small gaps)."""
    wd = WORK / p["slug"] / "ids"
    wd.mkdir(parents=True, exist_ok=True)
    ids = [i[0].lower() + i[1:] for i in p["override_ids"]]
    try:
        core.osmium("getid", "-r", pbf, *ids, "-o", wd / "ids.osm.pbf", "--overwrite")
    except RuntimeError as e:
        if "not find" in str(e).lower() or "not found" in str(e).lower():
            raise RuntimeError(f"ID not in {pbf.name}. If you edited it in OSM recently, the extract is "
                               f"older than your edit: delete that file and re-run tomorrow to re-download")
        raise
    for gtype in ("polygon", "linestring"):
        core.osmium("export", wd / "ids.osm.pbf", "-o", wd / f"ids_{gtype}.geojson", "--overwrite",
                    "-a", "type,id", f"--geometry-types={gtype}")
    polys = read(wd / "ids_polygon.geojson", "EPSG:4326")
    want = {(i[0].upper(), i[1:]) for i in p["override_ids"]}
    if len(polys):
        mine = polys[[(str(t)[:1].upper(), str(n)) in want
                      for t, n in zip(core.col(polys, "@type"), core.col(polys, "@id"))]]
        if len(mine):
            return unary_union([x.buffer(0) for x in mine.geometry]), override_method(p)
        return unary_union([x.buffer(0) for x in polys.geometry]), override_method(p) + " from members"
    lines = read(wd / "ids_linestring.geojson", "EPSG:4326")
    if len(lines):
        from shapely.ops import polygonize
        rings = list(polygonize(unary_union(list(lines.geometry))))
        if rings:
            return unary_union(rings), override_method(p) + " from member ways"
    return None, None


def boundary_from_coords(p, pbf):
    size_m2 = p["size_ha"] * 1e4
    radius = math.sqrt(size_m2) * 1.5 + 1500
    pt = Point(p["lon"], p["lat"])
    wd = WORK / p["slug"] / "areas"
    export_layers(pbf, bounds_with_margin(pt, radius), wd, {"areas": (AREA_FILTERS, "polygon")})
    areas = read(wd / "areas.geojson", "EPSG:4326")
    if not len(areas):
        return None, None
    utm = gpd.GeoSeries([pt], crs="EPSG:4326").estimate_utm_crs()
    a = areas.to_crs(utm)
    ptu = gpd.GeoSeries([pt], crs="EPSG:4326").to_crs(utm).iloc[0]
    cand = a[a.contains(ptu)]
    if not len(cand):
        cand = a[a.distance(ptu) < 500]
    if not len(cand):
        return None, None
    cand = cand.assign(score=(cand.area / size_m2).apply(lambda r: abs(math.log(max(r, 1e-6)))))
    best = cand.sort_values("score").iloc[0]
    geom = gpd.GeoSeries([best.geometry], crs=utm).to_crs("EPSG:4326").iloc[0]
    return geom, f"coordinates: {best.get('@type')}/{best.get('@id')} {best.get('name') or ''}".strip()


def measure(p, geom_wgs, method, pbf, snap=False):
    park_gs = gpd.GeoSeries([geom_wgs], crs="EPSG:4326")
    utm = park_gs.estimate_utm_crs()
    park = park_gs.to_crs(utm).iloc[0].buffer(0)
    wd = WORK / p["slug"]
    export_layers(pbf, bounds_with_margin(geom_wgs, MARGIN_M), wd, {
        "roads":   ([f"w/highway={','.join(core.HW_TO_CAT)}"], "linestring"),
        "parking": (["nwr/amenity=parking"], "polygon"),
        "paths":   ([f"w/highway={','.join(core.PATH_TYPES)}"], "linestring"),
    })
    roads = read(wd / "roads.geojson", utm, cols=("highway",))
    parking = read(wd / "parking.geojson", utm)
    paths = read(wd / "paths.geojson", utm, cols=("highway",))
    if snap:   # snap to a 1 cm grid: fixes rare GEOS "non-noded intersection" errors
        fix = lambda g: g.set_geometry(gpd.GeoSeries(g.geometry.apply(lambda x: shapely.set_precision(x, 0.01)), crs=utm))
        park = shapely.set_precision(park, 0.01)
        roads, parking, paths = fix(roads), fix(parking), fix(paths)

    col = core.col
    ok = roads["highway"].isin(core.HW_TO_CAT) \
        & (col(roads, "area") != "yes") & (col(roads, "access") != "private") \
        & (col(roads, "motor_vehicle") != "no") & (col(roads, "motorcar") != "no") \
        & ~((roads["highway"] == "service") & col(roads, "service").isin(core.SERVICE_EXCLUDED))
    near = gpd.clip(roads[ok], park.buffer(core.NEAR_M)) if ok.any() else roads.iloc[0:0].copy()
    near = near[~near.geometry.is_empty].copy()
    near["category"] = near["highway"].map(core.HW_TO_CAT)
    if len(near):
        w = near.apply(core.road_width, axis=1, result_type="expand")
        near["width_m"], near["width_source"] = w[0], w[1]
    else:
        near["width_m"], near["width_source"] = pd.Series(dtype=float), pd.Series(dtype=object)
    near["in_tunnel"] = col(near, "tunnel").apply(core.tag_on).astype(bool) | col(near, "covered").apply(core.tag_on).astype(bool)
    near["on_bridge"] = col(near, "bridge").apply(core.tag_on).astype(bool)
    paths_near = gpd.clip(paths, park.buffer(core.NEAR_M)) if len(paths) else paths
    paths_near = paths_near[~paths_near.geometry.is_empty]

    df, lines, surfaces, pk = core.car_space(near, parking, park)
    try:
        lines, speed_table, coverage, over30 = core.speeds(lines)
    except Exception:
        lines = lines.assign(maxspeed_kmh=None, speed_band="no limit mapped")
        speed_table, coverage, over30 = pd.DataFrame(), None, 0.0
    pieces = core.pieces_of(near, park)
    crossings, _ = core.crossings_of(near, paths_near, park, pieces)

    area_m2 = park.area
    car_km2 = float(df["surface_km2"].sum())
    ratio = (area_m2 / 1e4) / p["size_ha"] if p.get("size_ha") else None
    hw_km = float(lines.loc[lines["category"].isin(core.BARRIER_CATS) & ~lines["in_tunnel"].astype(bool),
                            "length_m"].sum() / 1000) if len(lines) else 0.0
    cats = df.set_index("category")
    real = crossings[crossings["type"] != "unmarked"] if len(crossings) else crossings
    walk = real[real["via"] == "path"] if len(real) else real
    halves_linked = int(sum(1 for lp in (walk["links_pieces"] if len(walk) else [])
                            if {"1", "2"} <= set(x.strip() for x in str(lp).split(","))))

    s = {
        "slug": p["slug"], "name": p["name"], "city": p["city"], "country": p["country"],
        "region": p["region"]["id"] if isinstance(p.get("region"), dict) else None,
        "continent": p.get("region_name") or p.get("region_label"),
        "boundary_method": method, "size_km2_wiki": round(p["size_ha"] / 100, 3),
        "park_area_km2": round(area_m2 / 1e6, 3),
        "area_ratio_osm_wiki": round(ratio, 2) if ratio else None,
        "needs_review": bool(ratio is None or not AREA_OK[0] <= ratio <= AREA_OK[1] or not method.startswith("OSM")),
        "car_space_km2": round(car_km2, 4), "car_space_pct": round(100 * car_km2 * 1e6 / area_m2, 3),
        "road_km": round(float(df["length_km"].fillna(0).sum()), 2),
        "football_fields": round(car_km2 * 1e6 / (105 * 68)),
        "highway_km": round(hw_km, 2),
        "pieces": int(len(pieces)),
        "largest_piece_km2": float(pieces["area_km2"].iloc[0]) if len(pieces) else 0.0,
        "second_piece_km2": float(pieces["area_km2"].iloc[1]) if len(pieces) > 1 else 0.0,
        "walking_crossings": int(len(walk)),
        "walking_crossings_over_motorway": int(walk["over_motorway"].sum()) if len(walk) else 0,
        "walking_crossings_between_two_largest": halves_linked,
        "road_crossings": int(len(real) - len(walk)),
        "unmarked_to_check": int((crossings["type"] == "unmarked").sum()) if len(crossings) else 0,
        "highway_km_per_walking_crossing": round(hw_km / len(walk), 2) if len(walk) else None,
        "speed_limit_coverage_pct": coverage, "road_km_above_30": over30,
    }
    for c in ("highways", "main roads", "local roads", "service roads", "parking"):
        key = c.split()[0]
        s[f"{key}_km"] = None if c == "parking" else float(cats.loc[c, "length_km"])
        s[f"{key}_km2"] = float(cats.loc[c, "surface_km2"])
        s[f"{key}_pct"] = float(cats.loc[c, "pct_of_park"])

    parts = [gpd.GeoDataFrame({"layer": ["park"]}, geometry=[park], crs=utm)]
    for cat, surf in surfaces.items():
        parts.append(gpd.GeoDataFrame({"layer": ["road_surface"], "category": [cat]}, geometry=[surf], crs=utm))
    if pk is not None and not pk.is_empty:
        parts.append(gpd.GeoDataFrame({"layer": ["parking"], "category": ["parking"]}, geometry=[pk], crs=utm))
    if len(lines):
        ln = lines[[c for c in ("highway", "category", "ref", "name", "oneway", "width_m", "width_source",
                                "in_tunnel", "maxspeed_kmh", "speed_band", "geometry") if c in lines.columns]].copy()
        ln["layer"] = "road_line"
        parts.append(ln)
    if len(pieces):
        pc = pieces[["piece_id", "area_km2", "geometry"]].copy()
        pc["layer"] = "piece"
        parts.append(pc)
    if len(crossings):
        cr = crossings[[c for c in ("cid", "type", "how", "via", "over_motorway", "crosses", "along",
                                    "links_pieces", "geometry") if c in crossings.columns]].copy()
        cr["layer"] = "crossing"
        parts.append(cr)
    gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=utm).to_crs("EPSG:4326").to_file(
        GEO_DIR / f"{p['slug']}.geojson", driver="GeoJSON", COORDINATE_PRECISION=6)
    return s


def phase_measure(plan, refresh, delete_pbf):
    print("\n3/3 Measuring")
    for rid, item in sorted(plan.items()):
        todo = [p for p in item["parks"] if refresh or not (SUM_DIR / f"{p['slug']}.json").exists()]
        if not todo:
            continue
        print(f"\n== {rid} ({len(todo)} parks)")
        try:
            pbf = ensure_pbf(item["region"])
        except Exception as e:
            print(f"  Download failed ({type(e).__name__}); skipping this region. Re-run to retry.")
            continue
        for p in todo:
            t0 = time.time()
            try:
                geom, method = load_boundary(p["slug"])
                if p.get("override_ids") and not str(method or "").startswith(override_method(p)):
                    geom, method = boundary_from_ids(p, pbf)
                    if geom is None:
                        raise RuntimeError("override IDs not found in the extract either; check overrides.json")
                    geom = close_gaps(geom, p)
                    save_boundary(p["slug"], geom, method)
                if geom is None and p.get("lat") is not None:
                    geom, method = boundary_from_coords(p, pbf)
                    if geom is not None:
                        save_boundary(p["slug"], geom, method)
                if geom is None:
                    raise RuntimeError("no boundary found")
                try:
                    s = measure(p, geom, method, pbf)
                except Exception as e:
                    if "GEOS" not in type(e).__name__ and "Topology" not in str(e):
                        raise
                    s = measure(p, geom, method, pbf, snap=True)
                s["status"] = "ok"
                flag = "  NEEDS REVIEW" if s["needs_review"] else ""
                print(f"  {p['name'][:40]:<42} {s['car_space_pct']:>6.2f}%  {s['road_km']:>7.1f} km  "
                      f"{s['pieces']:>2} pieces  {s['walking_crossings']:>2} crossings  "
                      f"({time.time()-t0:.0f}s){flag}")
            except Exception as e:
                s = {"slug": p["slug"], "name": p["name"], "city": p["city"], "country": p["country"],
                     "status": "failed", "error": f"{type(e).__name__}: {str(e)[:200]}"}
                print(f"  {p['name'][:40]:<42} FAILED: {s['error'][:90]}")
            (SUM_DIR / f"{p['slug']}.json").write_text(json.dumps(s, indent=2, ensure_ascii=False, default=float))
        if delete_pbf:
            pbf.unlink(missing_ok=True)
            print(f"  Deleted {pbf.name}")


def write_review_layer(df):
    """One GeoJSON with the boundary of every park still needing review, to check
    them all at once in QGIS over an OpenStreetMap basemap."""
    if "needs_review" not in df:
        return
    flagged = df[(df.get("status") == "ok") & (df["needs_review"] == True)]
    feats = []
    for _, r in flagged.iterrows():
        geom, method = load_boundary(r["slug"])
        if geom is None:
            continue
        feats.append({"slug": r["slug"], "name": r["name"], "city": r["city"],
                      "wiki_km2": r.get("size_km2_wiki"), "osm_km2": r.get("park_area_km2"),
                      "ratio": r.get("area_ratio_osm_wiki"), "car_space_pct": r.get("car_space_pct"),
                      "pieces": r.get("pieces"), "method": method, "geometry": geom})
    if feats:
        path = RUN / "review_boundaries.geojson"
        gpd.GeoDataFrame(feats, geometry="geometry", crs="EPSG:4326").to_file(path, driver="GeoJSON")
        print(f"Review layer: {path.relative_to(BASE)} ({len(feats)} parks) - load it in QGIS over an OSM basemap")


def write_results(parks):
    rows = []
    for p in parks:
        if p["skip"]:          # skipped in overrides.json: leave out, even if measured before
            continue
        f = SUM_DIR / f"{p['slug']}.json"
        if f.exists():
            row = json.loads(f.read_text())
            if row.get("status") == "ok":
                ratio = row.get("area_ratio_osm_wiki")
                row["verified_by_hand"] = p["verified"]
                row["needs_review"] = not p["verified"] and (
                    ratio is None or not AREA_OK[0] <= ratio <= AREA_OK[1]
                    or not str(row.get("boundary_method", "")).startswith("OSM"))
            rows.append(row)
        else:
            rows.append({"slug": p["slug"], "name": p["name"], "city": p["city"],
                         "country": p["country"], "status": "not run"})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    write_review_layer(df)
    ok = df[df.get("status") == "ok"] if "status" in df else df.iloc[0:0]
    print(f"\nSaved {RESULTS_CSV.relative_to(BASE)}: {len(ok)} measured, "
          f"{int((df['status'] == 'failed').sum())} failed, {int((df['status'] == 'not run').sum())} not run")
    if len(ok):
        print(f"Needs review (area mismatch or not matched by OSM ID): {int(ok['needs_review'].sum())}")
        print("\nTop 10 by share of park given to cars:")
        for _, r in ok.sort_values("car_space_pct", ascending=False).head(10).iterrows():
            print(f"  {r['name'][:40]:<42} {r['city'][:16]:<18} {r['car_space_pct']:>6.2f}%"
                  f"{'  (review)' if r['needs_review'] else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="boundaries and download plan only")
    ap.add_argument("--only", help="comma-separated slugs")
    ap.add_argument("--refresh", action="store_true", help="re-measure parks already done")
    ap.add_argument("--delete-pbf", action="store_true", help="delete each extract after use")
    ap.add_argument("--results-only", action="store_true", help="just rebuild parks_results.csv")
    args = ap.parse_args()

    parks = load_parks()
    if args.results_only:
        write_results(parks)
        return
    for p in parks:
        p["region_label"] = p.get("region")          # continent from the Wikipedia list
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        missing = wanted - {p["slug"] for p in parks}
        if missing:
            raise SystemExit(f"Unknown slug(s): {', '.join(sorted(missing))}")
        parks = [p for p in parks if p["slug"] in wanted]

    phase_boundaries(parks)
    plan = phase_regions(parks)
    print_plan(plan)
    if args.plan:
        return
    phase_measure(plan, args.refresh, args.delete_pbf)
    write_results(load_parks())       # always rebuilt from every park's saved result


if __name__ == "__main__":
    main()
