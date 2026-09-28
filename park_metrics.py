"""
park_metrics.py
=================
Shared measurement functions for every park (used by run_all_parks.py).
Run directly, it measures Monsanto alone from the Portugal extract, which is
how the method was first validated.

Measures car space, fragmentation, crossings and speed limits inside
Monsanto, fully offline (Geofabrik Portugal extract + osmium). Every metric
is computed from a park polygon plus the OSM roads and paths around it, so
the same functions will run unchanged on every park in the comparison.

1. Car space by road category (length km, surface km², % of park)
     highways       motorway, trunk and their links
     main roads     primary, secondary and their links
     local roads    tertiary, unclassified, residential, living_street, road
     service roads  service (excl. parking aisles, private, emergency access)
     parking        amenity=parking polygons
   Width per way: width tag > lanes x lane width + shoulders > default lanes.
   Surfaces are buffered lines, clipped to the park and merged, so overlaps
   count once. Tunnels add no surface; bridges do. Medians are not counted.

2. Pieces: the park minus highway surfaces. Tunnels and viaducts of
   VIADUCT_MIN_M or more don't cut the forest (open ground over or under them);
   short bridges over a road or path do, and the underpass counts as a crossing. Pieces
   under MIN_PIECE_M2 (medians, the inside of ramps) are ignored.

3. Crossings: places where a walkable path, or another road, crosses a
   highway inside the park. Intersections within CLUSTER_M of each other are
   one crossing (both carriageways of a motorway). Types:
     over/under   a bridge or tunnel on either way (or different layers)
     marked       footway=crossing or a crossing=* tag (at grade)
     unmarked     at grade with no crossing tag: check it; across a motorway
                  this is usually a mapping gap, not a real crossing
   "via path" = footway, path, steps, cycleway... ; "via road" = walking along
   another road. "over motorway" = crosses a motorway or motorway ramp.

4. Connections: which pieces each crossing links (pieces within CONNECT_M).

5. Speed limits: road km by band from maxspeed (incl. implicit PT:urban etc.),
   with the share of road length that actually has a limit mapped.

Outputs
  monsanto_summary.json          all headline numbers (for the web page)
  monsanto_by_category.csv       car space by category
  monsanto_crossings.csv         one row per crossing
  geojson_osmium/<slug>.geojson  layers: park, road_surface, parking,
                                 road_line (with speed), piece, crossing

Setup (once):
  brew install osmium-tool
  mkdir -p pbf
  curl -L --retry 3 -f -o pbf/portugal-latest.osm.pbf \
    https://download.geofabrik.de/europe/portugal-latest.osm.pbf

Run:         python park_metrics.py
Re-extract:  python park_metrics.py --refresh
"""

import itertools
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

BASE    = Path(__file__).resolve().parent
PBF     = BASE / "pbf" / "portugal-latest.osm.pbf"
WORK    = BASE / "osmium_work"
OUT_DIR = BASE / "geojson_osmium"

LISBON_BBOX = "-9.26,38.68,-9.08,38.80"      # left,bottom,right,top
PARK_NAME   = "Monsanto"
SLUG        = "084-lisbon-monsanto-forest-park"

CATEGORIES = [
    ("highways",      ["motorway", "motorway_link", "trunk", "trunk_link"]),
    ("main roads",    ["primary", "primary_link", "secondary", "secondary_link"]),
    ("local roads",   ["tertiary", "tertiary_link", "unclassified", "residential",
                       "living_street", "road"]),
    ("service roads", ["service"]),
]
HW_TO_CAT = {hw: cat for cat, hws in CATEGORIES for hw in hws}
SERVICE_EXCLUDED = {"parking", "parking_aisle", "emergency_access", "private"}

BARRIER_CATS   = {"highways"}                 # roads that cut the park into pieces
MOTORWAY_TYPES = {"motorway", "motorway_link"}
PATH_TYPES     = ["footway", "path", "pedestrian", "steps", "cycleway",
                  "bridleway", "track"]

MIN_PIECE_M2 = 10_000     # 1 ha: smaller pieces are medians or ramp islands
VIADUCT_MIN_M = 100      # highway bridges this long leave open ground below;
                         # shorter ones (over a road or path) still cut the forest
CLUSTER_M    = 25         # intersections closer than 2 x this = one crossing
CONNECT_M    = 60         # pieces within this distance of a crossing are linked
NEAR_M       = 60         # context kept around the park boundary

LANE_WIDTH = {"motorway": 3.75, "trunk": 3.75, "primary": 3.5, "secondary": 3.5}
DEFAULT_LANE_WIDTH = 3.0
SHOULDERS = {"motorway": 4.0, "trunk": 3.0, "primary": 1.0, "secondary": 1.0, "tertiary": 0.5}
LINK_SHOULDERS = 3.5
DEFAULT_LANES = {"motorway": 2, "trunk": 2, "primary": 2, "secondary": 2, "tertiary": 2,
                 "unclassified": 2, "residential": 2, "living_street": 1, "road": 2,
                 "service": 1}
TWO_WAY_HIGHWAY_LANES = 4

IMPLICIT_SPEEDS = {"urban": 50, "rural": 90, "trunk": 100, "motorway": 120,
                   "living_street": 20, "walk": 10}
SPEED_BANDS = [(30, "up to 30"), (50, "31 to 50"), (70, "51 to 70"),
               (90, "71 to 90"), (999, "over 90")]


# ── extraction ─────────────────────────────────────────────────────────

def osmium(*args):
    cmd = ["osmium", *map(str, args)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{r.stderr[-800:]}")


def prepare(refresh):
    WORK.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    names = ("parks", "roads", "parking", "paths")
    if not refresh and all((WORK / f"{n}.geojson").exists() for n in names):
        print("Using existing extracts in osmium_work/ (run with --refresh to redo)")
        return
    if not shutil.which("osmium"):
        raise SystemExit("osmium not found: run `brew install osmium-tool` first.")
    if not PBF.exists():
        raise SystemExit(f"{PBF} not found: download it first (see the docstring).")
    lisbon = WORK / "lisbon.osm.pbf"
    print("Extracting Lisbon from the Portugal extract...")
    osmium("extract", "-b", LISBON_BBOX, "-s", "smart", PBF, "-o", lisbon, "--overwrite")
    print("Filtering and exporting parks, roads, parking and paths...")
    layers = {
        "parks":   (["nwr/leisure=park,nature_reserve", "nwr/boundary=protected_area",
                     "nwr/landuse=forest"], "polygon"),
        "roads":   ([f"w/highway={','.join(HW_TO_CAT)}"], "linestring"),
        "parking": (["nwr/amenity=parking"], "polygon"),
        "paths":   ([f"w/highway={','.join(PATH_TYPES)}"], "linestring"),
    }
    for name, (filters, geom_type) in layers.items():
        filtered = WORK / f"{name}.osm.pbf"
        osmium("tags-filter", lisbon, *filters, "-o", filtered, "--overwrite")
        osmium("export", filtered, "-o", WORK / f"{name}.geojson", "--overwrite",
               "-a", "type,id", f"--geometry-types={geom_type}")


# ── tag helpers ────────────────────────────────────────────────────────

def col(df, name):
    return df[name] if name in df.columns else pd.Series([None] * len(df), index=df.index)


def missing(v):
    return v is None or (isinstance(v, float) and pd.isna(v))


def tag_on(v):
    """True for bridge/tunnel/covered style tags that are set to anything but no."""
    return not missing(v) and str(v).strip().lower() not in ("no", "false", "0", "")


def layer_of(v):
    try:
        return int(str(v).split(";")[0])
    except (TypeError, ValueError):
        return 0


def parse_width(v):
    if missing(v):
        return None
    s = str(v).strip().lower().replace(",", ".")
    if "'" in s or '"' in s or "ft" in s:
        return None
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*m?$", s)
    if not m:
        return None
    w = float(m.group(1))
    return w if 2 <= w <= 40 else None


def parse_lanes(v):
    if missing(v):
        return None
    m = re.match(r"^\s*(\d+)", str(v))
    n = int(m.group(1)) if m else None
    return n if n and 1 <= n <= 10 else None


def parse_speed(v):
    """Returns (km/h or None, source) where source is tag, implicit or None."""
    if missing(v):
        return None, None
    s = str(v).strip().lower()
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(mph|km/h|kmh|kph)?", s)
    if m:
        n = float(m.group(1)) * (1.609 if m.group(2) == "mph" else 1)
        return round(n), "tag"
    m = re.match(r"^[a-z]{2}:([a-z_]+)", s)
    if m and m.group(1) in IMPLICIT_SPEEDS:
        return IMPLICIT_SPEEDS[m.group(1)], "implicit"
    return None, None


def speed_band(kmh):
    if kmh is None or pd.isna(kmh):
        return "no limit mapped"
    for top, label in SPEED_BANDS:
        if kmh <= top:
            return label
    return SPEED_BANDS[-1][1]


def road_width(row):
    hw = row["highway"]
    base = hw.replace("_link", "")
    is_link = hw.endswith("_link")
    oneway = str(row.get("oneway")).lower() in ("yes", "true", "1", "-1")
    w = parse_width(row.get("width"))
    if w:
        return w, "width tag"
    lane_w = LANE_WIDTH.get(base, DEFAULT_LANE_WIDTH)
    shoulder = LINK_SHOULDERS if is_link and base in ("motorway", "trunk") else SHOULDERS.get(base, 0.0)
    lanes = parse_lanes(row.get("lanes"))
    if lanes:
        return lanes * lane_w + shoulder, "lanes tag"
    if is_link:
        lanes = 1
    elif base in ("motorway", "trunk") and not oneway:
        lanes = TWO_WAY_HIGHWAY_LANES
    else:
        lanes = DEFAULT_LANES.get(base, 2)
    return lanes * lane_w + shoulder, "default"


def walkable(df):
    foot = col(df, "foot").fillna("").astype(str).str.lower()
    access = col(df, "access").fillna("").astype(str).str.lower()
    return foot.isin(["yes", "designated", "permissive"]) | (
        ~foot.isin(["no", "private"]) & ~access.isin(["no", "private"]))


def points_of(geom):
    if geom.is_empty:
        return []
    if geom.geom_type == "Point":
        return [geom]
    if hasattr(geom, "geoms"):
        return [g if g.geom_type == "Point" else g.representative_point() for g in geom.geoms]
    return [geom.representative_point()]


# ── metrics (generic: park polygon + roads + paths, all in a metric CRS) ─

def car_space(roads_near, parking, park):
    lines = gpd.clip(roads_near, park)
    lines = lines[~lines.geometry.is_empty].copy()
    lines["length_m"] = lines.geometry.length

    above = roads_near[~roads_near["in_tunnel"]]
    polys = above.geometry.buffer(above["width_m"] / 2, cap_style="flat")
    taken, surfaces, rows = None, {}, []
    for cat, _ in CATEGORIES:
        sel = polys[above["category"] == cat]
        area = 0.0
        if len(sel):
            surf = sel.union_all().intersection(park)
            if taken is not None:
                surf = surf.difference(taken)
            taken = surf if taken is None else taken.union(surf)
            surfaces[cat] = surf
            area = surf.area
        cl = lines[lines["category"] == cat]
        src = cl[~cl["in_tunnel"]].groupby("width_source")["length_m"].sum()
        src_total = src.sum() or 1
        rows.append({
            "category": cat, "length_km": round(cl["length_m"].sum() / 1000, 2),
            "tunnel_km": round(cl.loc[cl["in_tunnel"], "length_m"].sum() / 1000, 2),
            "surface_km2": round(area / 1e6, 4), "pct_of_park": round(100 * area / park.area, 2),
            "pct_width_tag": round(100 * src.get("width tag", 0) / src_total),
            "pct_lanes_tag": round(100 * src.get("lanes tag", 0) / src_total),
            "pct_default": round(100 * src.get("default", 0) / src_total),
        })
    parking = parking[parking.geometry.intersects(park)]
    pk = parking.geometry.union_all().intersection(park) if len(parking) else None
    if pk is not None and taken is not None:
        pk = pk.difference(taken)
    pk_m2 = pk.area if pk is not None else 0.0
    rows.append({"category": "parking", "length_km": None, "tunnel_km": None,
                 "surface_km2": round(pk_m2 / 1e6, 4), "pct_of_park": round(100 * pk_m2 / park.area, 2),
                 "pct_width_tag": None, "pct_lanes_tag": None, "pct_default": None})
    return pd.DataFrame(rows), lines, surfaces, pk


def speeds(lines):
    parsed = col(lines, "maxspeed").apply(parse_speed)
    lines = lines.copy()
    lines["maxspeed_kmh"] = parsed.apply(lambda x: x[0])
    lines["speed_source"] = parsed.apply(lambda x: x[1])
    lines["speed_band"] = lines["maxspeed_kmh"].apply(speed_band)
    table = (lines.groupby(["category", "speed_band"])["length_m"].sum() / 1000).round(2).unstack(fill_value=0)
    order = [b for _, b in SPEED_BANDS] + ["no limit mapped"]
    table = table.reindex(columns=[c for c in order if c in table.columns])
    mapped = lines["maxspeed_kmh"].notna()
    coverage = 100 * lines.loc[mapped, "length_m"].sum() / (lines["length_m"].sum() or 1)
    over30 = lines.loc[lines["maxspeed_kmh"] > 30, "length_m"].sum() / 1000
    return lines, table, round(coverage, 1), round(over30, 2)


def pieces_of(roads_near, park):
    viaduct = roads_near["on_bridge"] & (roads_near.geometry.length >= VIADUCT_MIN_M)
    bar = roads_near[roads_near["category"].isin(BARRIER_CATS)
                     & ~roads_near["in_tunnel"] & ~viaduct]
    if not len(bar):
        rest = park
    else:
        # round caps seal the small gaps between consecutive ways
        barrier = bar.geometry.buffer(bar["width_m"] / 2 + 0.5, cap_style="round").union_all()
        rest = park.difference(barrier)
    p = gpd.GeoDataFrame(geometry=[rest], crs=roads_near.crs).explode(index_parts=False)
    p["area_m2"] = p.geometry.area
    p = p[p["area_m2"] >= MIN_PIECE_M2].sort_values("area_m2", ascending=False).reset_index(drop=True)
    p["piece_id"] = range(1, len(p) + 1)            # 1 = largest
    p["area_km2"] = (p["area_m2"] / 1e6).round(3)
    return p


def crossings_of(roads_near, paths_near, park, pieces):
    barriers = roads_near[roads_near["category"].isin(BARRIER_CATS)].reset_index(drop=True)
    others = roads_near[~roads_near["category"].isin(BARRIER_CATS)].copy()
    others["via"] = "road"
    paths = paths_near.copy()
    paths["via"] = "path"
    crossers = pd.concat([paths, others], ignore_index=True)
    crossers = gpd.GeoDataFrame(crossers[walkable(crossers)], crs=roads_near.crs).reset_index(drop=True)
    if not len(barriers) or not len(crossers):
        return gpd.GeoDataFrame(geometry=[], crs=roads_near.crs), pd.DataFrame()

    sj = gpd.sjoin(crossers[["geometry"]], barriers[["geometry"]], predicate="intersects")
    recs = []
    zone = park.buffer(20)
    for i, j in zip(sj.index, sj["index_right"]):
        c, b = crossers.loc[i], barriers.loc[j]
        for pt in points_of(c.geometry.intersection(b.geometry)):
            if not zone.contains(pt):
                continue
            separated = (tag_on(c.get("bridge")) or tag_on(c.get("tunnel")) or tag_on(c.get("covered"))
                         or tag_on(b.get("bridge")) or tag_on(b.get("tunnel"))
                         or layer_of(c.get("layer")) != layer_of(b.get("layer")))
            marked = (str(c.get("footway")).lower() == "crossing") or not missing(c.get("crossing"))
            how = None
            if separated:
                if (tag_on(c.get("bridge")) or tag_on(b.get("tunnel"))
                        or layer_of(c.get("layer")) > layer_of(b.get("layer"))):
                    how = "over"                      # footbridge / road bridge over the highway
                elif (tag_on(c.get("tunnel")) or tag_on(c.get("covered")) or tag_on(b.get("bridge"))
                        or layer_of(c.get("layer")) < layer_of(b.get("layer"))):
                    how = "under"                     # underpass
            recs.append({
                "geometry": pt, "via": c["via"],
                "type": "over/under" if separated else ("marked" if marked else "unmarked"),
                "how": how,
                "over_motorway": b["highway"] in MOTORWAY_TYPES,
                "crosser": c.get("name") or c["highway"],
                "barrier": b.get("ref") or b.get("name") or b["highway"],
            })
    if not recs:
        return gpd.GeoDataFrame(geometry=[], crs=roads_near.crs), pd.DataFrame()

    pts = gpd.GeoDataFrame(recs, geometry="geometry", crs=roads_near.crs)
    blobs = gpd.GeoSeries([pts.buffer(CLUSTER_M).union_all()], crs=pts.crs).explode(index_parts=False)
    blobs = gpd.GeoDataFrame({"cid": range(1, len(blobs) + 1)}, geometry=blobs.values, crs=pts.crs)
    pts = gpd.sjoin(pts, blobs, predicate="within").drop(columns="index_right")

    rank = {"over/under": 0, "marked": 1, "unmarked": 2}
    rows, links = [], []
    for cid, g in pts.groupby("cid"):
        centre = g.geometry.union_all().centroid
        near_pieces = sorted(pieces.loc[pieces.intersects(centre.buffer(CONNECT_M)), "piece_id"].tolist())
        ctype = min(g["type"], key=rank.get)
        via = "path" if (g["via"] == "path").any() else "road"
        rows.append({
            "cid": cid, "geometry": centre, "type": ctype, "via": via,
            "over_motorway": bool(g["over_motorway"].any()),
            "how": next((h for h in g["how"] if h), None),
            "crosses": ", ".join(sorted(set(map(str, g["barrier"])))),
            "along": ", ".join(sorted(set(map(str, g["crosser"])))[:3]),
            "links_pieces": ", ".join(map(str, near_pieces)),
        })
        for a, b in itertools.combinations(near_pieces, 2):
            links.append({"piece_a": a, "piece_b": b, "type": ctype, "via": via})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=pts.crs), pd.DataFrame(links)


# ── main ───────────────────────────────────────────────────────────────

def main():
    prepare("--refresh" in sys.argv)

    parks = gpd.read_file(WORK / "parks.geojson")
    cand = parks[col(parks, "name").str.contains(PARK_NAME, case=False, na=False)].copy()
    if not len(cand):
        raise SystemExit(f"No polygon named like '{PARK_NAME}' in the Lisbon extract.")
    utm = cand.estimate_utm_crs()
    cand = cand.to_crs(utm)
    cand["km2"] = cand.area / 1e6
    best = cand.sort_values("km2", ascending=False).iloc[0]
    park = best.geometry.buffer(0)
    print(f"\nPark: {best.get('name')} ({best.get('@type')}/{best.get('@id')}), {park.area/1e6:.2f} km²")

    roads = gpd.read_file(WORK / "roads.geojson").to_crs(utm)
    ok = roads["highway"].isin(HW_TO_CAT) \
        & (col(roads, "area") != "yes") & (col(roads, "access") != "private") \
        & (col(roads, "motor_vehicle") != "no") & (col(roads, "motorcar") != "no") \
        & ~((roads["highway"] == "service") & col(roads, "service").isin(SERVICE_EXCLUDED))
    near = gpd.clip(roads[ok], park.buffer(NEAR_M))
    near = near[~near.geometry.is_empty].copy()
    near["category"] = near["highway"].map(HW_TO_CAT)
    w = near.apply(road_width, axis=1, result_type="expand")
    near["width_m"], near["width_source"] = w[0], w[1]
    near["in_tunnel"] = col(near, "tunnel").apply(tag_on) | col(near, "covered").apply(tag_on)
    near["on_bridge"] = col(near, "bridge").apply(tag_on)

    parking = gpd.read_file(WORK / "parking.geojson").to_crs(utm)
    paths = gpd.read_file(WORK / "paths.geojson").to_crs(utm)
    paths_near = gpd.clip(paths, park.buffer(NEAR_M))
    paths_near = paths_near[~paths_near.geometry.is_empty]

    # 1. car space
    df, lines, surfaces, pk = car_space(near, parking, park)
    total_km2 = df["surface_km2"].sum()
    print("\n" + "=" * 100 + "\nCAR SPACE\n" + "=" * 100)
    print(f"{'Category':<15} {'Length km':>10} {'Tunnel km':>10} {'Surface km²':>12} {'% of park':>10}"
          f"   Width from: {'tag':>5} {'lanes':>6} {'default':>8}")
    print("-" * 100)
    for _, r in df.iterrows():
        if r["category"] == "parking":
            print(f"{r['category']:<15} {'':>10} {'':>10} {r['surface_km2']:>12.3f} {r['pct_of_park']:>9.2f}%")
        else:
            print(f"{r['category']:<15} {r['length_km']:>10.2f} {r['tunnel_km']:>10.2f} "
                  f"{r['surface_km2']:>12.3f} {r['pct_of_park']:>9.2f}%   {'':>11} "
                  f"{r['pct_width_tag']:>4}% {r['pct_lanes_tag']:>5}% {r['pct_default']:>7}%")
    print("-" * 100)
    print(f"{'TOTAL':<15} {df['length_km'].sum():>10.2f} {'':>10} {total_km2:>12.3f} "
          f"{100*total_km2*1e6/park.area:>9.2f}%")

    # 2. speed limits
    lines, speed_table, coverage, over30 = speeds(lines)
    print("\n" + "=" * 100 + "\nSPEED LIMITS (road km)\n" + "=" * 100)
    print(speed_table.to_string())
    print(f"\nRoad length with a speed limit mapped: {coverage}%")
    print(f"Road km with a limit above 30 km/h:    {over30} km")

    # 3. pieces
    pieces = pieces_of(near, park)
    largest = pieces["area_km2"].iloc[0] if len(pieces) else 0
    print("\n" + "=" * 100 + "\nPIECES (park minus at-grade highways, pieces of 1 ha or more)\n" + "=" * 100)
    print(f"Pieces: {len(pieces)}   Largest: {largest:.2f} km² "
          f"({100*largest*1e6/park.area:.0f}% of the park)")
    for _, p in pieces.head(10).iterrows():
        print(f"  piece {p['piece_id']:>2}  {p['area_km2']:>7.3f} km²")
    if len(pieces) > 10:
        print(f"  ... and {len(pieces) - 10} smaller pieces")

    # 4. crossings and connections
    crossings, links = crossings_of(near, paths_near, park, pieces)
    hw_km = lines.loc[lines["category"].isin(BARRIER_CATS) & ~lines["in_tunnel"], "length_m"].sum() / 1000
    summary_cross = {}
    print("\n" + "=" * 100 + "\nCROSSINGS OF HIGHWAYS INSIDE THE PARK\n" + "=" * 100)
    if len(crossings):
        real = crossings[crossings["type"] != "unmarked"]
        walk = real[real["via"] == "path"]
        mw_walk = walk[walk["over_motorway"]]
        print(crossings.groupby(["via", "type"]).size().unstack(fill_value=0).to_string())
        print(f"\nWalking crossings (paths, over/under or marked): {len(walk)}"
              f"   of which over a motorway: {len(mw_walk)}")
        print(f"Crossings along another road (over/under or marked): {len(real) - len(walk)}")
        print(f"Unmarked at-grade intersections to check: {int((crossings['type'] == 'unmarked').sum())}")
        per = f"{hw_km/len(walk):.1f} km" if len(walk) else "no walking crossing"
        print(f"Highway km inside the park: {hw_km:.1f}  ->  one walking crossing every {per}")
        print(f"\n  {'#':>3}  {'type':<11} {'via':<5} {'motorway':<9} {'crosses':<18} {'along':<30} links pieces")
        for _, c in crossings.sort_values(["type", "via"]).iterrows():
            print(f"  {c['cid']:>3}  {c['type']:<11} {c['via']:<5} {'yes' if c['over_motorway'] else '':<9} "
                  f"{c['crosses'][:17]:<18} {c['along'][:29]:<30} {c['links_pieces']}")
        if len(links):
            real_links = links[links["type"] != "unmarked"]
            pairs = real_links.groupby(["piece_a", "piece_b"]).agg(
                crossings=("type", "size"), walking=("via", lambda s: int((s == "path").sum()))).reset_index()
            print("\nConnections between pieces (over/under or marked crossings):")
            for _, r in pairs.iterrows():
                a = pieces.loc[pieces["piece_id"] == r["piece_a"], "area_km2"].iloc[0]
                b = pieces.loc[pieces["piece_id"] == r["piece_b"], "area_km2"].iloc[0]
                print(f"  piece {r['piece_a']} ({a:.2f} km²) <-> piece {r['piece_b']} ({b:.2f} km²): "
                      f"{r['crossings']} crossing(s), {r['walking']} on paths")
        summary_cross = {
            "walking_crossings": int(len(walk)),
            "walking_crossings_over_motorway": int(len(mw_walk)),
            "road_crossings": int(len(real) - len(walk)),
            "unmarked_to_check": int((crossings["type"] == "unmarked").sum()),
            "highway_km_per_walking_crossing": round(hw_km / len(walk), 2) if len(walk) else None,
        }
    else:
        print("No crossings found.")

    # save
    summary = {
        "park": best.get("name"), "osm": f"{best.get('@type')}/{best.get('@id')}",
        "park_area_km2": round(park.area / 1e6, 3),
        "car_space_km2": round(total_km2, 3),
        "car_space_pct": round(100 * total_km2 * 1e6 / park.area, 2),
        "road_km": round(df["length_km"].sum(), 2),
        "football_fields": round(total_km2 * 1e6 / (105 * 68)),
        "by_category": df.set_index("category")[["length_km", "surface_km2", "pct_of_park"]]
                         .where(pd.notna(df.set_index("category")[["length_km", "surface_km2", "pct_of_park"]]), None)
                         .to_dict(orient="index"),
        "speed_km_by_band": speed_table.sum().round(2).to_dict(),
        "speed_limit_coverage_pct": coverage,
        "road_km_above_30": over30,
        "highway_km": round(hw_km, 2),
        "pieces": int(len(pieces)),
        "largest_piece_km2": float(largest),
        **summary_cross,
    }
    (BASE / "monsanto_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    df.to_csv(BASE / "monsanto_by_category.csv", index=False)
    if len(crossings):
        crossings.drop(columns="geometry").to_csv(BASE / "monsanto_crossings.csv", index=False)

    parts = [gpd.GeoDataFrame({"layer": ["park"]}, geometry=[park], crs=utm)]
    for cat, surf in surfaces.items():
        parts.append(gpd.GeoDataFrame({"layer": ["road_surface"], "category": [cat]}, geometry=[surf], crs=utm))
    if pk is not None and not pk.is_empty:
        parts.append(gpd.GeoDataFrame({"layer": ["parking"], "category": ["parking"]}, geometry=[pk], crs=utm))
    ln = lines[["highway", "category", "width_m", "width_source", "in_tunnel",
                "maxspeed_kmh", "speed_band", "geometry"]].copy()
    ln["layer"] = "road_line"
    parts.append(ln)
    pc = pieces[["piece_id", "area_km2", "geometry"]].copy()
    pc["layer"] = "piece"
    parts.append(pc)
    if len(crossings):
        cr = crossings[["cid", "type", "via", "over_motorway", "crosses", "along", "links_pieces", "geometry"]].copy()
        cr["layer"] = "crossing"
        parts.append(cr)
    out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=utm).to_crs("EPSG:4326")
    out.to_file(OUT_DIR / f"{SLUG}.geojson", driver="GeoJSON", COORDINATE_PRECISION=6)
    print(f"\nSaved monsanto_summary.json, monsanto_by_category.csv, monsanto_crossings.csv")
    print(f"      {OUT_DIR.name}/{SLUG}.geojson (layers: park, road_surface, parking, road_line, piece, crossing)")


if __name__ == "__main__":
    main()
