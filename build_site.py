"""
build_site.py
===============
Builds the "Parks and cars" page from the measurement results, so the page
always reflects the data: add or drop a park, re-run the measurement, re-run
this script.

Reads
  run/parks_results.csv      one row per park (from run_all_parks.py)
  run/geojson/<slug>.geojson per-park map layers (from run_all_parks.py)
  overrides.json             parks with "skip": true are left out
  site_template.html         the page (layout, text, scripts)

Writes
  docs/index.html               the page (GitHub Pages serves the docs/ folder)
  docs/data/maps/<slug>.json    one compact map file per park, loaded only when
                                someone opens that park
  preview/parks-and-cars-single.html   with --single-file: one self-contained file
                                with all maps embedded (for previews; not committed)

Map files are simplified for the web: coordinates rounded to about 1 m, lines
and outlines simplified with a SIMPLIFY_M tolerance, road lines replaced by
precomputed speed-limit sign positions (rounded to the nearest
5 km/h, so limits converted from mph read naturally, e.g. 35 mph -> 55 km/h).

Run: python build_site.py [--single-file]
Only uses the Python standard library.
"""

import argparse
import csv
import json
import math
from pathlib import Path

BASE      = Path(__file__).resolve().parent
RESULTS   = BASE / "run" / "parks_results.csv"
GEO_IN    = BASE / "run" / "geojson"
CYC_IN    = BASE / "run" / "cycling"                # from cycling_infra.py (optional)
CYC_VERSION = 3                                    # cycling results older than this are ignored
CYC_MAP_CLASSES = {"separated", "shared", "painted"}   # what's drawn and counted
CYC_CSV   = BASE / "run" / "cycling_results.csv"

# ── classification (change these and every label, filter and sentence follows) ──
CAR_LOW_PCT      = 1.0    # under this share of the park given to cars: "mostly car-free"
CAR_HEAVY_PCT    = 4.0    # this share or more: "car-heavy"; in between: "some car infrastructure"
BIKES_SHARE_PER100 = 50   # km of cycling per 100 km of road for "bikes get a share"
MIN_ROAD_KM_RATIO  = 5    # parks with less road show cycling km but no ratio
OVERRIDES = BASE / "overrides.json"
TEMPLATE  = BASE / "site_template.html"
OUT       = BASE / "docs"                  # served by GitHub Pages
PREVIEW   = BASE / "preview"               # single-file preview (not committed)
MAPS_OUT  = OUT / "data" / "maps"
SPEED_OUT = OUT / "data" / "speed"          # per-park speed-limit lines, loaded when the Speed limits layer is switched on

HOME_SLUG   = "084-lisbon-monsanto-forest-park"   # the park the story is about
WIKI_LIST_SIZE = 193   # parks ranked 1 to 193 come from Wikipedia's list; higher ranks were added by hand
SIMPLIFY_M  = 1.5                                  # outline simplification tolerance
# ── speed limits (change these and the page follows) ──
SPEED_EDGES = (32.5, 52.0, 72.5)   # km/h upper edges of the bands: safe (30 km/h or 20 mph), 31-50, 51-70, over 70
SPEED_MIN_COVER_PCT = 25           # share of road length with a mapped limit needed to compare shares between parks
SPEED_MIN_KM = 3                   # ... and at least this many km of road with a limit mapped
SPEED_SIMPLIFY_M = 2.5             # tolerance for the speed lines
# Pedestrian risk: chance (%) that a person on foot dies when hit at this speed (km/h). These are the values reported by
# Hussain et al. (2019), the meta-analysis of 20 studies that the WHO speed management manual (2023) cites for its pedestrian
# figures. We read the risk straight off these points and join them with straight lines; below 30 km/h we use 5%, above 80
# km/h we use 90% (the highest value reported, which understates the risk of the fastest limits).
SPEED_RISK_POINTS = ((30, 5), (37, 10), (59, 50), (69, 75), (80, 90))
# Access-controlled roads where people on foot and bike are not allowed (Guide for Safe Speeds, 2024: motorways have no
# intersections or traffic lights). Their risk to pedestrians is not rated. Trunk roads are rated: they can have crossings.
SPEED_UNRATED = ("motorway", "motorway_link")
RISK_SAFE_BELOW = 10               # average risk under this %: "mostly safe speeds" (the published risk at 37 km/h)
EDIT_ZOOM = 20                     # the editor link for one road opens this close, so the road is unmistakable


def risk_at(kmh):
    pts = SPEED_RISK_POINTS
    if kmh <= pts[0][0]:
        return float(pts[0][1])
    if kmh >= pts[-1][0]:
        return float(pts[-1][1])
    for (a, ra), (b, rb) in zip(pts, pts[1:]):
        if a <= kmh <= b:
            return ra + (rb - ra) * (kmh - a) / (b - a)


RISK_UNSAFE_FROM = round(risk_at(50))   # average risk at least that of a road posted at 50 km/h (Safe System limit where side impacts are possible): "heavy on unsafe speeds"
OSM_HASHTAG = "#urbanparksandcars" # prefilled in the OpenStreetMap editor link, so edits can be traced
HW_SIMPLIFY_M = 1.0                                # tolerance for highway centre lines (drawn at real width)
REGION_CODES = {"Europe": "eu", "North America": "na", "South America": "sa",
                "Asia": "as", "Africa": "af", "Oceania": "oc"}
SPEED_CAT = {"highways": "h", "main roads": "m", "local roads": "l", "service roads": "s"}   # line thickness on the speed layer
CATEGORY_CODES = {"highways": "highways", "main roads": "main", "local roads": "local",
                  "service roads": "service"}
CROSSING_TYPES = {"over/under": "grade", "marked": "marked", "unmarked": "unmarked"}


# ── helpers ────────────────────────────────────────────────────────────

def truthy(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def number(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


class Local:
    """Equirectangular projection around a park, in metres (good enough for
    simplification and sign spacing inside one park)."""
    def __init__(self, lat0):
        self.kx = 111_320 * math.cos(math.radians(lat0))
        self.ky = 110_574

    def xy(self, c):
        return (c[0] * self.kx, c[1] * self.ky)

    def dist(self, a, b):
        (x1, y1), (x2, y2) = self.xy(a), self.xy(b)
        return math.hypot(x2 - x1, y2 - y1)


def simplify(points, proj, tol):
    """Douglas-Peucker on a list of [lon, lat]; keeps the first and last point."""
    if len(points) < 3:
        return points
    pts = [proj.xy(p) for p in points]
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        (x1, y1), (x2, y2) = pts[a], pts[b]
        dx, dy = x2 - x1, y2 - y1
        seg = dx * dx + dy * dy
        best, idx = 0.0, None
        for i in range(a + 1, b):
            px, py = pts[i]
            if seg == 0:
                d = math.hypot(px - x1, py - y1)
            else:
                t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / seg))
                d = math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))
            if d > best:
                best, idx = d, i
        if idx is not None and best > tol:
            keep[idx] = True
            stack += [(a, idx), (idx, b)]
    return [p for p, k in zip(points, keep) if k]


def rnd(p):
    return [round(p[0], 5), round(p[1], 5)]


def simplify_geom(geom, proj, tol=SIMPLIFY_M):
    t, c = geom["type"], geom["coordinates"]

    def ring(r):
        s = simplify(r, proj, tol)
        return [rnd(p) for p in (s if len(s) >= 4 else r)]

    def line(l):
        return [rnd(p) for p in simplify(l, proj, tol)]

    if t == "Polygon":
        return {"type": t, "coordinates": [ring(r) for r in c]}
    if t == "MultiPolygon":
        return {"type": t, "coordinates": [[ring(r) for r in poly] for poly in c]}
    if t == "LineString":
        return {"type": t, "coordinates": line(c)}
    if t == "MultiLineString":
        return {"type": t, "coordinates": [line(l) for l in c]}
    if t == "Point":
        return {"type": t, "coordinates": rnd(c)}
    return geom


def first_coord(geom):
    c = geom["coordinates"]
    while isinstance(c[0], list):
        c = c[0]
    return c


# ── speed-limit signs ─────────────────────────────────────────────────

def midpoint(line, proj):
    segs = [proj.dist(line[i], line[i + 1]) for i in range(len(line) - 1)]
    total, acc = sum(segs), 0.0
    for i, s in enumerate(segs):
        if acc + s >= total / 2 and s > 0:
            f = (total / 2 - acc) / s
            return total, [line[i][0] + f * (line[i + 1][0] - line[i][0]),
                           line[i][1] + f * (line[i + 1][1] - line[i][1])]
        acc += s
    return total, line[0]


def place_signs(lines, proj):
    """Tier 1 signs are widely spaced (shown at every zoom); tier 2 fill in
    when zoomed. Longest roads get their sign first."""
    cands = []
    for kmh, geom in lines:
        parts = [geom["coordinates"]] if geom["type"] == "LineString" else geom["coordinates"]
        for part in parts:
            if len(part) >= 2:
                length, mid = midpoint(part, proj)
                cands.append({"v": kmh, "len": length, "pt": mid})
    cands.sort(key=lambda c: -c["len"])
    placed = []
    for tier, same_gap, any_gap, min_len in ((1, 900, 250, 120), (2, 320, 110, 40)):
        for c in cands:
            if c["len"] < min_len or c.get("used"):
                continue
            if any(proj.dist(c["pt"], s["pt"]) < any_gap or
                   (s["v"] == c["v"] and proj.dist(c["pt"], s["pt"]) < same_gap) for s in placed):
                continue
            c["used"] = True
            placed.append({"v": c["v"], "pt": c["pt"], "t": tier})
    return [{"v": s["v"], "t": s["t"], "lon": round(s["pt"][0], 5), "lat": round(s["pt"][1], 5)} for s in placed]


# ── per-park map file ─────────────────────────────────────────────────

def clean_text(v):
    t = "" if v is None else str(v).strip()
    return "" if t.lower() in ("nan", "none", "null") else t


def speed_band(kmh):
    """x = no limit mapped; 1 = safe (30 km/h or 20 mph or less); 2, 3, 4 = faster bands."""
    if not kmh:
        return "x"
    return "1" if kmh <= SPEED_EDGES[0] else ("2" if kmh <= SPEED_EDGES[1] else ("3" if kmh <= SPEED_EDGES[2] else "4"))


def edit_start(feats, geom, proj):
    """Where the park-level editor link opens: the middle of the longest main or local road with no speed limit mapped
    (service roads if there is none), close enough to see which road it is: [lon, lat, zoom]."""
    def longest(cats):
        best, best_len = None, 0.0
        for f in feats:
            if f["b"] != "x" or f["c"] not in cats:
                continue
            for line in f["g"]:
                n = sum(proj.dist(line[i], line[i + 1]) for i in range(len(line) - 1))
                if n > best_len:
                    best, best_len = line, n
        return best
    line = longest(("m", "l")) or longest(("s",))
    if line:
        lon, lat = line[len(line) // 2]
    else:                                         # nothing to map: the middle of the park
        xs, ys = [], []

        def walk(c):
            if c and isinstance(c[0], (int, float)):
                xs.append(c[0]); ys.append(c[1])
            else:
                for x in c:
                    walk(x)
        walk(geom["coordinates"])
        lon, lat = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    return [round(lon, 5), round(lat, 5), EDIT_ZOOM - 1]


def compact_map(path):
    g = json.loads(path.read_text())
    feats = [f for f in g["features"] if f.get("geometry") and f["geometry"].get("coordinates")]
    park = next((f for f in feats if f["properties"].get("layer") == "park"), None)
    if park is None:
        return None
    proj = Local(first_coord(park["geometry"])[1])
    out = {"park": {"type": "Feature", "properties": {}, "geometry": simplify_geom(park["geometry"], proj)},
           "pieces": {"type": "FeatureCollection", "features": []},
           "surfaces": {"type": "FeatureCollection", "features": []},
           "parking": None, "crossings": [], "signs": []}
    speed_lines = []
    hw_groups = {}               # highway centre lines grouped by width (m), drawn one band per carriageway
    max_speed = {"hw": 0, "road": 0}    # highest mapped limit on highways / on other roads
    ref_km = {}                  # km of each numbered highway inside the park (e.g. "A 5")
    spd_feats, spd_km = [], {}                   # speed layer: one record per road; km by band (rated roads only)
    road_all = mw_km = rated_all = risk_sum = 0.0
    for f in feats:
        p, geom, layer = f["properties"], f["geometry"], f["properties"].get("layer")
        if layer == "piece":
            out["pieces"]["features"].append({"type": "Feature",
                "properties": {"id": int(number(p.get("piece_id")) or 0), "km2": number(p.get("area_km2"))},
                "geometry": simplify_geom(geom, proj)})
        elif layer == "road_surface" and p.get("category") in CATEGORY_CODES:
            out["surfaces"]["features"].append({"type": "Feature",
                "properties": {"c": CATEGORY_CODES[p["category"]]}, "geometry": simplify_geom(geom, proj)})
        elif layer == "parking":
            out["parking"] = {"type": "Feature", "properties": {}, "geometry": simplify_geom(geom, proj)}
        elif layer == "road_line":
            ref = (p.get("ref") or "").strip()
            if ref and p.get("category") == "highways" and geom["type"] in ("LineString", "MultiLineString"):
                parts = [geom["coordinates"]] if geom["type"] == "LineString" else geom["coordinates"]
                km = sum(proj.dist(l[i], l[i + 1]) for l in parts for i in range(len(l) - 1)) / 1000
                # a one-way way is one carriageway: two of them make one km of motorway
                ref_km[ref] = ref_km.get(ref, 0) + (km / 2 if truthy(p.get("oneway")) else km)
            if (p.get("category") == "highways" and not truthy(p.get("in_tunnel"))
                    and geom["type"] in ("LineString", "MultiLineString")):
                wm = number(p.get("width_m"))
                if wm:
                    key = round(round(wm * 4) / 4, 2)            # width buckets of 0.25 m
                    parts = [geom["coordinates"]] if geom["type"] == "LineString" else geom["coordinates"]
                    for part in parts:
                        if len(part) >= 2:
                            hw_groups.setdefault(key, []).append([rnd(c) for c in simplify(part, proj, HW_SIMPLIFY_M)])
            kmh = number(p.get("maxspeed_kmh"))
            if not truthy(p.get("in_tunnel")) and geom["type"] in ("LineString", "MultiLineString"):
                parts_ = [geom["coordinates"]] if geom["type"] == "LineString" else geom["coordinates"]
                km_ = sum(proj.dist(l[i], l[i + 1]) for l in parts_ for i in range(len(l) - 1)) / 1000
                hwy = clean_text(p.get("highway"))
                road_all += km_
                if hwy in SPEED_UNRATED:                 # access-controlled: drawn, but not rated
                    band = "m"; mw_km += km_
                else:
                    band = speed_band(kmh); rated_all += km_
                    spd_km[band] = spd_km.get(band, 0.0) + km_
                    if kmh:
                        risk_sum += km_ * risk_at(kmh)
                lines = [[rnd(c) for c in simplify(l, proj, SPEED_SIMPLIFY_M)] for l in parts_ if len(l) >= 2]
                lines = [l for l in lines if len(l) >= 2]
                if lines:
                    rec = {"b": band, "c": SPEED_CAT.get(p.get("category"), "l"), "g": lines}
                    if kmh:
                        rec["k"] = int(round(kmh))
                    for key, val in (("t", hwy), ("n", clean_text(p.get("name"))), ("r", clean_text(p.get("ref")))):
                        if val:
                            rec[key] = val
                    if truthy(p.get("oneway")):
                        rec["o"] = 1
                    spd_feats.append(rec)
            if kmh:
                k = "hw" if p.get("category") == "highways" else "road"
                max_speed[k] = max(max_speed[k], int(5 * round(kmh / 5)))
            if kmh and not truthy(p.get("in_tunnel")) and geom["type"] in ("LineString", "MultiLineString"):
                # nearest 5 km/h: limits converted from mph (25 mph = 40.2) read naturally
                speed_lines.append((int(5 * round(kmh / 5)), geom))
        elif layer == "crossing":
            out["crossings"].append({
                "t": CROSSING_TYPES.get(p.get("type"), "unmarked"), "v": p.get("via"), "h": p.get("how"),
                "mw": truthy(p.get("over_motorway")), "x": p.get("crosses") or "", "a": p.get("along") or "",
                "lp": [int(float(x)) for x in str(p.get("links_pieces") or "").split(",")
                       if x.strip() not in ("", "nan", "None")],
                "lon": round(geom["coordinates"][0], 5), "lat": round(geom["coordinates"][1], 5)})
    out["pieces"]["features"].sort(key=lambda f: f["properties"]["id"])
    if hw_groups:
        out["hwl"] = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"w": w},
             "geometry": {"type": "MultiLineString", "coordinates": lines}}
            for w, lines in sorted(hw_groups.items())]}
        # the merged highway shape is no longer drawn when the lines are there
        out["surfaces"]["features"] = [f for f in out["surfaces"]["features"] if f["properties"]["c"] != "highways"]
    out["signs"] = place_signs(speed_lines, proj)
    out["refs"] = {k: round(v, 2) for k, v in ref_km.items()}
    out["maxspd"] = {k: v or None for k, v in max_speed.items()}
    mapped_km = sum(v for b, v in spd_km.items() if b != "x")
    out["spd"] = {"road": round(road_all, 2), "mw": round(mw_km, 2), "rated": round(rated_all, 2), "mapped": round(mapped_km, 2),
                  **{f"b{b}": round(spd_km.get(b, 0.0), 2) for b in "1234"}, "risk": round(risk_sum, 3)}
    out["edit"] = edit_start(spd_feats, park["geometry"], proj)
    out["_speed"] = {"st": out["spd"], "l": spd_feats}
    return out


# ── park table ────────────────────────────────────────────────────────

_cyc_rows = None


def cycling_record(slug):
    """Cycling figures for a park: from run/cycling_results.csv, else run/cycling/<slug>.json."""
    global _cyc_rows
    if _cyc_rows is None:
        _cyc_rows = {}
        if CYC_CSV.exists():
            with open(CYC_CSV, newline="", encoding="utf-8") as f:
                _cyc_rows = {r["slug"]: r for r in csv.DictReader(f)}
    c = _cyc_rows.get(slug)
    if c is None:
        f = CYC_IN / f"{slug}.json"
        if not f.exists():
            return None
        c = json.loads(f.read_text())
    if c.get("status") != "ok" or (number(c.get("method_version")) or 0) < CYC_VERSION:
        return None
    c = {k: (number(v) if k != "status" else v) for k, v in c.items()}
    keys = ["separated_km", "shared_km", "painted_km", "advisory_km", "sharrow_km", "dedicated_km",
            "dedicated_per_road_km", "ecf_ratio_main_roads_pct", "main_road_km"]
    return {k: c.get(k) for k in keys}


def cycling_layer(slug):
    """Compact cycling lines for the map, or None."""
    f = CYC_IN / f"{slug}.geojson"
    if not f.exists():
        return None
    g = json.loads(f.read_text())
    feats = [x for x in g.get("features", []) if x.get("geometry") and x["geometry"].get("coordinates")
             and x["properties"].get("cls") in CYC_MAP_CLASSES
             and x["geometry"]["type"] in ("LineString", "MultiLineString")]
    if not feats:
        return {"type": "FeatureCollection", "features": []}
    proj = Local(first_coord(feats[0]["geometry"])[1])
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"c": x["properties"]["cls"], **({"n": clean_text(x["properties"].get("name"))} if clean_text(x["properties"].get("name")) else {})},
         "geometry": simplify_geom(x["geometry"], proj)} for x in feats]}


def classify(pct, road_km, cyc):
    """Car group from car space; cycling tag only for parks with car infrastructure."""
    group = "free" if pct < CAR_LOW_PCT else ("heavy" if pct >= CAR_HEAVY_PCT else "some")
    per100 = None
    if cyc is not None:
        ded = cyc.get("dedicated_km") or 0
        if road_km and road_km > 0:
            per100 = 100 * ded / road_km
        elif ded > 0:
            per100 = float("inf")                   # cycling but no road at all
        else:
            per100 = 0.0
    tag = None
    if group != "free" and per100 is not None:
        tag = "share" if per100 >= BIKES_SHARE_PER100 else "little"
    shown = per100 if (per100 is not None and road_km and road_km >= MIN_ROAD_KM_RATIO) else None
    return group, tag, (round(shown, 1) if shown is not None else None)


def park_record(r):
    n = number
    return {
        "id": r["slug"], "name": r["name"], "city": r["city"], "country": r["country"],
        "region": REGION_CODES.get(r.get("continent"), "other"),
        "km2": n(r["park_area_km2"]), "wiki_km2": n(r["size_km2_wiki"]),
        "pct": n(r["car_space_pct"]), "car_km2": n(r["car_space_km2"]),
        "road_km": n(r["road_km"]), "fields": n(r["football_fields"]),
        "cat_km2": {"highways": n(r["highways_km2"]), "main": n(r["main_km2"]), "local": n(r["local_km2"]),
                    "service": n(r["service_km2"]), "parking": n(r["parking_km2"])},
        "cat_km": {"highways": n(r["highways_km"]), "main": n(r["main_km"]), "local": n(r["local_km"]),
                   "service": n(r["service_km"])},
        "cat_pct": {"highways": n(r["highways_pct"]), "main": n(r["main_pct"]), "local": n(r["local_pct"]),
                    "service": n(r["service_pct"]), "parking": n(r["parking_pct"])},
        "hw_km": n(r["highway_km"]), "pieces": int(n(r["pieces"]) or 0),
        "largest_km2": n(r["largest_piece_km2"]), "second_km2": n(r["second_piece_km2"]),
        "walk": int(n(r["walking_crossings"]) or 0),
        "walk_mw": int(n(r["walking_crossings_over_motorway"]) or 0),
        "walk_between": int(n(r["walking_crossings_between_two_largest"]) or 0),
        "road_cross": int(n(r["road_crossings"]) or 0),
        "hw_per_walk": n(r["highway_km_per_walking_crossing"]),
        "speed_cov": n(r["speed_limit_coverage_pct"]), "above30_km": n(r["road_km_above_30"]),
        "osm_date": r.get("osm_data_date") or None,
        "cyc": cycling_record(r["slug"]),
        "added": int(r["slug"].split("-")[0]) > WIKI_LIST_SIZE,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single-file", action="store_true",
                    help="also write one self-contained HTML with all maps embedded")
    args = ap.parse_args()

    overrides = json.loads(OVERRIDES.read_text()) if OVERRIDES.exists() else {}
    skipped = {s for s, e in overrides.items() if e.get("skip")}
    with open(RESULTS, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status") == "ok" and r["slug"] not in skipped]
    parks = [park_record(r) for r in rows]
    for p in parks:
        p["group"], p["bikes"], p["cyc_per100"] = classify(p["pct"] or 0, p["road_km"], p["cyc"])
    print(f"Parks: {len(parks)} (skipped in overrides.json: {len(skipped)})")

    MAPS_OUT.mkdir(parents=True, exist_ok=True)
    included = {p["id"] for p in parks}
    for old in MAPS_OUT.glob("*.json"):          # drop maps of parks no longer included
        if old.stem not in included:
            old.unlink()
    SPEED_OUT.mkdir(parents=True, exist_ok=True)
    for old in SPEED_OUT.glob("*.json"):
        if old.stem not in included:
            old.unlink()
    maps, speed, total, kept = {}, {}, 0, 0
    for p in parks:
        src, dst = GEO_IN / f"{p['id']}.geojson", MAPS_OUT / f"{p['id']}.json"
        m = None
        spd_layer = None
        if src.exists():
            m = compact_map(src)
            spd_layer = m.pop("_speed", None) if m else None
        elif dst.exists():
            kept += 1                           # layers not on this machine: keep the published map
            m = json.loads(dst.read_text(encoding="utf-8"))
        if m is not None:
            cyc = cycling_layer(p["id"])
            if cyc is not None:
                m["cycling"] = cyc
            dst.write_text(json.dumps(m, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
        sp_file = SPEED_OUT / f"{p['id']}.json"
        if spd_layer is not None:
            sp_file.write_text(json.dumps(spd_layer, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
        if sp_file.exists():
            speed[p["id"]] = json.loads(sp_file.read_text(encoding="utf-8"))
        p["has_map"] = dst.exists()
        p["has_spd"] = sp_file.exists()
        p["spd"], p["edit"] = (m or {}).get("spd"), (m or {}).get("edit")
        spd = (m or {}).get("maxspd") or {}
        p["max_hw"], p["max_road"] = spd.get("hw"), spd.get("road")
        if dst.exists():
            text = dst.read_text(encoding="utf-8")
            maps[p["id"]] = json.loads(text)
            total += len(text)
    if kept:
        print(f"Kept {kept} existing map files (their layers aren't in run/geojson here)")
    print(f"Cycling: {sum(1 for p in parks if p.get('cyc'))} parks measured")
    from collections import Counter
    print("Groups:", dict(Counter(p["group"] for p in parks)), "| cycling tags:", dict(Counter(p["bikes"] for p in parks if p["bikes"])))
    print(f"Maps: {len(maps)} files, {total/1e6:.1f} MB in {MAPS_OUT.relative_to(BASE)}")
    print(f"Speed layers: {len(speed)} files, {sum(p.stat().st_size for p in SPEED_OUT.glob('*.json'))/1e6:.1f} MB in {SPEED_OUT.relative_to(BASE)}")

    dates = sorted(p["osm_date"] for p in parks if p.get("osm_date"))
    site = {"home": HOME_SLUG, "parks": parks, "skipped": len(skipped),
            "rules": {"car_low": CAR_LOW_PCT, "car_heavy": CAR_HEAVY_PCT,
                      "bikes_share": BIKES_SHARE_PER100, "min_road_km": MIN_ROAD_KM_RATIO,
                      "speed": {"min_cover": SPEED_MIN_COVER_PCT, "min_km": SPEED_MIN_KM, "safe_kmh": 30,
                                "risk_safe_below": RISK_SAFE_BELOW, "risk_unsafe_from": RISK_UNSAFE_FROM,
                                "risk_points": [list(x) for x in SPEED_RISK_POINTS], "unrated": list(SPEED_UNRATED)},
                      "edit_zoom": EDIT_ZOOM,
                      "hashtag": OSM_HASHTAG},
            "dates": [dates[0], dates[-1]] if dates else None}
    print(f"OpenStreetMap data dates: {dates[0]} to {dates[-1]}" if dates else "No data dates recorded yet")
    template = TEMPLATE.read_text(encoding="utf-8")
    site_json = json.dumps(site, separators=(",", ":"), ensure_ascii=False)
    page = template.replace("__SITE__", site_json).replace("__MAPS__", "{}").replace("__MAPS_URL__", "data/maps/") \
                .replace("__SPEED__", "{}").replace("__SPEED_URL__", "data/speed/")
    (OUT / "index.html").write_text(page, encoding="utf-8")
    (OUT / ".nojekyll").write_text("")          # serve files as they are, no Jekyll build
    print(f"Page: {(OUT / 'index.html').relative_to(BASE)} ({len(page)/1e3:.0f} KB)")

    if args.single_file:
        maps_json = json.dumps(maps, separators=(",", ":"), ensure_ascii=False)
        speed_json = json.dumps(speed, separators=(",", ":"), ensure_ascii=False)
        single = template.replace("__SITE__", site_json).replace("__MAPS__", maps_json).replace("__MAPS_URL__", "") \
                         .replace("__SPEED__", speed_json).replace("__SPEED_URL__", "")
        PREVIEW.mkdir(exist_ok=True)
        path = PREVIEW / "parks-and-cars-single.html"
        path.write_text(single, encoding="utf-8")
        print(f"Single file: {path.relative_to(BASE)} ({len(single)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
