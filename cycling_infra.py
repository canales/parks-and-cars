"""
cycling_infra.py
==================
Measures dedicated cycling infrastructure inside every park, with the same
boundaries and extracts as the car-space measurement. Writes its own outputs
only: the published results and the site are not touched.

Definitions follow the European Cyclists' Federation's QECIO 2.1 methodology
(Quantifying Europe's Cycling Infrastructure using OSM), so our figures can be
compared with theirs:
https://european-cyclists-federation.github.io/Documents/Methodology.pdf

What counts
  separated   ECF "cycle tracks": highway=cycleway unless shared with pedestrians
              (foot=designated and not segregated=yes); footway/path/pedestrian
              with bicycle=designated and segregated=yes; a road with
              cycleway(:side)=track or opposite_track
  shared      ECF "cycle and pedestrian tracks": footway/path/pedestrian with
              bicycle=designated and not segregated; highway=cycleway with
              foot=designated and not segregated
  painted     ECF "cycle lanes": a road with cycleway(:side)=lane or opposite_lane
Reported separately, never added to the totals
  advisory    painted lanes cars may use (cycleway(:side):lane=advisory). ECF
              counts these as cycle lanes; we don't, because they aren't
              reserved for bikes. This is our one deliberate difference from ECF.
  sharrow     cycleway(:side)=shared_lane (ECF doesn't count these either)
Not counted: bus lanes (share_busway), shoulders, plain permission to cycle
(bicycle=yes), highway=track and bridleways, and anything with private access.

Lengths (as in ECF): each side of a road with a lane or track is one feature;
one-way infrastructure counts half its length, so a one-way lane on each side
of a road adds up to the road's length. Standalone paths count in full unless
one-way. All lengths are clipped to the park.

Ratios
  dedicated_per_road_km       dedicated km / km of all roads in the park (ours)
  ecf_ratio_main_roads_pct    dedicated km / km of main roads x 100, main roads
                              being motorway to tertiary with links (ECF's
                              "segregated cycling infrastructure to main roads")

Outputs
  run/cycling_results.csv          one row per park
  run/cycling/<slug>.json          per park (makes the run resumable)
  run/cycling/<slug>.geojson       the cycling lines, by type (for a map layer later)

Usage
  python cycling_infra.py --only 084-lisbon-monsanto-forest-park   test on one park
  python cycling_infra.py                                          all parks (resumable)
  python cycling_infra.py --delete-pbf                             delete each extract after use
  python cycling_infra.py --refresh                                re-measure parks already done
"""

import argparse
import json
import time

import geopandas as gpd
import pandas as pd

import park_metrics as core
import run_all_parks as rp

OUT_DIR = rp.RUN / "cycling"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = rp.RUN / "cycling_results.csv"

ROAD_TYPES = set(core.HW_TO_CAT)
PATH_TYPES = {"path", "footway", "pedestrian"}                     # ECF's set
MAIN_ROADS = ["motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
              "secondary", "secondary_link", "tertiary", "tertiary_link"]      # ECF "main roads"
SIDE_KEYS = ["cycleway", "cycleway:both", "cycleway:left", "cycleway:right"]
OSMIUM_FILTERS = ["w/highway=cycleway", "w/bicycle=designated"] + [f"w/{k}" for k in SIDE_KEYS]
METHOD_VERSION = 3   # bump when definitions change: older results are re-measured automatically
CLASSES = ["separated", "shared", "painted", "advisory", "sharrow"]
COUNTED = ["separated", "shared", "painted"]


def tag(row, key):
    v = row.get(key)
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip().lower()


def public(row):
    """ECF's access check: leave out cycling infrastructure that isn't public."""
    access = tag(row, "access")
    if access in ("private", "no"):
        return tag(row, "bicycle") in ("yes", "designated", "permissive")
    return True


def classify(row):
    """List of (class, length factor) features for one OSM way; empty if nothing counts."""
    if not public(row):
        return []
    hw = tag(row, "highway")
    one_way = tag(row, "oneway") in ("yes", "1", "-1") or tag(row, "oneway:bicycle") == "yes"
    factor = 0.5 if one_way else 1.0
    not_segregated = tag(row, "segregated") != "yes"
    if hw == "cycleway":
        shared = tag(row, "foot") == "designated" and not_segregated
        return [("shared" if shared else "separated", factor)]
    if hw in PATH_TYPES and tag(row, "bicycle") == "designated":
        return [("shared" if not_segregated else "separated", factor)]
    if hw not in ROAD_TYPES:
        return []

    # one feature per side of the road that has a lane or track
    road_oneway = tag(row, "oneway") in ("yes", "1", "-1")
    sides = {}
    plain = tag(row, "cycleway")
    if plain:
        for side in (["right"] if road_oneway else ["left", "right"]):
            sides[side] = (plain, "cycleway")
    if tag(row, "cycleway:both"):
        for side in ("left", "right"):
            sides[side] = (tag(row, "cycleway:both"), "cycleway:both")
    for side in ("left", "right"):
        if tag(row, f"cycleway:{side}"):
            sides[side] = (tag(row, f"cycleway:{side}"), f"cycleway:{side}")

    features = []
    for side, (value, key) in sides.items():
        v = value.split(";")[0]
        two_way = tag(row, f"{key}:oneway") == "no" or tag(row, f"cycleway:{side}:oneway") == "no"
        f = 1.0 if two_way else 0.5
        if v in ("track", "opposite_track"):
            features.append(("separated", f))
        elif v in ("lane", "opposite_lane"):
            advisory = "advisory" in (tag(row, f"{key}:lane"), tag(row, f"cycleway:{side}:lane"))
            features.append(("advisory" if advisory else "painted", f))
        elif v == "shared_lane":
            features.append(("sharrow", f))
    return features


def measure_park(p, pbf, road_km):
    geom, method = rp.load_boundary(p["slug"])
    if geom is None:
        raise RuntimeError("no boundary saved (measure the park with run_all_parks.py first)")
    park_gs = gpd.GeoSeries([geom], crs="EPSG:4326")
    utm = park_gs.estimate_utm_crs()
    park = park_gs.to_crs(utm).iloc[0].buffer(0)

    wd = rp.WORK / p["slug"] / "cycling"
    rp.export_layers(pbf, rp.bounds_with_margin(geom, 60), wd, {
        "cycling":   (OSMIUM_FILTERS, "linestring"),
        "mainroads": ([f"w/highway={','.join(MAIN_ROADS)}"], "linestring"),
    })
    lines = rp.read(wd / "cycling.geojson", utm, cols=("highway",))
    mains = rp.read(wd / "mainroads.geojson", utm, cols=("highway",))
    if len(mains):
        mains = gpd.clip(mains, park)
        mains = mains[~mains.geometry.is_empty]
    main_road_km = round(mains.geometry.length.sum() / 1000, 2) if len(mains) else 0.0

    out = {"method_version": METHOD_VERSION,
           "slug": p["slug"], "name": p["name"], "city": p["city"], "country": p["country"],
           "park_area_km2": round(park.area / 1e6, 3), "road_km": road_km, "main_road_km": main_road_km,
           "osm_data_date": rp.extract_date(pbf)}
    for c in CLASSES:
        out[f"{c}_km"] = 0.0

    kept = None
    if len(lines):
        lines = lines.copy()
        lines["features"] = lines.apply(classify, axis=1)
        lines = lines[lines["features"].map(len) > 0]
        if len(lines):
            kept = gpd.clip(lines, park)
            kept = kept[~kept.geometry.is_empty].copy()
            kept["length_m"] = kept.geometry.length
            totals = {c: 0.0 for c in CLASSES}
            for length, features in zip(kept["length_m"], kept["features"]):
                for c, f in features:
                    totals[c] += length * f
            for c in CLASSES:
                out[f"{c}_km"] = round(totals[c] / 1000, 2)
            # the best feature on each way, for the map layer
            rank = {c: i for i, c in enumerate(CLASSES)}
            kept["cls"] = kept["features"].map(lambda fs: min((c for c, _ in fs), key=rank.get))

    dedicated = sum(out[f"{c}_km"] for c in COUNTED)
    out["dedicated_km"] = round(dedicated, 2)
    out["ecf_ratio_main_roads_pct"] = round(100 * dedicated / main_road_km, 1) if main_road_km else None

    if kept is not None and len(kept):
        cols = [c for c in ("cls", "highway", "name", "geometry") if c in kept.columns]
        kept[cols].to_crs("EPSG:4326").to_file(OUT_DIR / f"{p['slug']}.geojson",
                                                driver="GeoJSON", COORDINATE_PRECISION=6)
    return out


def current(slug):
    """True if this park was already measured with the current definitions."""
    f = OUT_DIR / f"{slug}.json"
    if not f.exists():
        return False
    s = json.loads(f.read_text())
    return s.get("status") == "ok" and s.get("method_version") == METHOD_VERSION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated slugs")
    ap.add_argument("--refresh", action="store_true", help="re-measure parks already done")
    ap.add_argument("--delete-pbf", action="store_true", help="delete each extract after use")
    args = ap.parse_args()

    results = pd.read_csv(rp.RESULTS_CSV)
    road_km = dict(zip(results["slug"], results["road_km"]))
    parks = [p for p in rp.load_parks() if not p["skip"] and p["slug"] in road_km]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        parks = [p for p in parks if p["slug"] in wanted]
        if not parks:
            raise SystemExit("None of those slugs are measured parks.")
    for p in parks:
        p["region_label"] = p.get("region")

    plan = rp.phase_regions(parks)
    rp.print_plan(plan)
    print("\nMeasuring cycling infrastructure")
    for rid, item in sorted(plan.items()):
        todo = [p for p in item["parks"] if args.refresh or not current(p["slug"])]
        if not todo:
            continue
        print(f"\n== {rid} ({len(todo)} parks)")
        try:
            pbf = rp.ensure_pbf(item["region"])
        except Exception as e:
            print(f"  Download failed ({type(e).__name__}); skipping this region. Re-run to retry.")
            continue
        for p in todo:
            t0 = time.time()
            try:
                s = measure_park(p, pbf, road_km.get(p["slug"]))
                s["status"] = "ok"
                print(f"  {p['name'][:40]:<42} separated {s['separated_km']:>6.2f} km  shared {s['shared_km']:>6.2f} km  "
                      f"painted {s['painted_km']:>5.2f} km  (advisory {s['advisory_km']:.2f}, sharrows {s['sharrow_km']:.2f})  "
                      f"({time.time()-t0:.0f}s)")
            except Exception as e:
                s = {"slug": p["slug"], "name": p["name"], "status": "failed",
                     "error": f"{type(e).__name__}: {str(e)[:200]}"}
                print(f"  {p['name'][:40]:<42} FAILED: {s['error'][:90]}")
            (OUT_DIR / f"{p['slug']}.json").write_text(json.dumps(s, indent=2, ensure_ascii=False, default=float))
        if args.delete_pbf:
            pbf.unlink(missing_ok=True)

    rows = [json.loads(f.read_text()) for f in sorted(OUT_DIR.glob("*.json"))]
    rows = [r for r in rows if r.get("method_version") == METHOD_VERSION or r.get("status") == "failed"]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    ok = df[df["status"] == "ok"] if "status" in df else df.iloc[0:0]
    print(f"\nSaved {OUT_CSV.relative_to(rp.BASE)}: {len(ok)} parks")
    if len(ok):
        print(f"\n{'Park':<40} {'Road km':>8} {'Separated':>10} {'Shared':>8} {'Painted':>8} {'Per road km':>12} {'ECF %':>7}")
        for _, r in ok.sort_values("dedicated_per_road_km", ascending=False).head(15).iterrows():
            per = r["dedicated_per_road_km"]
            ecf = r["ecf_ratio_main_roads_pct"]
            print(f"{r['name'][:38]:<40} {r['road_km']:>8.1f} {r['separated_km']:>10.2f} {r['shared_km']:>8.2f} "
                  f"{r['painted_km']:>8.2f} {per if pd.notna(per) else '':>12} {ecf if pd.notna(ecf) else '':>7}")
        home = ok[ok["slug"] == "084-lisbon-monsanto-forest-park"]
        if len(home):
            h = home.iloc[0]
            print(f"\nMonsanto: {h['separated_km']} km separated, {h['shared_km']} km shared paths, "
                  f"{h['painted_km']} km painted lanes (not counted: {h['advisory_km']} km advisory lanes, "
                  f"{h['sharrow_km']} km sharrows), for {h['road_km']} km of road, "
                  f"{h['main_road_km']} km of it main roads (ECF ratio {h['ecf_ratio_main_roads_pct']}%)")


if __name__ == "__main__":
    main()
