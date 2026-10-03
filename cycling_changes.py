"""
cycling_changes.py
====================
How did Monsanto's dedicated cycling infrastructure change between dated
OpenStreetMap snapshots? Measures every snapshot you give it with the same
rules as cycling_infra.py (ECF definitions, one-way infrastructure counts
half), then compares the "before" snapshot with the newest one, lane by lane
(see lane_diff.py): which stretches are new, which are the same lane, which
have disappeared.

Usage (any number of snapshots, in any order; the newest is "now"):
  python cycling_changes.py pbf/history/portugal-260101.osm.pbf pbf/portugal-latest.osm.pbf
  python cycling_changes.py pbf/history/portugal-250101.osm.pbf pbf/history/portugal-260101.osm.pbf pbf/portugal-latest.osm.pbf

"Before" is the snapshot closest to 2026-01-02 (--baseline-date), or --baseline PATH.
Needs the park already measured: run/boundaries/<slug>.geojson and run/parks_results.csv.

Lanes you have seen being removed on the ground, but that OpenStreetMap does not
show yet, go in monsanto_lane_status.json (see that file). They are reported
separately and drawn as "being removed". Once OpenStreetMap is updated, the
comparison finds them by itself and the entry is no longer needed.

Writes, in run/cycling_changes/:
  <slug>.json        the numbers: each snapshot, added / kept / removed km, per 100 km of road
  <slug>.geojson     every stretch with its status (unchanged, added, removed, removal)
  <slug>_ways.csv    every OSM way with its status and a link, to pick the way IDs
                     of lanes being removed
What the dates mean: a lane is "added" when it appears in OpenStreetMap between
two snapshots, which is when it was mapped, not necessarily when it was built.
"""

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd

import cycling_infra as ci
import lane_diff
import run_all_parks as rp

HOME = "084-lisbon-monsanto-forest-park"
OUT_DIR = rp.RUN / "cycling_changes"
STATUS_FILE = rp.BASE / "monsanto_lane_status.json"
DEFAULT_BASELINE = "2026-01-02"


# ── helpers ────────────────────────────────────────────────────────────

def snapshot_date(pbf):
    """Date of the OpenStreetMap data in a snapshot: its header, else its file name."""
    d = rp.extract_date(pbf)
    if d:
        return d
    m = re.search(r"-(\d{2})(\d{2})(\d{2})\.osm\.pbf$", pbf.name)
    if m:
        return f"20{m[1]}-{m[2]}-{m[3]}"
    return datetime.fromtimestamp(pbf.stat().st_mtime).date().isoformat()


def load_park(slug):
    geom, _ = rp.load_boundary(slug)
    if geom is None:
        raise SystemExit(f"No saved boundary for {slug}. Run run_all_parks.py for it first.")
    return geom


def clean_id(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def clean_name(v):
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def line_parts(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "LineString":
        return [geom]
    return [g for g in getattr(geom, "geoms", []) if g.geom_type == "LineString"]


def measure_snapshot(pbf, slug, park_wgs):
    """Cycling ways inside the park in one snapshot: a list of stretches (one per way part)
    and the counted km per class, with the same rules as cycling_infra.py."""
    park_gs = gpd.GeoSeries([park_wgs], crs="EPSG:4326")
    utm = park_gs.estimate_utm_crs()
    park = park_gs.to_crs(utm).iloc[0].buffer(0)

    wd = rp.WORK / slug / ("changes-" + pbf.name.replace(".osm.pbf", ""))
    rp.export_layers(pbf, rp.bounds_with_margin(park_wgs, 60), wd, {"cycling": (ci.OSMIUM_FILTERS, "linestring")})
    lines = rp.read(wd / "cycling.geojson", utm, cols=("highway",))

    feats, totals = [], {c: 0.0 for c in ci.CLASSES}
    if len(lines):
        lines = lines.copy()
        lines["features"] = lines.apply(ci.classify, axis=1)
        lines = lines[lines["features"].map(len) > 0]
    if len(lines):
        kept = gpd.clip(lines, park)
        kept = kept[~kept.geometry.is_empty].copy()
        kept_wgs = kept.to_crs("EPSG:4326")
        rank = {c: i for i, c in enumerate(ci.CLASSES)}
        for (_, row), geom_wgs in zip(kept.iterrows(), kept_wgs.geometry):
            parts_utm, parts_wgs = line_parts(row["geometry"]), line_parts(geom_wgs)
            for part_utm, part_wgs in zip(parts_utm, parts_wgs):
                length_m = part_utm.length
                if length_m <= 0:
                    continue
                factors = row["features"]                       # [(class, length factor), ...]
                counted_m = 0.0
                for cls, f in factors:
                    totals[cls] += length_m * f / 1000
                    if cls in ci.COUNTED:
                        counted_m += length_m * f
                best = min((c for c, _ in factors), key=rank.get)
                feats.append({
                    "id": clean_id(row.get("@id")), "name": clean_name(row.get("name")),
                    "cls": best, "group": "counted" if best in ci.COUNTED else "other",
                    "coords": [[x, y] for x, y, *_ in part_wgs.coords],
                    "length_km": length_m / 1000, "counted_km": counted_m / 1000,
                })
    return feats, {c: round(v, 3) for c, v in totals.items()}


def load_status():
    """Lanes seen being removed on the ground, from monsanto_lane_status.json."""
    if not STATUS_FILE.exists():
        return {}
    data = json.loads(STATUS_FILE.read_text())
    return {int(x["way"]): x for x in data.get("being_removed", []) if x.get("way")}


def dedicated(totals):
    return round(sum(totals[c] for c in ci.COUNTED), 3)


def per100(km, road_km):
    return round(100 * km / road_km, 2) if road_km else None


# ── main ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("snapshots", nargs="+", type=Path, help="two or more .osm.pbf snapshots, any order")
    ap.add_argument("--park", default=HOME)
    ap.add_argument("--baseline", type=Path, help="which snapshot is 'before' (default: closest to --baseline-date)")
    ap.add_argument("--baseline-date", default=DEFAULT_BASELINE)
    ap.add_argument("--tolerance", type=float, default=8.0, help="metres within which two lines are the same lane")
    args = ap.parse_args()

    for p in args.snapshots:
        if not p.exists():
            raise SystemExit(f"File not found: {p}")
    snaps = sorted(({"path": p, "date": snapshot_date(p)} for p in args.snapshots), key=lambda s: s["date"])
    print("Snapshots (oldest first):")
    for s in snaps:
        print(f"  {s['date']}  {s['path']}")

    current = snaps[-1]
    if args.baseline:
        baseline = next((s for s in snaps if s["path"].resolve() == args.baseline.resolve()), None)
        if baseline is None:
            raise SystemExit("--baseline must be one of the snapshots you listed.")
    else:
        target = datetime.fromisoformat(args.baseline_date)
        baseline = min(snaps, key=lambda s: abs((datetime.fromisoformat(s["date"]) - target).days))
        gap = abs((datetime.fromisoformat(baseline["date"]) - target).days)
        if gap > 30:
            raise SystemExit(f"No snapshot within 30 days of {args.baseline_date} (closest is {baseline['date']}). "
                             f"Use --baseline-date or --baseline.")
    if baseline is current:
        raise SystemExit("The newest snapshot is also the baseline, so there is nothing to compare. "
                         "Add a current extract (for example pbf/portugal-latest.osm.pbf).")
    print(f"\nBefore: {baseline['date']}   Now: {current['date']}")

    park = load_park(args.park)
    for s in snaps:
        print(f"Measuring {s['path'].name} ...")
        s["feats"], s["totals"] = measure_snapshot(s["path"], args.park, park)

    cur, base = current["feats"], baseline["feats"]
    lane_diff.compare(cur, base, park.centroid.y, tol=args.tolerance)
    added, unchanged, removed = lane_diff.km_by_status(cur, base)

    # lanes seen being removed on the ground, not yet in OpenStreetMap
    status = load_status()
    seen = set()
    for f in cur:
        if f["id"] in status and f["group"] == "counted":
            f["removal"], f["observed"] = True, status[f["id"]].get("observed")
            seen.add(f["id"])
    for way in sorted(set(status) - seen):
        print(f"  WARNING: way {way} in monsanto_lane_status.json is not a counted lane in the newest snapshot")
    being_removed_km = sum(f["counted_km"] for f in cur if f.get("removal"))

    road_km = None
    if rp.RESULTS_CSV.exists():
        res = pd.read_csv(rp.RESULTS_CSV)
        res = res[res["slug"] == args.park]
        road_km = float(res["road_km"].iloc[0]) if len(res) else None

    base_km, now_km = dedicated(baseline["totals"]), dedicated(current["totals"])
    after_km = round(now_km - being_removed_km, 3)
    summary = {
        "slug": args.park, "road_km": road_km, "tolerance_m": args.tolerance,
        "baseline": {"date": baseline["date"], "file": baseline["path"].name, **baseline["totals"],
                     "dedicated_km": base_km, "per_100_km_road": per100(base_km, road_km)},
        "current": {"date": current["date"], "file": current["path"].name, **current["totals"],
                    "dedicated_km": now_km, "per_100_km_road": per100(now_km, road_km)},
        "added_km": round(added, 3), "unchanged_km": round(unchanged, 3), "removed_in_osm_km": round(removed, 3),
        "other_change_km": round((now_km - base_km) - (added - removed), 3),
        "being_removed_km": round(being_removed_km, 3),
        "being_removed_ways": sorted(seen),
        "observed": sorted({str(status[w].get("observed")) for w in seen if status[w].get("observed")}),
        "dedicated_after_removal_km": after_km, "per_100_km_road_after_removal": per100(after_km, road_km),
        "snapshots": [{"date": s["date"], "file": s["path"].name, **s["totals"], "dedicated_km": dedicated(s["totals"])}
                      for s in snaps],
    }

    # outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{args.park}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    def feature(st, f, run):
        return {"type": "Feature",
                "properties": {"status": st, "cls": f["cls"], "counted": f["group"] == "counted",
                               "way": f["id"], "name": f["name"], "m": round(run["length_m"])},
                "geometry": {"type": "LineString", "coordinates": [[round(x, 6), round(y, 6)] for x, y in run["coords"]]}}
    out = []
    for f in cur:
        for r in f["runs"]:
            out.append(feature("removal" if f.get("removal") else ("unchanged" if r["covered"] else "added"), f, r))
    for f in base:
        for r in f["runs"]:
            if not r["covered"]:
                out.append(feature("removed", f, r))
    (OUT_DIR / f"{args.park}.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": out}))

    with open(OUT_DIR / f"{args.park}_ways.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "status", "way", "name", "class", "counted_km", "same_in_other_snapshot_pct", "osm_url"])
        rows = ([("now", "being removed (yours)" if f.get("removal") else f["status"], f) for f in cur]
                + [("before", f["status"], f) for f in base if f["status"] == "removed" or any(not r["covered"] for r in f["runs"])])
        for src, st, f in sorted(rows, key=lambda r: (r[0] != "now", r[1], -r[2]["counted_km"])):
            w.writerow([src, st, f["id"], f["name"], f["cls"], round(f["counted_km"], 3), round(100 * f["covered"]),
                        f"https://www.openstreetmap.org/way/{f['id']}" if f["id"] else ""])

    # report
    print("\n" + "=" * 78 + "\nDEDICATED CYCLING INFRASTRUCTURE IN THE PARK (counted km)\n" + "=" * 78)
    print(f"{'Snapshot':<14} {'Separated':>10} {'Shared':>8} {'Painted':>8} {'Total':>8} {'Per 100 km road':>16}")
    for s in summary["snapshots"]:
        mark = "  <- before" if s["date"] == baseline["date"] else ("  <- now" if s["date"] == current["date"] else "")
        print(f"{s['date']:<14} {s['separated']:>10.2f} {s['shared']:>8.2f} {s['painted']:>8.2f} "
              f"{s['dedicated_km']:>8.2f} {per100(s['dedicated_km'], road_km) if road_km else '':>16}{mark}")
    print(f"\nSince {baseline['date']}: {added:.2f} km added, {removed:.2f} km gone from OpenStreetMap, "
          f"{unchanged:.2f} km the same lane (other changes, such as one-way: {summary['other_change_km']:+.2f} km)")
    if status:
        print(f"Seen being removed on the ground (monsanto_lane_status.json): {being_removed_km:.2f} km "
              f"in {len(seen)} ways -> {after_km:.2f} km left, {per100(after_km, road_km)} per 100 km of road")
    else:
        print("No lanes listed yet in monsanto_lane_status.json as being removed.")
    mixed = [f for f in cur + base if 0.2 < f["covered"] < 0.8]
    if mixed:
        print(f"\nStretches that are only partly the same lane in both snapshots (check on the map): {len(mixed)}")
        for f in mixed[:8]:
            print(f"  way {f['id']} {f['name'] or '(no name)'}: {round(100 * f['covered'])}% in the other snapshot")
    top = sorted((f for f in cur if f["status"] == "added" and f["group"] == "counted"), key=lambda f: -f["counted_km"])
    if top:
        print("\nLargest additions:")
        for f in top[:6]:
            print(f"  {f['counted_km']:.2f} km  {f['cls']:<10} way {f['id']}  {f['name'] or '(no name)'}")
    print(f"\nSaved in {OUT_DIR.relative_to(rp.BASE)}/: {args.park}.json, .geojson and _ways.csv")


if __name__ == "__main__":
    main()
