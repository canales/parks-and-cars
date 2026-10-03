"""
lane_diff.py
==============
Compares two sets of cycling lines (for example Monsanto's lanes in a January
2026 snapshot and in a current extract) and says which stretches are new,
which have disappeared and which are the same lane in both.

Two lines count as the same lane when one runs within TOL metres of the other.
That is deliberately geometric, not by OSM way ID: mappers split and merge
ways all the time, and an ID comparison would call a re-mapped lane "removed
and added". Pure Python, no dependencies.

Each feature is a dict with:
  coords      list of [lon, lat]
  group       "counted" or "other": lanes are only compared within their group
              (a painted route that becomes a separated track is "added" as a
              counted lane, not "unchanged")
The functions add to each feature:
  covered     0 to 1, the share of its length that has a counterpart in the
              other set
  status      "unchanged", "added" (current set) or "removed" (baseline set),
              by majority of its length
  runs        the feature split into stretches that are covered or not, so a
              lane that is only partly new (or partly gone) is drawn that way;
              each run: {"covered": bool, "length_m": float, "coords": [[lon, lat], ...]}
"""

import math

M_PER_DEG_LAT = 110574.0


def to_xy(coords, lat0):
    """[lon, lat] list to local metres (equirectangular around lat0)."""
    kx = 111320.0 * math.cos(math.radians(lat0))
    return [(c[0] * kx, c[1] * M_PER_DEG_LAT) for c in coords]


def from_xy(pts, lat0):
    kx = 111320.0 * math.cos(math.radians(lat0))
    return [[x / kx, y / M_PER_DEG_LAT] for x, y in pts]


def densify(pts, step=5.0):
    """Points along a polyline about every `step` metres, including every vertex."""
    out = [pts[0]]
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        d = math.hypot(x2 - x1, y2 - y1)
        if d == 0:
            continue
        for i in range(1, int(d // step) + 1):
            f = i * step / d
            if f < 1.0:
                out.append((x1 + (x2 - x1) * f, y1 + (y2 - y1) * f))
        out.append((x2, y2))
    return out


class SegIndex:
    """Grid index of polyline segments, to ask 'is this point within tol of any line?'."""

    def __init__(self, polylines, tol):
        self.tol = tol
        self.cell = max(tol * 4.0, 20.0)
        self.grid = {}
        for pts in polylines:
            for (ax, ay), (bx, by) in zip(pts, pts[1:]):
                x0, x1 = min(ax, bx), max(ax, bx)
                y0, y1 = min(ay, by), max(ay, by)
                for ix in range(int((x0 - tol) // self.cell), int((x1 + tol) // self.cell) + 1):
                    for iy in range(int((y0 - tol) // self.cell), int((y1 + tol) // self.cell) + 1):
                        self.grid.setdefault((ix, iy), []).append((ax, ay, bx, by))

    def near(self, x, y):
        segs = self.grid.get((int(x // self.cell), int(y // self.cell)))
        if not segs:
            return False
        t2 = self.tol * self.tol
        for ax, ay, bx, by in segs:
            dx, dy = bx - ax, by - ay
            l2 = dx * dx + dy * dy
            u = 0.0 if l2 == 0 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / l2))
            px, py = ax + u * dx, ay + u * dy
            if (x - px) ** 2 + (y - py) ** 2 <= t2:
                return True
        return False


def covered_fraction(pts, index, step=5.0):
    """Share of the polyline's length that lies within the index's tolerance of a line."""
    samples = densify(pts, step)
    if len(samples) < 2:
        return 0.0
    total = covered = 0.0
    for (x1, y1), (x2, y2) in zip(samples, samples[1:]):
        d = math.hypot(x2 - x1, y2 - y1)
        total += d
        if index.near(x1, y1):
            covered += d
    return covered / total if total else 0.0


def split_runs(pts, index, lat0, step=5.0, min_run=12.0):
    """Split a polyline into consecutive stretches that are (or are not) near a line in `index`.
    Stretches shorter than min_run take the label of their longer neighbour, so a single
    junction or a gap in the data doesn't leave confetti."""
    samples = densify(pts, step)
    if len(samples) < 2:
        return []
    cov = [index.near(x, y) for x, y in samples]
    lab = [cov[i] if cov[i] != cov[i + 1] else cov[i] for i in range(len(samples) - 1)]
    seg = [math.hypot(samples[i + 1][0] - samples[i][0], samples[i + 1][1] - samples[i][1]) for i in range(len(lab))]

    def group(labels):
        out, start = [], 0
        for i in range(1, len(labels) + 1):
            if i == len(labels) or labels[i] != labels[start]:
                out.append([start, i, labels[start], sum(seg[start:i])])
                start = i
        return out

    runs = group(lab)
    while len(runs) > 1:
        short = [k for k, r in enumerate(runs) if r[3] < min_run]
        if not short:
            break
        k = short[0]
        left = runs[k - 1] if k > 0 else None
        right = runs[k + 1] if k + 1 < len(runs) else None
        pick = max((n for n in (left, right) if n), key=lambda n: n[3])
        for i in range(runs[k][0], runs[k][1]):
            lab[i] = pick[2]
        runs = group(lab)
    return [{"covered": r[2], "length_m": r[3], "coords": from_xy(samples[r[0]:r[1] + 1], lat0)} for r in runs]


def compare(current, baseline, lat0, tol=8.0, step=5.0, same_min=0.6):
    """Fill `covered` and `status` on every feature of both sets (in place)."""
    for f in current + baseline:
        f["_xy"] = to_xy(f["coords"], lat0)
    groups = {f["group"] for f in current + baseline}
    idx_base = {g: SegIndex([f["_xy"] for f in baseline if f["group"] == g], tol) for g in groups}
    idx_cur = {g: SegIndex([f["_xy"] for f in current if f["group"] == g], tol) for g in groups}
    for f in current:
        f["covered"] = covered_fraction(f["_xy"], idx_base[f["group"]], step)
        f["status"] = "unchanged" if f["covered"] >= same_min else "added"
        f["runs"] = split_runs(f["_xy"], idx_base[f["group"]], lat0, step)
    for f in baseline:
        f["covered"] = covered_fraction(f["_xy"], idx_cur[f["group"]], step)
        f["status"] = "unchanged" if f["covered"] >= same_min else "removed"
        f["runs"] = split_runs(f["_xy"], idx_cur[f["group"]], lat0, step)
    for f in current + baseline:
        del f["_xy"]


def km_by_status(current, baseline, key="counted_km"):
    """Counted km that are new, unchanged and gone, using the share of each line that
    has no counterpart, so a line that is only partly new contributes only its new part."""
    added = sum(f[key] * (1 - f["covered"]) for f in current if f["group"] == "counted")
    unchanged = sum(f[key] * f["covered"] for f in current if f["group"] == "counted")
    removed = sum(f[key] * (1 - f["covered"]) for f in baseline if f["group"] == "counted")
    return added, unchanged, removed
