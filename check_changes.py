"""
check_changes.py
==================
Compares a park's results before and after a re-measure, and writes a
Markdown table of what changed (used as the pull request description by the
monthly GitHub Action).

Usage: python check_changes.py before.json after.json changes.md
Prints changed=true/false, and sets it as a GitHub Actions step output.
"""

import json
import os
import sys

# Bookkeeping fields that change on every run without the numbers changing
IGNORE = {"osm_data_date", "measured_on", "boundary_method", "region", "continent"}


def main():
    before_path, after_path, out_path = sys.argv[1:4]
    before = json.load(open(before_path))
    after = json.load(open(after_path))

    changes = []
    for key in sorted(set(before) | set(after)):
        if key in IGNORE:
            continue
        old, new = before.get(key), after.get(key)
        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
            if abs(old - new) < 1e-9:
                continue
        elif old == new:
            continue
        changes.append((key, old, new))

    name = after.get("name", after.get("slug", "park"))
    lines = [f"Monthly re-measure of **{name}** with the latest OpenStreetMap data "
             f"(extract of {after.get('osm_data_date') or 'unknown date'}; "
             f"previous: {before.get('osm_data_date') or 'unknown'}).", ""]
    if changes:
        lines += ["| Value | Before | After |", "|---|---|---|"]
        lines += [f"| `{k}` | {o} | {n} |" for k, o, n in changes]
        lines += ["", "Check the numbers, then merge to publish them. Close the pull request to keep the current ones."]
    else:
        lines += ["No number changed."]
    open(out_path, "w").write("\n".join(lines) + "\n")

    changed = "true" if changes else "false"
    print(f"changed={changed}")
    for k, o, n in changes:
        print(f"  {k}: {o} -> {n}")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"changed={changed}\n")


if __name__ == "__main__":
    main()
