"""
resolve_park_ids.py
=====================
Finds the exact OpenStreetMap feature for every park, so the extraction
script no longer depends on English-name geocoding.

For each park in parks_data.json:
  1. Wikipedia article  - from the links in the "List of urban parks by size"
                          table (row matched by park name + city); falls back
                          to Wikipedia search, marked for review
  2. Wikidata ID + coordinates - from the article's page properties
  3. OSM ID             - Wikidata property P402 (OSM relation ID) if present,
                          otherwise an Overpass lookup for the OSM feature
                          tagged wikidata=<ID> near the park's coordinates
  4. Manual overrides   - overrides.json, applied last (optional)

Output: parks_resolved.json  (parks_data.json + wiki_title, qid, lat, lon,
        osm_id like "R123456" or "W987", and where each value came from)

overrides.json format (only for parks you need to fix by hand):
  {
    "084-lisbon-monsanto-forest-park": {"osm_id": "R1234567"},
    "020-lahore-changa-manga": {"wiki_title": "Changa Manga", "lat": 31.08, "lon": 73.97}
  }
Slugs are printed in the summary and stored in parks_resolved.json.

Run: python resolve_park_ids.py
Requires: requests, beautifulsoup4
"""

import difflib
import json
import re
import time
import unicodedata
from pathlib import Path
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup

BASE           = Path(__file__).resolve().parent
DATA_PATH      = BASE / "parks_data.json"
OUT_PATH       = BASE / "parks_resolved.json"
OVERRIDES_PATH = BASE / "overrides.json"

LIST_PAGE    = "List_of_urban_parks_by_size"
WIKI_API     = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

# Approved by the user for Wikipedia/Wikidata requests only.
WIKIMEDIA_UA = "park-roads-script/0.1 (https://github.com/canales)"
# OpenStreetMap requests stay generic and non-identifying.
OVERPASS_UA  = "park-roads-script/0.1"

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api",
    "https://overpass.private.coffee/api",
    "https://maps.mail.ru/osm/tools/overpass/api",
    "https://overpass.kumi.systems/api",
]
AROUND_M        = 30000     # search radius around the park's coordinates
WIKI_DELAY_S    = 0.5
OVERPASS_DELAY_S = 2.0

wm = requests.Session()
wm.headers["User-Agent"] = WIKIMEDIA_UA
op = requests.Session()
op.headers["User-Agent"] = OVERPASS_UA


# ── helpers ────────────────────────────────────────────────────────────

def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def make_slug(park):
    raw = f"{park['wiki_rank']:03d}-{park['city']}-{park['name']}"
    ascii_ = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_.lower()).strip("-")


def chunks(items, n):
    items = list(items)
    for i in range(0, len(items), n):
        yield items[i:i + n]


def api_get(session, url, params, label):
    params = {**params, "format": "json", "formatversion": 2}
    for attempt in range(4):
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 429:
                print(f"    {label} rate limit, waiting...")
                time.sleep(15 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"    X {label}: {type(e).__name__}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{label} API unreachable after 4 attempts")


# ── step 1: Wikipedia article per park ─────────────────────────────────

def titles_from_list_page(parks):
    html = api_get(wm, WIKI_API, {"action": "parse", "page": LIST_PAGE, "prop": "text"},
                   "Wikipedia")["parse"]["text"]
    soup = BeautifulSoup(html, "html.parser")

    rows = []
    for tr in soup.select("table.wikitable tr"):
        links = []
        for a in tr.select("a[href^='/wiki/']"):
            target = unquote(a["href"][6:].split("#")[0])
            if ":" in target:          # File:, Help:, Category: ...
                continue
            links.append((norm(a.get_text()), target.replace("_", " ")))
        if links:
            rows.append((norm(tr.get_text(" ")), links))

    found = {}
    for p in parks:
        name, city = norm(p["name"]), norm(p["city"])
        cands = [r for r in rows if name in r[0]]
        if len(cands) > 1:                         # e.g. two "Forest Park" rows
            cands = [r for r in cands if city in r[0]] or cands
        for _, links in cands:
            best = difflib.get_close_matches(name, [t for t, _ in links], n=1, cutoff=0.8)
            if best:
                found[p["slug"]] = next(title for t, title in links if t == best[0])
                break
    return found


GENERIC_WORDS = {"park", "parks", "the", "and", "of", "national", "state", "regional", "area",
                 "preserve", "reservation", "forest", "lake", "county", "recreation", "center",
                 "nature", "de", "da", "do", "la", "le"}


def plausible(park_name, title):
    """A search hit must share at least half of the park's distinctive words,
    otherwise it's usually the city (e.g. 'Beaman Park' -> 'Nashville, Tennessee')."""
    a = set(norm(re.sub(r"\(.*?\)", "", park_name)).split()) - GENERIC_WORDS
    b = set(norm(re.sub(r"\(.*?\)", "", title or "")).split()) - GENERIC_WORDS
    return bool(a) and len(a & b) / len(a) >= 0.5


def search_title(park):
    res = api_get(wm, WIKI_API, {"action": "query", "list": "search", "srlimit": 1,
                                 "srsearch": f"{park['name']} {park['city']}"}, "Wikipedia")
    hits = res.get("query", {}).get("search", [])
    return hits[0]["title"] if hits else None


# ── step 2: Wikidata ID + coordinates ──────────────────────────────────

def qids_and_coords(titles):
    out = {}
    for batch in chunks(sorted(titles), 50):
        res = api_get(wm, WIKI_API, {"action": "query", "titles": "|".join(batch),
                                     "prop": "pageprops|coordinates",
                                     "ppprop": "wikibase_item", "redirects": 1}, "Wikipedia")
        q = res.get("query", {})
        alias = {m["from"]: m["to"] for m in q.get("normalized", []) + q.get("redirects", [])}
        by_title = {}
        for page in q.get("pages", []):
            if page.get("missing"):
                continue
            c = (page.get("coordinates") or [{}])[0]
            by_title[page["title"]] = {"qid": page.get("pageprops", {}).get("wikibase_item"),
                                       "lat": c.get("lat"), "lon": c.get("lon")}
        for t in batch:
            final = t
            for _ in range(3):                    # normalized -> redirected
                final = alias.get(final, final)
            if final in by_title:
                out[t] = by_title[final]
        time.sleep(WIKI_DELAY_S)
    return out


# ── step 3: OSM relation ID from Wikidata (P402) ───────────────────────

def wikidata_claims(qids):
    out = {}
    for batch in chunks(sorted(qids), 50):
        res = api_get(wm, WIKIDATA_API, {"action": "wbgetentities", "ids": "|".join(batch),
                                         "props": "claims"}, "Wikidata")
        for qid, ent in res.get("entities", {}).items():
            claims = ent.get("claims", {})

            def val(pid):
                try:
                    return claims[pid][0]["mainsnak"]["datavalue"]["value"]
                except (KeyError, IndexError, TypeError):
                    return None

            coord = val("P625")
            out[qid] = {"osm_rel": val("P402"),
                        "lat": coord.get("latitude") if coord else None,
                        "lon": coord.get("longitude") if coord else None}
        time.sleep(WIKI_DELAY_S)
    return out


# ── step 4: Overpass lookup by wikidata tag ────────────────────────────

_good_mirror = 0


def overpass(query):
    global _good_mirror
    order = list(range(_good_mirror, len(OVERPASS_MIRRORS))) + list(range(0, _good_mirror))
    for idx in order:
        base = OVERPASS_MIRRORS[idx]
        host = base.split("//")[1].split("/")[0]
        try:
            r = op.post(base + "/interpreter", data={"data": query}, timeout=90)
            if r.status_code in (429, 504):
                print(f"      X {host}: HTTP {r.status_code}")
                time.sleep(15)
                continue
            r.raise_for_status()
            _good_mirror = idx
            return r.json()
        except (requests.RequestException, ValueError) as e:
            print(f"      X {host}: {type(e).__name__}")
    return None


def osm_id_by_wikidata(qid, lat, lon):
    around = f"(around:{AROUND_M},{lat},{lon})" if lat is not None else ""
    query = (f'[out:json][timeout:60];'
             f'(relation["wikidata"="{qid}"]{around};way["wikidata"="{qid}"]{around};);'
             f'out ids tags;')
    data = overpass(query)
    if data is None:
        return None, "Overpass unreachable"
    els = data.get("elements", [])
    if not els:
        return None, "no OSM feature tagged with this Wikidata ID"
    area_keys = ("leisure", "boundary", "landuse", "natural")
    els.sort(key=lambda e: (e["type"] != "relation",
                            not any(k in e.get("tags", {}) for k in area_keys)))
    e = els[0]
    return ("R" if e["type"] == "relation" else "W") + str(e["id"]), "overpass wikidata tag"


# ── main ───────────────────────────────────────────────────────────────

def main():
    parks = json.loads(DATA_PATH.read_text())
    overrides = json.loads(OVERRIDES_PATH.read_text()) if OVERRIDES_PATH.exists() else {}
    prev = ({p["slug"]: p for p in json.loads(OUT_PATH.read_text())}
            if OUT_PATH.exists() else {})
    for p in parks:
        p["slug"] = make_slug(p)

    print(f"1/4 Matching {len(parks)} parks to Wikipedia articles...")
    from_list = titles_from_list_page(parks)
    for p in parks:
        ov = overrides.get(p["slug"], {})
        if ov.get("wiki_title"):
            p["wiki_title"], p["wiki_title_source"] = ov["wiki_title"], "override"
        elif p["slug"] in from_list:
            p["wiki_title"], p["wiki_title_source"] = from_list[p["slug"]], "list page"
        else:
            t = search_title(p)
            time.sleep(WIKI_DELAY_S)
            if t and not plausible(p["name"], t):
                p["rejected_title"], t = t, None
            p["wiki_title"], p["wiki_title_source"] = t, ("search" if t else None)
    n_list = sum(p["wiki_title_source"] == "list page" for p in parks)
    n_search = sum(p["wiki_title_source"] == "search" for p in parks)
    print(f"    list page: {n_list}, search: {n_search}, "
          f"none: {sum(not p['wiki_title'] for p in parks)}")

    print("2/4 Getting Wikidata IDs and coordinates...")
    info = qids_and_coords({p["wiki_title"] for p in parks if p["wiki_title"]})
    for p in parks:
        i = info.get(p["wiki_title"], {})
        p["qid"], p["lat"], p["lon"] = i.get("qid"), i.get("lat"), i.get("lon")
    print(f"    with Wikidata ID: {sum(bool(p['qid']) for p in parks)}")

    print("3/4 Reading OSM relation IDs from Wikidata...")
    claims = wikidata_claims({p["qid"] for p in parks if p["qid"]})
    for p in parks:
        c = claims.get(p["qid"], {})
        if p["lat"] is None and c.get("lat") is not None:
            p["lat"], p["lon"] = c["lat"], c["lon"]
        if c.get("osm_rel"):
            p["osm_id"], p["osm_id_source"] = f"R{c['osm_rel']}", "wikidata P402"
        else:
            p["osm_id"], p["osm_id_source"] = None, None
    print(f"    OSM ID from Wikidata: {sum(bool(p['osm_id']) for p in parks)}")

    todo = [p for p in parks if not p["osm_id"] and p["qid"]]
    print(f"4/4 Looking up {len(todo)} remaining parks in OSM by Wikidata tag...")
    for i, p in enumerate(todo, 1):
        old = prev.get(p["slug"])
        if old and old.get("osm_id") and old.get("qid") == p["qid"]:
            p["osm_id"], p["osm_id_source"] = old["osm_id"], old["osm_id_source"]
            print(f"  [{i}/{len(todo)}] {p['name']}: {p['osm_id']} (cached)")
            continue
        osm_id, how = osm_id_by_wikidata(p["qid"], p["lat"], p["lon"])
        if osm_id:
            p["osm_id"], p["osm_id_source"] = osm_id, how
        print(f"  [{i}/{len(todo)}] {p['name']}: {osm_id or how}")
        time.sleep(OVERPASS_DELAY_S)

    for p in parks:
        ov = overrides.get(p["slug"], {})
        if ov.get("osm_id"):
            p["osm_id"], p["osm_id_source"] = ov["osm_id"], "override"
        if ov.get("lat") is not None:
            p["lat"], p["lon"] = ov["lat"], ov["lon"]

    OUT_PATH.write_text(json.dumps(parks, indent=2, ensure_ascii=False))

    print("\n" + "=" * 70 + "\nSUMMARY\n" + "=" * 70)
    sources = {}
    for p in parks:
        sources[p["osm_id_source"] or "not resolved"] = sources.get(p["osm_id_source"] or "not resolved", 0) + 1
    for s, n in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {s:<25} {n}")

    unresolved = [p for p in parks if not p["osm_id"]]
    if unresolved:
        print(f"\nNo OSM ID ({len(unresolved)}); these fall back to name search, "
              f"then a buffer around the coordinates:")
        for p in unresolved:
            why = "no Wikipedia article" if not p["wiki_title"] else \
                  "no Wikidata ID" if not p["qid"] else "not tagged in OSM"
            coords = "has coords" if p["lat"] is not None else "NO coords"
            print(f"  {p['slug']:<55} {why}, {coords}")

    searched = [p for p in parks if p["wiki_title_source"] == "search"]
    if searched:
        print(f"\nArticle found by search, check it's the right one ({len(searched)}):")
        for p in searched:
            print(f"  {p['name']} ({p['city']}) -> {p['wiki_title']}")

    print(f"\nSaved {OUT_PATH}")
    print("Fix any wrong match in overrides.json and re-run (OSM lookups are cached).")


if __name__ == "__main__":
    main()
