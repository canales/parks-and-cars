# Parks and cars

How much of the world's great urban parks is given to cars? This project measures
roads and parking inside every park on Wikipedia's
[list of urban parks by size](https://en.wikipedia.org/wiki/List_of_urban_parks_by_size),
with the same method for all of them, using OpenStreetMap data. It includes a
closer look at Monsanto Forest Park in Lisbon.

The site is in `docs/` and published with GitHub Pages.

## How it's built

| Step | Script | Output |
|---|---|---|
| 1. Match each park to its Wikipedia article, Wikidata item and OpenStreetMap ID | `resolve_park_ids.py` | `parks_resolved.json` |
| 2. Get each park's boundary, download its Geofabrik extract and measure it | `run_all_parks.py` (uses `park_metrics.py`) | `run/parks_results.csv`, per-park map layers |
| 3. Build the web page and the per-park map files | `build_site.py` (uses `site_template.html`) | `docs/` |

Manual corrections (a park's OpenStreetMap ID, or leaving a park out) go in `overrides.json`.

To rebuild after a change to one park:

```bash
python run_all_parks.py --only <slug> --refresh
python build_site.py
```

Requirements: Python 3.9+, `pip install -r requirements.txt`, and
[osmium-tool](https://osmcode.org/osmium-tool/) (`brew install osmium-tool`).
Extracts (about 25 GB) and caches are downloaded into `pbf/` and `run/`, and are
not committed.

## Data

Map data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright),
available under the Open Database License (ODbL). Extracts from
[Geofabrik](https://download.geofabrik.de/). Park list from Wikipedia.
