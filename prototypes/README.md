# Front-end flow-viz prototypes

Two self-contained prototypes that animate **real Leonia baseline flow**
(the GWB-approach corridors from a SUMO `edge_history.parquet`) using newer
WebGL mapping stacks. Both run **token-free** on a MapLibre / CARTO basemap,
matching the project's current no-API-key, offline-friendly model.

| File | Stack | Technique | Data |
| --- | --- | --- | --- |
| `maplibre_flow.html` | MapLibre GL JS | Data-driven line colour/width by vph + animated ant-path dashes | aggregated edge flow |
| `deckgl_flow.html` | deck.gl + MapLibre | GPU `PathLayer` + `TripsLayer` with **synthesised** comets | aggregated edge flow |
| `deckgl_fcd.html` | deck.gl + MapLibre | `TripsLayer` driven by **real per-vehicle trajectories** | SUMO `--fcd-output` |

`maplibre_flow.html` / `deckgl_flow.html` read `leonia_flow_data.js` (network
skeleton + per-15-min vehicles-per-hour for the active edges).
`deckgl_fcd.html` additionally reads `leonia_fcd_data.js` (real vehicle x/y/t).

## Regenerate the data

```bash
# Aggregated edge flow (maplibre_flow.html, deckgl_flow.html)
PYTHONPATH=. venv/bin/python prototypes/extract_flow_data.py
# options: --run <edge_history.parquet> --min-vph 6 --frame-minutes 15

# Real per-vehicle trajectories (deckgl_fcd.html)
#   Peak-AM slice (07:00-08:00, ~219 vehicles, dense):
PYTHONPATH=. venv/bin/python prototypes/run_fcd.py
#   Full 24h weekday baseline (All-Days residential routes), subsampled for
#   the browser via a per-origin cap (keeps every street, thins busy ones):
PYTHONPATH=. venv/bin/python prototypes/run_fcd.py \
    --routes data/processed/sumo/leonia.routes_prototype_weekday_allday.xml \
    --begin 0 --end 86400 --period 4 --per-origin-cap 10
#   --reuse-fcd re-parses the last /tmp FCD without re-running SUMO (fast
#   iteration on --per-origin-cap / --coord-precision).

# Full 24h weekday baseline (maplibre_flow.html, deckgl_flow.html)
PYTHONPATH=. venv/bin/python prototypes/run_baseline_24h.py
# --za-shape all_days (default) -> ~152 residential blocks (All-Days hourly shape)
# --za-shape weekday           -> ~59 blocks (matches the production demand)
# runs SUMO mesoscopically on weekday Bridge-OD + ZA residential demand
# and writes per-15-min edge flow; production demand is left untouched.
```

## View

CDN scripts + the local data file mean you want a tiny static server
(double-clicking `file://` blocks some browsers):

```bash
cd prototypes && python3 -m http.server 8800
# then open:
#   http://localhost:8800/maplibre_flow.html
#   http://localhost:8800/deckgl_flow.html
#   http://localhost:8800/deckgl_fcd.html
```

Internet access is required (MapLibre, deck.gl, and the basemap load from
unpkg / CARTO CDNs). For a fully offline build these can be vendored locally.

## Notes / honesty

- The two `*_flow.html` animations use the **link-level** data you already
  produce; the MapLibre one needs nothing new from the simulation.
- `deckgl_flow.html`'s comet trails are **synthesised** from edge flow.
- `deckgl_fcd.html` is the real deal: `run_fcd.py` adds `--fcd-output
  --fcd-output.geo true` to a SUMO run, so each comet is an actual simulated
  vehicle with its true path, timing, and speed (comet colour = speed, so red
  = crawling / congested). The FCD file for a 1-hour, ~219-vehicle slice is
  ~1 MB; a full 24-hour town-wide run would be far larger, so production use
  would subsample (`--device.fcd.period`), clip the window, or tile by hour.
