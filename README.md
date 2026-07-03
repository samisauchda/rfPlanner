# wifisim — a 5 GHz WiFi coverage simulation framework

A modular, extensible link-/system-level simulation tool for **5 GHz WiFi**,
built around **NVIDIA Sionna RT** ray tracing. Place transmitters, edit them,
move them, assign antenna patterns, and run coverage / SINR simulations —
interactively in a browser or programmatically. **Per-transmitter results are
cached**, so changing one access point never recomputes the others.

```
┌─────────────┐   place / edit / move    ┌──────────────┐
│  Web UI     │ ───────────────────────► │  Simulator   │
│  (Flask +   │ ◄─────────────────────── │  orchestrator│
│   canvas)   │   heatmap + stats        └──────┬───────┘
└─────────────┘                                 │ per-TX layer (cache miss)
                                         ┌───────▼────────┐   ┌──────────────┐
                                         │  LayerCache    │   │  Engine      │
                                         │ (mem + disk)   │◄──│  Sionna RT   │
                                         └────────────────┘   └──────────────┘
```

## Why it's structured this way

The expensive unit of work is computing **one transmitter's** coverage layer.
Everything is organised so that unit can be cached and reused:

* **`wifisim.models`** — frozen, hashable specs (`SceneConfig`, `GridSpec`,
  `Transmitter`, `AntennaConfig`). Each exposes a `signature` derived *only*
  from physics-affecting fields, so renaming or recolouring a transmitter is
  free, while moving or re-powering it invalidates just that transmitter.
* **`wifisim.engines`** — `SionnaRTEngine`: geometry-aware ray tracing via
  Sionna RT (`load_scene` → `RadioMapSolver`), grouped by frequency/antenna for
  efficient solves.
* **`wifisim.cache`** — content-addressed two-tier (memory + disk) cache keyed
  on `H(engine | scene | grid | transmitter)`.
* **`wifisim.combine`** — aggregates per-TX layers into RSS, best-server RSRP,
  best-server index, and **SINR with co-channel interference** (WiFi channel
  reuse is modelled: same channel interferes, orthogonal channels don't).
* **`wifisim.simulator`** — ties it together; `run()` only calls the engine for
  cache misses.

## Install

```bash
pip install -r requirements.txt
```

Sionna RT runs on **CPU via the Dr.Jit LLVM backend** (no GPU required) or on
GPU via CUDA for large scenes. The `sionna-rt` package is included in
`requirements.txt`. If you need the full Sionna PHY/SYS stack:

```bash
pip install sionna
```

## Use it: web interface

```bash
python run_web.py            # http://127.0.0.1:5000
```

* Add access points with **+ Add AP** (at the view centre) or **double-click**
  a spot on the map.
* **Drag** a marker to move it; **click** it to edit power, channel, bandwidth,
  antenna pattern, azimuth, beamwidth, height.
* **Pan** by dragging empty space; **scroll** to zoom toward the cursor;
  **Reset view** re-fits the scene.
* **Ray-tracing solver** panel: tune `max depth`, `samples / TX`, `refraction`,
  `diffraction` live. Changing these invalidates cached layers (different solver
  settings are different results).
* **Scene geometry** card: load a Mitsuba `.xml` scene (drives Sionna ray
  tracing) and/or a `.ply`/`.obj` **prediction mesh**. A **measurement surface**
  toggle chooses what defines the coverage grid: *Bounding box* (the geometry
  extent, or your manual Scene & grid bounds) or *Prediction mesh* (the mesh's
  area and mid-height). Click *Load & fit view* and the display auto-fits; the
  geometry/mesh footprint is drawn on the map (blue = geometry, red = mesh).
  Drop files in `geometry/` (or set `WIFISIM_GEOMETRY`).
* **Hover** anywhere on the map to read the coordinate and the value at that
  point (RSSI in dBm, or SINR in dB) for the displayed metric.
* Drag the **divider** between map and sidebar to resize it (remembered).
* Pick a **metric** (best-server RSRP / aggregate RSS / SINR / best server) and
  hit **Run** (or leave *auto* on to recompute on every change).
* The **cache panel** shows hit-rate and per-AP cached/dirty state — edit one AP
  and watch only that one go dirty.

Layout autosaves and computed layers persist on disk, so restarts are instant.

## Use it: library / scripts

```python
from wifisim import Simulator, SceneConfig, GridSpec, Transmitter, AntennaConfig
from wifisim import viz

sim = Simulator(
    scene=SceneConfig(name="empty"),
    grid=GridSpec(x_min=-30, x_max=30, y_min=-30, y_max=30, z=1.5, cell_size=0.5),
    engine="sionna_rt",
    cache=".wifisim_cache",
)

sim.add_transmitter(Transmitter("AP1", (-12, -10, 3), tx_power_dbm=20, channel=36,
                                antenna=AntennaConfig("tr38901"), orientation=(45, 0, 0)))
sim.place("AP2", 10, 8, channel=44, antenna=AntennaConfig("iso"))

result = sim.run()                         # cold: computes both
print(result.summary())

sim.move("AP2", 4, 4)
result = sim.run()                         # only AP2 recomputed (1 hit / 1 miss)

viz.save_figure(result, sim.transmitters, metric="sinr", path="sinr.png")
```

`run_cli.py` is a runnable version of the above.

## Measured antenna patterns (.msi)

Real principal-plane antenna files (`.msi`, horizontal + vertical cuts) are a
first-class antenna type. The 3D pattern is reconstructed with the standard
sum-of-planes method; the boresight gain is pinned to the file's `GAIN` value
and the implied directivity/efficiency is reported so you can judge the
approximation. The Sionna engine registers it as a DrJit-traceable
`PlanarArray(pattern=...)`.

Library:

```python
from wifisim import Simulator, Transmitter, make_msi_antenna

ant = make_msi_antenna("patterns/MA-WA55-4QP13 V.msi",
                       azimuth_ccw=True)           # flip if your files use compass az
sim.add_transmitter(Transmitter("sector-A", (0, 0, 6),
                                orientation=(120, 0, 0),   # steer boresight by yaw
                                antenna=ant))
```

Web UI: drop `.msi` files into the `patterns/` folder (or set
`WIFISIM_PATTERNS=/path/to/folder`); they appear in the antenna dropdown of the
transmitter editor. The server resolves the file, hashes its contents, and reads
the peak gain. **The cache keys on the file hash**, so editing or swapping a
`.msi` recomputes only the transmitters that use it.

Conventions (see `wifisim/antenna_msi.py`): pass `azimuth_ccw=False` for
clockwise/compass azimuth exports, and `vertical_down_positive=False` if your
files measure positive vertical angles upward. Verify with an asymmetric
pattern. The `make_msi_antenna` builder and the `wifisim.antenna_msi` module
(parse / evaluate / register / stats) are fully documented.

## Conventions

* Positions in metres, frame `(x, y, z)` with `z` up.
* Antenna `orientation = (yaw, pitch, roll)` in **degrees**; yaw 0° points +x,
  90° points +y. Boresight is local +x.
* `tx_power_dbm` is total radiated power; antenna gain is applied per direction.
* Channels follow `f = 5000 + 5·n` MHz; noise floor is
  `-174 + 10·log10(B) + NF`.

## Reproducibility

Results are pure functions of the (hashable) spec objects and the engine
signature. The same project + engine settings always yield the same maps, and
the cache key encodes every dependency, so a stale result can never be silently
served.

## Extending

* **New propagation model** → subclass `engines.base.PropagationEngine`,
  implement `signature` + `compute_layer` (optionally `compute_layers`), and
  register it in `engines.make_engine`.
* **New antenna pattern** → add it to `models.ANTENNA_PATTERNS` and map it in
  `SionnaRTEngine._planar_array`.
* **New metric** → add an array to `SimulationResult`, compute it in
  `combine.aggregate`, and register a `MetricSpec` in `viz.METRIC_SPECS`.
* **Real geometry** → set `SceneConfig(name="simple_street_canyon")` or a path
  to a Mitsuba `.xml` scene.

## Tests

```bash
python -m pytest tests/ -q          # or: python tests/test_core.py
```

Core tests use a dependency-free stub engine and cover signature
stability/sensitivity, cache key composition, disk round-trip, the "only
recompute what changed" guarantee, aggregation/SINR, and rendering. Geometry
and MSI tests are also engine-independent.

## Notes & limitations

* `SionnaRTEngine` targets the **Sionna RT 1.x / 2.x** solver API
  (`RadioMapSolver`). The radio-map row order can differ between Sionna
  versions; `flip_rows=True` (non-default) corrects a vertically-mirrored map.
* This is a planning/abstraction tool: SINR uses a co-channel power-sum model,
  not a full PHY-layer BLER abstraction (that would live in a Sionna SYS layer).
