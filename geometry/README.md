# Geometry & prediction meshes

Drop files here to use them in the planner:

* **Geometry** — a Mitsuba `.xml` scene drives Sionna RT ray tracing. `.obj`/`.ply`
  meshes can also be selected as geometry to provide a 2D footprint and bounds
  (but Sionna ray tracing itself requires an `.xml` scene).
* **Prediction mesh** — a `.ply`/`.obj` mesh defines the measurement area and
  height. Its XY bounding box and mid-height set the coverage grid.

In the web UI, pick them in the **Scene geometry** card and click *Load & fit
view* — the display auto-fits to their extent. Point the app at a different
folder with the `WIFISIM_GEOMETRY` env var. Only ASCII OBJ/PLY are parsed for
bounds/footprint.
