"""Flask web backend exposing the simulator as a small REST API.

This is a single-user research tool: one :class:`~wifisim.Simulator` instance
lives in the app and is autosaved to ``<cache>/project.json`` after every
mutation so a restart restores the layout.  The expensive per-transmitter
layers live in the on-disk :class:`~wifisim.cache.LayerCache`, so even a fresh
process re-uses previous computations.

Endpoints
---------
GET  /                         the single-page UI
GET  /api/options              antenna patterns, channels, engines, metrics
GET  /api/state                full project + cache stats
POST /api/scene                update scene + grid
POST /api/transmitter          create/replace a transmitter
POST /api/transmitter/<n>/move {x, y[, z]}
DELETE /api/transmitter/<n>    remove a transmitter
POST /api/simulate             {metric} -> overlay image + summary + per-TX cache
POST /api/cache/clear          empty the layer cache
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request

from wifisim import GridSpec, SceneConfig, Simulator, Transmitter, AntennaConfig
from wifisim import config as cfg
from wifisim import viz, antenna_msi, geometry as geo
from wifisim import routes as wroutes
from wifisim.cache import layer_key
from wifisim.engines import available_engines
from wifisim.models import ANTENNA_PATTERNS


def create_app(cache_dir: str = ".wifisim_cache", engine: str = "auto") -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    cache_dir = os.environ.get("WIFISIM_CACHE", cache_dir)
    project_path = Path(cache_dir) / "project.json"
    # Folder scanned for .msi antenna pattern files (drop files here).
    patterns_dir = Path(os.environ.get("WIFISIM_PATTERNS", "patterns"))
    patterns_dir.mkdir(parents=True, exist_ok=True)
    # Folder scanned for geometry (.xml/.ply/.obj) and prediction meshes.
    geometry_dir = Path(os.environ.get("WIFISIM_GEOMETRY", "geometry"))
    geometry_dir.mkdir(parents=True, exist_ok=True)

    # Optional Sionna solver tuning via environment (handy on CPU-only hosts):
    #   WIFISIM_SAMPLES_PER_TX (default 1e6), WIFISIM_MAX_DEPTH (default 5)
    engine_kwargs = {}
    if "WIFISIM_SAMPLES_PER_TX" in os.environ:
        engine_kwargs["samples_per_tx"] = int(float(os.environ["WIFISIM_SAMPLES_PER_TX"]))
    if "WIFISIM_MAX_DEPTH" in os.environ:
        engine_kwargs["max_depth"] = int(os.environ["WIFISIM_MAX_DEPTH"])

    sim = Simulator(
        scene=SceneConfig(name="empty", path_loss_exponent=2.8, shadowing_std_db=2.0),
        grid=GridSpec(x_min=-30, x_max=30, y_min=-30, y_max=30, z=1.5, cell_size=0.5),
        engine=engine,
        cache=cache_dir,
        engine_kwargs=engine_kwargs or None,
    )

    # Restore previous project if present, else seed a couple of APs.
    if project_path.exists():
        try:
            sim.load_dict(json.loads(project_path.read_text()))
        except Exception:
            pass
    if not sim.transmitters:
        sim.place("AP1", -10, -8, 3.0, tx_power_dbm=20, channel=36,
                  antenna=AntennaConfig(pattern="iso"))
        sim.place("AP2", 10, 8, 3.0, tx_power_dbm=20, channel=44,
                  antenna=AntennaConfig(pattern="iso"))

    def autosave() -> None:
        try:
            project_path.parent.mkdir(parents=True, exist_ok=True)
            project_path.write_text(json.dumps(sim.to_dict(), indent=2))
        except OSError:
            pass

    # ------------------------------------------------------------------ #
    def cache_status() -> Dict[str, bool]:
        """name -> True if this transmitter's layer is already cached."""
        status = {}
        for tx in sim.transmitters:
            key = layer_key(sim.engine.signature, sim.scene, sim.grid, tx)
            status[tx.name] = sim.cache.contains(key)
        return status

    _SOLVER_FIELDS = ("max_depth", "samples_per_tx", "refraction", "diffraction")

    def engine_params() -> Dict[str, Any]:
        """Current ray-tracing solver params (present on the Sionna engine)."""
        p = {f: getattr(sim.engine, f, None) for f in _SOLVER_FIELDS}
        p["applies"] = sim.engine.name == "sionna_rt"
        return p

    def state_payload() -> Dict[str, Any]:
        return {
            "scene": _scene_dict(sim.scene),
            "grid": _grid_dict(sim.grid),
            "engine": {
                "name": sim.engine.name,
                "available": available_engines(),
                "params": engine_params(),
            },
            "transmitters": [tx.to_dict() for tx in sim.transmitters],
            "cache": sim.cache.stats(),
            "cache_status": cache_status(),
        }

    def _resolve_msi(td: dict) -> dict:
        """If a transmitter dict references an MSI pattern, fill in the file hash
        and peak gain server-side so the cache signature is correct."""
        ant = td.get("antenna") or {}
        if ant.get("pattern") != "msi":
            return td
        ref = ant.get("msi_file", "")
        if not ref:
            raise ValueError("MSI antenna requires 'msi_file'")
        path = Path(ref)
        if not path.is_absolute():
            path = patterns_dir / ref
        if not path.exists():
            raise FileNotFoundError(f"MSI file not found: {ref}")
        built = antenna_msi.make_msi_antenna(
            str(path),
            polarization=ant.get("polarization", "V"),
            num_rows=int(ant.get("num_rows", 1)),
            num_cols=int(ant.get("num_cols", 1)),
            azimuth_ccw=bool(ant.get("msi_azimuth_ccw", True)),
            vertical_down_positive=bool(ant.get("msi_vertical_down_positive", True)),
        )
        from dataclasses import asdict
        td = dict(td)
        td["antenna"] = asdict(built)
        return td

    # ------------------------------------------------------------------ #
    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/patterns")
    def patterns():
        items = []
        for p in sorted(patterns_dir.glob("*.msi")):
            try:
                items.append(antenna_msi.describe(str(p)))
            except Exception as exc:
                items.append({"file": str(p), "name": p.stem, "error": str(exc)})
        return jsonify({"dir": str(patterns_dir.resolve()), "patterns": items})

    def _resolve_geo(ref: str) -> Path:
        p = Path(ref)
        return p if p.is_absolute() else (geometry_dir / ref)

    @app.get("/api/geometry")
    def geometry_list():
        files = []
        for ext in ("*.xml", "*.ply", "*.obj"):
            for p in sorted(geometry_dir.glob(ext)):
                kind = "geometry" if p.suffix.lower() == ".xml" else "mesh"
                files.append({"file": p.name, "path": str(p), "kind": kind,
                              "ext": p.suffix.lower()})
        return jsonify({
            "dir": str(geometry_dir.resolve()),
            "files": files,
            "scene": {"geometry_file": sim.scene.geometry_file,
                      "mesh_file": sim.scene.mesh_file, "name": sim.scene.name},
        })

    @app.post("/api/geometry")
    def geometry_apply():
        """Load geometry and/or a prediction mesh, then auto-fit the grid.

        Body: {geometry_file?, mesh_file?, cell_size?}.  Empty strings clear.
        """
        data = request.get_json(force=True) or {}
        geo_ref = data.get("geometry_file", sim.scene.geometry_file)
        mesh_ref = data.get("mesh_file", sim.scene.mesh_file)
        cell_size = float(data.get("cell_size", sim.grid.cell_size))
        # Which surface defines the measurement grid:
        #   "mesh" -> fit to prediction mesh; "bbox" -> fit to geometry bbox;
        #   "none"/"manual" -> leave the grid as-is (manual Scene & grid).
        grid_source = data.get("grid_source", "mesh" if mesh_ref else "bbox")

        geo_bounds = mesh_bounds = None
        geo_path = mesh_path = ""
        geo_sha = mesh_sha = ""
        warning = ""
        try:
            if geo_ref:
                gp = _resolve_geo(geo_ref)
                if not gp.exists():
                    return jsonify({"error": f"geometry not found: {geo_ref}"}), 400
                geo_path, geo_sha = str(gp), geo.file_sha(str(gp))
                try:
                    geo_bounds = geo.geometry_info(str(gp)).bounds
                except Exception as exc:
                    geo_bounds = None  # ray-traces in Sionna, but no footprint/fit
                    warning = f"geometry footprint unavailable: {exc}"
            if mesh_ref:
                mp = _resolve_geo(mesh_ref)
                if not mp.exists():
                    return jsonify({"error": f"mesh not found: {mesh_ref}"}), 400
                m = geo.load_mesh(str(mp))
                mesh_path, mesh_sha, mesh_bounds = str(mp), m.sha, m.bounds
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        sim.scene = SceneConfig(
            name=("empty" if geo_path else sim.scene.name),
            path_loss_exponent=sim.scene.path_loss_exponent,
            shadowing_std_db=sim.scene.shadowing_std_db,
            noise_figure_db=sim.scene.noise_figure_db,
            geometry_file=geo_path, geometry_sha=geo_sha,
            mesh_file=mesh_path, mesh_sha=mesh_sha,
        )
        bounds = None
        z = None
        if grid_source == "mesh" and mesh_bounds:
            bounds = mesh_bounds
            z = 0.5 * (mesh_bounds["z_min"] + mesh_bounds["z_max"])
        elif grid_source == "bbox" and geo_bounds:
            bounds = geo_bounds
        elif grid_source in ("mesh", "bbox"):
            # requested source unavailable; fall back to whatever we have
            bounds = geo.union_bounds(mesh_bounds, geo_bounds)
            if mesh_bounds:
                z = 0.5 * (mesh_bounds["z_min"] + mesh_bounds["z_max"])
        fitted = False
        if bounds and grid_source != "manual":
            sim.grid = geo.grid_from_bounds(bounds, cell_size=cell_size, z=z)
            fitted = True
        autosave()
        payload = state_payload()
        n_fp = 0
        try:
            if geo_path:
                n_fp += len(geo.geometry_info(geo_path).segments)
            if mesh_path:
                n_fp += len(geo.footprint_segments(geo.load_mesh(mesh_path)))
        except Exception:
            pass
        payload["geometry"] = {"fitted": fitted, "bounds": bounds,
                               "grid_source": grid_source,
                               "has_mesh": bool(mesh_bounds),
                               "footprint_segments": n_fp,
                               "warning": warning}
        return jsonify(payload)

    @app.get("/api/footprint")
    def footprint():
        """XY-projected edges of the current geometry/mesh, for drawing."""
        out = {"geometry": [], "mesh": []}
        try:
            if sim.scene.geometry_file:
                out["geometry"] = geo.geometry_info(sim.scene.geometry_file).segments
        except Exception:
            pass
        try:
            if sim.scene.mesh_file:
                out["mesh"] = geo.footprint_segments(geo.load_mesh(sim.scene.mesh_file))
        except Exception:
            pass
        return jsonify(out)

    @app.get("/api/geometry3d")
    def geometry3d():
        """Triangle meshes of the loaded geometry / prediction mesh (3D view)."""
        from wifisim import mesh3d
        out: Dict[str, Any] = {"geometry": [], "mesh": []}
        try:
            if sim.scene.geometry_file:
                out["geometry"] = mesh3d.mesh_payload(sim.scene.geometry_file)
        except Exception as exc:
            out["geometry_error"] = str(exc)
        try:
            if sim.scene.mesh_file:
                out["mesh"] = mesh3d.mesh_payload(sim.scene.mesh_file)
        except Exception as exc:
            out["mesh_error"] = str(exc)
        return jsonify(out)

    @app.get("/api/options")
    def options():
        return jsonify({
            "antenna_patterns": list(ANTENNA_PATTERNS),
            "channels": sorted(cfg.WIFI5_CHANNELS_20MHZ.keys()),
            "bandwidths_mhz": [20, 40, 80, 160],
            "polarizations": ["V", "H", "cross"],
            "metrics": [
                {"key": "best_rsrp", "label": "Best-server RSRP"},
                {"key": "rss", "label": "Aggregate RSS"},
                {"key": "sinr", "label": "SINR"},
                {"key": "best_server", "label": "Best server"},
            ],
            "engines": available_engines(),
        })

    @app.get("/api/state")
    def state():
        return jsonify(state_payload())

    @app.post("/api/scene")
    def update_scene():
        data = request.get_json(force=True) or {}
        s = data.get("scene", {})
        g = data.get("grid", {})
        if s:
            sim.scene = SceneConfig(**{**_scene_dict(sim.scene), **s})
        if g:
            sim.grid = GridSpec(**{**_grid_dict(sim.grid), **g})
        autosave()
        return jsonify(state_payload())

    @app.post("/api/engine")
    def update_engine():
        """Update ray-tracing solver parameters (applies to the Sionna engine).

        Changing these changes the engine signature, so cached layers computed
        with the old settings are not reused.
        """
        data = request.get_json(force=True) or {}
        if "max_depth" in data:
            setattr(sim.engine, "max_depth", max(0, int(data["max_depth"])))
        if "samples_per_tx" in data:
            setattr(sim.engine, "samples_per_tx", max(1000, int(float(data["samples_per_tx"]))))
        if "refraction" in data:
            setattr(sim.engine, "refraction", bool(data["refraction"]))
        if "diffraction" in data:
            setattr(sim.engine, "diffraction", bool(data["diffraction"]))
        return jsonify(state_payload())

    @app.post("/api/transmitter")
    def upsert_tx():
        data = request.get_json(force=True) or {}
        try:
            data = _resolve_msi(data)
        except (ValueError, FileNotFoundError) as exc:
            return jsonify({"error": str(exc)}), 400
        tx = Transmitter.from_dict(data)
        sim.add_transmitter(tx)
        autosave()
        return jsonify({"transmitter": tx.to_dict(), "cache_status": cache_status()})

    @app.post("/api/transmitter/<name>/move")
    def move_tx(name: str):
        data = request.get_json(force=True) or {}
        if name not in [t.name for t in sim.transmitters]:
            return jsonify({"error": f"unknown transmitter {name}"}), 404
        tx = sim.move(name, data["x"], data["y"], data.get("z"))
        autosave()
        return jsonify({"transmitter": tx.to_dict(), "cache_status": cache_status()})

    @app.delete("/api/transmitter/<name>")
    def delete_tx(name: str):
        sim.remove(name)
        autosave()
        return jsonify({"ok": True, "cache_status": cache_status()})

    @app.post("/api/simulate")
    def simulate():
        data = request.get_json(force=True) or {}
        metric = data.get("metric", "best_rsrp")
        force = bool(data.get("force", False))
        # Snapshot which TXs were already cached (i.e. will be hits this run).
        pre = cache_status()
        result = sim.run(force=force)
        overlay = viz.render_overlay(result, metric,
                                     data.get("vmin"), data.get("vmax"))
        per_tx = [
            {"name": t.name, "cached_before_run": pre.get(t.name, False),
             "signature": t.signature}
            for t in sim.transmitters if t.enabled
        ]
        return jsonify({
            "overlay": overlay,
            "summary": result.summary(),
            "per_tx": per_tx,
            "cache": sim.cache.stats(),
        })

    @app.post("/api/cache/clear")
    def clear_cache():
        removed = sim.cache.clear()
        return jsonify({"removed": removed, "cache": sim.cache.stats()})

    # ------------------------------------------------------------------ #
    # Routes (CSV polylines) + metric profiles along them
    # ------------------------------------------------------------------ #
    routes_path = Path(cache_dir) / "routes.json"
    route_store: Dict[str, wroutes.Route] = {}          # slot "A"/"B" -> Route

    def _save_routes() -> None:
        try:
            routes_path.parent.mkdir(parents=True, exist_ok=True)
            routes_path.write_text(json.dumps({
                slot: {"name": r.name, "source": r.source, "csv": r.csv_text}
                for slot, r in route_store.items()}, indent=2))
        except OSError:
            pass

    if routes_path.exists():                            # restore across restarts
        try:
            for slot, rd in json.loads(routes_path.read_text()).items():
                route_store[slot] = wroutes.parse_route_csv(
                    rd["csv"], name=rd.get("name", f"Route {slot}"),
                    source=rd.get("source", ""))
        except Exception:
            route_store.clear()

    def _routes_payload() -> Dict[str, Any]:
        return {"routes": {slot: r.to_dict() for slot, r in route_store.items()}}

    @app.get("/api/routes")
    def routes_list():
        return jsonify(_routes_payload())

    @app.post("/api/routes/<slot>")
    def routes_upload(slot: str):
        """Load a route CSV into slot A or B. Body: {csv, name?, source?}."""
        slot = slot.upper()
        if slot not in ("A", "B"):
            return jsonify({"error": "slot must be A or B"}), 400
        data = request.get_json(force=True) or {}
        csv_text = data.get("csv", "")
        if not csv_text.strip():
            return jsonify({"error": "empty CSV"}), 400
        try:
            route = wroutes.parse_route_csv(
                csv_text, name=data.get("name") or f"Route {slot}",
                source=data.get("source", ""))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        route_store[slot] = route
        _save_routes()
        return jsonify(_routes_payload())

    @app.delete("/api/routes/<slot>")
    def routes_delete(slot: str):
        route_store.pop(slot.upper(), None)
        _save_routes()
        return jsonify(_routes_payload())

    @app.post("/api/routes/profile")
    def routes_profile():
        """Sample the current coverage along the loaded routes.

        Body: {metric, interval, radius, force?}.  The simulation itself is
        served from the layer cache when nothing changed.  Sampling happens on
        the coverage grid at grid.z; each sample is the (power-)average of all
        grid cells within `radius` of the resampled route point.
        """
        if not route_store:
            return jsonify({"error": "no routes loaded"}), 400
        data = request.get_json(force=True) or {}
        metric = data.get("metric", "best_rsrp")
        interval = max(0.1, float(data.get("interval", 1.0)))
        radius = max(0.0, float(data.get("radius", 0.5)))
        result = sim.run(force=bool(data.get("force", False)))
        profiles = []
        try:
            for slot in sorted(route_store):
                p = wroutes.route_profile(
                    result, sim.grid, metric, route_store[slot],
                    interval=interval, radius=radius)
                p["slot"] = slot
                profiles.append(p)
        except KeyError as exc:
            return jsonify({"error": f"unknown metric: {exc}"}), 400
        image = None
        if data.get("render"):               # optional matplotlib PNG export
            image = wroutes.render_profile_plot(
                profiles, title=f"engine {sim.engine.name} \u00b7 grid z "
                                f"{sim.grid.z:g} m \u00b7 radius {radius:g} m")
        return jsonify({"image": image, "profiles": profiles,
                        "metric": metric, "cache": sim.cache.stats()})

    return app


# --------------------------------------------------------------------------- #
def _scene_dict(s: SceneConfig) -> Dict[str, Any]:
    from dataclasses import asdict
    return asdict(s)


def _grid_dict(g: GridSpec) -> Dict[str, Any]:
    from dataclasses import asdict
    return asdict(g)


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=True)
