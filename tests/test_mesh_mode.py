"""Tests for mesh-native prediction mode (coverage on a mesh's own triangles,
no bounding box / cell size) -- signatures, cache-key isolation, aggregation,
and Simulator branching between grid and mesh mode.

Run with:  python -m pytest tests/  -q     (or)     python tests/test_mesh_mode.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wifisim import GridSpec, MeshSurface, SceneConfig, Simulator, Transmitter
from wifisim.cache import layer_key
from wifisim.engines.base import PropagationEngine
from wifisim.models import CoverageLayer
from wifisim import combine


class _StubEngine(PropagationEngine):
    """Flat -60 dBm layer for both grid and mesh mode; no Sionna needed."""

    name = "stub"
    n_faces = 40

    @property
    def signature(self) -> str:
        return "stub000000000"

    def compute_layer(self, scene, grid, tx):
        return self.compute_layers(scene, grid, [tx])[tx.signature]

    def compute_layers(self, scene, grid, txs):
        ny, nx = grid.shape
        return {
            tx.signature: CoverageLayer(
                rsrp_dbm=np.full((ny, nx), -60.0, dtype=np.float32),
                tx_signature=tx.signature, tx_name=tx.name,
                channel=tx.channel, engine=self.name,
            )
            for tx in txs
        }

    def compute_layers_mesh(self, scene, surface, txs):
        return {
            tx.signature: CoverageLayer(
                rsrp_dbm=np.full(self.n_faces, -60.0, dtype=np.float32),
                tx_signature=tx.signature, tx_name=tx.name,
                channel=tx.channel, engine=self.name,
            )
            for tx in txs
        }

    def mesh_cell_centers(self, scene, surface):
        return np.zeros((self.n_faces, 3), dtype=np.float32)


def test_mesh_surface_signature_never_collides_with_gridspec():
    m1 = MeshSurface(mesh_file="a.ply", mesh_sha="sha1")
    m2 = MeshSurface(mesh_file="a.ply", mesh_sha="sha2")
    assert m1.signature != m2.signature                 # content matters
    grids = [GridSpec(), GridSpec(cell_size=0.25), GridSpec(x_min=-10, x_max=10)]
    for g in grids:
        assert g.signature != m1.signature               # never collides


def test_layer_key_differs_between_grid_and_mesh_mode():
    eng = _StubEngine()
    scene = SceneConfig()
    tx = Transmitter("ap", (0, 0, 3))
    grid = GridSpec()
    mesh = MeshSurface(mesh_file="a.ply", mesh_sha="sha1")
    assert layer_key(eng.signature, scene, grid, tx) != layer_key(eng.signature, scene, mesh, tx)


def test_aggregate_on_mesh_surface_shapes_and_sinr():
    mesh = MeshSurface(mesh_file="a.ply", mesh_sha="sha1")
    scene = SceneConfig()
    eng = _StubEngine()
    txs = [Transmitter("A", (-5, 0, 3), channel=36),
           Transmitter("B", (5, 0, 3), channel=36),
           Transmitter("C", (0, 5, 3), channel=149)]
    layers = [eng.compute_layers_mesh(scene, mesh, [t])[t.signature] for t in txs]
    res = combine.aggregate(mesh, scene, layers, txs, "stub")
    n = eng.n_faces
    assert res.rss_dbm.shape == (n,) == res.sinr_db.shape == res.best_server.shape
    assert res.best_server.min() >= -1 and res.best_server.max() < len(txs)
    assert np.isfinite(res.sinr_db).any()


def test_aggregate_empty_layers_mesh_mode():
    mesh = MeshSurface(mesh_file="a.ply", mesh_sha="sha1")
    res = combine.aggregate(mesh, SceneConfig(), [], [], "stub")
    assert res.rss_dbm.shape == (0,)
    assert res.tx_names == []


def test_simulator_mesh_mode_computes_and_caches():
    with tempfile.TemporaryDirectory() as d:
        sim = Simulator(SceneConfig(), GridSpec(), engine=_StubEngine(), cache=d)
        sim.mesh_surface = MeshSurface(mesh_file="a.ply", mesh_sha="sha1")
        sim.mode = "mesh"
        sim.place("A", -5, 0)
        sim.place("B", 5, 0)
        r1 = sim.run()
        assert (r1.cache_hits, r1.cache_misses) == (0, 2)
        assert r1.best_rsrp_dbm.shape == (_StubEngine.n_faces,)
        assert r1.cell_centers is not None and r1.cell_centers.shape == (_StubEngine.n_faces, 3)
        r2 = sim.run()
        assert (r2.cache_hits, r2.cache_misses) == (2, 0)   # cached the 2nd time


def test_simulator_switching_mode_does_not_cross_contaminate_cache():
    with tempfile.TemporaryDirectory() as d:
        sim = Simulator(SceneConfig(), GridSpec(x_min=-10, x_max=10, y_min=-10, y_max=10,
                                                 cell_size=1.0),
                         engine=_StubEngine(), cache=d)
        sim.place("A", 0, 0)
        r_grid = sim.run()
        assert r_grid.cache_misses == 1                     # grid mode: fresh compute

        sim.mesh_surface = MeshSurface(mesh_file="a.ply", mesh_sha="sha1")
        sim.mode = "mesh"
        r_mesh = sim.run()
        assert r_mesh.cache_misses == 1                      # different key -> not a false hit
        assert r_mesh.best_rsrp_dbm.shape == (_StubEngine.n_faces,)

        sim.mode = "grid"
        r_grid2 = sim.run()
        assert (r_grid2.cache_hits, r_grid2.cache_misses) == (1, 0)  # grid layer still cached


def test_simulator_to_dict_roundtrips_mesh_mode():
    sim = Simulator(SceneConfig(), GridSpec(), engine=_StubEngine(), cache=None)
    sim.mesh_surface = MeshSurface(mesh_file="a.ply", mesh_sha="sha1")
    sim.mode = "mesh"
    sim.place("A", 1, 2)
    d = sim.to_dict()
    assert d["mode"] == "mesh" and d["mesh_surface"] == {"mesh_file": "a.ply", "mesh_sha": "sha1"}

    sim2 = Simulator(SceneConfig(), GridSpec(), engine=_StubEngine(), cache=None)
    sim2.load_dict(d)
    assert sim2.mode == "mesh"
    assert sim2.mesh_surface == MeshSurface(mesh_file="a.ply", mesh_sha="sha1")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} mesh-mode tests passed")
