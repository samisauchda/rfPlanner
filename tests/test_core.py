"""Tests for the wifisim core: signatures, caching, aggregation, rendering.

Run with:  python -m pytest tests/  -q     (or)     python tests/test_core.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wifisim import AntennaConfig, GridSpec, SceneConfig, Simulator, Transmitter
from wifisim.cache import LayerCache, layer_key
from wifisim.engines import AnalyticalEngine, make_engine
from wifisim import combine, viz


def test_signature_stability_and_sensitivity():
    a = Transmitter("ap", (0, 0, 3), tx_power_dbm=20)
    b = Transmitter("ap", (0, 0, 3), tx_power_dbm=20)
    assert a.signature == b.signature                      # same physics -> same sig
    assert a.signature == a.with_(name="renamed").signature  # name is cosmetic
    assert a.signature == a.with_(color="#fff").signature    # colour is cosmetic
    assert a.signature != a.moved_to(1, 0).signature         # moving changes sig
    assert a.signature != a.with_(tx_power_dbm=10).signature  # power changes sig
    assert a.signature != a.with_(channel=149).signature      # channel changes sig


def test_layer_key_depends_on_all_parts():
    eng = AnalyticalEngine()
    s1, s2 = SceneConfig(path_loss_exponent=2.5), SceneConfig(path_loss_exponent=3.0)
    g1, g2 = GridSpec(cell_size=0.5), GridSpec(cell_size=1.0)
    tx = Transmitter("ap", (0, 0, 3))
    base = layer_key(eng.signature, s1, g1, tx)
    assert base != layer_key(eng.signature, s2, g1, tx)   # scene matters
    assert base != layer_key(eng.signature, s1, g2, tx)   # grid matters
    assert base != layer_key(eng.signature, s1, g1, tx.moved_to(5, 5))  # tx matters


def test_cache_hit_miss_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        cache = LayerCache(cache_dir=d)
        eng = AnalyticalEngine()
        scene, grid = SceneConfig(), GridSpec()
        tx = Transmitter("ap", (1, 2, 3))
        layer = eng.compute_layer(scene, grid, tx)
        key = layer_key(eng.signature, scene, grid, tx)
        assert cache.get(key) is None                     # miss
        cache.put(key, layer)
        again = cache.get(key)                             # hit (memory)
        assert again is not None
        np.testing.assert_allclose(again.rsrp_dbm, layer.rsrp_dbm)
        # Force reload from disk by clearing memory tier.
        cache._mem.clear()
        reloaded = cache.get(key)
        assert reloaded is not None
        np.testing.assert_allclose(reloaded.rsrp_dbm, layer.rsrp_dbm, rtol=1e-4)


def test_simulator_only_recomputes_changed_tx():
    with tempfile.TemporaryDirectory() as d:
        sim = Simulator(SceneConfig(), GridSpec(x_min=-10, x_max=10, y_min=-10, y_max=10,
                                                cell_size=1.0),
                        engine="analytical", cache=d)
        sim.place("A", -5, 0); sim.place("B", 5, 0)
        r1 = sim.run()
        assert (r1.cache_hits, r1.cache_misses) == (0, 2)
        r2 = sim.run()
        assert (r2.cache_hits, r2.cache_misses) == (2, 0)   # all cached
        sim.move("B", 5, 3)
        r3 = sim.run()
        assert (r3.cache_hits, r3.cache_misses) == (1, 1)   # only B recomputed
        sim.edit("A", tx_power_dbm=10)
        r4 = sim.run()
        assert (r4.cache_hits, r4.cache_misses) == (1, 1)   # only A recomputed


def test_aggregation_shapes_and_sinr():
    grid = GridSpec(x_min=-10, x_max=10, y_min=-10, y_max=10, cell_size=1.0)
    scene = SceneConfig()
    eng = AnalyticalEngine()
    # Two co-channel APs (interfere) + one on another channel.
    txs = [Transmitter("A", (-5, 0, 3), channel=36),
           Transmitter("B", (5, 0, 3), channel=36),
           Transmitter("C", (0, 5, 3), channel=149)]
    layers = [eng.compute_layer(scene, grid, t) for t in txs]
    res = combine.aggregate(grid, scene, layers, txs, "analytical")
    assert res.rss_dbm.shape == grid.shape == res.sinr_db.shape
    assert res.best_server.min() >= -1 and res.best_server.max() < len(txs)
    # SINR should be finite where there is coverage.
    assert np.isfinite(res.sinr_db).any()


def test_force_bypasses_cache():
    with tempfile.TemporaryDirectory() as d:
        sim = Simulator(SceneConfig(), GridSpec(cell_size=2.0), engine="analytical", cache=d)
        sim.place("A", 0, 0)
        sim.run()
        r = sim.run(force=True)
        assert r.cache_misses == 1 and r.cache_hits == 0


def test_render_overlay_returns_data_url():
    sim = Simulator(SceneConfig(), GridSpec(cell_size=2.0), engine="analytical", cache=None)
    sim.place("A", 0, 0)
    res = sim.run()
    ov = viz.render_overlay(res, "best_rsrp")
    assert ov["image"].startswith("data:image/png;base64,")
    assert ov["extent"] == list(sim.grid.extent)


def test_antenna_patterns_run():
    grid = GridSpec(cell_size=2.0)
    scene, eng = SceneConfig(), AnalyticalEngine()
    for pat in ("iso", "dipole", "tr38901", "patch", "sector"):
        tx = Transmitter("ap", (0, 0, 3), antenna=AntennaConfig(pattern=pat))
        layer = eng.compute_layer(scene, grid, tx)
        assert np.isfinite(layer.rsrp_dbm).all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
