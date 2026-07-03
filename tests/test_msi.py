"""Tests for MSI antenna-pattern parsing, evaluation, and cache integration.

A small synthetic .msi file is generated so the tests are self-contained.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wifisim import antenna_msi, AntennaConfig, GridSpec, SceneConfig, Simulator, Transmitter
from wifisim.engines.base import PropagationEngine
from wifisim.models import CoverageLayer


class _StubEngine(PropagationEngine):
    name = "stub"

    @property
    def signature(self) -> str:
        return "stub000000000"

    def compute_layer(self, scene, grid, tx):
        ny, nx = grid.shape
        return CoverageLayer(
            rsrp_dbm=np.full((ny, nx), -60.0, dtype=np.float32),
            tx_signature=tx.signature,
            tx_name=tx.name,
            channel=tx.channel,
            engine=self.name,
        )


def _write_msi(path: Path, gain_dbi=15.0, az_hpbw=65.0, el_hpbw=40.0, name="Test Sector"):
    """Write a synthetic sector-like MSI file (0 dB atten at boresight)."""
    def atten(d, hpbw, cap=25.0):
        dd = d if d <= 180 else d - 360                 # wrap to [-180,180]
        return min(12.0 * (dd / hpbw) ** 2, cap)
    lines = [f"NAME {name}", "FREQUENCY 5500", f"GAIN {gain_dbi} dBi", "HORIZONTAL 360"]
    lines += [f"{d} {atten(d, az_hpbw):.3f}" for d in range(360)]
    lines += ["VERTICAL 360"]
    lines += [f"{d} {atten(d, el_hpbw):.3f}" for d in range(360)]
    path.write_text("\n".join(lines) + "\n")


def test_parse_and_tables():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "ant.msi"
        _write_msi(f, gain_dbi=15.0)
        tab = antenna_msi.load_tables(str(f))
        assert tab.peak_gain_dbi == 15.0
        assert tab.gv_dbn.shape == (181,) and tab.gh_dbn.shape == (360,)
        # both cuts must peak at 0 dB (boresight)
        assert tab.gv_dbn.max() == 0.0 and tab.gh_dbn.max() == 0.0


def test_gain_boresight_and_backlobe():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "ant.msi"
        _write_msi(f, gain_dbi=15.0)
        ant = antenna_msi.make_msi_antenna(str(f))
        g0 = antenna_msi.gain_dbi(ant, np.array([0.0]), np.array([0.0]))[0]
        gback = antenna_msi.gain_dbi(ant, np.array([180.0]), np.array([0.0]))[0]
        gup = antenna_msi.gain_dbi(ant, np.array([0.0]), np.array([90.0]))[0]
        assert abs(g0 - 15.0) < 0.1                      # boresight == peak gain
        assert gback < g0 - 20                           # strong front-to-back
        assert gup < g0 - 20                             # nulls overhead


def test_make_msi_antenna_signature_tracks_file():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "ant.msi"
        _write_msi(f, gain_dbi=15.0)
        a1 = antenna_msi.make_msi_antenna(str(f))
        assert a1.pattern == "msi" and a1.msi_sha and a1.boresight_gain_dbi == 15.0
        sig1 = a1.signature
        _write_msi(f, gain_dbi=12.0)                     # change file contents
        a2 = antenna_msi.make_msi_antenna(str(f))
        assert a2.msi_sha != a1.msi_sha                  # hash changed
        assert a2.signature != sig1                      # cache will invalidate


def test_msi_requires_file():
    try:
        AntennaConfig(pattern="msi")                     # missing msi_file
    except ValueError:
        return
    raise AssertionError("AntennaConfig(pattern='msi') without file should raise")


def test_simulator_caches_and_invalidates_on_file_change():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "ant.msi"
        _write_msi(f, gain_dbi=15.0)
        sim = Simulator(SceneConfig(), GridSpec(cell_size=2.0),
                        engine=_StubEngine(), cache=str(Path(d) / "cache"))
        sim.add_transmitter(Transmitter("AP", (0, 0, 4),
                                        antenna=antenna_msi.make_msi_antenna(str(f))))
        r1 = sim.run()
        assert (r1.cache_hits, r1.cache_misses) == (0, 1)
        r2 = sim.run()
        assert (r2.cache_hits, r2.cache_misses) == (1, 0)
        # change the pattern file -> new antenna signature -> recompute
        _write_msi(f, gain_dbi=10.0)
        sim.edit("AP", antenna=antenna_msi.make_msi_antenna(str(f)))
        r3 = sim.run()
        assert (r3.cache_hits, r3.cache_misses) == (0, 1)


def test_directivity_efficiency_reported():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "ant.msi"
        _write_msi(f)
        tab = antenna_msi.load_tables(str(f))
        d_dbi, eff = antenna_msi.compute_pattern_stats(tab)
        assert np.isfinite(d_dbi) and eff > 0.0
        # NB: efficiency > 1 here only because the synthetic file's declared
        # GAIN exceeds the directivity its own (narrow) shape supports; real
        # MSI files have self-consistent gain.


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} MSI tests passed")
