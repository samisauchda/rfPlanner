"""MSI antenna-pattern support for wifisim.

Reads ``.msi`` principal-plane (horizontal + vertical cut) antenna files and
makes them usable as a first-class antenna type across the framework:

* :func:`make_msi_antenna` -> an :class:`~wifisim.models.AntennaConfig` with a
  content hash, so the layer cache invalidates only when the file changes.
* :func:`gain_dbi` -> absolute element gain (dBi) in wifisim's local
  ``(az, el)`` convention, used by the **analytical** engine (pure NumPy, no
  Sionna needed).
* :func:`register_in_sionna` -> registers a DrJit-traceable pattern with Sionna
  RT and returns the name to pass to ``PlanarArray(pattern=...)``, used by the
  **Sionna** engine.

The 3D pattern is reconstructed from the two cuts with the standard
sum-of-planes method ``G(theta, phi) = Gv(theta) + Gh(phi)`` (in dB).  This
tends to *overestimate* directivity for directive antennas; the boresight gain
is pinned to the file's GAIN value (the link-budget-relevant quantity) and the
implied efficiency is reported so the approximation error is visible.

Angle conventions
-----------------
Sionna / internal spherical:  theta=0 zenith, theta=pi/2 horizon, theta=pi
nadir; phi=0 boresight (+x), increasing counter-clockwise.
wifisim local antenna frame:  az=0 boresight (+x, CCW+), el=0 horizon (+up).
The mapping is therefore ``theta = 90 - el`` (deg) and ``phi = az`` (deg).

This module is import-safe without Sionna: only :func:`register_in_sionna`
imports ``drjit`` / ``mitsuba`` / ``sionna``, and it does so lazily.
"""
from __future__ import annotations

import hashlib
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .models import AntennaConfig

LOG10_OVER_20 = np.log(10.0) / 20.0


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_msi_file(filepath: str):
    """Parse an MSI file -> (horizontal, vertical, freq_mhz, gain_dbi, name).

    ``horizontal`` / ``vertical`` are attenuation arrays in dB (>= 0) sampled on
    a 360-point, 1-degree grid.
    """
    with open(filepath, "r", errors="replace") as f:
        lines = f.readlines()

    name, freq_mhz, gain_dbi = "", 0.0, 0.0
    h_start = v_start = None
    h_count = v_count = 360

    for i, line in enumerate(lines):
        s = line.strip()
        up = s.upper()
        if up.startswith("NAME"):
            name = s[4:].strip()
        elif up.startswith("FREQUENCY"):
            try:
                freq_mhz = float(s.split()[1])
            except (IndexError, ValueError):
                pass
        elif up.startswith("GAIN"):
            parts = s.split()
            try:
                gain_value = float(parts[1])
                if len(parts) >= 3 and parts[2].lower() == "dbd":
                    gain_dbi = gain_value + 2.15
                else:
                    gain_dbi = gain_value
            except (IndexError, ValueError):
                pass
        elif up.startswith("HORIZONTAL"):
            parts = s.split()
            if len(parts) >= 2:
                h_count = int(float(parts[1]))
            h_start = i + 1
        elif up.startswith("VERTICAL"):
            parts = s.split()
            if len(parts) >= 2:
                v_count = int(float(parts[1]))
            v_start = i + 1

    if h_start is None or v_start is None:
        raise ValueError(f"No HORIZONTAL/VERTICAL block found in {filepath}")

    def extract(start, count):
        angles, values = [], []
        for j in range(start, start + count):
            toks = lines[j].split()[:2]
            angles.append(float(toks[0]))
            values.append(float(toks[1]))
        angles = np.asarray(angles, dtype=float)
        values = np.asarray(values, dtype=float)
        if count != 360 or not np.allclose(angles, np.arange(360)):
            grid = np.arange(360.0)
            values = np.interp(grid, angles, values, period=360.0)
        return values

    return extract(h_start, h_count), extract(v_start, v_count), freq_mhz, gain_dbi, name


# --------------------------------------------------------------------------- #
# Tables (memoised)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MSITables:
    gv_dbn: np.ndarray          # (181,) theta = 0..180 deg, <= 0 dB
    gh_dbn: np.ndarray          # (360,) MSI azimuth degree index, <= 0 dB
    peak_gain_dbi: float
    peak_gain_linear: float
    azimuth_ccw: bool
    min_dbn: float
    sha: str
    name: str
    freq_mhz: float


_TABLE_CACHE: Dict[Tuple, MSITables] = {}


def file_sha(filepath: str) -> str:
    """SHA-256 (16 hex) of the file contents - drives cache invalidation."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_tables(filepath: str, *, azimuth_ccw: bool = True,
                vertical_down_positive: bool = True,
                min_dbn: float = -60.0) -> MSITables:
    """Build (and memoise) normalised-gain lookup tables for an MSI file."""
    path = os.path.abspath(filepath)
    sha = file_sha(path)
    key = (path, sha, azimuth_ccw, vertical_down_positive, min_dbn)
    cached = _TABLE_CACHE.get(key)
    if cached is not None:
        return cached

    horizontal, vertical, freq_mhz, gain_dbi, name = parse_msi_file(path)

    theta_deg = np.arange(181)
    if vertical_down_positive:
        msi_idx = (270 + theta_deg) % 360     # theta=0 -> MSI 270 (zenith)
    else:
        msi_idx = (90 - theta_deg) % 360
    gv_dbn = np.maximum(-vertical[msi_idx].astype(np.float64), min_dbn)
    gh_dbn = np.maximum(-horizontal.astype(np.float64), min_dbn)

    if gv_dbn.max() < -0.5 or gh_dbn.max() < -0.5:
        warnings.warn(
            f"MSI cuts do not peak at 0 dB (V {gv_dbn.max():.2f}, "
            f"H {gh_dbn.max():.2f}); check angle conventions for {name!r}.")

    tables = MSITables(
        gv_dbn=gv_dbn, gh_dbn=gh_dbn,
        peak_gain_dbi=float(gain_dbi),
        peak_gain_linear=float(10.0 ** (gain_dbi / 10.0)),
        azimuth_ccw=azimuth_ccw, min_dbn=float(min_dbn),
        sha=sha, name=name or os.path.splitext(os.path.basename(path))[0],
        freq_mhz=float(freq_mhz),
    )
    _TABLE_CACHE[key] = tables
    return tables


# --------------------------------------------------------------------------- #
# Pure-NumPy evaluation (sum-of-planes)
# --------------------------------------------------------------------------- #
def gain_dbn_numpy(theta, phi, gv_dbn, gh_dbn, azimuth_ccw=True, min_dbn=-60.0):
    """Normalised sum-of-planes gain [dB <= 0] at (theta, phi) in radians."""
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)

    t = np.clip(theta, 0.0, np.pi) / np.pi * 180.0
    i0 = np.minimum(np.floor(t), 179.0)
    wv = t - i0
    i0 = i0.astype(np.int64)
    gv = gv_dbn[i0] * (1.0 - wv) + gv_dbn[i0 + 1] * wv

    p = np.degrees(phi) if azimuth_ccw else -np.degrees(phi)
    p = np.mod(p, 360.0)
    j0 = np.floor(p)
    wh = p - j0
    j0 = j0.astype(np.int64) % 360
    j1 = (j0 + 1) % 360
    gh = gh_dbn[j0] * (1.0 - wh) + gh_dbn[j1] * wh
    return np.maximum(gv + gh, min_dbn)


def gain_dbi(ant: AntennaConfig, az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
    """Absolute element gain (dBi) for an MSI antenna in wifisim's (az, el).

    ``az_deg`` = azimuth offset from boresight (0 = boresight, CCW positive),
    ``el_deg`` = elevation (0 = horizon, +90 = up).
    """
    tab = load_tables(ant.msi_file, azimuth_ccw=ant.msi_azimuth_ccw,
                      vertical_down_positive=ant.msi_vertical_down_positive)
    theta = np.radians(90.0 - np.asarray(el_deg, dtype=np.float64))   # el -> theta
    phi = np.radians(np.asarray(az_deg, dtype=np.float64))            # az -> phi
    g_dbn = gain_dbn_numpy(theta, phi, tab.gv_dbn, tab.gh_dbn,
                           tab.azimuth_ccw, tab.min_dbn)
    return tab.peak_gain_dbi + g_dbn


def compute_pattern_stats(tab: MSITables, n_theta=361, n_phi=720):
    """Implied directivity (dBi) and radiation efficiency of the reconstruction."""
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(theta, phi, indexing="ij")
    g_lin = 10.0 ** (gain_dbn_numpy(T, P, tab.gv_dbn, tab.gh_dbn,
                                    tab.azimuth_ccw, tab.min_dbn) / 10.0)
    integral = np.sum(g_lin * np.sin(theta)[:, None]) * (theta[1] - theta[0]) * (phi[1] - phi[0])
    directivity = 4.0 * np.pi * g_lin.max() / integral
    efficiency = tab.peak_gain_linear / directivity
    return float(10.0 * np.log10(directivity)), float(efficiency)


# --------------------------------------------------------------------------- #
# AntennaConfig builder
# --------------------------------------------------------------------------- #
def make_msi_antenna(filepath: str, *, polarization: str = "V",
                     num_rows: int = 1, num_cols: int = 1,
                     azimuth_ccw: bool = True,
                     vertical_down_positive: bool = True,
                     min_dbn: float = -60.0) -> AntennaConfig:
    """Build an :class:`AntennaConfig` for an MSI pattern file.

    The returned config carries the file's content hash and peak gain, so it has
    a stable cache signature and works in both engines.
    """
    tab = load_tables(filepath, azimuth_ccw=azimuth_ccw,
                      vertical_down_positive=vertical_down_positive, min_dbn=min_dbn)
    return AntennaConfig(
        pattern="msi",
        num_rows=num_rows, num_cols=num_cols, polarization=polarization,
        boresight_gain_dbi=tab.peak_gain_dbi,
        msi_file=os.path.abspath(filepath), msi_sha=tab.sha,
        msi_azimuth_ccw=azimuth_ccw,
        msi_vertical_down_positive=vertical_down_positive,
    )


def describe(filepath: str) -> dict:
    """Lightweight header summary for UI listings (no heavy table build)."""
    horizontal, vertical, freq_mhz, gain_dbi_v, name = parse_msi_file(filepath)
    return {
        "file": os.path.abspath(filepath),
        "name": name or os.path.splitext(os.path.basename(filepath))[0],
        "gain_dbi": round(float(gain_dbi_v), 2),
        "freq_mhz": float(freq_mhz),
    }


# --------------------------------------------------------------------------- #
# Sionna RT registration (lazy drjit/mitsuba/sionna import)
# --------------------------------------------------------------------------- #
_SIONNA_REGISTERED: Dict[str, str] = {}   # cache-key -> registered pattern name


def register_in_sionna(ant: AntennaConfig, verbose: bool = False) -> str:
    """Register an MSI pattern with Sionna RT; return the PlanarArray name.

    Idempotent: a given (file, conventions) is registered once per process.
    """
    import drjit as dr
    import mitsuba as mi
    from sionna.rt import register_antenna_pattern
    try:
        from sionna.rt import PolarizedAntennaPattern
        _have_polarized = True
    except ImportError:                       # pragma: no cover
        from sionna.rt import AntennaPattern
        _have_polarized = False

    tab = load_tables(ant.msi_file, azimuth_ccw=ant.msi_azimuth_ccw,
                      vertical_down_positive=ant.msi_vertical_down_positive)
    pattern_name = f"msi_{tab.sha}"
    if pattern_name in _SIONNA_REGISTERED:
        return pattern_name

    gv_buf = mi.Float(tab.gv_dbn.astype(np.float32))
    gh_buf = mi.Float(tab.gh_dbn.astype(np.float32))
    peak_amp = float(np.sqrt(tab.peak_gain_linear))
    phi_sign = 1.0 if tab.azimuth_ccw else -1.0
    rad2deg = float(180.0 / np.pi)
    floor_dbn = float(tab.min_dbn)

    def v_msi_pattern(theta, phi):
        t = dr.clip(theta * rad2deg, 0.0, 180.0)
        i0f = dr.minimum(dr.floor(t), 179.0)
        wv = t - i0f
        i0 = mi.UInt32(i0f)
        gv = dr.lerp(dr.gather(mi.Float, gv_buf, i0),
                     dr.gather(mi.Float, gv_buf, i0 + 1), wv)
        p = phi_sign * phi * rad2deg
        p = p - 360.0 * dr.floor(p / 360.0)
        j0f = dr.floor(p)
        wh = p - j0f
        j0 = mi.UInt32(j0f) % 360
        j1 = (j0 + 1) % 360
        gh = dr.lerp(dr.gather(mi.Float, gh_buf, j0),
                     dr.gather(mi.Float, gh_buf, j1), wh)
        g_dbn = dr.maximum(gv + gh, floor_dbn)
        amp = peak_amp * dr.exp(g_dbn * LOG10_OVER_20)
        return mi.Complex2f(amp, 0.0)

    if _have_polarized:
        def factory(*, polarization="V", polarization_model="tr38901_2"):
            return PolarizedAntennaPattern(
                v_pattern=v_msi_pattern, polarization=polarization,
                polarization_model=polarization_model)
        register_antenna_pattern(pattern_name, factory)
    else:                                     # pragma: no cover
        class _MSIPattern(AntennaPattern):
            def __init__(self, **kwargs):
                if kwargs.get("polarization", "V") not in (None, "V"):
                    raise ValueError("fallback supports polarization='V' only")
                super().__init__()

            @property
            def patterns(self):
                return [v_msi_pattern]
        register_antenna_pattern(pattern_name, _MSIPattern)

    _SIONNA_REGISTERED[pattern_name] = pattern_name
    if verbose:
        d_dbi, eff = compute_pattern_stats(tab)
        print(f"Registered MSI pattern '{pattern_name}' ({tab.name}): "
              f"peak {tab.peak_gain_dbi:.2f} dBi, implied directivity "
              f"{d_dbi:.2f} dBi, efficiency {eff*100:.1f}%")
    return pattern_name
