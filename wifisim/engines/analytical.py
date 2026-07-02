"""Dependency-light analytical propagation engine.

This engine implements a vectorised **log-distance path-loss** model with proper
**antenna-pattern** directivity.  It does not trace rays and therefore ignores
scene geometry (walls, reflections); instead it is parameterised by a
path-loss exponent and optional spatially-correlated log-normal shadowing.

It exists so the full framework - caching, aggregation, web UI - runs and is
testable on any machine, and so results degrade gracefully when Sionna / a GPU
is unavailable.  For geometry-aware results, use :class:`SionnaRTEngine`.

All maths is vectorised over the grid for speed.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Tuple

import numpy as np

from .. import config as cfg
from ..models import AntennaConfig, CoverageLayer, GridSpec, SceneConfig, Transmitter
from .base import PropagationEngine

_MIN_DISTANCE_M = 1.0  # reference distance; avoids the 1/d^2 singularity at d->0


def _rotation_local_to_global(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Rotation matrix R (3x3) mapping antenna-local vectors to global frame.

    Intrinsic yaw (z) -> pitch (y) -> roll (x).  Boresight is local +x.
    """
    a, b, g = (math.radians(yaw_deg), math.radians(pitch_deg), math.radians(roll_deg))
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    cg, sg = math.cos(g), math.sin(g)
    rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rx = np.array([[1, 0, 0], [0, cg, -sg], [0, sg, cg]])
    return rz @ ry @ rx


def _smooth(field: np.ndarray, passes: int = 2) -> np.ndarray:
    """Cheap separable 3x3 box blur (a few passes) for spatial correlation."""
    out = field.astype(np.float64)
    for _ in range(passes):
        out = (out
               + np.pad(out, ((1, 0), (0, 0)))[:-1]
               + np.pad(out, ((0, 1), (0, 0)))[1:]
               + np.pad(out, ((0, 0), (1, 0)))[:, :-1]
               + np.pad(out, ((0, 0), (0, 1)))[:, 1:]) / 5.0
    return out


# --------------------------------------------------------------------------- #
# Antenna patterns -> gain in dBi as a function of local az / el (degrees)
# --------------------------------------------------------------------------- #
def antenna_gain_dbi(
    ant: AntennaConfig, az_deg: np.ndarray, el_deg: np.ndarray
) -> np.ndarray:
    """Vectorised element gain (dBi).

    ``az_deg`` is the azimuth offset from boresight (0 = boresight, local +x),
    ``el_deg`` the elevation offset (0 = horizon, +90 = local +z / up).
    """
    az = np.asarray(az_deg, dtype=np.float64)
    el = np.asarray(el_deg, dtype=np.float64)

    if ant.pattern == "iso":
        return np.zeros_like(az)

    if ant.pattern == "dipole":
        # Half-wave dipole, axis along local z. theta = angle from axis.
        theta = np.radians(90.0 - el)            # el=90 -> along axis -> null
        s = np.sin(theta)
        s = np.where(np.abs(s) < 1e-6, 1e-6, s)
        f = np.cos(0.5 * math.pi * np.cos(theta)) / s
        directivity = 1.64 * f * f               # peak 1.64 (2.15 dBi) at theta=90
        directivity = np.maximum(directivity, 1e-6)
        return 10.0 * np.log10(directivity)

    if ant.pattern == "tr38901":
        # 3GPP TR 38.901 single-element sector pattern (max gain 8 dBi default).
        theta = 90.0 - el                        # 90 = horizon in TR38.901 convention
        a_v = -np.minimum(12.0 * ((theta - 90.0) / 65.0) ** 2, 30.0)
        a_h = -np.minimum(12.0 * (az / 65.0) ** 2, 30.0)
        a = -np.minimum(-(a_v + a_h), 30.0)
        return ant.boresight_gain_dbi + a

    if ant.pattern in ("patch", "sector"):
        # Parametric pattern from half-power beamwidths + front-to-back ratio.
        az_term = 12.0 * (az / ant.az_hpbw_deg) ** 2
        el_term = 12.0 * (el / ant.el_hpbw_deg) ** 2
        atten = np.minimum(az_term + el_term, ant.front_to_back_db)
        return ant.boresight_gain_dbi - atten

    if ant.pattern == "msi":
        # Real principal-plane pattern (sum-of-planes), evaluated from the file.
        from .. import antenna_msi
        return antenna_msi.gain_dbi(ant, az, el)

    raise ValueError(f"Unsupported pattern {ant.pattern!r}")


def array_gain_db(ant: AntennaConfig) -> float:
    """Coherent broadside array gain (dB) from element count.

    A planar array of ``N`` elements yields up to ``10*log10(N)`` dB of array
    gain at broadside.  This is an upper bound used by the analytical model.
    """
    n = max(1, ant.num_rows * ant.num_cols)
    return 10.0 * math.log10(n)


class AnalyticalEngine(PropagationEngine):
    """Log-distance propagation with antenna directivity."""

    name = "analytical"
    version = "1.0"

    @property
    def signature(self) -> str:
        return hashlib.sha256(
            json.dumps({"engine": self.name, "version": self.version}).encode()
        ).hexdigest()[:12]

    def compute_layer(
        self, scene: SceneConfig, grid: GridSpec, tx: Transmitter
    ) -> CoverageLayer:
        X, Y = grid.cell_centers()                 # (ny, nx)
        tx_x, tx_y, tx_z = tx.position
        dx = X - tx_x
        dy = Y - tx_y
        dz = grid.z - tx_z
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        dist = np.maximum(dist, _MIN_DISTANCE_M)

        # --- log-distance path loss --------------------------------------- #
        lam = cfg.wavelength_m(tx.frequency_hz)
        pl_d0 = 20.0 * np.log10(4.0 * math.pi * _MIN_DISTANCE_M / lam)  # FSPL @ 1 m
        pl = pl_d0 + 10.0 * scene.path_loss_exponent * np.log10(dist / _MIN_DISTANCE_M)

        # --- antenna directivity ------------------------------------------ #
        # direction TX->cell in antenna-local frame
        u = np.stack([dx, dy, np.full_like(dx, dz)], axis=-1)  # (ny, nx, 3)
        u = u / np.linalg.norm(u, axis=-1, keepdims=True)
        rot = _rotation_local_to_global(*tx.orientation)
        u_local = u @ rot                                       # u_local = R^T u
        az = np.degrees(np.arctan2(u_local[..., 1], u_local[..., 0]))
        el = np.degrees(np.arcsin(np.clip(u_local[..., 2], -1.0, 1.0)))
        gain = antenna_gain_dbi(tx.antenna, az, el) + array_gain_db(tx.antenna)

        rsrp = tx.tx_power_dbm + gain - pl

        # --- optional spatially-correlated shadowing ---------------------- #
        if scene.shadowing_std_db > 0:
            seed = int(hashlib.sha256(
                (tx.signature + grid.signature + scene.signature).encode()
            ).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            noise = _smooth(rng.standard_normal(rsrp.shape), passes=3)
            noise *= scene.shadowing_std_db / (noise.std() + 1e-9)
            rsrp = rsrp + noise

        return CoverageLayer(
            rsrp_dbm=rsrp.astype(np.float32),
            tx_signature=tx.signature,
            tx_name=tx.name,
            channel=tx.channel,
            engine=self.name,
            meta={
                "path_loss_exponent": scene.path_loss_exponent,
                "frequency_hz": tx.frequency_hz,
                "peak_gain_dbi": float(np.nanmax(gain)),
            },
        )
