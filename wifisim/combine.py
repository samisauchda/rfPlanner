"""Aggregate per-transmitter coverage layers into system-level metrics.

Given one :class:`CoverageLayer` per (enabled) transmitter, compute, per grid
cell:

* **RSS**          - total received signal strength (linear power sum).
* **best RSRP**    - power from the strongest single transmitter.
* **best server**  - index of that strongest transmitter.
* **SINR**         - for the best server, using *co-channel* transmitters as
                     interference plus the thermal noise floor.

Co-channel interference is what makes this useful for WiFi planning: two APs on
the same channel interfere, on orthogonal channels they do not.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from . import config as cfg
from .models import CoverageLayer, GridSpec, MeshSurface, SceneConfig, SimulationResult, Transmitter

_NO_COVERAGE_DBM = -120.0  # below this, a cell is considered unserved


def _dbm_to_mw(dbm: np.ndarray) -> np.ndarray:
    return np.power(10.0, dbm / 10.0)


def aggregate(
    grid: "GridSpec | MeshSurface",
    scene: SceneConfig,
    layers: Sequence[CoverageLayer],
    txs: Sequence[Transmitter],
    engine_name: str = "unknown",
) -> SimulationResult:
    """Combine layers (parallel to ``txs``) into a :class:`SimulationResult`.

    ``grid`` may be a raster :class:`GridSpec` (layers shaped ``(ny, nx)``) or
    a mesh-native :class:`MeshSurface` (layers shaped ``(N,)``, one value per
    triangle) -- the arithmetic below is elementwise and shape-agnostic, so
    ``out_shape`` is the only place the two modes are distinguished.
    """
    out_shape = layers[0].rsrp_dbm.shape if layers else grid.shape
    if not layers:
        nan = np.full(out_shape, np.nan, dtype=np.float32)
        return SimulationResult(
            grid=grid, rss_dbm=nan.copy(), best_rsrp_dbm=nan.copy(),
            best_server=np.full(out_shape, -1, dtype=np.int32),
            sinr_db=nan.copy(), tx_names=[], engine=engine_name,
        )

    k = len(layers)
    dbm = np.stack([np.asarray(l.rsrp_dbm, dtype=np.float64) for l in layers], axis=0)  # (k, ny, nx)
    lin = _dbm_to_mw(dbm)                                                                # mW
    channels = np.array([l.channel for l in layers])
    bandwidths = np.array([tx.bandwidth_hz for tx in txs])

    # --- aggregate RSS ------------------------------------------------- #
    rss_lin = lin.sum(axis=0)
    rss_dbm = 10.0 * np.log10(np.maximum(rss_lin, 1e-30))

    # --- best server --------------------------------------------------- #
    best_server = np.argmax(dbm, axis=0).astype(np.int32)        # (ny, nx)
    best_dbm = np.take_along_axis(dbm, best_server[None], axis=0)[0]
    best_lin = _dbm_to_mw(best_dbm)

    # --- co-channel interference --------------------------------------- #
    # Sum of linear power per channel, then for the serving cell subtract the
    # serving power so only *other* co-channel TXs count as interference.
    unique_channels = np.unique(channels)
    per_channel_lin = np.zeros((len(unique_channels),) + out_shape)
    for i, ch in enumerate(unique_channels):
        mask = channels == ch
        per_channel_lin[i] = lin[mask].sum(axis=0)
    ch_index = {int(ch): i for i, ch in enumerate(unique_channels)}

    serving_channel = channels[best_server]                       # (ny, nx)
    serving_ch_idx = np.vectorize(ch_index.get)(serving_channel)
    same_ch_total = np.take_along_axis(
        per_channel_lin, serving_ch_idx[None], axis=0
    )[0]
    interference_lin = np.maximum(same_ch_total - best_lin, 0.0)

    # --- noise floor (per serving-TX bandwidth) ------------------------ #
    serving_bw = bandwidths[best_server]
    noise_dbm = np.vectorize(lambda b: cfg.noise_floor_dbm(b, scene.noise_figure_db))(serving_bw)
    noise_lin = _dbm_to_mw(noise_dbm)

    sinr_lin = best_lin / (interference_lin + noise_lin)
    sinr_db = 10.0 * np.log10(np.maximum(sinr_lin, 1e-30))

    # --- mark unserved cells ------------------------------------------- #
    unserved = best_dbm < _NO_COVERAGE_DBM
    best_server = np.where(unserved, -1, best_server).astype(np.int32)
    best_dbm = np.where(unserved, np.nan, best_dbm)
    sinr_db = np.where(unserved, np.nan, sinr_db)

    return SimulationResult(
        grid=grid,
        rss_dbm=rss_dbm.astype(np.float32),
        best_rsrp_dbm=best_dbm.astype(np.float32),
        best_server=best_server,
        sinr_db=sinr_db.astype(np.float32),
        tx_names=[t.name for t in txs],
        engine=engine_name,
    )
