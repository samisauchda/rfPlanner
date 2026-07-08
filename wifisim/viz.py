"""Rendering of simulation results.

Two entry points:

* :func:`render_overlay` - a clean, axis-free, transparency-aware PNG sized to
  the grid, returned as a base64 data URL plus the metadata (value range,
  colormap, geographic extent) a front-end needs to place and label it.  The
  image is **north-up**: its top row corresponds to ``y_max``.
* :func:`render_figure` - a fully annotated matplotlib figure (heatmap +
  colorbar + transmitter markers) for notebooks / CLI quick-looks.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.image as mimage
import matplotlib.pyplot as plt
import numpy as np

from .models import SimulationResult, Transmitter


# Signal-strength (RSSI) colormap: dBm -> RGB control points.  Classic RF
# planning ramp (blue = weak ... magenta = strong).  Registered with matplotlib
# under the name "rssi" so it can be referenced like any built-in colormap.
RSSI_STOPS = [
    (-120, (0, 0, 255)),
    (-110, (0, 125, 255)),
    (-100, (0, 255, 251)),
    (-90, (0, 255, 66)),
    (-80, (130, 255, 0)),
    (-70, (220, 255, 0)),
    (-60, (255, 219, 0)),
    (-50, (255, 137, 0)),
    (-40, (255, 0, 55)),
    (-30, (255, 0, 255)),
]
RSSI_VMIN, RSSI_VMAX = RSSI_STOPS[0][0], RSSI_STOPS[-1][0]


def _make_rssi_cmap() -> mcolors.LinearSegmentedColormap:
    lo, hi = RSSI_VMIN, RSSI_VMAX
    pts = [((dbm - lo) / (hi - lo), (r / 255, g / 255, b / 255))
           for dbm, (r, g, b) in RSSI_STOPS]
    return mcolors.LinearSegmentedColormap.from_list("rssi", pts, N=1024)


RSSI_CMAP = _make_rssi_cmap()
try:  # register once; ignore if already present (e.g. module re-import)
    matplotlib.colormaps.register(RSSI_CMAP, name="rssi")
except (ValueError, AttributeError):
    pass


def rssi_css_gradient() -> str:
    """CSS linear-gradient matching the RSSI colormap, for the web legend."""
    lo, hi = RSSI_VMIN, RSSI_VMAX
    stops = [f"rgb({r},{g},{b}) {((dbm - lo) / (hi - lo) * 100):.1f}%"
             for dbm, (r, g, b) in RSSI_STOPS]
    return "linear-gradient(90deg," + ",".join(stops) + ")"


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    unit: str
    vmin: float
    vmax: float
    cmap: str


METRIC_SPECS: Dict[str, MetricSpec] = {
    "best_rsrp": MetricSpec("best_rsrp", "Best-server RSRP", "dBm", RSSI_VMIN, RSSI_VMAX, "rssi"),
    "rss": MetricSpec("rss", "Aggregate RSS", "dBm", RSSI_VMIN, RSSI_VMAX, "rssi"),
    "sinr": MetricSpec("sinr", "Best-server SINR", "dB", -5.0, 35.0, "turbo"),
    "best_server": MetricSpec("best_server", "Best server", "tx", 0.0, 1.0, "tab20"),
}


def _metric_array(result: SimulationResult, metric: str) -> np.ndarray:
    return {
        "best_rsrp": result.best_rsrp_dbm,
        "rss": result.rss_dbm,
        "sinr": result.sinr_db,
        "best_server": result.best_server.astype(np.float32),
    }[metric]


def _norm_and_cmap(spec: MetricSpec, vmin: Optional[float], vmax: Optional[float]):
    vmin = spec.vmin if vmin is None else vmin
    vmax = spec.vmax if vmax is None else vmax
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    try:
        cmap = matplotlib.colormaps[spec.cmap]
    except (KeyError, AttributeError):
        cmap = cm.get_cmap(spec.cmap)
    return norm, cmap


def colorize(
    data: np.ndarray, spec: MetricSpec, vmin: Optional[float] = None, vmax: Optional[float] = None
) -> np.ndarray:
    """Map an array (any shape) to an RGBA uint8 image (NaN -> fully transparent)."""
    norm, cmap = _norm_and_cmap(spec, vmin, vmax)
    rgba = cmap(norm(np.ma.masked_invalid(data)))  # (..., 4) floats
    rgba[..., 3] = np.where(np.isfinite(data), rgba[..., 3], 0.0)
    return (rgba * 255).astype(np.uint8)


def render_overlay(
    result: SimulationResult,
    metric: str = "best_rsrp",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> dict:
    """Render a transparent, north-up heatmap overlay.

    Returns a dict with ``image`` (data URL), ``vmin``/``vmax``, ``cmap``,
    ``label``, ``unit``, and ``extent`` ``[x_min, x_max, y_min, y_max]``.
    """
    spec = METRIC_SPECS[metric]
    data = _metric_array(result, metric)
    rgba = colorize(data, spec, vmin, vmax)
    rgba = np.flipud(rgba)  # row 0 -> y_max (north up)

    buf = io.BytesIO()
    mimage.imsave(buf, rgba, format="png")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # Canonical value grid (row 0 = y_min) for client-side hover read-out.
    vals = np.ascontiguousarray(data, dtype=np.float32)
    vals_b64 = base64.b64encode(vals.tobytes()).decode("ascii")

    return {
        "mode": "grid",
        "image": f"data:image/png;base64,{b64}",
        "metric": metric,
        "label": spec.label,
        "unit": spec.unit,
        "vmin": spec.vmin if vmin is None else vmin,
        "vmax": spec.vmax if vmax is None else vmax,
        "cmap": spec.cmap,
        "extent": list(result.grid.extent),
        "values": vals_b64,                 # float32, row-major, canonical
        "shape": [int(vals.shape[0]), int(vals.shape[1])],  # [ny, nx]
    }


def render_mesh_overlay(
    result: SimulationResult,
    metric: str = "best_rsrp",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> dict:
    """Serialize a mesh-native result: one value + colour per triangle, plus
    the triangle centroids -- there is no rectangular image to rasterize, so
    (unlike :func:`render_overlay`) this returns scattered data for the
    front-end to draw itself (2-D dots, 3-D per-face mesh colouring).
    """
    spec = METRIC_SPECS[metric]
    data = _metric_array(result, metric)          # (N,)
    norm, cmap = _norm_and_cmap(spec, vmin, vmax)
    rgba = cmap(norm(np.ma.masked_invalid(data)))  # (N, 4) floats
    rgba[..., 3] = np.where(np.isfinite(data), rgba[..., 3], 0.0)
    colors = (rgba * 255).astype(np.uint8)

    vals_b64 = base64.b64encode(np.ascontiguousarray(data, dtype=np.float32).tobytes()).decode("ascii")
    colors_b64 = base64.b64encode(np.ascontiguousarray(colors, dtype=np.uint8).tobytes()).decode("ascii")
    centers = np.ascontiguousarray(result.cell_centers, dtype=np.float32)
    centers_b64 = base64.b64encode(centers.tobytes()).decode("ascii")

    return {
        "mode": "mesh",
        "metric": metric,
        "label": spec.label,
        "unit": spec.unit,
        "vmin": spec.vmin if vmin is None else vmin,
        "vmax": spec.vmax if vmax is None else vmax,
        "cmap": spec.cmap,
        "values": vals_b64,          # float32 (N,)
        "colors": colors_b64,        # uint8 (N,4) RGBA
        "cell_centers": centers_b64,  # float32 (N,3)
        "n_cells": int(data.shape[0]),
    }


def render_figure(
    result: SimulationResult,
    transmitters: Optional[List[Transmitter]] = None,
    metric: str = "best_rsrp",
    title: Optional[str] = None,
    figsize=(7.5, 6.5),
):
    """Return a fully annotated matplotlib ``Figure`` for the chosen metric."""
    spec = METRIC_SPECS[metric]
    data = _metric_array(result, metric)
    grid = result.grid

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        data, origin="lower", extent=grid.extent, cmap=spec.cmap,
        vmin=spec.vmin, vmax=spec.vmax, interpolation="nearest", aspect="equal",
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(f"{spec.label} [{spec.unit}]")

    if transmitters:
        for tx in transmitters:
            if not tx.enabled:
                continue
            x, y, _ = tx.position
            ax.plot(x, y, marker="^", color="white", markeredgecolor="black",
                    markersize=10, zorder=5)
            ax.annotate(tx.name, (x, y), textcoords="offset points", xytext=(6, 6),
                        color="white", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5))

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title or f"{spec.label} - engine: {result.engine}")
    fig.tight_layout()
    return fig


def save_figure(result, transmitters, metric, path, **kwargs) -> str:
    fig = render_figure(result, transmitters, metric, **kwargs)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
