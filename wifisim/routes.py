"""Route profiles: sample coverage metrics along CSV-defined walk/drive routes.

A *route* is an ordered polyline given as a CSV file with columns
``point_index,x,y,z`` (header optional).  This module

* parses route CSVs (:func:`parse_route_csv` / :func:`load_route_csv`),
* resamples the polyline at a fixed arc-length **interval**
  (:func:`resample_polyline`), and
* samples a metric grid not at the exact sample point but averaged over an
  adjustable **radius** around it (:func:`sample_grid`), which mimics a real
  measurement device moving around the nominal point.

For dB-valued metrics (RSRP/RSS in dBm, SINR in dB) the average inside the
radius is a *power* average (linear mean, then back to dB); the min/max inside
the radius are returned as an uncertainty band.  ``best_server`` uses the
modal (most frequent) server index inside the radius.

Everything is duck-typed against :class:`~wifisim.models.GridSpec`
(``cell_centers()``, ``z``) and :class:`~wifisim.models.SimulationResult`
(metric arrays), so it has no import-time dependency on heavy modules.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "Route", "parse_route_csv", "load_route_csv", "resample_polyline",
    "sample_grid", "route_profile", "render_profile_plot", "metric_array",
    "METRIC_INFO",
]

#: metric key -> (label, unit, is_dB_quantity)
METRIC_INFO = {
    "best_rsrp":  ("Best-server RSRP", "dBm", True),
    "rss":        ("Aggregate RSS",    "dBm", True),
    "sinr":       ("SINR",             "dB",  True),
    "best_server": ("Best server",     "index", False),
}

# Candidate attribute names on SimulationResult for each metric key.
_METRIC_ATTRS = {
    "best_rsrp":  ("best_rsrp_dbm", "best_rsrp", "rsrp_dbm", "best_rsrp_db"),
    "rss":        ("rss_dbm", "rss"),
    "sinr":       ("sinr_db", "sinr"),
    "best_server": ("best_server", "best_server_idx"),
}


def metric_array(result, metric: str) -> np.ndarray:
    """Return the (ny, nx) array for ``metric`` from a SimulationResult.

    Prefers :data:`wifisim.viz.METRIC_SPECS` when it can name the attribute,
    then falls back to well-known attribute names.
    """
    try:  # optional: use the viz registry if it exposes the attribute name
        from . import viz
        spec = getattr(viz, "METRIC_SPECS", {}).get(metric)
        if spec is not None:
            for a in ("attr", "field", "array", "key", "name"):
                nm = getattr(spec, a, None)
                if isinstance(nm, str) and hasattr(result, nm):
                    return np.asarray(getattr(result, nm))
            getter = getattr(spec, "getter", None) or getattr(spec, "get", None)
            if callable(getter):
                return np.asarray(getter(result))
    except Exception:
        pass
    for nm in _METRIC_ATTRS.get(metric, ()):
        if hasattr(result, nm):
            return np.asarray(getattr(result, nm))
    raise KeyError(f"metric {metric!r} not found on result "
                   f"(tried {_METRIC_ATTRS.get(metric, ())})")


# --------------------------------------------------------------------------- #
# Route parsing
# --------------------------------------------------------------------------- #
@dataclass
class Route:
    """An ordered 3D polyline (metres) with a display name."""
    name: str
    points: np.ndarray                      # (N, 3) float64, ordered
    source: str = ""                        # original filename, if any
    csv_text: str = field(default="", repr=False)  # kept for persistence

    @property
    def length_m(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(self.points, axis=0), axis=1)))

    def to_dict(self) -> dict:
        return {"name": self.name, "source": self.source,
                "n_points": int(len(self.points)),
                "length_m": round(self.length_m, 2),
                "points": self.points.round(4).tolist()}


def parse_route_csv(text: str, name: str = "route", source: str = "") -> Route:
    """Parse CSV text with columns ``point_index,x,y,z`` into a :class:`Route`.

    Tolerates a header row, blank lines, ``;`` or whitespace separators, and
    unsorted point indices (rows are sorted by ``point_index``).  A 3-column
    file (``x,y,z`` without an index) is accepted in file order.
    """
    rows: List[List[float]] = []
    for raw in io.StringIO(text):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for sep in (",", ";", "\t"):
            line = line.replace(sep, " ")
        toks = line.split()
        try:
            vals = [float(t) for t in toks]
        except ValueError:
            continue                         # header or junk line
        if len(vals) >= 4:
            rows.append(vals[:4])
        elif len(vals) == 3:
            rows.append([len(rows)] + vals)  # no index column
    if len(rows) < 2:
        raise ValueError(
            f"route {name!r}: need at least 2 valid 'point_index,x,y,z' rows "
            f"(got {len(rows)})")
    arr = np.asarray(rows, dtype=np.float64)
    arr = arr[np.argsort(arr[:, 0], kind="stable")]
    pts = arr[:, 1:4]
    # Drop exact consecutive duplicates (zero-length segments break nothing,
    # but are pointless).
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9
    return Route(name=name, points=pts[keep], source=source, csv_text=text)


def load_route_csv(path: str, name: Optional[str] = None) -> Route:
    import os
    with open(path, "r", errors="replace") as f:
        text = f.read()
    nm = name or os.path.splitext(os.path.basename(path))[0]
    return parse_route_csv(text, name=nm, source=os.path.abspath(path))


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #
def resample_polyline(points: np.ndarray, interval: float):
    """Sample a polyline every ``interval`` metres of 3D arc length.

    Returns ``(samples, dists)`` where ``samples`` is (M, 3) and ``dists`` the
    cumulative distance of each sample.  The first vertex and the exact final
    vertex are always included.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return pts.copy(), np.zeros(len(pts))
    interval = max(float(interval), 1e-3)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    d = np.arange(0.0, total, interval)
    if total - (d[-1] if len(d) else 0.0) > 1e-9:
        d = np.append(d, total)             # always include the end point
    # locate each sample distance on the polyline
    idx = np.clip(np.searchsorted(cum, d, side="right") - 1, 0, len(seg) - 1)
    t = (d - cum[idx]) / np.where(seg[idx] > 0, seg[idx], 1.0)
    samples = pts[idx] + (pts[idx + 1] - pts[idx]) * t[:, None]
    return samples, d


# --------------------------------------------------------------------------- #
# Grid sampling (mean/min/max within a radius)
# --------------------------------------------------------------------------- #
def sample_grid(values: np.ndarray, grid, pts: np.ndarray, radius: float,
                db_quantity: bool = True, categorical: bool = False):
    """Sample ``values`` (shape ``grid.shape``) at XY points within ``radius``.

    For each point, all grid cells whose centre lies within ``radius`` (in the
    XY plane) contribute; if none do (radius smaller than half a cell), the
    nearest cell is used.  Returns ``(mean, vmin, vmax)`` arrays of length
    ``len(pts)`` (NaN where only NaN cells fall inside).

    * ``db_quantity=True``  -> the mean is a linear-power mean re-expressed in dB.
    * ``categorical=True``  -> the "mean" is the modal value (e.g. best server).
    """
    values = np.asarray(values)
    X, Y = grid.cell_centers()               # (ny, nx)
    xf, yf, vf = X.ravel(), Y.ravel(), values.ravel().astype(np.float64)
    pts = np.asarray(pts, dtype=np.float64)
    r2 = float(radius) ** 2
    n = len(pts)
    mean = np.full(n, np.nan)
    vmin = np.full(n, np.nan)
    vmax = np.full(n, np.nan)

    for i, (px, py) in enumerate(pts[:, :2]):
        d2 = (xf - px) ** 2 + (yf - py) ** 2
        if r2 > 0:
            sel = d2 <= r2
            if not sel.any():
                sel = np.zeros_like(d2, dtype=bool)
                sel[np.argmin(d2)] = True
        else:
            sel = np.zeros_like(d2, dtype=bool)
            sel[np.argmin(d2)] = True
        v = vf[sel]
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        vmin[i], vmax[i] = float(v.min()), float(v.max())
        if categorical:
            ints = v.astype(np.int64)
            mean[i] = float(np.bincount(ints - ints.min()).argmax() + ints.min())
        elif db_quantity:
            mean[i] = float(10.0 * np.log10(np.mean(10.0 ** (v / 10.0))))
        else:
            mean[i] = float(v.mean())
    return mean, vmin, vmax


def route_profile(result, grid, metric: str, route: Route,
                  interval: float, radius: float) -> dict:
    """Full pipeline: resample the route, sample the metric, return a series."""
    label, unit, is_db = METRIC_INFO.get(metric, (metric, "", True))
    arr = metric_array(result, metric)
    samples, dists = resample_polyline(route.points, interval)
    mean, vmin, vmax = sample_grid(arr, grid, samples, radius,
                                   db_quantity=is_db,
                                   categorical=(metric == "best_server"))
    clean = lambda a: [None if not np.isfinite(v) else round(float(v), 2) for v in a]
    return {
        "name": route.name, "metric": metric, "label": label, "unit": unit,
        "interval": float(interval), "radius": float(radius),
        "distance": [round(float(d), 2) for d in dists],
        "value": clean(mean), "vmin": clean(vmin), "vmax": clean(vmax),
        "samples": samples.round(3).tolist(),
        "length_m": round(float(dists[-1]) if len(dists) else 0.0, 2),
    }


# --------------------------------------------------------------------------- #
# Plot rendering
# --------------------------------------------------------------------------- #
_ROUTE_COLORS = ("#008f95", "#ef7c00", "#a862a4", "#00963f")   # U-line palette


def render_profile_plot(profiles: Sequence[dict], title: str = "") -> str:
    """Render route profiles into a PNG data URL (min-max band + mean line)."""
    import base64
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.6, 3.4), dpi=110)
    unit = profiles[0]["unit"] if profiles else ""
    label = profiles[0]["label"] if profiles else ""
    for i, p in enumerate(profiles):
        c = _ROUTE_COLORS[i % len(_ROUTE_COLORS)]
        d = np.asarray(p["distance"], dtype=float)
        m = np.asarray([np.nan if v is None else v for v in p["value"]], float)
        lo = np.asarray([np.nan if v is None else v for v in p["vmin"]], float)
        hi = np.asarray([np.nan if v is None else v for v in p["vmax"]], float)
        if np.isfinite(lo).any():
            ax.fill_between(d, lo, hi, color=c, alpha=0.16, linewidth=0,
                            label=None)
        ax.plot(d, m, color=c, lw=1.8,
                label=f"{p['name']}  ({p['length_m']:.0f} m)")
    ax.set_xlabel("distance along route [m]")
    ax.set_ylabel(f"{label} [{unit}]" if unit else label)
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, lw=0.5)
    if profiles:
        ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
