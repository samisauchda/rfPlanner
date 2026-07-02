"""Propagation engines and a small factory.

Use :func:`make_engine` to obtain an engine by name.  ``"auto"`` returns the
Sionna RT engine if it can be imported, otherwise the analytical engine.
"""
from __future__ import annotations

from typing import Any

from .base import PropagationEngine
from .analytical import AnalyticalEngine
from .sionna_engine import SionnaRTEngine, EngineUnavailable

__all__ = [
    "PropagationEngine",
    "AnalyticalEngine",
    "SionnaRTEngine",
    "EngineUnavailable",
    "make_engine",
    "available_engines",
]


def available_engines() -> dict:
    """Report which engines can run in the current environment."""
    info = {"analytical": True, "sionna_rt": False}
    try:
        import sionna.rt  # noqa: F401
        info["sionna_rt"] = True
    except Exception:
        info["sionna_rt"] = False
    return info


def make_engine(name: str = "auto", **kwargs: Any) -> PropagationEngine:
    """Construct an engine.

    Parameters
    ----------
    name:
        ``"analytical"``, ``"sionna_rt"``, or ``"auto"``.
    **kwargs:
        Forwarded to the engine constructor (e.g. ``max_depth``,
        ``samples_per_tx`` for Sionna).
    """
    name = (name or "auto").lower()
    if name == "analytical":
        return AnalyticalEngine()
    if name == "sionna_rt":
        return SionnaRTEngine(**kwargs)
    if name == "auto":
        try:
            return SionnaRTEngine(**kwargs)
        except EngineUnavailable:
            return AnalyticalEngine()
    raise ValueError(f"Unknown engine {name!r}")
