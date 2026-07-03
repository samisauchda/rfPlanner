"""Propagation engines and a small factory.

Use :func:`make_engine` to obtain an engine by name.
"""
from __future__ import annotations

from typing import Any

from .base import PropagationEngine
from .sionna_engine import SionnaRTEngine, EngineUnavailable

__all__ = [
    "PropagationEngine",
    "SionnaRTEngine",
    "EngineUnavailable",
    "make_engine",
]


def make_engine(name: str = "sionna_rt", **kwargs: Any) -> PropagationEngine:
    """Construct an engine.

    Parameters
    ----------
    name:
        ``"sionna_rt"`` (or the alias ``"auto"``).
    **kwargs:
        Forwarded to :class:`SionnaRTEngine` (e.g. ``max_depth``,
        ``samples_per_tx``).
    """
    name = (name or "sionna_rt").lower()
    if name in ("sionna_rt", "auto"):
        return SionnaRTEngine(**kwargs)
    raise ValueError(f"Unknown engine {name!r}; expected 'sionna_rt'")
