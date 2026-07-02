"""Propagation engine interface.

An engine turns *one* transmitter (in a scene, on a grid) into a
:class:`~wifisim.models.CoverageLayer`.  Keeping the unit of work at
single-transmitter granularity is what makes per-transmitter caching natural:
the simulator asks the cache first and only calls the engine for the
transmitters whose signature changed.

Engines may override :meth:`compute_layers` to compute several transmitters in
one pass (the Sionna back-end does this, since a single ray-tracing solve can
return one layer per transmitter).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

from ..models import CoverageLayer, GridSpec, SceneConfig, Transmitter


class PropagationEngine(ABC):
    """Base class for all propagation back-ends."""

    #: Short human-readable name, also used in cache keys.
    name: str = "base"

    @property
    @abstractmethod
    def signature(self) -> str:
        """Stable hash of everything about the engine that affects results.

        Must incorporate the engine name, its version, and any solver settings
        (e.g. ``max_depth``, ``samples_per_tx``).  Changing solver settings must
        change this signature so stale cache entries are not reused.
        """

    @abstractmethod
    def compute_layer(
        self, scene: SceneConfig, grid: GridSpec, tx: Transmitter
    ) -> CoverageLayer:
        """Compute the coverage layer for a single transmitter."""

    def compute_layers(
        self, scene: SceneConfig, grid: GridSpec, txs: List[Transmitter]
    ) -> Dict[str, CoverageLayer]:
        """Compute layers for several transmitters.

        Default implementation loops over :meth:`compute_layer`.  Returns a dict
        keyed by transmitter ``signature``.
        """
        out: Dict[str, CoverageLayer] = {}
        for tx in txs:
            out[tx.signature] = self.compute_layer(scene, grid, tx)
        return out

    def close(self) -> None:  # pragma: no cover - optional resource cleanup
        """Release any heavy resources (GPU contexts, scenes).  Optional."""
