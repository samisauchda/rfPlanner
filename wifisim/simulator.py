"""Top-level orchestration.

:class:`Simulator` owns the scene, grid, engine, and cache, and exposes a small
API to place / edit / move / remove transmitters and run the simulation.  On
each :meth:`run`, it asks the cache for every enabled transmitter's layer and
calls the engine only for the cache misses, then aggregates the layers.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from . import combine
from .cache import LayerCache, layer_key
from .engines import PropagationEngine, make_engine
from .models import GridSpec, SceneConfig, SimulationResult, Transmitter


class Simulator:
    """Manage a set of transmitters and run cached coverage simulations.

    Parameters
    ----------
    scene, grid:
        The environment and measurement plane.
    engine:
        A :class:`PropagationEngine`, or a name (``"auto"``/``"analytical"``/
        ``"sionna_rt"``) passed to :func:`wifisim.engines.make_engine`.
    cache:
        A :class:`LayerCache`, or ``None`` to disable caching.
    """

    def __init__(
        self,
        scene: Optional[SceneConfig] = None,
        grid: Optional[GridSpec] = None,
        engine: "PropagationEngine | str" = "sionna_rt",
        cache: "LayerCache | None | str" = ".wifisim_cache",
        engine_kwargs: Optional[dict] = None,
    ) -> None:
        self.scene = scene or SceneConfig()
        self.grid = grid or GridSpec()
        self.engine: PropagationEngine = (
            engine if isinstance(engine, PropagationEngine)
            else make_engine(engine, **(engine_kwargs or {}))
        )
        if cache is None:
            self.cache = LayerCache(enabled=False)
        elif isinstance(cache, LayerCache):
            self.cache = cache
        else:
            self.cache = LayerCache(cache_dir=cache)
        self._txs: "Dict[str, Transmitter]" = {}

    # ------------------------------------------------------------------ #
    # Transmitter management
    # ------------------------------------------------------------------ #
    def add_transmitter(self, tx: Transmitter) -> Transmitter:
        """Add (or replace by name) a transmitter."""
        self._txs[tx.name] = tx
        return tx

    def place(self, name: str, x: float, y: float, z: float = 3.0, **kwargs) -> Transmitter:
        """Convenience: create and add a transmitter at ``(x, y, z)``."""
        tx = Transmitter(name=name, position=(float(x), float(y), float(z)), **kwargs)
        return self.add_transmitter(tx)

    def get(self, name: str) -> Transmitter:
        return self._txs[name]

    def move(self, name: str, x: float, y: float, z: Optional[float] = None) -> Transmitter:
        tx = self._txs[name].moved_to(x, y, z)
        self._txs[name] = tx
        return tx

    def edit(self, name: str, **changes) -> Transmitter:
        """Edit any transmitter field, e.g. ``edit("ap1", tx_power_dbm=23)``."""
        tx = self._txs[name].with_(**changes)
        self._txs[name] = tx
        return tx

    def remove(self, name: str) -> None:
        self._txs.pop(name, None)

    def clear_transmitters(self) -> None:
        self._txs.clear()

    @property
    def transmitters(self) -> List[Transmitter]:
        return list(self._txs.values())

    # ------------------------------------------------------------------ #
    # Running
    # ------------------------------------------------------------------ #
    def run(self, force: bool = False) -> SimulationResult:
        """Compute the aggregated coverage result.

        Parameters
        ----------
        force:
            If ``True``, bypass cached layers and recompute every transmitter.
        """
        t0 = time.perf_counter()
        self.cache.reset_counters()
        active = [t for t in self._txs.values() if t.enabled]

        layers = {}
        misses: List[Transmitter] = []
        n_hits = 0
        for tx in active:
            key = layer_key(self.engine.signature, self.scene, self.grid, tx)
            cached = None if force else self.cache.get(key)
            if cached is not None:
                layers[tx.signature] = cached
                n_hits += 1
            else:
                misses.append(tx)

        if misses:
            computed = self.engine.compute_layers(self.scene, self.grid, misses)
            for tx in misses:
                layer = computed[tx.signature]
                layers[tx.signature] = layer
                key = layer_key(self.engine.signature, self.scene, self.grid, tx)
                self.cache.put(key, layer)

        ordered_layers = [layers[t.signature] for t in active]
        result = combine.aggregate(
            self.grid, self.scene, ordered_layers, active, engine_name=self.engine.name
        )
        result.cache_hits = n_hits
        result.cache_misses = len(misses)
        result.timing_s = time.perf_counter() - t0
        return result

    # ------------------------------------------------------------------ #
    # (De)serialisation of the whole project
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        from dataclasses import asdict
        return {
            "scene": asdict(self.scene),
            "grid": asdict(self.grid),
            "engine": self.engine.name,
            "transmitters": [t.to_dict() for t in self._txs.values()],
        }

    def load_dict(self, data: dict) -> None:
        self.scene = SceneConfig.from_dict(data.get("scene", {}))
        self.grid = GridSpec(**data.get("grid", {}))
        self._txs = {}
        for td in data.get("transmitters", []):
            tx = Transmitter.from_dict(td)
            self._txs[tx.name] = tx
