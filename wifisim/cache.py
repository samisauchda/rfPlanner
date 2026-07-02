"""Per-transmitter coverage-layer cache.

The expensive unit of work is computing one transmitter's coverage layer.  We
cache each layer under a key that is the hash of everything it depends on::

    key = H( engine_signature | scene_signature | grid_signature | tx_signature )

Because a transmitter's signature is derived only from its physics-affecting
fields (position, orientation, power, channel, antenna), moving or editing one
transmitter changes only its key - every other transmitter is a cache hit.  The
same layer is reused across runs (disk) and within a run (memory).

Layers are stored on disk as compressed ``.npz`` files, so they survive process
restarts.  An in-memory ``OrderedDict`` provides a fast LRU in front of disk.
"""
from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np

from .models import CoverageLayer, GridSpec, SceneConfig, Transmitter


def layer_key(
    engine_signature: str,
    scene: SceneConfig,
    grid: GridSpec,
    tx: Transmitter,
) -> str:
    """Deterministic cache key for one transmitter's layer."""
    raw = "|".join([engine_signature, scene.signature, grid.signature, tx.signature])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


class LayerCache:
    """Two-tier (memory + disk) cache of :class:`CoverageLayer` objects.

    Parameters
    ----------
    cache_dir:
        Directory for ``.npz`` layer files.  Created if missing.
    mem_capacity:
        Maximum number of layers kept in the in-memory LRU.
    enabled:
        If ``False``, every lookup misses (useful for benchmarking).
    """

    def __init__(
        self,
        cache_dir: str | os.PathLike = ".wifisim_cache",
        mem_capacity: int = 256,
        enabled: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.mem_capacity = mem_capacity
        self.enabled = enabled
        self._mem: "OrderedDict[str, CoverageLayer]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------ #
    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.npz"

    def _mem_put(self, key: str, layer: CoverageLayer) -> None:
        self._mem[key] = layer
        self._mem.move_to_end(key)
        while len(self._mem) > self.mem_capacity:
            self._mem.popitem(last=False)

    # ------------------------------------------------------------------ #
    def get(self, key: str) -> Optional[CoverageLayer]:
        """Return a cached layer or ``None``.  Updates hit/miss counters."""
        if not self.enabled:
            self.misses += 1
            return None
        if key in self._mem:
            self._mem.move_to_end(key)
            self.hits += 1
            return self._mem[key]
        path = self._path(key)
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as npz:
                    layer = CoverageLayer.from_npz(npz)
                self._mem_put(key, layer)
                self.hits += 1
                return layer
            except Exception:
                # Corrupt entry: drop it and treat as a miss.
                try:
                    path.unlink()
                except OSError:
                    pass
        self.misses += 1
        return None

    def put(self, key: str, layer: CoverageLayer) -> None:
        """Store a layer in memory and on disk."""
        if not self.enabled:
            return
        self._mem_put(key, layer)
        # NB: np.savez* appends ".npz" if the name lacks it, so the temp file
        # must already end in ".npz" to keep the rename target predictable.
        tmp = self.cache_dir / f".{key}.tmp.npz"
        try:
            np.savez_compressed(tmp, **layer.to_npz_dict())
            os.replace(tmp, self._path(key))  # atomic on the same filesystem
        except Exception:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass

    def contains(self, key: str) -> bool:
        return key in self._mem or self._path(key).exists()

    # ------------------------------------------------------------------ #
    def reset_counters(self) -> None:
        self.hits = 0
        self.misses = 0

    def clear(self, disk: bool = True) -> int:
        """Empty the cache.  Returns the number of disk files removed."""
        self._mem.clear()
        removed = 0
        if disk:
            for p in self.cache_dir.glob("*.npz"):
                try:
                    p.unlink()
                    removed += 1
                except OSError:
                    pass
        return removed

    def stats(self) -> dict:
        n_disk = sum(1 for _ in self.cache_dir.glob("*.npz"))
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "mem_entries": len(self._mem),
            "disk_entries": n_disk,
            "cache_dir": str(self.cache_dir),
        }
