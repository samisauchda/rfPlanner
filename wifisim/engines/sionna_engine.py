"""NVIDIA Sionna RT ray-tracing propagation engine.

Written against the **Sionna RT 1.x / 2.x** API, where propagation is computed
with solver objects (``RadioMapSolver`` / ``PathSolver``) rather than the legacy
``scene.compute_paths`` / ``scene.coverage_map`` methods.

Sionna (and its Dr.Jit / Mitsuba / TensorFlow stack and a GPU) is an optional
dependency: the import is performed lazily inside :meth:`__init__` so that the
rest of ``wifisim`` works without it.  If Sionna is missing, constructing this
engine raises ``EngineUnavailable`` and the simulator falls back to the
analytical engine.

Per-transmitter caching note
----------------------------
A single ``RadioMapSolver`` call returns one map layer per transmitter, but all
transmitters in that call share ``scene.frequency`` and ``scene.tx_array``.  We
therefore group the *dirty* transmitters by ``(frequency, antenna)`` and run one
solve per group, slicing out each transmitter's layer for the cache.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from ..models import AntennaConfig, CoverageLayer, GridSpec, SceneConfig, Transmitter
from .base import PropagationEngine

_RSS_FLOOR_DBM = -200.0  # value assigned to cells with no received power


def _to_numpy(x):
    """Convert a Sionna/Dr.Jit/Mitsuba/torch/TF tensor to a numpy array.

    Sionna RT returns different tensor types depending on the installed backend
    (Mitsuba ``TensorXf`` with Dr.Jit, PyTorch, or TensorFlow).  Try the common
    conversion paths in order and fall back to ``np.asarray``.
    """
    for attr in ("numpy", "cpu"):
        if hasattr(x, attr):
            try:
                y = getattr(x, attr)()
                return np.asarray(y.numpy() if hasattr(y, "numpy") else y)
            except Exception:
                pass
    try:
        import drjit as dr  # type: ignore
        return np.asarray(dr.numpy(x))
    except Exception:
        return np.asarray(x)


class EngineUnavailable(RuntimeError):
    """Raised when Sionna RT cannot be imported / initialised."""


class SionnaRTEngine(PropagationEngine):
    """Ray-tracing engine backed by Sionna RT.

    Parameters
    ----------
    max_depth:
        Maximum number of interactions (reflections/refractions) per ray.
    samples_per_tx:
        Monte-Carlo rays shot per transmitter for the radio-map solver.
    refraction, diffraction:
        Toggle the corresponding propagation mechanisms.
    flip_rows:
        Whether to vertically flip the Sionna radio map.  Sionna's radio-map
        plane (with zero orientation) already indexes row 0 = ``y_min``, which
        matches :meth:`GridSpec.cell_centers` (the canonical convention used by
        the renderer), so this defaults to ``False``.  Set it ``True`` only if a
        future Sionna version mirrors the row order and maps look flipped.
    """

    name = "sionna_rt"

    def __init__(
        self,
        max_depth: int = 5,
        samples_per_tx: int = 1_000_000,
        refraction: bool = False,
        diffraction: bool = False,
        flip_rows: bool = False,
    ) -> None:
        try:
            import sionna  # noqa: F401
            import sionna.rt as rt  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on environment
            raise EngineUnavailable(
                "Sionna RT is not installed. Install with `pip install sionna` "
                "(requires a CUDA GPU + Dr.Jit/Mitsuba for best performance)."
            ) from exc

        self._rt = rt
        self._sionna_version = getattr(sionna, "__version__", "unknown")
        self.max_depth = max_depth
        self.samples_per_tx = samples_per_tx
        self.refraction = refraction
        self.diffraction = diffraction
        self.flip_rows = flip_rows

    # ------------------------------------------------------------------ #
    @property
    def signature(self) -> str:
        payload = {
            "engine": self.name,
            "sionna": self._sionna_version,
            "max_depth": self.max_depth,
            "samples_per_tx": self.samples_per_tx,
            "refraction": self.refraction,
            "diffraction": self.diffraction,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    # ------------------------------------------------------------------ #
    def _load_scene(self, scene_cfg: SceneConfig):
        rt = self._rt
        # An explicit geometry file (Mitsuba .xml) takes priority over `name`.
        if scene_cfg.geometry_file and scene_cfg.geometry_file.endswith(".xml"):
            return rt.load_scene(scene_cfg.geometry_file)
        name = scene_cfg.name
        if name in ("empty", "", None):
            return rt.load_scene()  # empty scene (free space + ground if any)
        if isinstance(name, str) and name.endswith(".xml"):
            return rt.load_scene(name)
        # Built-in integrated scene, e.g. "simple_street_canyon", "munich".
        builtin = getattr(rt.scene, name, None)
        if builtin is None:
            raise ValueError(f"Unknown Sionna scene {name!r}")
        return rt.load_scene(builtin)

    def _planar_array(self, ant: AntennaConfig):
        rt = self._rt
        # Map our pattern names to Sionna's; parametric ones fall back to a
        # directive Sionna pattern of similar character.
        pattern = ant.pattern
        if pattern in ("patch", "sector"):
            pattern = "tr38901"
        elif pattern == "msi":
            # Register the measured pattern with Sionna (idempotent) and use it.
            from .. import antenna_msi
            pattern = antenna_msi.register_in_sionna(ant)
        return rt.PlanarArray(
            num_rows=ant.num_rows,
            num_cols=ant.num_cols,
            vertical_spacing=ant.v_spacing,
            horizontal_spacing=ant.h_spacing,
            pattern=pattern,
            polarization=ant.polarization,
        )

    def _radio_map_plane(self, grid: GridSpec):
        cx = 0.5 * (grid.x_min + grid.x_max)
        cy = 0.5 * (grid.y_min + grid.y_max)
        center = [cx, cy, grid.z]
        size = [grid.x_max - grid.x_min, grid.y_max - grid.y_min]
        return center, size

    def _rss_to_dbm_grid(self, rss_layer: np.ndarray) -> np.ndarray:
        """Convert one transmitter's linear RSS map (W) to a (ny, nx) dBm grid."""
        rss = np.asarray(rss_layer, dtype=np.float64)
        with np.errstate(divide="ignore"):
            dbm = 10.0 * np.log10(np.where(rss > 0, rss, np.nan)) + 30.0
        dbm = np.where(np.isfinite(dbm), dbm, _RSS_FLOOR_DBM)
        if self.flip_rows:
            dbm = dbm[::-1, :]
        return dbm.astype(np.float32)

    # ------------------------------------------------------------------ #
    def compute_layer(
        self, scene: SceneConfig, grid: GridSpec, tx: Transmitter
    ) -> CoverageLayer:
        return self.compute_layers(scene, grid, [tx])[tx.signature]

    def compute_layers(
        self, scene_cfg: SceneConfig, grid: GridSpec, txs: List[Transmitter]
    ) -> Dict[str, CoverageLayer]:
        rt = self._rt
        results: Dict[str, CoverageLayer] = {}

        # Group transmitters that can share one solve.
        groups: Dict[Tuple[float, str], List[Transmitter]] = defaultdict(list)
        for tx in txs:
            groups[(tx.frequency_hz, tx.antenna.signature)].append(tx)

        center, size = self._radio_map_plane(grid)

        for (freq_hz, _ant_sig), group in groups.items():
            scene = self._load_scene(scene_cfg)
            scene.frequency = float(freq_hz)
            scene.tx_array = self._planar_array(group[0].antenna)
            # Reference isotropic, vertically-polarised receiver for the map.
            scene.rx_array = rt.PlanarArray(
                num_rows=1, num_cols=1, vertical_spacing=0.5,
                horizontal_spacing=0.5, pattern="iso", polarization="V",
            )

            ordered: List[Transmitter] = []
            for tx in group:
                sio_tx = rt.Transmitter(
                    name=f"tx_{tx.signature}",
                    position=list(tx.position),
                    orientation=[math.radians(a) for a in tx.orientation],
                    power_dbm=float(tx.tx_power_dbm),
                )
                scene.add(sio_tx)
                ordered.append(tx)

            rm_solver = rt.RadioMapSolver()
            rm = rm_solver(
                scene,
                max_depth=self.max_depth,
                cell_size=[grid.cell_size, grid.cell_size],
                # center, orientation and size define the measurement plane.
                # Sionna requires all three together (or all omitted), so the
                # horizontal plane is given an explicit zero orientation.
                center=center,
                orientation=[0.0, 0.0, 0.0],
                size=size,
                samples_per_tx=int(self.samples_per_tx),
                refraction=self.refraction,
            )

            # rm.rss: per-transmitter received signal strength (linear, W),
            # shape [num_tx, num_cells_y, num_cells_x]. Convert to numpy.
            rss = _to_numpy(rm.rss)
            if rss.ndim == 2:
                rss = rss[None, ...]

            for idx, tx in enumerate(ordered):
                dbm = self._rss_to_dbm_grid(rss[idx])
                results[tx.signature] = CoverageLayer(
                    rsrp_dbm=dbm,
                    tx_signature=tx.signature,
                    tx_name=tx.name,
                    channel=tx.channel,
                    engine=self.name,
                    meta={
                        "sionna_version": self._sionna_version,
                        "frequency_hz": tx.frequency_hz,
                        "max_depth": self.max_depth,
                        "samples_per_tx": self.samples_per_tx,
                    },
                )

        return results
