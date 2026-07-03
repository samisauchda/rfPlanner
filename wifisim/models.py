"""Immutable, hashable description of a simulation.

The objects in this module are deliberately split into two kinds:

* **Specification objects** (``AntennaConfig``, ``Transmitter``, ``GridSpec``,
  ``SceneConfig``) are frozen dataclasses.  They describe *what* to simulate and
  expose a stable :pyattr:`signature` (a short hash) derived only from the
  fields that affect the physics.  These signatures are what the cache keys on,
  so moving one transmitter invalidates only that transmitter's cached layer.

* **Result objects** (``CoverageLayer``, ``SimulationResult``) carry numpy
  arrays plus the metadata needed to interpret and re-locate them on the grid.

Coordinate / unit conventions
------------------------------
* Positions are metres in a right-handed frame ``(x, y, z)`` with ``z`` up.
* Antenna orientation is ``(yaw, pitch, roll)`` in **degrees**.  ``yaw`` rotates
  the boresight in the horizontal plane (0 deg = +x axis, 90 deg = +y axis).
* Transmit power is total radiated power in **dBm** (antenna gain is applied on
  top, per direction, by the engine).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict, replace
from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import config as cfg


def _hash_obj(obj: Any) -> str:
    """Stable short hash of a JSON-serialisable object (12 hex chars)."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Antenna
# --------------------------------------------------------------------------- #
#: Antenna patterns understood by the framework.  The names that overlap with
#: Sionna RT ("iso", "dipole", "tr38901") are forwarded verbatim to Sionna's
#: ``PlanarArray``; the analytical engine implements all of them parametrically.
ANTENNA_PATTERNS = ("iso", "dipole", "tr38901", "patch", "sector", "msi")


@dataclass(frozen=True)
class AntennaConfig:
    """Antenna / array description for one transmitter.

    Parameters
    ----------
    pattern:
        One of :data:`ANTENNA_PATTERNS`.
    num_rows, num_cols:
        Planar-array geometry.  ``1x1`` is a single element.
    h_spacing, v_spacing:
        Element spacing in wavelengths.
    polarization:
        ``"V"``, ``"H"``, or ``"cross"`` (forwarded to Sionna).
    boresight_gain_dbi:
        Peak gain for the parametric ``patch`` / ``sector`` patterns.  Ignored
        for physically-defined patterns ("iso", "dipole", "tr38901").
    az_hpbw_deg, el_hpbw_deg:
        Half-power beamwidths (azimuth / elevation) for ``sector`` / ``patch``.
    front_to_back_db:
        Front-to-back ratio for ``sector`` / ``patch``.
    msi_file, msi_sha:
        For ``pattern="msi"``: path to a ``.msi`` principal-plane pattern file
        and a hash of its contents.  ``msi_sha`` is what the cache keys on, so a
        changed file invalidates only the transmitters that use it.  Build these
        with :func:`wifisim.antenna_msi.make_msi_antenna` rather than by hand.
    msi_azimuth_ccw, msi_vertical_down_positive:
        MSI angle conventions (see :mod:`wifisim.antenna_msi`).
    """

    pattern: str = "iso"
    num_rows: int = 1
    num_cols: int = 1
    h_spacing: float = 0.5
    v_spacing: float = 0.5
    polarization: str = "V"
    boresight_gain_dbi: float = 8.0
    az_hpbw_deg: float = 65.0
    el_hpbw_deg: float = 65.0
    front_to_back_db: float = 25.0
    msi_file: str = ""
    msi_sha: str = ""
    msi_azimuth_ccw: bool = True
    msi_vertical_down_positive: bool = True

    def __post_init__(self) -> None:
        if self.pattern not in ANTENNA_PATTERNS:
            raise ValueError(
                f"Unknown antenna pattern {self.pattern!r}; "
                f"expected one of {ANTENNA_PATTERNS}"
            )
        if self.pattern == "msi" and not self.msi_file:
            raise ValueError("pattern='msi' requires msi_file (use "
                             "wifisim.antenna_msi.make_msi_antenna)")

    @property
    def signature(self) -> str:
        return _hash_obj(asdict(self))


# --------------------------------------------------------------------------- #
# Transmitter
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Transmitter:
    """A WiFi access point / transmitter.

    The ``signature`` deliberately excludes ``name`` and ``color`` (presentation
    only) so that renaming or recolouring a transmitter does not invalidate its
    cached coverage layer.
    """

    name: str
    position: Tuple[float, float, float] = (0.0, 0.0, 3.0)
    orientation: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # yaw, pitch, roll (deg)
    tx_power_dbm: float = 20.0
    channel: int = cfg.DEFAULT_CHANNEL
    bandwidth_mhz: float = cfg.DEFAULT_BANDWIDTH_MHZ
    antenna: AntennaConfig = field(default_factory=AntennaConfig)
    enabled: bool = True
    color: str = "#39d0d8"

    # ---- derived RF quantities -------------------------------------------- #
    @property
    def frequency_hz(self) -> float:
        return cfg.channel_frequency_hz(self.channel)

    @property
    def bandwidth_hz(self) -> float:
        return self.bandwidth_mhz * 1e6

    # ---- hashing ---------------------------------------------------------- #
    @property
    def physics_dict(self) -> Dict[str, Any]:
        """Only the fields that change the computed coverage layer."""
        return {
            "position": list(self.position),
            "orientation": list(self.orientation),
            "tx_power_dbm": self.tx_power_dbm,
            "channel": self.channel,
            "bandwidth_mhz": self.bandwidth_mhz,
            "antenna": asdict(self.antenna),
        }

    @property
    def signature(self) -> str:
        return _hash_obj(self.physics_dict)

    # ---- convenience editing ---------------------------------------------- #
    def moved_to(self, x: float, y: float, z: Optional[float] = None) -> "Transmitter":
        z = self.position[2] if z is None else z
        return replace(self, position=(float(x), float(y), float(z)))

    def with_(self, **changes: Any) -> "Transmitter":
        return replace(self, **changes)

    # ---- (de)serialisation ------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["position"] = list(self.position)
        d["orientation"] = list(self.orientation)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Transmitter":
        d = dict(d)
        ant = d.pop("antenna", None)
        antenna = AntennaConfig(**ant) if ant else AntennaConfig()
        d["position"] = tuple(d.get("position", (0.0, 0.0, 3.0)))
        d["orientation"] = tuple(d.get("orientation", (0.0, 0.0, 0.0)))
        return cls(antenna=antenna, **d)


# --------------------------------------------------------------------------- #
# Measurement grid
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GridSpec:
    """Horizontal measurement plane on which coverage is sampled.

    The plane spans ``[x_min, x_max] x [y_min, y_max]`` at height ``z`` and is
    discretised into square cells of side ``cell_size`` (metres).
    """

    x_min: float = -25.0
    x_max: float = 25.0
    y_min: float = -25.0
    y_max: float = 25.0
    z: float = 1.5
    cell_size: float = 0.5

    @property
    def nx(self) -> int:
        return max(1, int(round((self.x_max - self.x_min) / self.cell_size)))

    @property
    def ny(self) -> int:
        return max(1, int(round((self.y_max - self.y_min) / self.cell_size)))

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """matplotlib-style extent (left, right, bottom, top)."""
        return (self.x_min, self.x_max, self.y_min, self.y_max)

    def cell_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(X, Y)`` meshgrids of cell-centre coordinates, shape (ny, nx)."""
        xs = self.x_min + (np.arange(self.nx) + 0.5) * self.cell_size
        ys = self.y_min + (np.arange(self.ny) + 0.5) * self.cell_size
        return np.meshgrid(xs, ys)

    @property
    def signature(self) -> str:
        return _hash_obj(asdict(self))


# --------------------------------------------------------------------------- #
# Scene
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SceneConfig:
    """Propagation environment.

    ``name`` selects a built-in Sionna scene (e.g. ``"simple_street_canyon"``)
    or, if it ends in ``.xml``, a Mitsuba scene file path.
    """

    name: str = "empty"
    noise_figure_db: float = cfg.DEFAULT_NOISE_FIGURE_DB
    geometry_file: str = ""              # Mitsuba .xml / mesh used for ray tracing
    geometry_sha: str = ""               # content hash (cache invalidation)
    mesh_file: str = ""                  # prediction-mesh file (measurement surface)
    mesh_sha: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SceneConfig":
        """Construct from a dict, ignoring unrecognised keys (e.g. from old project.json)."""
        valid = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in valid})

    @property
    def signature(self) -> str:
        return _hash_obj(asdict(self))


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass
class CoverageLayer:
    """Per-transmitter received-power map on a :class:`GridSpec`.

    Attributes
    ----------
    rsrp_dbm:
        ``(ny, nx)`` array of received power in dBm that a reference isotropic
        receiver would observe from this transmitter alone (signal only, no
        interference, no noise).
    tx_signature, tx_name:
        Identify the transmitter that produced the layer.
    channel:
        WiFi channel of the transmitter (used for co-channel interference).
    engine, meta:
        Provenance for reproducibility.
    """

    rsrp_dbm: np.ndarray
    tx_signature: str
    tx_name: str
    channel: int
    engine: str = "unknown"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_npz_dict(self) -> Dict[str, Any]:
        return {
            "rsrp_dbm": self.rsrp_dbm.astype(np.float32),
            "_meta": json.dumps(
                {
                    "tx_signature": self.tx_signature,
                    "tx_name": self.tx_name,
                    "channel": self.channel,
                    "engine": self.engine,
                    "meta": self.meta,
                }
            ),
        }

    @classmethod
    def from_npz(cls, npz) -> "CoverageLayer":
        meta = json.loads(str(npz["_meta"]))
        return cls(
            rsrp_dbm=npz["rsrp_dbm"],
            tx_signature=meta["tx_signature"],
            tx_name=meta["tx_name"],
            channel=meta["channel"],
            engine=meta.get("engine", "unknown"),
            meta=meta.get("meta", {}),
        )


@dataclass
class SimulationResult:
    """Aggregated result for a whole scene.

    Arrays are all ``(ny, nx)`` aligned to ``grid``.
    """

    grid: GridSpec
    rss_dbm: np.ndarray                 # aggregate received signal strength
    best_rsrp_dbm: np.ndarray           # strongest single-TX power
    best_server: np.ndarray             # index of strongest TX (-1 = none)
    sinr_db: np.ndarray                 # SINR of the best server
    tx_names: list
    cache_hits: int = 0
    cache_misses: int = 0
    engine: str = "unknown"
    timing_s: float = 0.0

    def summary(self) -> Dict[str, Any]:
        finite = np.isfinite(self.best_rsrp_dbm)
        cov = lambda thr: float(np.mean(self.best_rsrp_dbm[finite] >= thr) * 100.0) if finite.any() else 0.0
        return {
            "engine": self.engine,
            "grid_shape": list(self.grid.shape),
            "n_transmitters": len(self.tx_names),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "timing_s": round(self.timing_s, 4),
            "coverage_pct_-67dBm": round(cov(-67.0), 1),   # good WiFi
            "coverage_pct_-80dBm": round(cov(-80.0), 1),   # usable WiFi
            "median_sinr_db": round(float(np.nanmedian(self.sinr_db)), 2) if np.isfinite(self.sinr_db).any() else None,
        }
