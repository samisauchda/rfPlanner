"""wifisim - A modular link/system-level WiFi (5 GHz) simulation framework.

The package is organised around three replaceable layers:

* ``wifisim.models``     - immutable, hashable description of *what* to simulate
                           (scene, transmitters, antennas, measurement grid).
* ``wifisim.engines``    - pluggable propagation back-ends that turn a single
                           transmitter into a per-transmitter coverage layer.
                           ``SionnaRTEngine`` uses NVIDIA Sionna RT ray tracing;
                           ``AnalyticalEngine`` is a dependency-light fallback.
* ``wifisim.simulator``  - orchestration, per-transmitter caching, and
                           aggregation of layers into RSS / best-server / SINR.

See ``README.md`` for the full design rationale.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .models import (
    AntennaConfig,
    GridSpec,
    SceneConfig,
    Transmitter,
    CoverageLayer,
    SimulationResult,
)
from .simulator import Simulator
from .cache import LayerCache
from . import antenna_msi
from .antenna_msi import make_msi_antenna
from . import geometry

__all__ = [
    "__version__",
    "AntennaConfig",
    "GridSpec",
    "SceneConfig",
    "Transmitter",
    "CoverageLayer",
    "SimulationResult",
    "Simulator",
    "LayerCache",
    "antenna_msi",
    "make_msi_antenna",
    "geometry",
]
