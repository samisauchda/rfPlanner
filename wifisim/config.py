"""5 GHz WiFi RF constants, channel plan, and noise-floor helpers.

Everything here is pure data / pure functions so it can be imported by any
layer (engines, simulator, web) without side effects.

References
---------
* IEEE 802.11 5 GHz (U-NII) channelisation.
* Thermal noise floor:  N0 = kT B  ->  N[dBm] = -174 + 10log10(B) + NF.
"""
from __future__ import annotations

from dataclasses import dataclass

SPEED_OF_LIGHT = 299_792_458.0  # m/s
BOLTZMANN = 1.380649e-23        # J/K
T0 = 290.0                      # K, reference temperature

#: Thermal noise power spectral density at T0 in dBm/Hz (= -173.98).
THERMAL_NOISE_PSD_DBM_HZ = 10.0 * 2.62  # placeholder, overwritten below


def _thermal_psd_dbm_hz() -> float:
    import math
    return 10.0 * math.log10(BOLTZMANN * T0 * 1e3)  # W->mW => +30 dB inside log


THERMAL_NOISE_PSD_DBM_HZ = _thermal_psd_dbm_hz()  # ~ -173.98 dBm/Hz


@dataclass(frozen=True)
class WifiChannel:
    """A single 5 GHz WiFi channel."""

    number: int
    center_mhz: float
    bandwidth_mhz: float

    @property
    def center_hz(self) -> float:
        return self.center_mhz * 1e6

    @property
    def bandwidth_hz(self) -> float:
        return self.bandwidth_mhz * 1e6


# A compact, representative subset of the 5 GHz channel plan.  Center
# frequencies follow the 802.11 formula  f = 5000 + 5*n  MHz.  We expose the
# common 20 MHz primary channels plus a few 40/80 MHz aggregates.
def _ch(n: int, bw: float = 20.0) -> WifiChannel:
    return WifiChannel(number=n, center_mhz=5000.0 + 5.0 * n, bandwidth_mhz=bw)


WIFI5_CHANNELS_20MHZ = {
    n: _ch(n, 20.0)
    for n in (36, 40, 44, 48, 52, 56, 60, 64,
              100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144,
              149, 153, 157, 161, 165)
}

#: Default channel used when a transmitter does not specify one.
DEFAULT_CHANNEL = 36
DEFAULT_BANDWIDTH_MHZ = 20.0
DEFAULT_FREQUENCY_HZ = WIFI5_CHANNELS_20MHZ[DEFAULT_CHANNEL].center_hz  # 5.18 GHz

#: Typical receiver noise figure for consumer 5 GHz WiFi front-ends.
DEFAULT_NOISE_FIGURE_DB = 7.0


def channel_frequency_hz(channel: int) -> float:
    """Return the centre frequency in Hz for a 5 GHz WiFi channel number."""
    return (5000.0 + 5.0 * channel) * 1e6


def noise_floor_dbm(bandwidth_hz: float, noise_figure_db: float = DEFAULT_NOISE_FIGURE_DB) -> float:
    """Receiver thermal noise floor in dBm for a given bandwidth.

    N[dBm] = -174 + 10*log10(B[Hz]) + NF[dB]
    """
    import math
    return THERMAL_NOISE_PSD_DBM_HZ + 10.0 * math.log10(bandwidth_hz) + noise_figure_db


def wavelength_m(frequency_hz: float) -> float:
    return SPEED_OF_LIGHT / frequency_hz
