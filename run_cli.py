#!/usr/bin/env python3
"""Scriptable example: build a scene, run, demonstrate caching, save figures.

Run::

    python run_cli.py --engine auto --out coverage.png

This is also the place to start when embedding wifisim in your own research
scripts or notebooks.
"""
from __future__ import annotations

import argparse

from wifisim import AntennaConfig, GridSpec, SceneConfig, Simulator, Transmitter
from wifisim import viz
from wifisim.engines import available_engines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default="auto", choices=["auto", "analytical", "sionna_rt"])
    p.add_argument("--metric", default="best_rsrp",
                   choices=["best_rsrp", "rss", "sinr", "best_server"])
    p.add_argument("--out", default="coverage.png")
    p.add_argument("--cache", default=".wifisim_cache")
    args = p.parse_args()

    print("available engines:", available_engines())

    sim = Simulator(
        scene=SceneConfig(name="empty", path_loss_exponent=2.8, shadowing_std_db=2.0),
        grid=GridSpec(x_min=-30, x_max=30, y_min=-30, y_max=30, z=1.5, cell_size=0.5),
        engine=args.engine,
        cache=args.cache,
    )

    # Place a few 5 GHz APs with different antennas / channels.
    sim.add_transmitter(Transmitter("AP-lobby", (-12, -10, 3), tx_power_dbm=20,
                                    channel=36, antenna=AntennaConfig("iso")))
    sim.add_transmitter(Transmitter("AP-hall", (10, 6, 3), tx_power_dbm=20,
                                    channel=44, antenna=AntennaConfig("tr38901"),
                                    orientation=(200, 0, 0)))
    sim.add_transmitter(Transmitter("AP-corner", (14, -14, 3), tx_power_dbm=23,
                                    channel=36, antenna=AntennaConfig("sector", az_hpbw_deg=80),
                                    orientation=(135, 0, 0)))

    r1 = sim.run()
    print("cold run :", r1.summary())
    r2 = sim.run()
    print("warm run :", {k: r2.summary()[k] for k in ("cache_hits", "cache_misses", "timing_s")})

    sim.move("AP-hall", 4, 4)
    r3 = sim.run()
    print("after move:", {k: r3.summary()[k] for k in ("cache_hits", "cache_misses")},
          "(only the moved AP recomputed)")

    viz.save_figure(r3, sim.transmitters, args.metric, args.out)
    print(f"saved {args.metric} map -> {args.out}")
    print("cache:", sim.cache.stats())


if __name__ == "__main__":
    main()
