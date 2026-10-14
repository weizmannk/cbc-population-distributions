"""Why are injections found at a simulated SNR the searches should not reach?

The found fraction of a bin does not fall to zero at low SNR, which no physical
selection explains. For every run and class, this lists the found injections
whose simulated network SNR, median over the noise draws, stays below
--snr-max: how many they are, which searches found them, and two bookkeeping
causes. "no LIGO" is the share whose observing detectors, per the GWOSC
segments, exclude H1 and L1; the release gives no Virgo SNR at all for O3 and
O4a, so their simulated network SNR is near zero while the searches found them
in LIGO data. "loud off" is the share whose largest optimal SNR belongs to a
detector the segments call off.

    python checks/low_snr_found.py
    python checks/low_snr_found.py --runs O3a O3b --snr-max 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import injections  # noqa: E402
import segments  # noqa: E402
from config import CLASSES, FAR_THRESHOLD, INJECTIONS, RUNS  # noqa: E402
from threshold import network_snr  # noqa: E402

CBC_SEARCHES = ("GstLAL", "MBTA", "PyCBC")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--injections", default=INJECTIONS)
    parser.add_argument("--runs", nargs="+", default=list(RUNS))
    parser.add_argument("--far", type=float, default=FAR_THRESHOLD)
    parser.add_argument("--snr-max", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    print(
        f"found injections with median simulated network SNR < {args.snr_max:g}\n"
        f"{'run':4} {'class':5} {'low':>7} {'of found':>9} {'share':>7} "
        f"{'cWB only':>9} {'1 CBC':>7} {'no LIGO':>8} {'loud off':>9} "
        f"{'median SNR':>11}"
    )
    for run in args.runs:
        inj = injections.load(args.injections, run)
        state = segments.observing(inj.gps, run)
        inside = np.any(list(state.values()), axis=0)
        inj = inj.subset(inside)
        state = {ifo: on[inside] for ifo, on in state.items()}

        snr = np.median(network_snr(inj.optimal_snr, state, rng), axis=1)
        found = inj.found(args.far)
        cwb = inj.found(args.far, "cWB")
        cbc = np.count_nonzero(
            [inj.found(args.far, name) for name in CBC_SEARCHES], axis=0
        )
        optimal = np.column_stack([inj.optimal_snr[ifo] for ifo in state])
        observing = np.column_stack([state[ifo] for ifo in state])
        no_ligo = ~(state["H1"] | state["L1"])
        # The detector with the largest optimal SNR, whether or not it observed.
        loudest_off = ~observing[np.arange(len(inj)), np.argmax(optimal, axis=1)]

        for cls in CLASSES:
            of_class = found & (inj.source_class == cls)
            low = of_class & (snr < args.snr_max)
            if not low.any():
                print(f"{run:4} {cls:5} {0:>7} {int(of_class.sum()):>9}")
                continue
            print(
                f"{run:4} {cls:5} {int(low.sum()):>7} {int(of_class.sum()):>9} "
                f"{low.sum() / of_class.sum():7.2%} "
                f"{np.mean(cwb[low] & (cbc[low] == 0)):9.2%} "
                f"{np.mean(cbc[low] == 1):7.2%} "
                f"{np.mean(no_ligo[low]):8.2%} "
                f"{np.mean(loudest_off[low]):9.2%} "
                f"{np.median(snr[low]):11.2f}"
            )

        for cls in CLASSES:
            low = found & (inj.source_class == cls) & (snr < args.snr_max)
            if not low.any():
                continue
            print(f"\n{run} {cls}, {int(low.sum())} injections:")
            print(
                f"  {'optimal SNR':>28} {'observing':>12} "
                f"{'net SNR':>8} {'searches':>26}"
            )
            order = np.argsort(snr[low])
            for k in np.flatnonzero(low)[order][:10]:
                rho = "  ".join(
                    f"{ifo} {inj.optimal_snr[ifo][k]:6.2f}" for ifo in state
                )
                on = "".join(ifo for ifo in state if state[ifo][k]) or "none"
                names = [
                    name
                    for name in (*CBC_SEARCHES, "cWB")
                    if inj.found(args.far, name)[k]
                ]
                print(f"  {rho:>28} {on:>12} {snr[k]:8.2f} {','.join(names):>26}")
            if low.sum() > 10:
                print(f"  ({int(low.sum()) - 10} more)")


if __name__ == "__main__":
    main()
