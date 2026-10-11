"""Closure tests: a planted threshold, then a soft selection.

First, injections are found when one independent noise realization of their
network SNR exceeds a planted threshold. rho_eq, measured from other
realizations as in measure.py, should return that threshold. The minimum SNR of
the found injections, a common shortcut, is printed for comparison.

Second, the selection is not a step: the found probability p(rho) of one bin is
read from histograms.csv and applied to a population with dN/drho proportional
to rho^-4. The step cut at the rho_eq of that bin is then compared with the
number the soft selection actually finds, which is what the simulations miss
when they apply a single threshold.

    python checks/closure.py
    python checks/closure.py --run O4a --source-class BNS --chirp-mass-min 1.12
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CURRENT_THRESHOLD, OUT_DIR, read_csv  # noqa: E402
from threshold import equivalent_threshold, network_snr  # noqa: E402

N = 100_000
PLANTED = (8.0, 10.0, 12.0)
POPULATION_MIN = 4.0  # smallest SNR of the drawn population
POPULATION_SLOPE = 4.0  # dN/drho proportional to rho^-POPULATION_SLOPE


def found_probability(run, source_class, chirp_mass_min):
    """p(rho) of one bin from histograms.csv: SNR bin centres and found fraction."""
    rows = [
        r
        for r in read_csv(OUT_DIR / "histograms.csv")
        if r["run"] == run
        and r["source_class"] == source_class
        and np.isclose(float(r["chirp_mass_min"]), chirp_mass_min)
    ]
    if not rows:
        raise ValueError(
            f"no histogram for {run} {source_class} {chirp_mass_min:g} "
            f"in {OUT_DIR / 'histograms.csv'}"
        )
    centre, probability = [], []
    for r in rows:
        found, missed = float(r["found"]), float(r["missed"])
        if found + missed > 0:
            centre.append(np.sqrt(float(r["snr_min"]) * float(r["snr_max"])))
            probability.append(found / (found + missed))
    return np.array(centre), np.array(probability)


def soft_selection(args, rng):
    """Step cut at rho_eq against the soft selection p(rho) of the same bin."""
    centre, probability = found_probability(
        args.run, args.source_class, args.chirp_mass_min
    )
    row = next(
        r
        for r in read_csv(OUT_DIR / "rho_eq.csv")
        if r["run"] == args.run
        and r["source_class"] == args.source_class
        and np.isclose(float(r["chirp_mass_min"]), args.chirp_mass_min)
    )
    rho_eq = float(row["rho_eq"])

    rho = POPULATION_MIN * rng.uniform(size=args.n) ** (-1 / (POPULATION_SLOPE - 1))
    p = np.interp(rho, centre, probability)
    soft = int(np.sum(rng.uniform(size=args.n) < p))
    step = int(np.sum(rho >= rho_eq))
    print(
        f"\n{args.run} {args.source_class} {args.chirp_mass_min:g}, "
        f"rho_eq = {rho_eq:.3f}, population dN/drho ~ rho^-{POPULATION_SLOPE:g} "
        f"above {POPULATION_MIN:g}"
    )
    # The simulations keep only events above a network SNR of 8, so the same
    # comparison is also reported on that subset.
    above = rho >= CURRENT_THRESHOLD
    soft_above = int(np.sum((rng.uniform(size=args.n) < p) & above))
    step_above = int(np.sum(above & (rho >= rho_eq)))
    print(
        f"{'population':>22} {'soft selection':>16} {'step at rho_eq':>16} {'ratio':>8}"
    )
    print(f"{'all':>22} {soft:16d} {step:16d} {soft / step:8.3f}")
    print(
        f"{f'SNR >= {CURRENT_THRESHOLD:g}':>22} {soft_above:16d} {step_above:16d} "
        f"{soft_above / step_above:8.3f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", default="O4b")
    parser.add_argument("--source-class", default="BBH")
    parser.add_argument("--chirp-mass-min", type=float, default=20.0)
    parser.add_argument("--n", type=int, default=1_000_000)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    # Optimal network SNR uniform in volume (p ~ rho^-4 above 3), split over two detectors.
    total = 3.0 * rng.uniform(size=N) ** (-1 / 3)
    share = rng.uniform(0.2, 0.8, N)
    optimal = {"H1": total * np.sqrt(share), "L1": total * np.sqrt(1 - share)}
    state = {ifo: np.ones(N, bool) for ifo in optimal}

    print(f"{'planted':>8} {'rho_eq':>8} {'min SNR of found':>17}")
    for planted in PLANTED:
        found = network_snr(optimal, state, rng, draws=1)[:, 0] >= planted
        snr = network_snr(optimal, state, rng)
        smallest = snr[found].min()
        print(
            f"{planted:8.1f} {equivalent_threshold(snr, found):8.2f} {smallest:17.2f}"
        )

    soft_selection(args, rng)


if __name__ == "__main__":
    main()
