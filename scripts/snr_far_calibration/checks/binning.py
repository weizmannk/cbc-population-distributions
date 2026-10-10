"""Does rho_eq depend on the choice of chirp-mass bins?

Each bin of config.CHIRP_MASS_EDGES is split at its geometric centre, and
rho_eq is measured in both halves. If the bins are narrow enough, the two
halves agree with each other and with the full bin within the errors. A
second grid, shifted by half a bin, tests the position of the edges.

    python checks/binning.py
    python checks/binning.py --runs O4a O4b
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import injections  # noqa: E402
import segments  # noqa: E402
from config import (  # noqa: E402
    CHIRP_MASS_EDGES,
    CLASSES,
    FAR_THRESHOLD,
    INJECTIONS,
    RUNS,
)
from threshold import bootstrap_error, equivalent_threshold, network_snr  # noqa: E402

MIN_FOUND = 30
CHECK_RESAMPLES = 100  # diagnostic: no need for the precision of the table


def measure(inj, state, cls, lo, hi, rng):
    index = np.flatnonzero(
        (inj.source_class == cls)
        & (inj.chirp_mass_det >= lo)
        & (inj.chirp_mass_det < hi)
    )
    found = inj.found(FAR_THRESHOLD)[index]
    if found.sum() < MIN_FOUND:
        return np.nan, np.nan
    snr = network_snr(
        {k: v[index] for k, v in inj.optimal_snr.items()},
        {k: v[index] for k, v in state.items()},
        rng,
    )
    return equivalent_threshold(snr, found), bootstrap_error(
        snr, found, rng, CHECK_RESAMPLES
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--injections", default=INJECTIONS)
    parser.add_argument("--runs", nargs="+", default=list(RUNS))
    args = parser.parse_args()

    rng = np.random.default_rng(2)
    edges = np.array(CHIRP_MASS_EDGES)
    centres = np.sqrt(edges[:-1] * edges[1:])

    print(
        f"{'run':4} {'class':5} {'Mc_det':>10} {'full bin':>13} {'lower half':>13} "
        f"{'upper half':>13} {'shifted bin':>13}"
    )
    for run in args.runs:
        inj = injections.load(args.injections, run)
        state = segments.observing(inj.gps, run)
        inside = np.any(list(state.values()), axis=0)
        inj = inj.subset(inside)
        state = {k: v[inside] for k, v in state.items()}
        for cls in CLASSES:
            for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                full = measure(inj, state, cls, lo, hi, rng)
                if np.isnan(full[0]):
                    continue
                cells = [
                    full,
                    measure(inj, state, cls, lo, centres[k], rng),
                    measure(inj, state, cls, centres[k], hi, rng),
                ]
                if k + 1 < len(centres):
                    cells.append(
                        measure(inj, state, cls, centres[k], centres[k + 1], rng)
                    )
                else:
                    cells.append((np.nan, np.nan))
                text = " ".join(
                    f"{v:6.2f} ± {e:4.2f}" if np.isfinite(v) else f"{'-':>13}"
                    for v, e in cells
                )
                print(f"{run:4} {cls:5} {lo:>5g}-{hi:<5g} {text}")


if __name__ == "__main__":
    main()
