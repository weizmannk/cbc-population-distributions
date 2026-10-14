"""Check the classes, the mass cut and the chirp-mass bins against the injections.

For each run and class: number of injections, range of the source-frame
component masses, range of the detector-frame chirp mass, the injections
removed by the cut at config.BH_MAX_MASS, and the injections left outside the
bins of config.CHIRP_MASS_EDGES. Found means FAR below the threshold.

    python checks/coverage.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import injections  # noqa: E402
from config import (  # noqa: E402
    BH_MAX_MASS,
    CHIRP_MASS_EDGES,
    CLASSES,
    FAR_THRESHOLD,
    INJECTIONS,
    NS_MAX_MASS,
    RUNS,
)


def span(x):
    return f"{x.min():7.2f}-{x.max():<8.2f}" if len(x) else f"{'-':>16}"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--injections", default=INJECTIONS)
    parser.add_argument("--runs", nargs="+", default=list(RUNS))
    args = parser.parse_args()

    low, high = CHIRP_MASS_EDGES[0], CHIRP_MASS_EDGES[-1]
    print(
        f"NS below {NS_MAX_MASS} Msun, components up to {BH_MAX_MASS:g} Msun (source frame); "
        f"bins {low}-{high:g} Msun (detector frame)\n"
    )
    print(
        f"{'run':4} {'class':5} {'kept':>8} {'heavier [Msun]':>16} {'lighter [Msun]':>16} "
        f"{'Mc_det [Msun]':>16} {'cut (found)':>15} {'outside (found)':>15}"
    )
    for run in args.runs:
        everything = injections.load(args.injections, run, max_mass=np.inf)
        found = everything.found(FAR_THRESHOLD)
        cut = everything.masses[:, 0] > BH_MAX_MASS
        for cls in CLASSES:
            sel = (everything.source_class == cls) & ~cut
            mc = everything.chirp_mass_det[sel]
            outside = (mc < low) | (mc >= high)
            removed = (everything.source_class == cls) & cut
            print(
                f"{run:4} {cls:5} {sel.sum():8d} "
                f"{span(everything.masses[sel, 0])} {span(everything.masses[sel, 1])} "
                f"{span(mc)} "
                f"{f'{removed.sum()} ({np.sum(found & removed)})':>15} "
                f"{f'{outside.sum()} ({np.sum(found[sel] & outside)})':>15}"
            )


if __name__ == "__main__":
    main()
