#!/usr/bin/env python
"""Detection rates per class from the detection-weights.dat files of a run.

Reads the per-event weights written by observing-scenarios-simulations'
`make detection-weights`, sums them per source class, and turns the counts into
predicted detection numbers with their 90% credible interval, exactly as
scripts/notebooks/detection_rate.ipynb does.

    python detection_rate_from_weights.py <runs dir> --runs O4a O4b
    python detection_rate_from_weights.py <runs dir> --runs O4b --no-redshift
"""

import argparse
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import Table
from poisson_rate_utils import (
    format_with_errorbars,
    poisson_lognormal_rate_quantiles,
)
from scipy import stats

#: Total CBC merger rate density [Gpc^-3 yr^-1], median and 90% bounds, from
#: data/derived/rate_summary.csv (GWTC-5.0 FullPop / PixelPop).
FIDUCIAL_RATE = {
    "fullpop": (100.07580795644503, 49.88807658685758, 202.77919891201657),
    "pixelpop": (68.27664566040039, 39.2487714767456, 138.97688369750972),
}

#: CBC injection rate density [Mpc^-3 yr^-1] of each simulated run, read from
#: the bayestar-inject comment in its events.sqlite.
SIM_RATES = {
    "fullpop": {
        "O4a": 4.241071050090774e-06,
        "O4b": 4.241071050090774e-06,
        "IR1HL": 4.331602735899129e-06,
        "IR1HLV": 4.331602735899129e-06,
        "O5a": 3.187082037941972e-06,
        "O5b": 2.67230947825585e-06,
        "O5c": 2.5207447621766638e-06,
    },
    "pixelpop": {
        "O4a": 3.1181168097568635e-06,
        "O4b": 3.1181168097568635e-06,
        "IR1HL": 3.1858542569816434e-06,
        "IR1HLV": 3.1858542569816434e-06,
        "O5a": 2.424990666626414e-06,
        "O5b": 2.092936520960754e-06,
        "O5c": 1.999871588465548e-06,
    },
}

#: Observing time [yr], the calendar length of each segment.
DURATIONS = {
    "O4a": (1389456018 - 1368975618) / (86400 * 365.25),
    "O4b": (1422118818 - 1396796418) / (86400 * 365.25),
    "IR1HL": 0.5,
    "IR1HLV": 0.5,
}

CLASSES = ("BNS", "NSBH", "BBH")


def mass_fractions(path, ns_max_mass):
    """BNS/NSBH/BBH fractions of a population sample, as the notebook computes."""
    t = Table.read(path)
    m1, m2 = np.asarray(t["mass1"]), np.asarray(t["mass2"])
    bns = np.sum((m1 < ns_max_mass) & (m2 < ns_max_mass))
    nsbh = np.sum((m1 >= ns_max_mass) & (m2 < ns_max_mass))
    return {
        "BNS": bns / len(t),
        "NSBH": nsbh / len(t),
        "BBH": 1 - (bns + nsbh) / len(t),
    }


def counts(path, redshift_weight):
    """Detections above rho_eq per class, and whether the weight was applied.

    Only the fullpop files carry redshift_weight, so a model written without it
    is counted unweighted rather than refused.
    """
    t = Table.read(path, format="ascii.fast_tab")
    keep = np.asarray(t["passes_rho_eq"], dtype=bool)
    weighted = redshift_weight and "redshift_weight" in t.colnames
    w = np.asarray(t["redshift_weight"]) if weighted else np.ones(len(t))
    cls = np.asarray(t["source_class"])
    return {c: float(np.sum(w[keep & (cls == c)])) for c in CLASSES}, weighted


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs", type=Path, help="directory holding <run>/<model>/")
    parser.add_argument("--runs", dest="names", nargs="+", default=["O4a", "O4b"])
    parser.add_argument("--models", nargs="+", default=["fullpop", "pixelpop"])
    parser.add_argument("--samples", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--ns-max-mass",
        type=float,
        default=2.5,
        help="NS/BH boundary for the mass fractions [default: %(default)s]",
    )
    parser.add_argument(
        "--no-redshift",
        action="store_true",
        help="ignore redshift_weight, count events instead of weighting them",
    )
    args = parser.parse_args()

    # The width of a 90% normal interval, to turn the rate bounds into sigma.
    width = np.diff(stats.norm.interval(0.9))[0]

    print(
        f"{'run':7} {'model':9} {'criterion':12} "
        + " ".join(f"{c:>18}" for c in CLASSES)
    )
    for model in args.models:
        sample = args.samples / (
            "fullpop_grid.h5" if model == "fullpop" else "pixelpop.h5"
        )
        fraction = mass_fractions(sample, args.ns_max_mass)
        mid, lo, hi = FIDUCIAL_RATE[model]
        for name in args.names:
            path = args.runs / name / model / "detection-weights.dat"
            if not path.exists():
                print(f"{name:7} {model:9} {'absent':12} {path}")
                continue
            detections, weighted = counts(path, not args.no_redshift)
            sim = (SIM_RATES[model][name] / (u.Mpc**3 * u.yr)).to(u.Gpc**-3 * u.yr**-1)
            cells = []
            for c in CLASSES:
                rate = sim.value * fraction[c]
                mu = (
                    np.log(mid * fraction[c])
                    + np.log(DURATIONS[name])
                    + np.log(detections[c] / rate)
                )
                sigma = (np.log(hi * fraction[c]) - np.log(lo * fraction[c])) / width
                q = poisson_lognormal_rate_quantiles(
                    np.array([0.05, 0.5, 0.95]), mu, sigma
                )
                m, a, b = format_with_errorbars(
                    int(np.round(q[1])), int(np.floor(q[0])), int(np.ceil(q[2]))
                )
                cells.append(f"{m}(-{a}+{b})".rjust(18))
            label = "rho_eq" + (", +z" if weighted else "")
            print(f"{name:7} {model:9} {label:12} " + " ".join(cells))


if __name__ == "__main__":
    main()
