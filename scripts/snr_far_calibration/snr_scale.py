"""Factors that convert the optimal SNRs of the release to those of the simulations.

The release computes the optimal SNR of each injection with the PSD of its
month: the 10% quantile, in each frequency bin, of the PSDs measured during that
month (Essick et al. 2025, Sec. V.A), hence more sensitive than the typical
data. The simulations compute it with the noise curves given to them, and with
other templates and another low-frequency cutoff (config.py). For one detector
and one source, the two SNRs differ by the ratio of the noise-weighted norms of
the signal,

    factor = sigma(simulation template, noise curve)
             / sigma(release template, release PSD of the month),

which depends on the masses through the frequency content of the signal. This
script evaluates it on a grid of detector-frame total mass and mass ratio, for
each detector and each month of the release PSDs, without spin, face on.
measure.py --snr-scale multiplies each optimal SNR of the release by the factor
of its detector, month and masses before measuring rho_eq.

The noise curves are ASD files as for scripts/pack-psds.py of the
observing-scenarios simulations, and are resampled the same way (1 Hz grid from
10 Hz, log-log interpolation) so that the SNRs are those of bayestar. Which
curves to give is a choice about what the thresholds mean, see README.md:
those of the simulations being compared with a run, or the curves that
represent the sensitivity of the run whose injections are measured.

    python snr_scale.py --runs O4a O4b -o outputs/snr_scale.csv \\
        --H1 ../../data/asd/o4b_h1_ref.txt \\
        --L1 ../../data/asd/o4b_l1_ref.txt \\
        --V1 ../../data/asd/o4b_v1_ref.txt

The o2-uberbank templates above 4 Msun need SEOBNRv4ROM_v3.0.hdf5 in
LAL_DATA_PATH.
"""

import argparse
import csv
import gzip
import logging
import shlex
import sys
from pathlib import Path

import numpy as np
from config import (
    F_HIGH,
    RELEASE_F_LOW,
    RELEASE_PSDS,
    RELEASE_WAVEFORM,
    RUN_DETECTORS,
    SIMULATION_F_LOW,
    SIMULATION_WAVEFORM,
    provenance,
    read_csv,
)
from sensitivity import O4_MONTHS

# Detector-frame total mass [Msun] from the lightest BNS of the release
# (1 + 1 Msun) to 100 + 100 Msun at z ~ 4, and mass ratio down to the
# lightest NSBH of the release (1 + 60 Msun). The o2-uberbank templates switch
# from TaylorF2 to SEOBNRv4_ROM at 4 Msun of total mass, where the SNR jumps by
# about 4%; a grid point on each side keeps the jump sharp.
TOTAL_MASS = np.union1d(np.geomspace(1.9, 1000.0, 49), [3.999, 4.001])
MASS_RATIO = np.array([0.015, 0.03, 0.06, 0.12, 0.25, 0.5, 1.0])
MIN_COMPONENT_MASS = 0.9

# Frequency grid of the PSDs written by scripts/pack-psds.py.
PACKED_GRID = np.arange(10.0, 4096.0, 1.0)


def read_release_psd(path):
    """(frequency, PSD) of a release file, comma- or space-separated, gzipped."""
    with gzip.open(path, "rt") as f:
        rows = [line.replace(",", " ").split() for line in f if line[0].isdigit()]
    frequency, psd = np.array(rows, float).T
    keep = frequency > 0
    return frequency[keep], psd[keep]


def read_asd(path, column=None):
    """(frequency, PSD) of an ASD file, resampled as scripts/pack-psds.py does."""
    if column is None:
        frequency, asd = np.loadtxt(path, unpack=True)
    else:
        data = np.genfromtxt(path, names=True)
        frequency, asd = data[data.dtype.names[0]], data[column]
    keep = frequency > 0
    psd = np.exp(
        np.interp(
            np.log(PACKED_GRID),
            np.log(frequency[keep]),
            np.log(np.square(asd[keep])),
        )
    )
    return PACKED_GRID, psd


def release_psd_files(directory, run):
    """{(ifo, month): path} of the release PSDs of one run; month is 'O3' for O3."""
    if run.startswith("O3"):
        months, pattern = ["O3"], "psd-O3-{d}.*gz"
    else:
        months, pattern = O4_MONTHS[run], "psd-O4-{m}_v1-{d}.*gz"
    files = {}
    for ifo in RUN_DETECTORS[run]:
        for month in months:
            found = sorted(Path(directory).glob(pattern.format(m=month, d=ifo[0])))
            if found:
                files[ifo, month] = found[0]
    return files


def template(waveform, total_mass, mass_ratio, f_low):
    """(frequency, |h(f)|^2, df) of a non-spinning face-on template, as in bayestar."""
    from ligo.skymap.bayestar.filter import abscissa, sngl_inspiral_psd

    mass1 = total_mass / (1 + mass_ratio)
    h = sngl_inspiral_psd(waveform, mass1, mass_ratio * mass1, f_min=f_low)
    frequency, power = abscissa(h), h.data.data
    keep = (frequency >= f_low) & (frequency < F_HIGH) & (power > 0)
    return frequency[keep], power[keep], h.deltaF


def sigma(signal, psd):
    """Noise-weighted norm sqrt(4 sum |h|^2 / S df), PSD interpolated in log-log."""
    frequency, power, df = signal
    noise = np.exp(np.interp(np.log(frequency), np.log(psd[0]), np.log(psd[1])))
    return np.sqrt(4 * df * np.sum(power / noise))


def nearest(month, available):
    """The month of ``available`` closest to ``month`` (both 'YYYY_MM')."""

    def index(m):
        year, number = map(int, m.split("_"))
        return 12 * year + number

    return min(available, key=lambda m: abs(index(m) - index(month)))


def norms(waveform, f_low, psds):
    """{key: sigma on the TOTAL_MASS x MASS_RATIO grid} for each PSD of ``psds``.

    One template per grid point, evaluated against every PSD. A cell whose
    lighter component would be below MIN_COMPONENT_MASS takes the value of the
    smallest valid mass ratio at the same total mass.
    """
    grids = {key: np.full((len(TOTAL_MASS), len(MASS_RATIO)), np.nan) for key in psds}
    for i, total_mass in enumerate(TOTAL_MASS):
        for j, mass_ratio in enumerate(MASS_RATIO):
            if mass_ratio * total_mass / (1 + mass_ratio) < MIN_COMPONENT_MASS:
                continue
            signal = template(waveform, total_mass, mass_ratio, f_low)
            for key, psd in psds.items():
                grids[key][i, j] = sigma(signal, psd)
    for grid in grids.values():
        for row in grid:
            row[: np.argmax(np.isfinite(row))] = row[np.isfinite(row)][0]
    return grids


def interpolator(grid, total_mass=TOTAL_MASS, mass_ratio=MASS_RATIO):
    """f(total_mass, mass_ratio), linear in log total mass and log mass ratio.

    Outside the grid, f keeps the values of its edges.
    """
    from scipy.interpolate import RegularGridInterpolator

    axes = np.log(total_mass), np.log(mass_ratio)
    function = RegularGridInterpolator(axes, grid)

    def f(m, q):
        m = np.clip(np.log(m), axes[0][0], axes[0][-1])
        q = np.clip(np.log(q), axes[1][0], axes[1][-1])
        return function(np.column_stack([m, q]))

    return f


def read(path):
    """{(run, ifo, month): factor(total_mass, mass_ratio)} from a table written here."""
    rows = read_csv(path)
    total_mass = np.unique([float(r["total_mass"]) for r in rows])
    mass_ratio = np.unique([float(r["mass_ratio"]) for r in rows])
    grids = {}
    for r in rows:
        grid = grids.setdefault(
            (r["run"], r["ifo"], r["month"]),
            np.full((len(total_mass), len(mass_ratio)), np.nan),
        )
        i = np.searchsorted(total_mass, float(r["total_mass"]))
        j = np.searchsorted(mass_ratio, float(r["mass_ratio"]))
        grid[i, j] = float(r["factor"])
    return {
        key: interpolator(grid, total_mass, mass_ratio) for key, grid in grids.items()
    }


def header(path, key):
    """Value of the comment line ``# key: value`` of a table written here."""
    with open(path) as f:
        for line in f:
            if line.startswith(f"# {key}: "):
                return line[len(f"# {key}: ") :].strip()
    raise ValueError(f"{path} has no '{key}' line")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", nargs="+", required=True, choices=list(RUN_DETECTORS))
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--release-psds", type=Path, default=RELEASE_PSDS)
    for ifo in ("H1", "L1", "V1"):
        parser.add_argument(f"--{ifo}", metavar="ASD.txt", help=f"noise curve of {ifo}")
        parser.add_argument(
            f"--{ifo}-column", metavar="COLUMN", help=f"column of the {ifo} file"
        )
    args = parser.parse_args()
    logging.getLogger("BAYESTAR").setLevel(logging.ERROR)

    targets = {
        ifo: read_asd(getattr(args, ifo), getattr(args, f"{ifo}_column"))
        for ifo in ("H1", "L1", "V1")
        if getattr(args, ifo)
    }
    release = {}
    for run in args.runs:
        missing = set(RUN_DETECTORS[run]) - set(targets)
        if missing:
            parser.error(f"{run} needs a noise curve for {', '.join(sorted(missing))}")
        files = release_psd_files(args.release_psds, run)
        if not files:
            parser.error(f"no release PSD for {run} in {args.release_psds}")
        release.update(
            {
                (run, ifo, month): read_release_psd(p)
                for (ifo, month), p in files.items()
            }
        )

    simulation = norms(SIMULATION_WAVEFORM, SIMULATION_F_LOW, targets)
    factors = {}
    for (run, ifo, month), norm in norms(
        RELEASE_WAVEFORM, RELEASE_F_LOW, release
    ).items():
        # A signal with no power above the release cutoff has no optimal SNR in
        # the release, so its factor does not matter; it is set to zero.
        factors[run, ifo, month] = np.divide(
            simulation[ifo], norm, out=np.zeros_like(norm), where=norm > 0
        )

    rows = [
        dict(
            run=run,
            ifo=ifo,
            month=month,
            total_mass=round(total_mass, 4),
            mass_ratio=mass_ratio,
            factor=round(grid[i, j], 5),
        )
        for (run, ifo, month), grid in factors.items()
        for i, total_mass in enumerate(TOTAL_MASS)
        for j, mass_ratio in enumerate(MASS_RATIO)
    ]

    curves = ", ".join(
        " ".join(
            filter(None, [ifo, getattr(args, ifo), getattr(args, f"{ifo}_column")])
        )
        for ifo in targets
    )
    # measure.py reads the first two lines back to name the units of rho_eq.
    comments = [
        f"target: {curves}",
        f"simulations: {SIMULATION_WAVEFORM} from {SIMULATION_F_LOW:g} Hz",
        f"release: {args.release_psds.name}, "
        f"{RELEASE_WAVEFORM} from {RELEASE_F_LOW:g} Hz",
        f"command: {shlex.join(sys.argv)}",
    ] + provenance()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        for line in comments:
            f.write(f"# {line}\n")
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")

    # Summary: factor of a 1.4 + 1.4 Msun and a 30 + 30 Msun source per month.
    print(f"\n{'run':4} {'ifo':3} {'month':7} {'1.4+1.4':>8} {'30+30':>8}")
    for (run, ifo, month), grid in factors.items():
        values = [grid[np.argmin(abs(np.log(TOTAL_MASS / m))), -1] for m in (2.8, 60.0)]
        print(f"{run:4} {ifo:3} {month:7} {values[0]:8.3f} {values[1]:8.3f}")


if __name__ == "__main__":
    main()
