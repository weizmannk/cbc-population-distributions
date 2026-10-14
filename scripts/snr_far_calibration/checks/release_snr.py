"""Does snr_scale.py compute the optimal SNRs of the release the way the release does?

snr_scale.py divides by sigma(release template, release PSD of the month), the
norm with which the release computed each optimal SNR. If that is right, then
for every injection

    rho_release = sigma * |(1 + cos^2 i) / 2 * F+ + i cos i * Fx| / D_L.

The release gives the sky position and the inclination but not the polarization
angle, on which F+ and Fx depend. For a face-on source, |cos i| = 1, the factor
reduces to sqrt(F+^2 + Fx^2), which does not depend on it. This check takes the
face-on injections with small aligned spins and prints the median and the
16-84% range of rho_release / rho_predicted per detector and chirp-mass bin.
A median of 1 means that the template, the low-frequency cutoff and the PSDs of
config.py reproduce the release; the spread comes from the precession and the
higher modes of the injected signals, absent from the template. The mass grid
of snr_scale.py limits the comparison to about 1% above 20 Msun of chirp mass
and 0.3% below.

Only the mixture-real file of the release has the sky position; by default it
is read next to config.INJECTIONS.

    python checks/release_snr.py --runs O4a O4b
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py
import lal
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import snr_scale  # noqa: E402
from config import (  # noqa: E402
    INJECTIONS,
    RELEASE_F_LOW,
    RELEASE_PSDS,
    RELEASE_WAVEFORM,
    RUN_DETECTORS,
    RUN_GPS,
)
from measure import months  # noqa: E402
from snr_scale import MASS_RATIO, TOTAL_MASS  # noqa: E402

REAL_INJECTIONS = INJECTIONS.with_name(
    "mixture-real_o3_o4a_o4b-cartesian_spins_20260410130052UTC-clipped.hdf"
)
MIN_COS_INCLINATION = 0.98  # the factor is then within 1% of sqrt(F+^2 + Fx^2)
MAX_CHI_EFF = 0.1
MIN_SNR = 4.0  # the release keeps injections above a network SNR of 5
CHIRP_MASS_EDGES = (0.85, 2.2, 7.0, 20.0, 60.0, 300.0)


def chirp_mass(total_mass, mass_ratio):
    return total_mass * (mass_ratio / (1 + mass_ratio) ** 2) ** 0.6


def antenna(ifo, ra, dec, gps):
    """sqrt(F+^2 + Fx^2) of ``ifo``, which does not depend on the polarization."""
    response = lal.cached_detector_by_prefix[ifo].response
    return np.array(
        [
            np.hypot(
                *lal.ComputeDetAMResponse(
                    response, a, d, 0.0, lal.GreenwichMeanSiderealTime(t)
                )
            )
            for a, d, t in zip(ra, dec, gps)
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--injections", type=Path, default=REAL_INJECTIONS)
    parser.add_argument("--runs", nargs="+", default=["O4a", "O4b"])
    parser.add_argument("--release-psds", type=Path, default=RELEASE_PSDS)
    args = parser.parse_args()
    logging.getLogger("BAYESTAR").setLevel(logging.ERROR)

    print(
        f"release template {RELEASE_WAVEFORM} from {RELEASE_F_LOW:g} Hz; "
        f"|cos i| >= {MIN_COS_INCLINATION}, |chi_eff| <= {MAX_CHI_EFF}"
    )
    print(f"{'run':4} {'ifo':3} {'Mc_det':>9} {'n':>6} {'median':>7} {'16-84%':>13}")
    for run in args.runs:
        files = snr_scale.release_psd_files(args.release_psds, run)
        psds = {key: snr_scale.read_release_psd(p) for key, p in files.items()}
        # sigma scales as chirp mass^(5/6) at leading order; the remainder is
        # smooth enough in the masses to be interpolated on the grid.
        grid_chirp = chirp_mass(*np.meshgrid(TOTAL_MASS, MASS_RATIO, indexing="ij"))
        norm = {
            key: snr_scale.interpolator(grid / grid_chirp ** (5 / 6))
            for key, grid in snr_scale.norms(
                RELEASE_WAVEFORM, RELEASE_F_LOW, psds
            ).items()
        }

        with h5py.File(args.injections, "r") as f:
            events = f["events"]
            gps = events["time_geocenter"][:]
            start, end = RUN_GPS[run]
            m1, m2 = events["mass1_source"][:], events["mass2_source"][:]
            chi_eff = (m1 * events["spin1z"][:] + m2 * events["spin2z"][:]) / (m1 + m2)
            rows = np.flatnonzero(
                (gps >= start)
                & (gps < end)
                & (np.abs(np.cos(events["inclination"][:])) >= MIN_COS_INCLINATION)
                & (np.abs(chi_eff) <= MAX_CHI_EFF)
            )
            data = {
                name: events[name][rows]
                for name in (
                    "time_geocenter",
                    "redshift",
                    "luminosity_distance",
                    "right_ascension",
                    "declination",
                    "inclination",
                    "mass1_source",
                    "mass2_source",
                )
            }
            for ifo in RUN_DETECTORS[run]:
                data[ifo] = events[f"estimated_optimal_snr_{ifo[0]}"][rows]

        heavy = np.maximum(data["mass1_source"], data["mass2_source"])
        light = np.minimum(data["mass1_source"], data["mass2_source"])
        total_mass = (heavy + light) * (1 + data["redshift"])
        mass_ratio = light / heavy
        chirp = chirp_mass(total_mass, mass_ratio)
        month = (
            np.full(len(rows), "O3")
            if run.startswith("O3")
            else np.char.replace(months(data["time_geocenter"]), "-", "_")
        )
        cos_i = np.cos(data["inclination"])
        orientation = np.sqrt(((1 + cos_i**2) / 2) ** 2 + cos_i**2) / np.sqrt(2)

        for ifo in RUN_DETECTORS[run]:
            available = sorted(m for d, m in files if d == ifo)
            if not available:
                continue
            predicted = np.full(len(rows), np.nan)
            for m in np.unique(month):
                use = m if m in available else snr_scale.nearest(m, available)
                selected = month == m
                predicted[selected] = norm[ifo, use](
                    total_mass[selected], mass_ratio[selected]
                ) * chirp[selected] ** (5 / 6)
            predicted *= (
                antenna(
                    ifo,
                    data["right_ascension"],
                    data["declination"],
                    data["time_geocenter"],
                )
                * orientation
                / data["luminosity_distance"]
            )
            ratio = data[ifo] / predicted
            usable = np.isfinite(ratio) & (data[ifo] >= MIN_SNR)
            for lo, hi in zip(CHIRP_MASS_EDGES[:-1], CHIRP_MASS_EDGES[1:]):
                values = ratio[usable & (chirp >= lo) & (chirp < hi)]
                if len(values) < 20:
                    continue
                q16, q50, q84 = np.percentile(values, [16, 50, 84])
                print(
                    f"{run:4} {ifo:3} {lo:>4g}-{hi:<4g} {len(values):6d} "
                    f"{q50:7.3f} {q16:6.3f}-{q84:<6.3f}"
                )


if __name__ == "__main__":
    main()
