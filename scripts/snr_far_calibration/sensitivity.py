"""BNS range of the PSDs used for the optimal SNRs of the injection release.

One PSD per detector for O3 (shared by O3a and O3b), one per month and
detector for O4. The range is that of a 1.4 + 1.4 Msun inspiral at SNR 8,
averaged over sky position and orientation (LIGO-T030276), from 10 Hz to the
innermost stable circular orbit.

    python sensitivity.py           # prints the table, writes outputs/release_ranges.csv
"""

import csv
import gzip
import re

import numpy as np
from config import OUT_DIR, RELEASE_PSDS, RUN_DETECTORS, RUNS

G, C, MSUN, MPC = 6.674e-11, 2.998e8, 1.989e30, 3.0857e22

# Calendar months of each O4 run (the release has one PSD per month).
O4_MONTHS = {
    "O4a": [f"2023_{m:02d}" for m in range(5, 13)] + ["2024_01"],
    "O4b": [f"2024_{m:02d}" for m in range(4, 13)] + ["2025_01"],
}


def bns_range(frequency, psd, f_min=10.0, mass=1.4):
    """Sky- and orientation-averaged range [Mpc] at SNR 8 (horizon / 2.264)."""
    total, chirp = 2 * mass * MSUN, mass * 2**-0.2 * MSUN
    f_isco = C**3 / (6**1.5 * np.pi * G * total)
    band = (frequency >= f_min) & (frequency <= f_isco) & (psd > 0)
    integral = np.trapezoid(frequency[band] ** (-7 / 3) / psd[band], frequency[band])
    amplitude = np.sqrt(5 / 24) * np.pi ** (-2 / 3) * (G * chirp) ** (5 / 6) / C**1.5
    return 2 * amplitude * np.sqrt(integral) / 8 / 2.264 / MPC


def read_psd(path):
    with gzip.open(path, "rt") as f:
        rows = [line.replace(",", " ").split() for line in f if line[0].isdigit()]
    return np.array(rows, float).T


def month_ranges():
    """{(ifo, month): range} of every O3 and O4 release PSD; month is 'O3' for O3."""
    ranges = {}
    for path in RELEASE_PSDS.glob("psd-O*gz"):
        match = re.match(r"psd-(?:O3|O4-(\d{4}_\d{2})_v1)-([HLV])\.", path.name)
        if match:
            ranges[f"{match[2]}1", match[1] or "O3"] = bns_range(*read_psd(path))
    return ranges


def reference_range(path):
    """Range of a reference sensitivity curve stored as (frequency, ASD)."""
    frequency, asd = np.loadtxt(path, unpack=True)
    return bns_range(frequency, asd**2)


def psd_files(run, ifo):
    if run.startswith("O3"):
        patterns = [f"psd-O3-{ifo[0]}"]
    else:
        patterns = [f"psd-O4-{month}_v1-{ifo[0]}" for month in O4_MONTHS[run]]
    return [p for name in patterns for p in sorted(RELEASE_PSDS.glob(name + ".*gz"))]


def release_ranges():
    """{(run, ifo): array of ranges [Mpc], one per PSD file}."""
    return {
        (run, ifo): np.array([bns_range(*read_psd(p)) for p in psd_files(run, ifo)])
        for run in RUNS
        for ifo in RUN_DETECTORS[run]
        if psd_files(run, ifo)
    }


def network_range(ranges, run, ifos=("H1", "L1")):
    """Quadrature sum over ``ifos`` of the mean range of each detector."""
    return float(np.sqrt(sum(ranges[run, ifo].mean() ** 2 for ifo in ifos)))


def main():
    ranges = release_ranges()
    rows = []
    print(f"{'run':4} {'ifo':3} {'range [Mpc]':>12} {'PSDs':>5} {'min-max':>10}")
    for (run, ifo), values in ranges.items():
        rows.append(
            dict(
                run=run,
                ifo=ifo,
                range_mean=round(values.mean(), 1),
                range_min=round(values.min(), 1),
                range_max=round(values.max(), 1),
                n_psd=len(values),
            )
        )
        print(
            f"{run:4} {ifo:3} {values.mean():12.0f} {len(values):5d} "
            f"{values.min():5.0f}-{values.max():<4.0f}"
        )
    print("\nHL network range (quadrature of the mean ranges):")
    for run in RUNS:
        print(f"  {run}: {network_range(ranges, run):.0f} Mpc")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "release_ranges.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_DIR / 'release_ranges.csv'}")


if __name__ == "__main__":
    main()
