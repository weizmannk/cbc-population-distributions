"""Panels (a) and (c) of figure.py on the SNRs of the simulations.

    python figure_simulation.py

(a) How rho_eq is set, for BNS in O4.
(b) rho_eq in every chirp-mass bin of O4a and O4b.

The thresholds are those copied to thresholds/rho_eq.csv of
observing-scenarios-simulations.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from figure import COLOR, DARK, MARKER, panel_principle, panel_thresholds, read
from matplotlib.lines import Line2D

UNITS = "# SNRs: optimal SNRs of the simulations"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rho-eq", type=Path, default=Path("outputs/rho_eq_simulations.csv")
    )
    parser.add_argument(
        "--histograms", type=Path, default=Path("outputs/histograms_simulations.csv")
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("outputs/snr_threshold_simulations.png"),
    )
    args = parser.parse_args()

    for path in (args.rho_eq, args.histograms):
        with open(path) as f:
            if not f.readline().startswith(UNITS):
                raise SystemExit(f"{path} is not on the SNRs of the simulations")

    # read() joins its argument to OUT_DIR, which keeps an absolute path as is.
    results = read(args.rho_eq.resolve())
    histograms = read(args.histograms.resolve())

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.9), layout="constrained")
    panel_principle(a, results, histograms)
    panel_thresholds(b, results)
    for ax, label in zip((a, b), "ab"):
        ax.set_title(f"({label})", loc="left", fontsize=9, fontweight="bold", pad=3)

    handles = [
        Line2D(
            [],
            [],
            color=COLOR[cls],
            marker=MARKER[cls],
            ls="none",
            ms=5,
            label=f"{cls}, O4a and O4b",
        )
        for cls in ("BNS", "NSBH", "BBH")
    ]
    handles.append(Line2D([], [], color=DARK, lw=0.8, ls=":", label="SNR = 8"))
    b.legend(handles=handles, loc="upper right", handletextpad=0.4)

    fig.savefig(args.output, dpi=200)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
