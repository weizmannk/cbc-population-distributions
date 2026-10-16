"""Write the paper table: rho_eq in release SNRs and in simulation SNRs.

One measurement in two units, side by side, rather than two tables that would
invite the question of which rho_eq is the real one. The simulation columns
exist for O4a and O4b only, the runs the simulations are calibrated on.

    python latex_table.py                 # outputs/rho_eq_table.tex
"""

import argparse
import csv

from config import OUT_DIR, TABLE_BINS

RUNS = ("O4a", "O4b")
LABEL = {"BNS": r"\textsc{bns}", "NSBH": r"\textsc{nsbh}", "BBH": r"\textsc{bbh}"}


def read(name):
    with open(OUT_DIR / name) as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    return {(r["run"], r["source_class"], float(r["chirp_mass_min"])): r for r in rows}


def cell(table, run, cls, lo):
    row = table.get((run, cls, lo))
    return (
        "--"
        if row is None
        else rf"${float(row['rho_eq']):.2f} \pm {float(row['error']):.2f}$"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-o", "--output", default="rho_eq_table.tex")
    args = parser.parse_args()

    release, simulations = read("rho_eq_release.csv"), read("rho_eq_simulations.csv")
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\caption{Network SNR threshold $\rho_\mathrm{eq}$ that selects as many "
        r"injections as a $\mathrm{FAR} < 1\,\mathrm{yr}^{-1}$ cut, per source class "
        r"and detector-frame chirp-mass bin. The two columns of each run are the same "
        r"measurement expressed in two units: the optimal SNRs of the injection "
        r"release, and the SNRs the simulations compute for the same signals, with "
        r"their templates and their \SI{25}{\hertz} low-frequency cutoff. Only the "
        r"simulation columns apply to the simulated counts. Uncertainties are the "
        r"bootstrap error over the injections.}",
        r"\label{tab:rho-eq}",
        r"\begin{tabular}{llcccc}",
        r"\hline",
        r"& & \multicolumn{2}{c}{O4a} & \multicolumn{2}{c}{O4b} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Class & $\mathcal{M}_\mathrm{c}^\mathrm{det}$ [$M_\odot$] "
        r"& Release & Simulations & Release & Simulations \\",
        r"\hline",
    ]
    previous = None
    for cls, lo, hi in TABLE_BINS:
        name = LABEL[cls] if cls != previous else ""
        previous = cls
        cells = [cell(t, run, cls, lo) for run in RUNS for t in (release, simulations)]
        lines.append(f"{name} & ${lo:g}$--${hi:g}$ & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}", ""]

    path = OUT_DIR / args.output
    path.write_text("\n".join(lines))
    print(f"wrote {path} ({len(TABLE_BINS)} bins)")


if __name__ == "__main__":
    main()
