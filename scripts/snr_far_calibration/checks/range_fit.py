"""Does rho_eq follow the detector sensitivity within a run?

For each calendar month of a run, takes the monthly rho_eq of one bin
(variants.csv, group "month") and the HL network BNS range of the release PSD
of that month, and fits ln(rho_eq) = alpha ln(R) + c by weighted least squares.
The uncertainty of a monthly rho_eq is the bootstrap error of the full bin
scaled by sqrt(n_found / n_found of the month). When the fit is poor
(chi2/dof > 1), the error on alpha is multiplied by sqrt(chi2/dof).

    python checks/range_fit.py                       # O4a, BBH 20-35
    python checks/range_fit.py --run O4b --source-class NSBH --chirp-mass-min 2.2
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sensitivity  # noqa: E402
from config import OUT_DIR  # noqa: E402


def read(name):
    with open(OUT_DIR / name) as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", default="O4a")
    parser.add_argument("--source-class", default="BBH")
    parser.add_argument("--chirp-mass-min", type=float, default=20.0)
    args = parser.parse_args()

    def same_bin(row):
        return (
            row["run"] == args.run
            and row["source_class"] == args.source_class
            and np.isclose(float(row["chirp_mass_min"]), args.chirp_mass_min)
        )

    full = next(r for r in read("rho_eq.csv") if same_bin(r))
    error, n_found = float(full["error"]), int(full["n_found"])
    ranges = sensitivity.month_ranges()

    rows = []
    for r in read("variants.csv"):
        month = r["variant"].replace("-", "_")
        if not (same_bin(r) and r["group"] == "month"):
            continue
        if ("H1", month) not in ranges or ("L1", month) not in ranges:
            continue
        hl = np.hypot(ranges["H1", month], ranges["L1", month])
        sigma = error * np.sqrt(n_found / int(r["n_found"]))
        rows.append((r["variant"], hl, float(r["rho_eq"]), int(r["n_found"]), sigma))

    print(
        f"{args.run} {args.source_class} {args.chirp_mass_min:g}: "
        f"full bin rho_eq = {float(full['rho_eq']):.3f} +- {error:.3f} ({n_found} found)\n"
    )
    print(f"{'month':8} {'R_HL [Mpc]':>11} {'rho_eq':>7} {'found':>7} {'sigma':>6}")
    for month, hl, rho, n, sigma in rows:
        print(f"{month:8} {hl:11.1f} {rho:7.3f} {n:7d} {sigma:6.3f}")

    _, hl, rho, _, sigma = map(np.array, zip(*rows))
    x, y = np.log(hl), np.log(rho)
    w = (rho / sigma) ** 2  # 1 / variance of ln(rho)
    design = np.column_stack([x, np.ones_like(x)])
    cov = np.linalg.inv(design.T @ (w[:, None] * design))
    alpha, c = cov @ design.T @ (w * y)
    chi2 = np.sum(w * (y - alpha * x - c) ** 2)
    dof = len(x) - 2
    stat = np.sqrt(cov[0, 0])
    scaled = stat * np.sqrt(max(1.0, chi2 / dof))

    print(f"\nalpha = {alpha:+.4f}")
    print(
        f"statistical error = {stat:.4f}, chi2/dof = {chi2:.1f}/{dof} = {chi2 / dof:.2f}"
    )
    print(f"error scaled by sqrt(chi2/dof) = {scaled:.4f}")
    print(f"result: alpha = {alpha:.2f} +- {scaled:.2f}")


if __name__ == "__main__":
    main()
