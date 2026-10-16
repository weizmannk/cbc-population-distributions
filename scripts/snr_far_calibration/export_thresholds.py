"""Copy the simulation thresholds into the observing-scenarios fork.

The fork reads one table, thresholds/rho_eq.csv, and it must be the
_simulations one: its SNRs are those the simulations compute, for the same
templates and low-frequency cutoff. Copying by hand is what let the two drift
apart, so this script does it and refuses the release table outright.

    python export_thresholds.py ../../observing-scenarios-simulations
    python export_thresholds.py <fork> --check    # report, write nothing
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

from config import OUT_DIR

SOURCE = "rho_eq_simulations.csv"
TARGET = Path("thresholds") / "rho_eq.csv"
DROP = ("found_below", "missed_above")
RUNS = ("O4a", "O4b")


def read(path):
    with open(path) as f:
        comments = [line for line in f if line.startswith("#")]
    with open(path) as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    return comments, rows


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "fork", type=Path, help="root of observing-scenarios-simulations"
    )
    parser.add_argument(
        "--check", action="store_true", help="compare only, write nothing"
    )
    args = parser.parse_args()

    source = OUT_DIR / SOURCE
    if not source.exists():
        parser.error(f"{source} is missing; run measure.py --snr-scale first")
    comments, rows = read(source)
    if not any("simulations" in line for line in comments):
        parser.error(f"{source} does not carry the simulation units; wrong table")

    rows = [r for r in rows if r["run"] in RUNS]
    fields = [k for k in rows[0] if k not in DROP]
    target = args.fork / TARGET

    if target.exists():
        _, old = read(target)
        stale = [
            (
                r["run"],
                r["source_class"],
                r["chirp_mass_min"],
                float(o["rho_eq"]),
                float(r["rho_eq"]),
            )
            for o, r in zip(old, rows)
            if abs(float(o["rho_eq"]) - float(r["rho_eq"])) > 1e-9
        ]
        if stale:
            worst = max(stale, key=lambda t: abs(t[4] - t[3]))
            print(
                f"{len(stale)}/{len(rows)} rows differ, largest {worst[4] - worst[3]:+.3f} "
                f"at {worst[0]} {worst[1]} {worst[2]}"
            )
        else:
            print(f"{target} is already up to date")
            return
    if args.check:
        sys.exit(1)

    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.with_suffix(".csv.bak")
    if target.exists():
        shutil.copy2(target, backup)
    with open(target, "w", newline="") as f:
        f.writelines(comments)
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"wrote {target} ({len(rows)} rows){f', previous kept as {backup.name}' if backup.exists() else ''}"
    )


if __name__ == "__main__":
    main()
