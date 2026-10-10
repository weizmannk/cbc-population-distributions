#!/usr/bin/env bash
# Run every step and check, and pack the outputs into results.zip.
#
#   bash run_all.sh
#
# Optional steps run only when their inputs exist:
#   ../../data/asd/o4b_*_ref.txt          thresholds for the reference curves
#   ../../data/runs/O4{a,b}/psds.xml      noise-model check (needs LAL_DATA_PATH)
set -uo pipefail
cd "$(dirname "$0")"

DATA=../../data
OUT=outputs
LOG=$OUT/run_all.log
mkdir -p "$OUT"
: > "$LOG"

step() {
    echo -e "\n===== $* =====" | tee -a "$LOG"
    "$@" 2>&1 | tee -a "$LOG"
}

step python segments.py
step python measure.py
step python sensitivity.py
step python figure.py
step python checks/coverage.py
step python checks/closure.py
step python checks/binning.py

skip() {
    echo -e "\n===== skipped: $* =====" | tee -a "$LOG"
}

if ls "$DATA"/asd/o4b_*_ref.txt > /dev/null 2>&1; then
    SNRFAR_OUT=$OUT/reference step python measure.py --runs O4a O4b --reference "$DATA/asd"
else
    skip "reference curves, no $DATA/asd/o4b_*_ref.txt"
fi

# O4a and O4b share the L1 curve of the simulations; H1 and V1 differ.
for check in "O4a L1" "O4b H1" "O4b V1"; do
    set -- $check
    if [ ! -f "$DATA/runs/$1/psds.xml" ]; then
        skip "noise model $1 $2, no $DATA/runs/$1/psds.xml"
    elif [ -z "${LAL_DATA_PATH:-}" ]; then
        skip "noise model $1 $2, LAL_DATA_PATH not set"
    else
        step python checks/noise_model.py "$DATA/runs/$1/psds.xml" --detector "$2"
    fi
done

rm -f results.zip
zip -qr results.zip "$OUT"
echo -e "\nSend results.zip ($(du -h results.zip | cut -f1))." | tee -a "$LOG"
