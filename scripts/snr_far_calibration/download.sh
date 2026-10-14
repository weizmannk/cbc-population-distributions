#!/usr/bin/env bash
# Download the data this directory reads into data/ at the repository root,
# skipping files that are already there. Run once, on a machine with internet
# access:
#
#   bash download.sh
#
# The GWOSC segments are downloaded by `python segments.py`. The waveform data
# file goes to $LAL_DATA_PATH, or to ~/lalsuite-waveform-data when that
# variable is not set.
set -euo pipefail

DATA="$(cd "$(dirname "$0")/../.." && pwd)/data"
ZENODO=https://zenodo.org/records/19500052/files
INJECTIONS=20260410130052UTC-clipped.hdf
WAVEFORMS=${LAL_DATA_PATH:-$HOME/lalsuite-waveform-data}

fetch() {  # fetch URL FILE
    if [ -s "$2" ]; then
        echo "present: $2"
    else
        echo "downloading: $2"
        mkdir -p "$(dirname "$2")"
        curl -fL --retry 3 -o "$2.part" "$1"
        mv "$2.part" "$2"
    fi
}

fetch "$ZENODO/mixture-semi_o1_o2-real_o3_o4a_o4b-cartesian_spins_$INJECTIONS" \
    "$DATA/injections/mixture-semi_o1_o2-real_o3_o4a_o4b-cartesian_spins_$INJECTIONS"
fetch "$ZENODO/mixture-real_o3_o4a_o4b-cartesian_spins_$INJECTIONS" \
    "$DATA/injections/mixture-real_o3_o4a_o4b-cartesian_spins_$INJECTIONS"

# The release PSDs go to data/psds-o1234ab/ whatever the layout of the archive.
if ! ls "$DATA"/psds-o1234ab/psd-O* > /dev/null 2>&1; then
    fetch "$ZENODO/psds-o1234ab.tar.gz" "$DATA/psds-o1234ab.tar.gz"
    unpacked=$(mktemp -d)
    tar -xzf "$DATA/psds-o1234ab.tar.gz" -C "$unpacked"
    mkdir -p "$DATA/psds-o1234ab"
    find "$unpacked" -type f -name 'psd-O*' -exec mv {} "$DATA/psds-o1234ab/" \;
    rm -rf "$unpacked" "$DATA/psds-o1234ab.tar.gz"
fi
echo "present: $DATA/psds-o1234ab ($(ls "$DATA"/psds-o1234ab/psd-O* | wc -l) PSDs)"

fetch https://dcc.ligo.org/T2500363-v1/public/o4b_h1_ref.txt "$DATA/asd/o4b_h1_ref.txt"
fetch https://dcc.ligo.org/T2500363-v1/public/o4b_l1_ref.txt "$DATA/asd/o4b_l1_ref.txt"
fetch https://dcc.ligo.org/T2500388-v1/public/o4b_v1_ref.txt "$DATA/asd/o4b_v1_ref.txt"

fetch https://zenodo.org/records/14999310/files/SEOBNRv4ROM_v3.0.hdf5 \
    "$WAVEFORMS/SEOBNRv4ROM_v3.0.hdf5"
if [ -z "${LAL_DATA_PATH:-}" ]; then
    echo "set LAL_DATA_PATH=$WAVEFORMS before running snr_scale.py"
fi
