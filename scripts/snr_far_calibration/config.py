"""Paths, observing runs and analysis choices.

Each value is taken from a data release, a catalogue paper or the
observing-scenarios pipeline, as noted next to it.
"""

import os
from pathlib import Path

REPO = Path(os.environ.get("CBCPOP_REPO", Path(__file__).resolve().parents[2]))
OUT_DIR = Path(
    os.environ.get("SNRFAR_OUT", Path(__file__).resolve().parent / "outputs")
)

# GWTC-5.0 cumulative sensitivity estimates, doi:10.5281/zenodo.19500052
# (Essick et al. 2025, arXiv:2508.10638): injections and the reference PSDs
# used for their optimal SNRs (psds-o1234ab.tar.gz, unpacked).
INJECTIONS = (
    REPO
    / "data"
    / "injections"
    / (
        "mixture-semi_o1_o2-real_o3_o4a_o4b-cartesian_spins_20260410130052UTC-clipped.hdf"
    )
)
RELEASE_PSDS = REPO / "data" / "psds-o1234ab"
SEGMENTS = REPO / "data" / "segments"

# GPS boundaries [start, end) of the runs, as in the injection release.
RUN_GPS = {
    "O3a": (1238166018, 1253977218),
    "O3b": (1256655618, 1269363618),
    "O4a": (1368975618, 1389456018),
    "O4b": (1396969218, 1422118818),
}
RUN_DETECTORS = {
    "O3a": ("H1", "L1", "V1"),
    "O3b": ("H1", "L1", "V1"),
    "O4a": ("H1", "L1"),
    "O4b": ("H1", "L1", "V1"),
}
RUNS = tuple(RUN_GPS)

# Fraction of calendar time per detector network (GWOSC). Used only for the
# duty-cycle variant; the measurement itself reads the segments.
# O4a: arXiv:2508.18079, Table 1. O4b: arXiv:2605.27090, Table 2.
DUTY_CYCLES = {
    "O4a": {("H1", "L1"): 0.5335, ("H1",): 0.1407, ("L1",): 0.1561},
    "O4b": {
        ("H1", "L1", "V1"): 0.311,
        ("H1", "L1"): 0.074,
        ("H1", "V1"): 0.075,
        ("L1", "V1"): 0.219,
        ("H1",): 0.027,
        ("L1",): 0.078,
        ("V1",): 0.103,
    },
}

# A detection is a FAR below 1/yr, as in the GWTC catalogues.
FAR_THRESHOLD = 1.0
# Threshold of the observing-scenarios simulations (bayestar-realize-coincs).
CURRENT_THRESHOLD = 8.0

# Source classes from source-frame component masses. The NS/BH boundary is
# 2.5 Msun, as in every comparison with LVK data in the paper (3 Msun is used
# only to classify the simulated events). It is the only boundary present in
# the injections: the O3 sets cap the spins at 0.4 below 2.5 Msun; the O4 sets
# draw spins independently of mass. Injections with a component above
# BH_MAX_MASS, the upper end of the FullPop mass range, are left out.
NS_MAX_MASS = 2.5
BH_MAX_MASS = 100.0
CLASSES = ("BNS", "NSBH", "BBH")

# Detector-frame chirp-mass bins [Msun], a factor ~1.7 wide, halved below
# 2.2 Msun where rho_eq varies within a bin in O4 (checks/binning.py). The
# first edge is below the lightest BNS (1 + 1 Msun, chirp mass 0.87 Msun); the
# last reaches 100 + 100 Msun BBH (source chirp mass 87 Msun) up to z ~ 2.4.
# checks/coverage.py counts the injections outside these edges.
CHIRP_MASS_EDGES = (
    0.85,
    1.12,
    1.4,
    1.75,
    2.2,
    4.0,
    7.0,
    12.0,
    20.0,
    35.0,
    60.0,
    120.0,
    300.0,
)

# Bins reported in the paper. The others are nearly empty in FullPop: BBH with
# both components in the lower mass gap, BNS with both components near 2.5 Msun.
TABLE_BINS = (
    ("BNS", 0.85, 1.12),
    ("BNS", 1.12, 1.4),
    ("BNS", 1.4, 1.75),
    ("BNS", 1.75, 2.2),
    ("NSBH", 1.4, 1.75),
    ("NSBH", 1.75, 2.2),
    ("NSBH", 2.2, 4.0),
    ("NSBH", 4.0, 7.0),
    ("BBH", 7.0, 12.0),
    ("BBH", 12.0, 20.0),
    ("BBH", 20.0, 35.0),
    ("BBH", 35.0, 60.0),
    ("BBH", 60.0, 120.0),
    ("BBH", 120.0, 300.0),
)

# One well-populated bin per class, used for the variants and the figure.
REFERENCE_BINS = (("BNS", 1.12, 1.4), ("NSBH", 2.2, 4.0), ("BBH", 20.0, 35.0))
