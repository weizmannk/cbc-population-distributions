"""GWOSC observing segments: which detectors were taking data at a given time.

Run once to download the segments (needs the ``gwosc`` package):

    python segments.py
"""

import argparse

import numpy as np
from config import RUN_DETECTORS, RUN_GPS, RUNS, SEGMENTS


def path(run, ifo):
    return SEGMENTS / f"{run}_{ifo}.txt"


def download(run):
    from gwosc.timeline import get_segments

    SEGMENTS.mkdir(parents=True, exist_ok=True)
    for ifo in RUN_DETECTORS[run]:
        if path(run, ifo).exists():
            continue
        segments = np.array(get_segments(f"{ifo}_DATA", *RUN_GPS[run]), dtype=np.int64)
        np.savetxt(
            path(run, ifo),
            segments.reshape(-1, 2),
            fmt="%d",
            header=f"{ifo}_DATA, {run}: start end [GPS s]",
        )
        print(f"{path(run, ifo)}: {len(segments)} segments")


def observing(gps, run):
    """{ifo: bool array}, True where ``ifo`` was observing at ``gps``."""
    result = {}
    for ifo in RUN_DETECTORS[run]:
        table = np.loadtxt(path(run, ifo), dtype=np.int64, ndmin=2)
        start, end = table[np.argsort(table[:, 0])].T
        i = np.searchsorted(start, gps, side="right") - 1
        result[ifo] = (i >= 0) & (gps < end[np.maximum(i, 0)])
    return result


def network(state):
    """Network label of each entry, e.g. 'H1L1V1', or '' when none observed."""
    labels = np.full(len(next(iter(state.values()))), "", dtype=object)
    for ifo, on in state.items():
        labels[on] += ifo
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=list(RUNS))
    for run in parser.parse_args().runs:
        download(run)
