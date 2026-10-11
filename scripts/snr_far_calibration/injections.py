"""LVK sensitivity injections of one observing run."""

from dataclasses import dataclass

import h5py
import numpy as np
from config import BH_MAX_MASS, NS_MAX_MASS, RUN_DETECTORS, RUN_GPS

# Search families. O3 had two PyCBC searches (broad and BBH-focused) and O4 one;
# taking the smallest FAR within a family makes the runs comparable.
SEARCHES = {"GstLAL": "gstlal", "MBTA": "mbta", "PyCBC": "pycbc", "cWB": "cwb"}


@dataclass
class Injections:
    chirp_mass_det: np.ndarray  # detector frame [Msun]
    source_class: np.ndarray  # 'BNS', 'NSBH' or 'BBH'
    masses: np.ndarray  # source-frame component masses, (n, 2), heavier first
    optimal_snr: dict  # {ifo: optimal SNR with the release PSD}
    release_on: dict  # {ifo: the release gives an SNR, so it observed}
    far: dict  # {search column: FAR [1/yr]}
    gps: np.ndarray

    def __len__(self):
        return len(self.gps)

    def subset(self, index):
        return Injections(
            self.chirp_mass_det[index],
            self.source_class[index],
            self.masses[index],
            {k: v[index] for k, v in self.optimal_snr.items()},
            {k: v[index] for k, v in self.release_on.items()},
            {k: v[index] for k, v in self.far.items()},
            self.gps[index],
        )

    def found(self, threshold, family=None):
        """Found below ``threshold`` by any search, or by one search family."""
        names = [n for n in self.far if family is None or SEARCHES[family] in n]
        if not names:
            return np.zeros(len(self), bool)
        return np.min([self.far[n] for n in names], axis=0) < threshold


def classify(mass1, mass2):
    heavy, light = np.maximum(mass1, mass2), np.minimum(mass1, mass2)
    return np.where(
        heavy < NS_MAX_MASS, "BNS", np.where(light < NS_MAX_MASS, "NSBH", "BBH")
    )


def load(path, run, max_mass=BH_MAX_MASS):
    """Injections of ``run`` with both source-frame components below ``max_mass``."""
    prefix = "o3_" if run.startswith("O3") else f"{run.lower()}_"
    with h5py.File(path, "r") as f:
        events = f["events"]
        gps = events["time_geocenter"][:]
        start, end = RUN_GPS[run]
        rows = np.flatnonzero((gps >= start) & (gps < end))

        def column(name):
            return events[name][rows]

        z = column("redshift")
        m1, m2 = column("mass1_source"), column("mass2_source")
        far = {
            name[len(prefix) : -len("_far")]: column(name)
            for name in events.dtype.names
            if name.startswith(prefix) and name.endswith("_far")
        }
        if not far:
            raise KeyError(f"no search results for {run} in {path}")
        raw = {
            ifo: column(f"estimated_optimal_snr_{ifo[0]}") for ifo in RUN_DETECTORS[run]
        }
        # The release leaves the SNR undefined where a detector did not observe.
        release_on = {ifo: np.isfinite(v) for ifo, v in raw.items()}
        snr = {ifo: np.nan_to_num(v) for ifo, v in raw.items()}

    source_chirp_mass = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    inj = Injections(
        chirp_mass_det=(1 + z) * source_chirp_mass,
        source_class=classify(m1, m2),
        masses=np.column_stack([np.maximum(m1, m2), np.minimum(m1, m2)]),
        optimal_snr=snr,
        release_on=release_on,
        far=far,
        gps=gps[rows],
    )
    return inj.subset(inj.masses[:, 0] <= max_mass)
