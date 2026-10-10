"""Closure test: recover a known threshold from synthetic injections.

Injections are found when one independent noise realization of their network
SNR exceeds a planted threshold. rho_eq, measured from other realizations as in
measure.py, should return that threshold. The minimum SNR of the found
injections, a common shortcut, is printed for comparison.

    python checks/closure.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from threshold import equivalent_threshold, network_snr  # noqa: E402

N = 100_000
PLANTED = (8.0, 10.0, 12.0)


def main():
    rng = np.random.default_rng(0)
    # Optimal network SNR uniform in volume (p ~ rho^-4 above 3), split over two detectors.
    total = 3.0 * rng.uniform(size=N) ** (-1 / 3)
    share = rng.uniform(0.2, 0.8, N)
    optimal = {"H1": total * np.sqrt(share), "L1": total * np.sqrt(1 - share)}
    state = {ifo: np.ones(N, bool) for ifo in optimal}

    print(f"{'planted':>8} {'rho_eq':>8} {'min SNR of found':>17}")
    for planted in PLANTED:
        found = network_snr(optimal, state, rng, draws=1)[:, 0] >= planted
        snr = network_snr(optimal, state, rng)
        smallest = snr[found].min()
        print(
            f"{planted:8.1f} {equivalent_threshold(snr, found):8.2f} {smallest:17.2f}"
        )


if __name__ == "__main__":
    main()
