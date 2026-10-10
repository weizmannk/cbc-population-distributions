"""Simulated network SNR and the SNR threshold equivalent to a FAR cut.

The observed SNR follows bayestar-realize-coincs (Singer & Price 2016): in each
observing detector, |rho_opt + n| with n a complex Gaussian of unit variance
per quadrature, plus the mean excess of its peak search; a detector counts
only above SINGLE_DETECTOR_MIN, and the network SNR is the quadrature sum.
"""

import numpy as np

NOISE_DRAWS = 8  # 8, 16 and 64 give the same rho_eq to 0.01
PEAK_OFFSET = 0.1  # measured with checks/noise_model.py
SINGLE_DETECTOR_MIN = 1.0  # bayestar-realize-coincs --snr-threshold
BOOTSTRAP = 200  # resamples of the injections; error known to ~5%


def network_snr(optimal_snr, state, rng, draws=NOISE_DRAWS, offset=PEAK_OFFSET):
    """Observed network SNR, shape (injections, draws).

    ``state`` maps each detector to a boolean array, of shape (injections,)
    or (injections, draws), True where the detector was observing.
    """
    n = len(next(iter(optimal_snr.values())))
    power = np.zeros((n, draws))
    for ifo, rho in optimal_snr.items():
        on = np.broadcast_to(np.reshape(state[ifo], (n, -1)), (n, draws))
        snr = (
            np.hypot(
                rho[:, None] + rng.standard_normal((n, draws)),
                rng.standard_normal((n, draws)),
            )
            + offset
        )
        power += np.where(on & (snr >= SINGLE_DETECTOR_MIN), snr**2, 0.0)
    return np.sqrt(power)


def equivalent_threshold(snr, found):
    """SNR above which the fraction of simulated values equals the found fraction."""
    return float(np.quantile(snr, 1.0 - found.mean()))


def bootstrap_error(snr, found, rng, resamples=BOOTSTRAP):
    estimates = []
    for _ in range(resamples):
        i = rng.integers(0, len(found), len(found))
        estimates.append(equivalent_threshold(snr[i], found[i]))
    return float(np.std(estimates))
