import numpy as np
import pandas as pd
from bilby.core.result import read_in_result


def _get_map_sample(hyperparams) -> pd.Series:
    """
    Return MAP sample if prior is informative, else ML sample.
    """
    post = hyperparams.posterior.copy()
    if "log_prior" in post and post["log_prior"].nunique() > 1:
        score = post.log_likelihood + post.log_prior
        return post.iloc[np.argmax(score)]
    else:
        return post.iloc[np.argmax(post.log_likelihood)]


hyperparams_file = (
    "../data/"
    "baseline5_widesigmachi2_mass_NotchFilterBinnedPairingMassDistribution_"
    "redshift_powerlaw_mag_iid_spin_magnitude_gaussian_tilt_"
    "iid_spin_orientation_result.hdf5"
)

# Load "Broken Power Law + 2 Peaks" model and extract MAP sample
hyperparams = read_in_result(hyperparams_file)
maxp_samp = _get_map_sample(hyperparams)
