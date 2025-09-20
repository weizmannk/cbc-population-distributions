"""
Spin population models for gravitational-wave sources from "https://github.com/ColmTalbot/gwpopulation".
Provides functions for evaluating spin orientation and spin magnitude distributions.
"""

from gwpopulation.utils import beta_dist, truncnorm, xp

def iid_spin_orientation_gaussian_isotropic(dataset, xi_spin, sigma_spin):
    r"""A mixture model of spin orientations with isotropic and normally
    distributed components. The distribution of primary and secondary spin
    orientations are expected to be identical and independent.

    https://arxiv.org/abs/1704.08370 Eq. (4)

    .. math::
        p(z_1, z_2 | \xi, \sigma) =
        \frac{(1 - \xi)^2}{4}
        + \xi \prod_{i\in\{1, 2\}} \mathcal{N}(z_i; \mu=1, \sigma=\sigma, z_\min=-1, z_\max=1)

    Where :math:`\mathcal{N}` is the truncated normal distribution.

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'cos_tilt_1' and 'cos_tilt_2'.
    xi_spin: float
        Fraction of black holes in preferentially aligned component (:math:`\xi`).
    sigma_spin: float
        Width of preferentially aligned component.
    """
    return independent_spin_orientation_gaussian_isotropic(
        dataset, xi_spin, sigma_spin, sigma_spin
    )


def independent_spin_orientation_gaussian_isotropic(dataset, xi_spin, sigma_1, sigma_2):
    r"""A mixture model of spin orientations with isotropic and normally
    distributed components.

    https://arxiv.org/abs/1704.08370 Eq. (4)

    .. math::
        p(z_1, z_2 | \xi, \sigma_1, \sigma_2) =
        \frac{(1 - \xi)^2}{4}
        + \xi \prod_{i\in\{1, 2\}} \mathcal{N}(z_i; \mu=1, \sigma=\sigma_i, z_\min=-1, z_\max=1)

    Where :math:`\mathcal{N}` is the truncated normal distribution.

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays for 'cos_tilt_1' and 'cos_tilt_2'.
    xi_spin: float
        Fraction of black holes in preferentially aligned component (:math:`\xi`).
    sigma_1: float
        Width of preferentially aligned component for the more
        massive black hole (:math:`\sigma_1`).
    sigma_2: float
        Width of preferentially aligned component for the less
        massive black hole (:math:`\sigma_2`).
    """
    prior = (1 - xi_spin) / 4 + xi_spin * truncnorm(
        dataset["cos_tilt_1"], 1, sigma_1, 1, -1
    ) * truncnorm(dataset["cos_tilt_2"], 1, sigma_2, 1, -1)
    return prior


def iid_spin_magnitude_beta(dataset, amax=1, alpha_chi=1, beta_chi=1):
    """Independent and identically distributed beta distributions for both spin magnitudes.

    https://arxiv.org/abs/1805.06442 Eq. (10)
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays containing 'a_1' and 'a_2'.
    alpha_chi, beta_chi: float
        Parameters of Beta distribution for both black holes.
    amax: float
        Maximum black hole spin.
    """
    return independent_spin_magnitude_beta(
        dataset, alpha_chi, alpha_chi, beta_chi, beta_chi, amax, amax
    )


def independent_spin_magnitude_beta(
    dataset, alpha_chi_1, alpha_chi_2, beta_chi_1, beta_chi_2, amax_1, amax_2
):
    """Independent beta distributions for both spin magnitudes.

    https://arxiv.org/abs/1805.06442 Eq. (10)
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html

    Parameters
    ----------
    dataset: dict
        Dictionary of numpy arrays containing 'a_1' and 'a_2'.
    alpha_chi_1, beta_chi_1: float
        Parameters of Beta distribution for more massive black hole.
    alpha_chi_2, beta_chi_2: float
        Parameters of Beta distribution for less massive black hole.
    amax_1, amax_2: float
        Maximum spin of the more/less massive black hole.
    """
    if alpha_chi_1 < 0 or beta_chi_1 < 0 or alpha_chi_2 < 0 or beta_chi_2 < 0:
        return 0

    # Restrict spins for neutron stars (mass < 3 M_sun)
    amax_1 = xp.where(dataset["mass_1"] < 3, 0.4, amax_1)
    amax_2 = xp.where(dataset["mass_2"] < 3, 0.4, amax_2)

    prior = beta_dist(
        dataset["a_1"], alpha_chi_1, beta_chi_1, scale=amax_1
    ) * beta_dist(dataset["a_2"], alpha_chi_2, beta_chi_2, scale=amax_2)

    return prior


def independent_spin_magnitude_gaussian(
    dataset, mu_chi_1, mu_chi_2, sigma_chi_1, sigma_chi_2, amax_1, amax_2
):
    """
    A model for the independent spin magnitude distribution of black holes.

    Parameters
    ----------
    dataset: dict
        Dictionary of arrays for 'a_1' and 'a_2'.
    mu_chi_1: float
        Mean of the spin magnitude Gaussian component for the primary black holes.
    mu_chi_2: float
        Mean of the spin magnitude Gaussian component for the secondary black holes.
    sigma_chi_1: float
        Standard deviation of the spin magnitude Gaussian component for the primary black holes.
    sigma_chi_2: float
        Standard deviation of the spin magnitude Gaussian component for the secondary black holes.
    amax_1: float
        Maximum spin magnitude for the primary black holes.
    amax_2: float
        Maximum spin magnitude for the secondary black holes.
    """
    amax_1 = xp.where(dataset["mass_1"] < 3, 0.4, amax_1)
    amax_2 = xp.where(dataset["mass_2"] < 3, 0.4, amax_2)
    p_a_1 = truncnorm(
        dataset["a_1"], mu=mu_chi_1, sigma=sigma_chi_1, high=amax_1, low=0
    )
    p_a_2 = truncnorm(
        dataset["a_2"], mu=mu_chi_2, sigma=sigma_chi_2, high=amax_2, low=0
    )
    return p_a_1 * p_a_2


def iid_spin_magnitude_gaussian(dataset, mu_chi, sigma_chi, amax):
    """
    A model for the independent and identically distributed spin magnitude distribution of black holes.

    Parameters
    ----------
    dataset: dict
        Dictionary of arrays for 'a_1' and 'a_2'.
    mu_chi: float
        Mean of the spin magnitude Gaussian component.
    sigma_chi: float
        Standard deviation of the spin magnitude Gaussian component.
    amax: float
        Maximum spin magnitude.
    """
    prior = independent_spin_magnitude_gaussian(
        dataset, mu_chi, mu_chi, sigma_chi, sigma_chi, amax, amax
    )
    return prior
