"""
Implements both independent and mass-ratio paired versions of the 2D mass model, plus the single-mass model.
"""

from gwpopulation.utils import truncnorm, xp


def power_law_dip_break_1d(
    mass,
    A,
    A2,
    NSmin,
    NSmax,
    BHmin,
    BHmax,
    UPPERmin,
    UPPERmax,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    alpha_1,
    alpha_2,
    alpha_dip,
    mu1,
    sig1,
    mix1,
    mu2,
    sig2,
    mix2,
    absolute_mmin=0.5,
    absolute_mmax=350,
):
    r"""
    The one-dimensional mass distribution considered in Fishbach, Essick, Holz. Does
    Matter Matter? ApJ Lett 899, 1 (2020) : arXiv:2006.13178

    .. math::
        p(m|\lambda) = n(m|\gamma_{\text{low}}, \gamma_{\text{high}}, A) \times
            l(m|m_{\text{max}}, \eta) \\
                 \times \begin{cases}
                         & m^{\alpha_1} \text{ if } m < \gamma_{\text{low}} \\
                         & m^{\alpha_2} \text{ if } m > \gamma_{\text{low}} \\
                         & 0 \text{ otherwise }
                 \end{cases}.
    
    where $l(m|m_{\text{max}}, \eta)$ is the low pass filter with powerlaw $\eta$
    applied at mass $m_{\text{max}}$,
    $n(m|\gamma_{\text{low}}, \gamma_{\text{high}}, A)$ is the notch
    filter with depth $A$ applied between $\gamma_{\text{low}}$ and 
    $\gamma_{\text{high}}$, and
    $\lambda$ is the subset of hyperparameters $\{ \gamma_{\text{low}},
    \gamma_{\text{high}}, A, \alpha_1, \alpha_2, m_{\text{min}}, m_{\
    text{max}}\}$.
    
    Parameters
    ----------
    mass: array-like
        Mass to evaluate probability at (:math:`m`).
    alpha_1: float
        Powerlaw exponent for compact objects below NSmax (:math:`\alpha_1`).
    alpha_2: float
        Powerlaw exponent for compact objects above BHmin (:math:`\alpha_2`).
    alpha_dip: float
        Powerlaw exponent for compact objects between NSmax and BHmin (:math: `\alpha_d`).
    NSmin: float
        Minimum compact object mass (:math:`m_\min`).
    NSmax: float
        Mass at which the notch filter starts for the lower mass gap (:math:`\gamma_{low1}`).
    BHmin: float
        Mass at which the notch filter ends for the lower mass gap (:math:`\gamma_{high1}`).
    BHmax: float
        Maximum mass in the powerlaw distributed component (:math:`m_\max`).
    A: float
        depth of the dip between NSmax and BHmin (A).
    UPPERmin: float
        Mass at which the notch filter starts for the upper mass gap (:math:`\gamma_{low2}`).
    UPPERmax: float
        Mass at which the notch filter ends for the upper mass gap (:math:`\gamma_{high2}`).
    mu1: float
        Location of the upper peak where an overdensity of merging compact objects is observed (:math:`\mu_{peak1}`).. 
    sig1: float
        Width of the upper peak where an overdensity of merging compact objects is observed (:math:`\sigma_{peak1}`)..
    mix1: float
        Mixing fraction of the first gaussian peak with the powerlaw + notches (:math: `c_1`)
    mu2: float
        Location of the lower peak where an overdensity of merging compact objects is observed (:math:`\mu_{peak2}`)..
    sig2: float
        Width of the lower peak where an overdensity of merging compact objects is observed (:math:`\mu_{peak2}`)..
    mix2: float
        Mixing fraction of the second gaussian peak with the powerlaw + notches (:math: `c_2`)
    absolute_mmin: float
        The minimum limit for the truncated normal distribution. 
    absolute_mmax: float
        The maximum limit for the truncated normal distribution. 
    n{0,5}:float
        Exponents to set the sharpness of the low mass cutoff and high mass cutoff, respectively (:math:`\eta_i`). 
    n{1,2}: float
        Exponents to set the sharpness of the lower edge and upper edge of the lower mass gap, respectively (:math:`\eta_i`). 
    n{3,4}: float
        Exponents to set the sharpness of the lower edge and upper edge of the upper mass gap, respectively (:math:`\eta_i`). 
        
    """

    gaussian_peak1 = truncnorm(mass, mu1, sig1, low=absolute_mmin, high=absolute_mmax)
    gaussian_peak2 = truncnorm(mass, mu2, sig2, low=absolute_mmin, high=absolute_mmax)

    condlist = [mass < NSmax, (mass >= NSmax) & (mass < BHmin), mass >= BHmin]
    choicelist = [
        mass**alpha_1,
        (mass**alpha_dip) * (NSmax ** (alpha_1 - alpha_dip)),
        (mass**alpha_2)
        * (NSmax ** (alpha_1 - alpha_dip))
        * (BHmin ** (alpha_dip - alpha_2)),
    ]
    plaw = xp.select(condlist, choicelist, default=0.0)

    highpass_lower = 1 + (NSmin / mass) ** n0
    notch_lower = 1.0 - A / ((1 + (NSmax / mass) ** n1) * (1 + (mass / BHmin) ** n2))
    notch_upper = 1.0 - A2 / (
        (1 + (UPPERmin / mass) ** n3) * (1 + (mass / UPPERmax) ** n4)
    )
    lowpass_upper = 1 + (mass / BHmax) ** n5

    return (
        (1 + mix1 * gaussian_peak1 + mix2 * gaussian_peak2)
        * plaw
        * notch_lower
        * notch_upper
        / highpass_lower
        / lowpass_upper
    )


def matter_matters_primary_secondary_independent(
    dataset,
    A,
    A2,
    NSmin,
    NSmax,
    BHmin,
    BHmax,
    UPPERmin,
    UPPERmax,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    alpha_1,
    alpha_2,
    alpha_dip,
    mu1,
    sig1,
    mix1,
    mu2,
    sig2,
    mix2,
    absolute_mmin,
    absolute_mmax,
):
    r"""
    Two-dimenstional mass distribution considered in Fishbach, Essick, Holz. Does
    Matter Matter? ApJ Lett 899, 1 (2020) : arXiv:2006.13178 modelling the
    primary and secondary masses as following independent distributions.

    Applies the 1D mass model to `mass_1` and `mass_2` independently, using :func:`power_law_dip_break_1d`.
    For parameter definitions, see :func:`power_law_dip_break_1d`.

    Parameters
    ----------
    dataset : dict
        Dictionary with 'mass_1' and 'mass_2' arrays.
    (All other parameters are as in  :func:`power_law_dip_break_1d`.)

    Returns
    -------
    prob : array-like
        Joint probability for each (mass_1, mass_2) pair.

    """
    p_m1 = power_law_dip_break_1d(
        dataset["mass_1"],
        A,
        A2,
        NSmin,
        NSmax,
        BHmin,
        BHmax,
        UPPERmin,
        UPPERmax,
        n0,
        n1,
        n2,
        n3,
        n4,
        n5,
        alpha_1,
        alpha_2,
        alpha_dip,
        mu1,
        sig1,
        mix1,
        mu2,
        sig2,
        mix2,
        absolute_mmin,
        absolute_mmax,
    )

    p_m2 = power_law_dip_break_1d(
        dataset["mass_2"],
        A,
        A2,
        NSmin,
        NSmax,
        BHmin,
        BHmax,
        UPPERmin,
        UPPERmax,
        n0,
        n1,
        n2,
        n3,
        n4,
        n5,
        alpha_1,
        alpha_2,
        alpha_dip,
        mu1,
        sig1,
        mix1,
        mu2,
        sig2,
        mix2,
        absolute_mmin,
        absolute_mmax,
    )

    prob = _primary_secondary_general(dataset, p_m1, p_m2)

    # get rid of areas where there are no injections
    prob = xp.where((dataset["mass_1"] > 60) * (dataset["mass_2"] < 3), 0, prob)
    return prob


def matter_matters_pairing(
    dataset,
    A,
    A2,
    NSmin,
    NSmax,
    BHmin,
    BHmax,
    UPPERmin,
    UPPERmax,
    n0,
    n1,
    n2,
    n3,
    n4,
    n5,
    alpha_1,
    alpha_2,
    alpha_dip,
    mu1,
    sig1,
    mix1,
    mu2,
    sig2,
    mix2,
    absolute_mmin,
    absolute_mmax,
    mbreak,
    beta_pair_1,
    beta_pair_2,
):
    r"""
    Two-dimenstional mass distribution considered in Fishbach, Essick, Holz. Does
    Matter Matter? ApJ Lett 899, 1 (2020) : arXiv:2006.13178 modelling the
    primary and secondary masses as following independent distributions.

    Compute the joint probability for a 2D mass model with mass-ratio dependent pairing.

    Applies the 1D mass distribution (see :func:`power_law_dip_break_1d`) to both
    primary and secondary masses and combines them with a mass-ratio dependent pairing.

    Parameters
    ----------
    dataset : dict
        Dictionary with 'mass_1' and 'mass_2' arrays.
    mbreak : float
        Break mass where the pairing power-law index changes.
    beta_pair_1 : float
        Pairing power-law index for `mass_2` below `mbreak`.
    beta_pair_2 : float
        Pairing power-law index for `mass_2` above `mbreak`.
    (Other parameters as in :func:`power_law_dip_break_1d`.)

    Returns
    -------
    prob : array-like
        Joint probability for each (mass_1, mass_2) pair.

    See Also
    --------
    power_law_dip_break_1d : 1D analytic mass distribution and full parameter list.

    Notes
    -----
    The analytic form and parameter definitions are described in :func:`power_law_dip_break_1d`.
    The mass-ratio pairing is implemented as a power-law in q, with an index that
    changes at `mbreak`.
    """

    p_m1 = power_law_dip_break_1d(
        dataset["mass_1"],
        A,
        A2,
        NSmin,
        NSmax,
        BHmin,
        BHmax,
        UPPERmin,
        UPPERmax,
        n0,
        n1,
        n2,
        n3,
        n4,
        n5,
        alpha_1,
        alpha_2,
        alpha_dip,
        mu1,
        sig1,
        mix1,
        mu2,
        sig2,
        mix2,
        absolute_mmin,
        absolute_mmax,
    )

    p_m2 = power_law_dip_break_1d(
        dataset["mass_2"],
        A,
        A2,
        NSmin,
        NSmax,
        BHmin,
        BHmax,
        UPPERmin,
        UPPERmax,
        n0,
        n1,
        n2,
        n3,
        n4,
        n5,
        alpha_1,
        alpha_2,
        alpha_dip,
        mu1,
        sig1,
        mix1,
        mu2,
        sig2,
        mix2,
        absolute_mmin,
        absolute_mmax,
    )

    beta_pair = xp.where(dataset["mass_2"] < mbreak, beta_pair_1, beta_pair_2)

    prob = _primary_secondary_plaw_pairing(dataset, p_m1, p_m2, beta_pair)
    # get rid of areas where there are no injections
    prob = xp.where((dataset["mass_1"] > 60) * (dataset["mass_2"] < 3), 0, prob)
    return prob


def _primary_secondary_general(dataset, p_m1, p_m2):
    return p_m1 * p_m2 * (dataset["mass_1"] >= dataset["mass_2"]) * 2


def _primary_secondary_plaw_pairing(dataset, p_m1, p_m2, beta_pair):
    q = dataset["mass_2"] / dataset["mass_1"]
    return _primary_secondary_general(dataset, p_m1, p_m2) * (q**beta_pair)
