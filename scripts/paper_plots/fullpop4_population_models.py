# coding: utf-8
"""
---------------------------------------------------------------------------------------------------
                                         ABOUT
@author         : Ramodgwendé Weizmann KIENDREBEOGO
@email          : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
@repo           : https://github.com/weizmannk/cbc-population-distributions.git
@createdOn      : February 2026
@description    : Visualises the compact binary coalescence (CBC) mass distribution
                  inferred from GWTC-4 using the FullPop-4.0 hierarchical Bayesian
                  model.  Three figures are produced:
                    1. 1D mass distribution - FullPop-4.0 vs. the GWTC-3
                       Power-Law + Dip + Break model.
                    2. 1D marginal distributions of the primary (m1) and
                       secondary (m2) masses compared to posterior-predictive
                       samples drawn from the population model.
                    3. 2D joint mass distribution p(m1, m2 | lambda) weighted by
                       m1 x m2 to highlight the GW-detectable region.
---------------------------------------------------------------------------------------------------
"""

# ============================================================================
# Standard-library imports
# ============================================================================
import gc
import logging
import os
import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.table import Table
from gwpopulation.utils import truncnorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from popsummary.popresult import PopulationResult
from scipy.integrate import simpson

# ============================================================================
# Logging / warning configuration
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="No LaTeX-compatible font")
logging.getLogger("matplotlib").setLevel(logging.ERROR)

# ============================================================================
# Matplotlib style
# ============================================================================
USE_LATEX = True if shutil.which("latex") is not None else False

# sns.set_style("whitegrid")

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "text.usetex": USE_LATEX,
        "font.size": 12,
        "legend.fontsize": 14,
        "axes.labelsize": 12,
        "axes.titlesize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "savefig.dpi": 300,
    }
)

# ============================================================================
# File-path constants
# ============================================================================
DATA_DIR: str = "./"

#: HDF5 file produced by popsummary containing the hierarchical posterior
#: samples for the FullPop-4.0 run on the full GWTC-4 CBC catalogue.
HYPER_FILE: str = os.path.join(DATA_DIR, "AllCBC_FullPop.h5")

#: HDF5 file (astropy Table) with posterior-predictive mass samples
#: (columns ``mass1``, ``mass2``) drawn from the FullPop-4.0 population.
CBC_MASS_DISTRIBUTION_FILE: str = os.path.join(DATA_DIR, "fullpop4.h5")

#: Output figure filenames (written in both PNG and PDF).
OUT_1D_MODELS: str = "1D_Population_models"
OUT_1D_MARGINALS: str = "FullPop-4_mass_distributions_vs_model"
OUT_2D_JOINT: str = "2D_FullPop_mass_distribution"

# ============================================================================
# Grid / sampling constants
# ============================================================================
#: Number of points for the 1D mass axis used in the single-model plot.
N_MASS_1D: int = 1_000_000

#: Number of points per axis for the 2D joint-distribution grid.
#: Increase for publication quality; decrease for faster development runs.
N_MASS_2D: int = 1_000

#: Log-spaced mass range [M_sun] used for all theoretical curves.
MASS_MIN: float = 1.0  # M_sun - hard lower boundary (injection bound)
MASS_MAX: float = 100.0  # M_sun - hard upper boundary

#: Number of histogram bins (log-spaced) for the 1D marginal plots.
N_HIST_BINS: int = 100

#: Percentile of the 2D weighted distribution used as the colour-scale
#: upper limit (guards against single-pixel spikes swamping the palette).
VMAX_PERCENTILE: float = 99.0

# ============================================================================
# GWTC-3 Power-Law + Dip + Break  -  fixed parameters (Table 2, Abbott+2023)
# ============================================================================
#: Spectral index below the gap.
PDB_ALPHA_1: float = -2.16
#: Spectral index above the gap.
PDB_ALPHA_2: float = -1.46
#: Gap depth (0 = no gap, 1 = complete gap).
PDB_A: float = 0.97
#: Lower edge of the mass gap [M_sun].
PDB_M_GAP_LO: float = 2.72
#: Upper edge of the mass gap [M_sun].
PDB_M_GAP_HI: float = 6.13
#: Sharpness of the lower gap edge.
PDB_ETA_GAP_LO: float = 50.0
#: Sharpness of the upper gap edge.
PDB_ETA_GAP_HI: float = 50.0
#: Sharpness of the low-mass cutoff.
PDB_ETA_MIN: float = 50.0
#: Sharpness of the high-mass cutoff.
PDB_ETA_MAX: float = 4.91
#: Minimum mass of the distribution [M_sun].
PDB_M_MIN: float = 1.16
#: Maximum mass of the distribution [M_sun].
PDB_M_MAX: float = 54.38

# ============================================================================
# Plot colour palette
# ============================================================================
COLORS: dict = {
    "m1": "#E74C3C",  # coral  - primary mass
    "m2": "#1ABC9C",  # teal   - secondary mass
    "model": "#2C3E50",  # slate  - theoretical curves
}


# ============================================================================
# Model functions
# ============================================================================


def _lopass(m: np.ndarray, m_crit: float, eta: float) -> np.ndarray:
    """Sigmoid low-pass filter - suppresses masses *above* ``m_crit``.

    Implements the building block shared by the tapering function ``l(m)``
    and both notch filters (see GWTC-4, Eq. B7):

    .. math::

        \\ell(m) = \\frac{1}{1 + \\left(\\dfrac{m}{m_{\\rm crit}}\\right)^{\\eta}}

    The transition is centred at ``m_crit`` (where the filter equals 0.5)
    and becomes a hard step function as ``eta --> infinity``.

    Parameters
    ----------
    m : np.ndarray
        Mass array [M_sun].
    m_crit : float
        Critical mass at which the filter equals 0.5 [M_sun].
        For ``l(m)`` this is ``m_max,BH``; for a notch upper edge it is
        ``m_min,BH`` or ``γ_high,2``.
    eta : float
        Steepness parameter (``n_5`` for ``l(m)``; ``n_2`` or ``n_4`` for
        notch upper edges).  Larger values give a sharper transition.

    Returns
    -------
    np.ndarray
        Filter values in the open interval (0, 1).
    """
    return 1.0 / (1.0 + (m / m_crit) ** eta)


def _hipass(m: np.ndarray, m_crit: float, eta: float) -> np.ndarray:
    """Sigmoid high-pass filter - suppresses masses *below* ``m_crit``.

    Implements the complement of :func:`_lopass`, used as the tapering
    function ``h(m)`` and as the lower-edge filter inside each notch
    (GWTC-4, Eq. B7):

    .. math::

        h(m) = 1 - \\frac{1}{1 + \\left(\\dfrac{m_{\\rm crit}}{m}\\right)^{\\eta}}
             = \\frac{1}{1 + \\left(\\dfrac{m_{\\rm crit}}{m}\\right)^{\\eta}}

    For ``h(m)`` (global low-mass taper), ``m_crit = m_min,NS`` and
    ``eta = n_i``.  For notch lower edges, ``m_crit`` is ``m_max,NS`` or
    ``γ_low,2`` and ``eta`` is ``η_1`` or ``η_3`` respectively.

    Parameters
    ----------
    m : np.ndarray
        Mass array [M_sun].
    m_crit : float
        Critical mass at which the filter equals 0.5 [M_sun].
    eta : float
        Steepness parameter.

    Returns
    -------
    np.ndarray
        Filter values in the open interval (0, 1).
    """
    return 1.0 - _lopass(m, m_crit, eta)


def _notch(
    m: np.ndarray,
    m_lo: float,
    m_hi: float,
    eta_lo: float,
    eta_hi: float,
    amp: float,
) -> np.ndarray:
    """Band-stop (notch) filter that suppresses a mass-gap region.

    Two notch functions appear in the FullPop-4.0 model (GWTC-4, Eq. B7):

    .. math::

        n_i(m) = 1 - \\frac{A_i}{
            \\left[1 + \\left(\\dfrac{m_{\\rm lo}}{m}\\right)^{\\eta_{\\rm lo}}\\right]
            \\left[1 + \\left(\\dfrac{m}{m_{\\rm hi}}\\right)^{\\eta_{\\rm hi}}\\right]}

    * **Notch 1** (NS–BH mass gap):
      ``m_lo = m_max,NS``, ``m_hi = m_min,BH``, depth ``A``,
      sharpness parameters ``η1`` (lower edge) and ``n2`` (upper edge).
    * **Notch 2** (pair-instability gap):
      ``m_lo = γ_low,2``, ``m_hi = γ_high,2``, depth ``A2``,
      sharpness parameters ``n3`` and ``n4``.

    The filter equals ``1 - amp`` deep inside the gap and approaches 1
    far outside it.  Setting ``amp = 1`` produces a complete suppression;
    ``amp = 0`` leaves the distribution untouched.

    Parameters
    ----------
    m : np.ndarray
        Mass array [M_sun].
    m_lo : float
        Lower edge of the notch [M_sun]  (``m_max,NS`` or ``γ_low,2``).
    m_hi : float
        Upper edge of the notch [M_sun]  (``m_min,BH`` or ``γ_high,2``).
    eta_lo : float
        Sharpness of the lower edge (``n1`` or ``n3``).
    eta_hi : float
        Sharpness of the upper edge (``n2`` or ``n4``).
    amp : float
        Fractional depth of the notch ``A or A2 ∈ [0, 1]``.  ``amp = 1``
        completely removes the mass-gap population.

    Returns
    -------
    np.ndarray
        Multiplier ≈ 1 outside the gap and ≈ ``1 - amp`` inside.
    """
    return 1.0 - amp * _hipass(m, m_lo, eta_lo) * _lopass(m, m_hi, eta_hi)


def power_law_dip_break(m: np.ndarray) -> np.ndarray:
    """GWTC-3 Power-Law + Dip + Break 1D mass distribution (un-normalised).

    This is the *predecessor* model from Abbott et al. (2023, GWTC-3),
    arXiv:2111.03634.  It is plotted alongside FullPop-4.0 in Figure 1
    to illustrate how the population inference evolved between the two
    catalogue releases.

    The model uses the same sigmoid filter algebra as FullPop-4.0 but
    with a single notch (the NS–BH mass gap) and no Gaussian subpopulation
    peaks.  All hyperparameter values are fixed to the GWTC-3 MAP
    estimates (module-level constants ``PDB_*``):

    .. math::

        p_{\\rm PDB}(m) \\propto
        \\underbrace{\\left[1 - A\\,
            h(m,\\,m_{\\rm gap,lo},\\,\\eta_{\\rm gap,lo})\\,
            l(m,\\,m_{\\rm gap,hi},\\,\\eta_{\\rm gap,hi})
        \\right]}_{\\text{bandpass (dip)}}
        \\times\\;
        h(m,\\,m_{\\rm min},\\,\\eta_{\\rm min})
        \\times\\;
        l(m,\\,m_{\\rm max},\\,\\eta_{\\rm max})
        \\times\\;
        \\left(\\frac{m}{m_{\\rm gap,hi}}\\right)^{\\alpha}

    where the spectral index switches from ``alpha_1`` below ``m_gap,hi`` to
    ``α₂`` above it (the "break").

    Fixed hyperparameters (GWTC-3 MAP)
    ------------------------------------
    ==============  =========  ==============================================
    Constant        Value      Description
    ==============  =========  ==============================================
    PDB_ALPHA_1     −2.16      Spectral index below the break.
    PDB_ALPHA_2     −1.46      Spectral index above the break.
    PDB_A            0.97      Gap depth (1 = complete suppression).
    PDB_M_GAP_LO     2.72      Lower gap edge [M_sun].
    PDB_M_GAP_HI     6.13      Upper gap edge / break mass [M_sun].
    PDB_ETA_GAP_LO  50.0       Sharpness of the lower gap edge.
    PDB_ETA_GAP_HI  50.0       Sharpness of the upper gap edge.
    PDB_ETA_MIN     50.0       Sharpness of the low-mass cutoff.
    PDB_ETA_MAX      4.91      Sharpness of the high-mass cutoff.
    PDB_M_MIN        1.16      Minimum mass [M_sun].
    PDB_M_MAX       54.38      Maximum mass [M_sun].
    ==============  =========  ==============================================

    Parameters
    ----------
    m : np.ndarray
        1-D mass array [M_sun] at which to evaluate the model.

    Returns
    -------
    np.ndarray
        Un-normalised probability density evaluated at *m*.

    References
    ----------
    LIGO-Virgo-KAGRA Collaboration, GWTC-3 Population Paper,
    arXiv:2111.03634, Table 2.

    Examples
    --------
    >>> import numpy as np
    >>> m = np.geomspace(1, 100, 500)
    >>> pdf = power_law_dip_break(m)
    >>> pdf.shape
    (500,)
    >>> float(pdf[pdf < 0].size)  # non-negative everywhere
    0.0
    """

    def _bandpass(m, m_lo, m_hi, eta_lo, eta_hi, A):
        return 1.0 - A * _hipass(m, m_lo, eta_lo) * _lopass(m, m_hi, eta_hi)

    return (
        _bandpass(m, PDB_M_GAP_LO, PDB_M_GAP_HI, PDB_ETA_GAP_LO, PDB_ETA_GAP_HI, PDB_A)
        * _hipass(m, PDB_M_MIN, PDB_ETA_MIN)
        * _lopass(m, PDB_M_MAX, PDB_ETA_MAX)
        * (m / PDB_M_GAP_HI) ** np.where(m < PDB_M_GAP_HI, PDB_ALPHA_1, PDB_ALPHA_2)
    )


def fullpop(m: np.ndarray, hyper: pd.Series) -> np.ndarray:
    """GWTC-4 FullPop-4.0 1D mass distribution (un-normalised).

    Implements Equation (B7) of the GWTC-4 population paper
    (arXiv:2508.18083).  The full model reads:

    .. math::

        p(m | \\lambda) \\propto
        \\Bigl[1 + \\sum_{i=1}^{2} c_i\\, G_i(m)\\Bigr]
        \\times S(m)
        \\times \\prod_{i=1}^{2} n_i(m)
        \\times h(m) \\times l(m)

    **Gaussian peaks** ``G_i(m)``
        Two truncated Gaussians
        :math:`\\mathcal{N}(\\mu_{\\mathrm{peak},i},\\,\\sigma_{\\mathrm{peak},i})`
        truncated over the *fixed* injection domain
        :math:`[m_{\\rm inj,min},\\, m_{\\rm inj,max}] = [1,\\,500]\\,M_\\odot`.
        These bounds are set by the injection campaign and are **not** fitted.
        The mixing fractions ``c_i`` control their relative contribution.

    **Broken power-law** ``S(m)``

    .. math::

        S(m) = \\begin{cases}
            m^{\\alpha_1}                        & m < m_{\\rm max,NS} \\\\
            K_1\\, m^{\\alpha_{\\rm dip}}         & m_{\\rm max,NS} \\le m < m_{\\rm min,BH} \\\\
            K_2\\, m^{\\alpha_2}                  & m \\ge m_{\\rm min,BH}
        \\end{cases}

    with continuity constants
    :math:`K_1 = m_{\\rm max,NS}^{\\alpha_1 - \\alpha_{\\rm dip}}` and
    :math:`K_2 = K_1 \\times m_{\\rm min,BH}^{\\alpha_{\\rm dip} - \\alpha_2}`.

    **Notch filters** ``n₁(m)``, ``n₂(m)``
        Suppress the two mass gaps (NS–BH gap and pair-instability gap);
        see :func:`_notch` for the analytic form.

    **Global tapering filters** ``h(m)``, ``l(m)``
        Smooth low- and high-mass cutoffs implemented via :func:`_hipass`
        and :func:`_lopass` respectively:

        .. math::

            h(m) = 1 + \\left(\\frac{m_{\\rm min,NS}}{m}\\right)^{\\eta_0}, \\qquad
            l(m) = 1 + \\left(\\frac{m}{m_{\\rm max,BH}}\\right)^{\\eta_5}

    Parameters
    ----------
    m : np.ndarray
        1-D mass array [M_sun] at which to evaluate the model.
    hyper : pd.Series
        MAP (or any posterior draw) of the hyperparameter vector.
        Required keys:

        ================  =====================================================
        Key               Physical meaning
        ================  =====================================================
        ``mu1``, ``sig1`` Mean and std of Gaussian peak 1 [M_sun].
        ``mu2``, ``sig2`` Mean and std of Gaussian peak 2 [M_sun].
        ``mix1``,``mix2`` Mixing fractions ``c_1``, ``c_2``.
        ``NSmin``         :math:`m_{\\rm min,NS}` — global low-mass cutoff.
        ``NSmax``         :math:`m_{\\rm max,NS}` — upper NS boundary.
        ``BHmin``         :math:`m_{\\rm min,BH}` — lower BH boundary.
        ``BHmax``         :math:`m_{\\rm max,BH}` — global high-mass cutoff.
        ``UPPERmin``      :math:`\\gamma_{\\rm low,2}` — lower PI-gap edge.
        ``UPPERmax``      :math:`\\gamma_{\\rm high,2}` — upper PI-gap edge.
        ``n0`` ... ``n5``   Sharpness parameters ``n1 ... n5``.
        ``A``, ``A2``     Notch depths for gap 1 and gap 2.
        ``alpha_1``       Power-law index in the NS regime.
        ``alpha_dip``     Power-law index inside the mass gap.
        ``alpha_2``       Power-law index in the BH regime.
        ================  =====================================================

    Returns
    -------
    np.ndarray
        Un-normalised probability density p(m | λ) evaluated at *m*.
        Normalise with ``∫ p dm = 1`` before use (see :func:`main`).

    References
    ----------
    LIGO-Virgo-KAGRA Collaboration, GWTC-4 Population Paper,
    arXiv:2508.18083, Eq. (B7).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> m = np.geomspace(1, 100, 1000)
    >>> pdf = fullpop(m, hyperparams)   # hyperparams loaded from HDF5
    >>> assert (pdf >= 0).all()
    """
    # Hard injection bounds - fixed by the injection campaign, NOT inferred.
    # They define the support of the truncated Gaussians G_i(m).
    inj_mmin: float = 1.0  # m_inj,min  [M_sun]
    inj_mmax: float = 500.0  # m_inj,max  [M_sun]

    # Truncated Gaussian subpopulation peaks
    peak1 = truncnorm(m, hyper.mu1, hyper.sig1, inj_mmax, inj_mmin)
    peak2 = truncnorm(m, hyper.mu2, hyper.sig2, inj_mmax, inj_mmin)

    # Global mass-window filters
    hi = _hipass(m, hyper.NSmin, hyper.n0)
    lo = _lopass(m, hyper.BHmax, hyper.n5)

    # Notch 1: NS-BH mass gap
    notch1 = _notch(m, hyper.NSmax, hyper.BHmin, hyper.n1, hyper.n2, hyper.A)
    # Notch 2: pair-instability gap
    notch2 = _notch(m, hyper.UPPERmin, hyper.UPPERmax, hyper.n3, hyper.n4, hyper.A2)

    # Three-segment broken power law
    powerlaw = np.piecewise(
        m,
        (
            m < hyper.NSmax,
            (m >= hyper.NSmax) & (m < hyper.BHmin),
        ),
        (
            lambda m: m**hyper.alpha_1,
            lambda m: (
                m**hyper.alpha_dip * hyper.NSmax ** (hyper.alpha_1 - hyper.alpha_dip)
            ),
            lambda m: (
                m**hyper.alpha_2
                * hyper.NSmax ** (hyper.alpha_1 - hyper.alpha_dip)
                * hyper.BHmin ** (hyper.alpha_dip - hyper.alpha_2)
            ),
        ),
    )

    return (
        (1.0 + hyper.mix1 * peak1 + hyper.mix2 * peak2)
        * notch1
        * notch2
        * hi
        * lo
        * powerlaw
    )


# ============================================================================
# Helper utilities
# ============================================================================


def normalise_log(m: np.ndarray, pdf: np.ndarray) -> np.ndarray:
    """Normalise *pdf* so that integral of p(m) d(ln(m)) = 1.

    This is the natural norm when the mass axis is log-spaced and one
    wishes to plot ``m × p(m)`` vs. ``ln(m)``.

    Parameters
    ----------
    m : np.ndarray
        Log-spaced mass array [M_sun].
    pdf : np.ndarray
        Un-normalised probability density (same shape as *m*).

    Returns
    -------
    np.ndarray
        Normalised probability density.
    """
    norm = np.trapezoid(m * pdf, np.log(m))
    return pdf / norm


def normalise(m: np.ndarray, pdf: np.ndarray) -> np.ndarray:
    """Normalise *pdf* so that integrate p(m) dm = 1.

    Parameters
    ----------
    m : np.ndarray
        Mass array [M_sun].
    pdf : np.ndarray
        Un-normalised probability density (same shape as *m*).

    Returns
    -------
    np.ndarray
        Normalised probability density.
    """
    norm = np.trapezoid(pdf, m)
    return pdf / norm


def save_figure(fig: plt.Figure, basename: str, dpi: int = 300) -> None:
    """Save *fig* as both PNG and PDF.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    basename : str
        Output filename **without** extension (e.g. ``"1D_models"``).
    dpi : int, optional
        Raster resolution for the PNG output (default: 300).
    """
    for ext in ("png", "pdf"):
        path = f"{basename}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved ==> %s", path)


# ============================================================================
# Figure 1 - 1D model comparison (FullPop-4.0 vs. Power-Law + Dip + Break)
# ============================================================================


def plot_1d_model_comparison(
    m: np.ndarray,
    model_fullpop: np.ndarray,
    model_pdb: np.ndarray,
    boundary_masses: list[float],
    boundary_labels: list[str],
) -> plt.Figure:
    """Plot FullPop-4.0 vs. Power-Law+Dip+Break 1D mass distributions.

    Both curves are normalised so that ∫ m p(m) d(ln m) = 1 and plotted
    as ``m p(m)`` on a log–log scale.  Vertical lines mark the FullPop-4.0
    model boundaries, which are also annotated on a secondary *x*-axis.

    Parameters
    ----------
    m : np.ndarray
        Log-spaced mass array [M_sun].
    model_fullpop : np.ndarray
        Un-normalised FullPop-4.0 PDF evaluated on *m*.
    model_pdb : np.ndarray
        Un-normalised Power-Law+Dip+Break PDF evaluated on *m*.
    boundary_masses : list of float
        *x*-positions of the vertical boundary lines [M_sun].
    boundary_labels : list of str
        LaTeX labels for each boundary (same order as *boundary_masses*).

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure.
    """
    fp_normed = normalise_log(m, model_fullpop)
    pdb_normed = normalise_log(m, model_pdb)

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.plot(
        m,
        m * fp_normed,
        color="#9400D3",
        linewidth=2.0,
        linestyle="-",
        label="GWTC-4: FullPop-4.0",
    )
    ax.plot(
        m,
        m * pdb_normed,
        color="#555555",
        linewidth=2.0,
        linestyle="--",
        label="GWTC-3: Power Law + Dip + Break",
    )

    ax.set_xlim(MASS_MIN, MASS_MAX)
    ax.set_ylim(1e-3 * 4, 100)
    ax.set_xlabel(r"Mass, $m\;[M_\odot]$", fontsize=16)
    ax.set_ylabel(r"$m\,p(m|\lambda)$", fontsize=16)
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=1.0,
        edgecolor="lightgray",
        fancybox=False,
    )
    ax.tick_params(which="both", direction="in", top=False, right=True)

    # Secondary x-axis with model boundary annotations
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xscale("log")
    ax2.set_xticks(boundary_masses)
    ax2.set_xticklabels(boundary_labels, fontsize=16)
    ax2.tick_params(which="both", direction="in")

    for xv in boundary_masses:
        ax.axvline(x=xv, color="gray", linewidth=0.7, alpha=0.5, zorder=0)

    fig.tight_layout(pad=0.4)
    plt.subplots_adjust(top=0.87)
    return fig


# ============================================================================
# Figure 2 - 1D marginal distributions compared to posterior-predictive samples
# ============================================================================


def plot_1d_marginals(
    m: np.ndarray,
    m1_samples: np.ndarray,
    m2_samples: np.ndarray,
    p_m1_marginal: np.ndarray,
    p_m2_marginal: np.ndarray,
) -> plt.Figure:
    """Compare FullPop-4.0 marginals against posterior-predictive samples.

    Two side-by-side panels show the histogram of *m1_samples* /
    *m2_samples* against the corresponding normalised 1-D marginal of
    the analytic 2D distribution.

    Parameters
    ----------
    m : np.ndarray
        Log-spaced mass array [M_sun] used for the analytic curves.
    m1_samples : np.ndarray
        Posterior-predictive primary-mass samples [M_sun].
    m2_samples : np.ndarray
        Posterior-predictive secondary-mass samples [M_sun].
    p_m1_marginal : np.ndarray
        Un-normalised marginal the integral of  p(m1, m2) dm2 evaluated on *m*.
    p_m2_marginal : np.ndarray
        Un-normalised marginal  the integral of p(m1, m2) dm1 evaluated on *m*.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure.
    """
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "legend.fontsize": 11,
        }
    )

    # Shared log-spaced histogram bins
    lo = max(MASS_MIN, min(m1_samples.min(), m2_samples.min(), m.min()))
    hi = min(MASS_MAX, max(m1_samples.max(), m2_samples.max(), m.max()))
    bins = np.geomspace(lo, hi, N_HIST_BINS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    panels = [
        (axes[0], m1_samples, p_m1_marginal.copy(), r"$m_1$", "m1"),
        (axes[1], m2_samples, p_m2_marginal.copy(), r"$m_2$", "m2"),
    ]

    for ax, samples, marginal, label, color_key in panels:
        color = COLORS[color_key]

        # --- Normalise: integral p(m) dm = 1  [M_sun^-1] ---
        marginal_normed = normalise(m, marginal)

        # Sanity check - must equal 1.0
        integral = np.trapezoid(marginal_normed, m)
        logger.info(f"Integral of p({label}) dm = {integral} (expected 1.0)")

        # --- Left axis: probability DENSITY p(m)  [M_sun^-1] ---
        sns.histplot(
            samples,
            bins=bins,
            stat="density",
            color=color,
            alpha=0.8,
            edgecolor="white",
            linewidth=1.2,
            ax=ax,
            label="GWTC-4: FullPop-4.0 samples",
        )
        ax.plot(
            m,
            marginal_normed,
            color=COLORS["model"],
            lw=3.5,
            label="Model PDF",
            zorder=10,
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(rf"{label} [$M_\odot$]", fontsize=15, fontweight="600")
        ax.grid(
            True,
            which="major",
            alpha=0.3,
            linestyle="-",
            linewidth=1.2,
            color="#D5DBDB",
        )
        ax.grid(
            True,
            which="minor",
            alpha=0.15,
            linestyle=":",
            linewidth=0.8,
            color="#E5E7E9",
        )

        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

        legend = ax.legend(
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=12,
            loc="upper right",
            edgecolor=color,
            facecolor="white",
            framealpha=0.95,
        )
        legend.get_frame().set_linewidth(2)

    axes[0].set_ylabel(
        r"Probability Density $p(m)$  $[M_\odot^{-1}]$", fontsize=15, fontweight="600"
    )

    fig.suptitle(
        "FullPop-4.0 Mass Distributions", fontsize=18, fontweight="bold", y=0.98
    )
    plt.tight_layout()
    return fig


# ============================================================================
# Figure 3 - 2D joint mass distribution
# ============================================================================


def plot_2d_joint(
    m: np.ndarray,
    weighted_dist: np.ndarray,
) -> plt.Figure:
    """Visualise the 2D FullPop-4.0 joint mass distribution.

    The colour map shows ``m1 × m2 × p(m1, m2 | Lambda)`` on a log-log grid.
    The unphysical region ``m2 > m1`` is masked in white; the equal-mass
    line ``m1 = m2`` is drawn as a dashed white line.

    Parameters
    ----------
    m : np.ndarray
        Log-spaced 1-D mass array [M_sun] (both axes share this grid).
    weighted_dist : np.ndarray
        2-D array of shape ``(len(m), len(m))`` containing
        ``m1 × m2 × p(m1, m2 | Lambda)``.

    Returns
    -------
    matplotlib.figure.Figure
        The completed figure.
    """
    fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(aspect=1))
    ax.set_xlim(MASS_MIN, MASS_MAX)
    ax.set_ylim(MASS_MIN, MASS_MAX)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_1$ [$M_\odot$]", fontsize=15, fontweight="600")
    ax.set_ylabel(r"$m_2$ [$M_\odot$]", fontsize=15, fontweight="600")

    vmax = np.percentile(weighted_dist[weighted_dist > 0], VMAX_PERCENTILE)

    img = ax.pcolormesh(
        m,
        m,
        weighted_dist,
        vmin=0,
        vmax=vmax,
        shading="gouraud",
        rasterized=True,
        cmap="magma",
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(img, cax=cax)
    cbar.set_label(
        r"$m_1 \times m_2 \times p(m_1, m_2 \mid \Lambda)$",
        fontsize=15,
        fontweight="600",
    )
    cbar.set_ticks([])

    # Mask unphysical region (m2 > m1)
    ax.fill_between(
        [MASS_MIN, MASS_MAX],
        [MASS_MIN, MASS_MAX],
        [MASS_MAX, MASS_MAX],
        color="white",
        linewidth=0,
        alpha=0.75,
        zorder=10,
    )

    # Equal-mass line
    ax.plot([MASS_MIN, MASS_MAX], [MASS_MIN, MASS_MAX], "--w", linewidth=2, zorder=11)

    fig.suptitle(r"2D mass distributions", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(pad=0.1)
    return fig


# ============================================================================
# Main execution
# ============================================================================


def main() -> None:
    """Load data, compute models, and produce all three figures.

    Workflow
    --------
    1. Load the FullPop-4.0 hyperparameter posterior from *HYPER_FILE* and
       identify the MAP (maximum a posteriori) sample.
    2. Evaluate the 1D FullPop-4.0 and Power-Law+Dip+Break models on a
       log-spaced mass grid and save **Figure 1**.
    3. Load posterior-predictive CBC mass samples from
       *CBC_MASS_DISTRIBUTION_FILE* and compute the 2D joint distribution
       ``p(m1, m2 | Lambda)`` including mass-ratio-dependent pairing.
    4. Marginalise the 2D distribution to obtain 1-D PDFs for m1 and m2,
       then save **Figure 2**.
    5. Weight the 2D distribution by ``m1 × m2`` and save **Figure 3**.
    """

    # ------------------------------------------------------------------
    # 1.  Load MAP hyperparameters
    # ------------------------------------------------------------------
    logger.info("Loading hyperparameter posterior from: %s", HYPER_FILE)
    popresult = PopulationResult(HYPER_FILE)

    df = pd.DataFrame(
        popresult.get_hyperparameter_samples(),
        columns=popresult.get_metadata("hyperparameters"),
    )
    hyperparams = df.iloc[(df.log_likelihood + df.log_prior).idxmax()]
    logger.info(
        "MAP hyperparameters loaded: %s ...", ", ".join(list(hyperparams.keys())[:5])
    )

    # ------------------------------------------------------------------
    # 2.  Figure 1 - 1D model comparison
    # ------------------------------------------------------------------
    logger.info("Computing 1D mass distributions (N = %d points) ...", N_MASS_1D)
    m_1d = np.geomspace(MASS_MIN, MASS_MAX, N_MASS_1D)

    boundary_masses = [
        hyperparams["NSmin"],
        hyperparams["NSmax"],
        hyperparams["BHmin"],
        hyperparams["UPPERmin"],
        hyperparams["UPPERmax"],
    ]
    boundary_labels = [
        r"$M_{\min}$",
        r"$\gamma_{\mathrm{low,1}}$",
        r"$\gamma_{\mathrm{high,1}}$",
        r"$\gamma_{\mathrm{low,2}}$",
        r"$\gamma_{\mathrm{high,2}}$",
    ]

    fig1 = plot_1d_model_comparison(
        m_1d,
        fullpop(m_1d, hyperparams),
        power_law_dip_break(m_1d),
        boundary_masses,
        boundary_labels,
    )
    save_figure(fig1, OUT_1D_MODELS)
    plt.close(fig1)
    gc.collect()

    # ------------------------------------------------------------------
    # 3.  Load posterior-predictive samples
    # ------------------------------------------------------------------
    logger.info("Loading CBC samples from: %s", CBC_MASS_DISTRIBUTION_FILE)
    table = Table.read(CBC_MASS_DISTRIBUTION_FILE)
    m1_samples = np.asarray(table["mass1"])
    m2_samples = np.asarray(table["mass2"])
    logger.info(
        "Loaded %d CBC samples  |  m1 ∈ [%.2f, %.2f]  |  m2 ∈ [%.2f, %.2f]",
        len(m1_samples),
        m1_samples.min(),
        m1_samples.max(),
        m2_samples.min(),
        m2_samples.max(),
    )

    # ------------------------------------------------------------------
    # 4.  Compute 2D joint distribution and marginals
    # ------------------------------------------------------------------
    logger.info("Building 2D mass grid (%d × %d) ...", N_MASS_2D, N_MASS_2D)
    m_2d = np.geomspace(MASS_MIN, MASS_MAX, N_MASS_2D)

    model_2d = fullpop(m_2d, hyperparams)
    pdf_1d = model_2d / simpson(model_2d, m_2d)  # integral of p dm = 1

    m1_grid, m2_grid = np.meshgrid(m_2d, m_2d)

    beta_pair_1 = hyperparams["beta_pair_1"]
    beta_pair_2 = hyperparams["beta_pair_2"]
    mbreak = hyperparams["mbreak"]

    # Joint PDF with mass-ratio-dependent pairing
    q = m2_grid / m1_grid  # mass ratio, q ∈ (0, 1] since m2 <= m1
    p_m1m2 = np.outer(pdf_1d, pdf_1d) * (
        (m2_grid < mbreak) * q**beta_pair_1 + (m2_grid >= mbreak) * q**beta_pair_2
    )
    # Here we didn't apply the extreme mass ratio cut.
    # Enforce m1 >= m2. To also cut extreme mass ratios (m1>60, m2<3), use:
    # p_m1m2 *= (m1_grid >= m2_grid) &  np.logical_not((m1_grid > 60) & (m2_grid < 3))
    p_m1m2 *= m1_grid >= m2_grid  # enforce m1 >= m2

    # Marginalize to get 1D distribution
    p_m1_marginal = np.trapezoid(p_m1m2, m_2d, axis=0)  # Integrate over m2
    p_m2_marginal = np.trapezoid(p_m1m2, m_2d, axis=1)  # Integrate over m1

    # ------------------------------------------------------------------
    # 5.  Figure 2 - 1D marginals vs. samples
    # ------------------------------------------------------------------
    fig2 = plot_1d_marginals(
        m_2d,
        m1_samples,
        m2_samples,
        p_m1_marginal,
        p_m2_marginal,
    )
    save_figure(fig2, OUT_1D_MARGINALS)
    plt.close(fig2)
    gc.collect()

    # ------------------------------------------------------------------
    # 6.  Figure 3 - 2D joint distribution
    # ------------------------------------------------------------------
    weighted_dist = m1_grid * m2_grid * p_m1m2
    fig3 = plot_2d_joint(m_2d, weighted_dist)
    save_figure(fig3, OUT_2D_JOINT)
    plt.close(fig3)
    gc.collect()

    logger.info("All figures saved successfully.")


if __name__ == "__main__":
    main()
