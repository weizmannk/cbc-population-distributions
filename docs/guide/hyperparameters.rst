.. _hyperparams:

======================================
Population Model (GWTC-4): FullPop-4.0
======================================


.. dropdown:: Primary Mass Distribution

    The figure below shows the one-dimensional FullPop-4.0 mass distribution
    :math:`p(m|\lambda)` for the primary black hole mass in the range
    :math:`[1, 100]\,M_\odot`. The model combines a broken power law with two Gaussian peaks
    and includes two characteristic features: the neutron star–black hole mass gap
    (between :math:`M^\mathrm{gap}_\mathrm{low}` and :math:`M^\mathrm{gap}_\mathrm{high}`),
    and the pair-instability gap (between :math:`M_\mathrm{PI,low}` and :math:`M_\mathrm{PI,high}`).

    .. math::
        p(m|\lambda) = n(m|\gamma_{\text{low}}, \gamma_{\text{high}}, A) \times
            l(m|m_{\text{max}}, \eta) \\
                    \times \begin{cases}
                            & m^{\alpha_1} \text{ if } m < \gamma_{\text{low}} \\
                            & m^{\alpha_2} \text{ if } m > \gamma_{\text{low}} \\
                            & 0 \text{ otherwise }
                    \end{cases}.

    where :math:`l(m \mid m_{\text{max}}, \eta)` is the low-pass filter with power-law
    :math:`\eta` applied at mass :math:`m_{\text{max}}`,
    :math:`n(m \mid \gamma_{\text{low}}, \gamma_{\text{high}}, A)` is the notch filter with
    depth :math:`A` applied between :math:`\gamma_{\text{low}}` and :math:`\gamma_{\text{high}}`,
    and :math:`\lambda` is the subset of hyperparameters
    :math:`\{\gamma_{\text{low}}, \gamma_{\text{high}}, A, \alpha_1, \alpha_2,
    m_{\min}, m_{\text{max}}\}`.

    .. plot::
        :caption: Population model for the primary mass distribution.
        :include-source: False

        import numpy as np
        import matplotlib.pyplot as plt
        from gwpopulation.utils import truncnorm, xp

        # =========================================================
        # GWTC-3  Power-Law + Dip + Break  (Abbott et al. 2023)
        # =========================================================
        PDB_ALPHA_1    = -2.16
        PDB_ALPHA_2    = -1.46
        PDB_A          =  0.97
        PDB_M_GAP_LO   =  2.72
        PDB_M_GAP_HI   =  6.13
        PDB_ETA_GAP_LO = 50.0
        PDB_ETA_GAP_HI = 50.0
        PDB_ETA_MIN    = 50.0
        PDB_ETA_MAX    =  4.91
        PDB_M_MIN      =  1.16
        PDB_M_MAX      = 54.38

        def _lopass(m, m_crit, eta):
            return 1.0 / (1.0 + (m / m_crit) ** eta)

        def _hipass(m, m_crit, eta):
            return 1.0 - _lopass(m, m_crit, eta)

        def power_law_dip_break(m):
            bandpass = (
                1.0 - PDB_A
                * _hipass(m, PDB_M_GAP_LO, PDB_ETA_GAP_LO)
                * _lopass(m, PDB_M_GAP_HI, PDB_ETA_GAP_HI)
            )
            return (
                bandpass
                * _hipass(m, PDB_M_MIN, PDB_ETA_MIN)
                * _lopass(m, PDB_M_MAX, PDB_ETA_MAX)
                * (m / PDB_M_GAP_HI) ** np.where(m < PDB_M_GAP_HI, PDB_ALPHA_1, PDB_ALPHA_2)
            )

        # =========================================================
        # GWTC-4  FullPop-4.0 (https://arxiv.org/abs/2508.18083)
        # =========================================================
        A          = 0.091462
        A2         = 0.828165
        BH_MAX     = 152.055979
        BH_MIN     = 7.763955
        NS_MAX     = 4.094744
        NS_MIN     = 1.176367
        UPPER_MAX  = 66.576705
        UPPER_MIN  = 38.277415
        ALPHA_1    = -4.509283
        ALPHA_2    = -0.902035
        ALPHA_DIP  = -1.679769
        MIX1       = 735.473276
        MIX2       = 211.733327
        MU1        = 37.811196
        MU2        = 8.897742
        SIG1       = 17.126431
        SIG2       = 1.044693
        N0, N1, N2, N3, N4, N5 = 50.0, 50.0, 50.0, 30.0, 30.0, 10.041072
        INJ_MMIN, INJ_MMAX = 1.0, 500.0

        def fullpop(m):
            peak1 = truncnorm(m, MU1, SIG1, low=INJ_MMIN, high=INJ_MMAX)
            peak2 = truncnorm(m, MU2, SIG2, low=INJ_MMIN, high=INJ_MMAX)

            hi     = _hipass(m, NS_MIN,    N0)
            lo     = _lopass(m, BH_MAX,    N5)
            notch1 = 1.0 - A  * _hipass(m, NS_MAX,    N1) * _lopass(m, BH_MIN,    N2)
            notch2 = 1.0 - A2 * _hipass(m, UPPER_MIN, N3) * _lopass(m, UPPER_MAX, N4)

            condlist   = [m < NS_MAX, (m >= NS_MAX) & (m < BH_MIN), m >= BH_MIN]
            choicelist = [
                m**ALPHA_1,
                m**ALPHA_DIP  * NS_MAX**(ALPHA_1  - ALPHA_DIP),
                m**ALPHA_2    * NS_MAX**(ALPHA_1  - ALPHA_DIP) * BH_MIN**(ALPHA_DIP - ALPHA_2),
            ]
            plaw = xp.select(condlist, choicelist, default=0.0)

            return (1.0 + MIX1 * peak1 + MIX2 * peak2) * plaw * notch1 * notch2 * hi * lo

        # =========================================================
        # Grid + normalisation
        # =========================================================
        m = np.geomspace(1, 100, 100_000)

        def normalise_log(m, pdf):
            norm = np.trapezoid(m * pdf, np.log(m))
            return pdf / norm

        fp4  = normalise_log(m, fullpop(m))
        pdb  = normalise_log(m, power_law_dip_break(m))

        # =========================================================
        # Boundary ticks (FullPop-4.0 landmarks)
        # =========================================================
        boundary_masses  = [NS_MIN, NS_MAX, BH_MIN, UPPER_MIN, UPPER_MAX]
        boundary_labels  = [
            r"$M_{\min}$",
            r"$\gamma_{\mathrm{low,1}}$",
            r"$\gamma_{\mathrm{high,1}}$",
            r"$\gamma_{\mathrm{low,2}}$",
            r"$\gamma_{\mathrm{high,2}}$",
        ]

        # =========================================================
        # Figure
        # =========================================================
        fig, ax = plt.subplots()
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.plot(m, m * fp4,
                color="#9400D3", linewidth=2.0, linestyle="-",
                label="GWTC-4: FullPop-4.0")
        ax.plot(m, m * pdb,
                color="#555555", linewidth=2.0, linestyle="--",
                label="GWTC-3: Power Law + Dip + Break")

        ax.set_xlim(1, 100)
        ax.set_ylim(4e-3, 100)
        ax.set_xlabel(r"Mass $m\,[M_\odot]$")
        ax.set_ylabel(r"$m\,p(m|\lambda)$")
        ax.legend(loc="upper right", frameon=True, framealpha=1.0,
                edgecolor="lightgray", fancybox=False)
        ax.tick_params(which="both", direction="in", top=False, right=True)

        # Vertical boundary lines
        for xv in boundary_masses:
            ax.axvline(x=xv, color="gray", linewidth=0.7, alpha=0.5, zorder=0)

        # Secondary x-axis with boundary labels
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xscale("log")
        ax2.set_xticks(boundary_masses)
        ax2.set_xticklabels(boundary_labels)
        ax2.tick_params(which="both", direction="in")

        fig.tight_layout(pad=0.4)
        plt.subplots_adjust(top=0.87)
        fig.show()


.. dropdown:: Hyperparameters Value

    .. tab-set::

        .. tab-item:: Hyperparameters

            .. include:: ./hyperparams_table.rst
