.. _hyperparams:

=========================
Population Model (GWTC-4)
=========================


.. dropdown:: Primary Mass Distribution

    The figure below shows the one-dimensional Broken Power Law + 2 Peaks (BP2P) mass distribution
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

        def GWTC4_broken_powerlaw_peaks(m):
            """
            GWTC-4 mass distribution.
            Broken power-law with two truncated Gaussian peaks and smooth filters.
            """

            # --- GWTC-4 model parameters ---
            A = 0.091462
            A2 = 0.828165
            BH_MAX = 152.055979
            BH_MIN = 7.763955
            NS_MAX = 4.094744
            NS_MIN = 1.176367
            UPPER_MAX = 66.576705
            UPPER_MIN = 38.277415

            ALPHA_1 = -4.509283
            ALPHA_2 = -0.902035
            ALPHA_DIP = -1.679769
            # ALPHA_CHIi=  -0.013141

            MIX1 = 735.473276
            MIX2 = 211.733327
            MU1 = 37.811196
            MU2 = 8.897742
            SIG1 = 17.126431
            SIG2 = 1.044693

            N0 = 50.0
            N1 = 50.0
            N2 = 50.0
            N3 = 30.0
            N4 = 30.0
            N5 = 10.041072

            # amax=        1.000000
            # beta_chi=   -0.942731
            # beta_pair_1= 0.964138
            # beta_pair_2= 2.160036
            # lamb       = 2.406658
            # mbreak=     5.000000

            ABS_MMIN = 0.5
            ABS_MMAX = 350.0

            # --- Truncated Gaussian peaks ---
            gaussian_peak1 = truncnorm(m, MU1, SIG1, low=ABS_MMIN, high=ABS_MMAX)
            gaussian_peak2 = truncnorm(m, MU2, SIG2, low=ABS_MMIN, high=ABS_MMAX)

            # --- Broken power-law with dip between NS_MAX and BH_MIN ---
            condlist = [m < NS_MAX, (m >= NS_MAX) & (m < BH_MIN), m >= BH_MIN]
            choicelist = [
                m**ALPHA_1,
                (m**ALPHA_DIP) * (NS_MAX ** (ALPHA_1 - ALPHA_DIP)),
                (m**ALPHA_2)
                * (NS_MAX ** (ALPHA_1 - ALPHA_DIP))
                * (BH_MIN ** (ALPHA_DIP - ALPHA_2)),
            ]
            plaw = xp.select(condlist, choicelist, default=0.0)

            # --- Smooth filters (notches + cutoffs) ---
            highpass_lower = 1.0 + (NS_MIN / m) ** N0
            notch_lower = 1.0 - A / ((1.0 + (NS_MAX / m) ** N1) * (1.0 + (m / BH_MIN) ** N2))
            notch_upper = 1.0 - A2 / (
                (1.0 + (UPPER_MIN / m) ** N3) * (1.0 + (m / UPPER_MAX) ** N4)
            )
            lowpass_upper = 1.0 + (m / BH_MAX) ** N5

            # --- Combine all components ---
            base = (
                (1.0 + MIX1 * gaussian_peak1 + MIX2 * gaussian_peak2)
                * plaw
                * notch_lower
                * notch_upper
                / highpass_lower
                / lowpass_upper
            )

            return base

        # mass grid
        m = np.geomspace(1, 100, 100000)
        gwtc4 = GWTC4_broken_powerlaw_peaks(m)

        BH_MIN = 7.763955
        NS_MAX = 4.094744
        NS_MIN = 1.176367
        UPPER_MAX = 66.576705
        UPPER_MIN = 38.277415

        # figure setup
        fig, ax1 = plt.subplots()
        ax1.set_xscale("log")
        ax1.set_yscale("log")

        # violet: '#9400D3', navy: '#001F75'

        ax1.plot(
            m,
            m * gwtc4,
            color="#001F75", #"#9400D3",
            linewidth=2,
            linestyle="--",
            label="GWTC-4: Broken Power Law + 2 Peaks",
        )

        # limits and labels
        ax1.set_xlim(1, 100)
        ax1.set_ylim(0.01, 100)
        ax1.set_xlabel(r"Mass $m\,[M_\odot]$")
        ax1.set_ylabel(r"$m\,p(m|\lambda)$")
        ax1.legend()

        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xscale(ax1.get_xscale())
        ax2.set_xticks([NS_MIN, NS_MAX, BH_MIN, UPPER_MIN, UPPER_MAX])
        ax2.set_xticklabels([
            r"$M_\mathrm{min}$",
            r"$M^\mathrm{gap}_\mathrm{low}$",
            r"$M^\mathrm{gap}_\mathrm{high}$",
            r"$M_\mathrm{PI,low}$",
            r"$M_\mathrm{PI,high}$"
        ])

        ax2.grid(axis="x")
        fig.tight_layout()
        fig.show()



.. dropdown:: Hyperparameters Value

    .. tab-set::

        .. tab-item:: Hyperparameters

            .. table:: Hyperparameters of the BP2P model

                .. include:: ./hyperparams_table.rst
