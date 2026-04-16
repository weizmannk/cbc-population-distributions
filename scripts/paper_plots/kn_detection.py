# coding: utf-8
"""
---------------------------------------------------------------------------------------------------
                                    ABOUT
@author         : Ramodgwendé Weizmann KIENDREBEOGO
@email          : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
@repo           : https://github.com/weizmannk/bns-inspiral-range
@createdOn      : February 2026
@description    : Module for estimating the number of gravitational-wave compact binary
                  coalescence (CBC) events whose kilonova (KN) counterpart is detectable
                  by optical telescopes, given a GW170817-like luminosity.
                  Covers IR1 and O5 observing runs.
"""

import os
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import Distance
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
from astropy.table import Table

# ---------------------------------------------------------------------------
# Plotting defaults
# ---------------------------------------------------------------------------
matplotlib.rcParams["xtick.labelsize"] = 12.0
matplotlib.rcParams["ytick.labelsize"] = 12.0
matplotlib.rcParams["legend.fontsize"] = 18
matplotlib.rcParams["axes.titlesize"] = 18

COLOR_BNS = "crimson"  # Binary Neutron Star
COLOR_NSBH = "steelblue"  # Neutron Star - Black Hole


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def populations_bool(table, pop, ns_max_mass=3.0):
    """
    Classify CBC injections into BNS, NSBH, or BBH based on source-frame mass.

    Parameters
    ----------
    table : astropy.table.Table
        Injection catalogue with columns ``mass1``, ``mass2``, ``distance`` (Mpc).
    pop : str
        One of ``'BNS'``, ``'NSBH'``, or ``'BBH'``.
    ns_max_mass : float
        Maximum neutron-star mass in solar masses (default: 3.0).

    Returns
    -------
    numpy.ndarray of bool
    """
    z = z_at_value(cosmo.luminosity_distance, table["distance"] * u.Mpc).to_value(
        u.dimensionless_unscaled
    )
    zp1 = z + 1
    source_mass1 = table["mass1"] / zp1
    source_mass2 = table["mass2"] / zp1

    if pop == "BNS":
        return (source_mass1 < ns_max_mass) & (source_mass2 < ns_max_mass)
    elif pop == "NSBH":
        return (source_mass1 >= ns_max_mass) & (source_mass2 < ns_max_mass)
    else:  # BBH
        return (source_mass1 >= ns_max_mass) & (source_mass2 >= ns_max_mass)


def ztf_distance_limit(Mabs=-16, mlim=22):
    """
    Compute the maximum luminosity distance at which ZTF can detect a KN.

    Parameters
    ----------
    Mabs : float
        Absolute magnitude of the KN (default: -16, GW170817-like).
    mlim : float
        ZTF limiting magnitude (default: 22, clear sky / 300 s exposures).

    Returns
    -------
    float
        Distance limit in Mpc.
    """
    distmod = mlim - Mabs
    d = Distance(distmod=distmod, unit=u.Mpc)
    return d.value


def run_realizations(allsky_df, N_events, d_max_mpc, n_realizations=100_000, seed=42):
    """
    Monte-Carlo: draw ``N_events`` random distances from the catalogue and
    count how many fall within ``d_max_mpc``.

    Parameters
    ----------
    allsky_df : pandas.DataFrame
        All-sky catalogue with column ``distmean``.
    N_events : int
        Expected number of detections for this run.
    d_max_mpc : float
        ZTF detection horizon in Mpc.
    n_realizations : int
        Number of Monte-Carlo draws (default: 100 000).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of int
        Number of KN-detectable events for each realization.
    """
    rng = np.random.default_rng(seed)
    distances = allsky_df["distmean"].to_numpy()
    return [
        int(np.sum(rng.choice(distances, N_events) < d_max_mpc))
        for _ in range(n_realizations)
    ]


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def run_analysis(
    datapath="../data/runs",
    run_names=("HL", "HLV"),
    Number_BNS=None,
    Number_NSBH=None,
    ns_max_mass=3.0,
    Mabs=-16,
    mlim=22,
    n_realizations=100_000,
    seed=42,
    telescope_name="telescope",
    verbose=True,
):
    """
    Run the full KN detectability analysis for one or several GW network runs.

    Parameters
    ----------
    datapath : str
        Root path to the simulation data (contains ``<run>`` folders).
    run_names : list of str
        List of detector network labels, e.g. ``['HL', 'HLV']``.
    Number_BNS : dict
        Expected number of detected BNS events per run,
        e.g. ``{'HL': 4, 'HLV': 6}``.
    Number_NSBH : dict
        Expected number of detected NSBH events per run.
    ns_max_mass : float
        Maximum NS mass in solar masses (default: 3.0).
    Mabs : float
        KN absolute magnitude (default: -16).
    mlim : float
        ZTF limiting magnitude (default: 22).
    n_realizations : int
        Monte-Carlo realizations (default: 100 000).
    seed : int
        Random seed (default: 42).
    telescope_name: str
        Telescope name
    verbose : bool
        Print a summary table to stdout (default: True).

    Returns
    -------
    dict
        Nested dictionary keyed by run name, then ``'BNS'`` / ``'NSBH'``,
        containing the list of realization counts and summary statistics.
    """
    # Default event counts (Kiendrebeogo et al. 2026)
    if Number_BNS is None:
        Number_BNS = {"HL": 1, "HLV": 2}
    if Number_NSBH is None:
        Number_NSBH = {"HL": 2, "HLV": 2}

    d_max = ztf_distance_limit(Mabs=Mabs, mlim=mlim)

    if verbose:
        print(f"{telescope_name} KN detection horizon : {d_max:.1f} Mpc")
        print(f"  (m_lim={mlim}, M_abs={Mabs})\n")

    results = {}

    for run_name in run_names:
        path = Path(datapath) / run_name
        allsky = Table.read(str(path / "allsky.dat"), format="ascii.fast_tab")
        injections = Table.read(str(path / "injections.dat"), format="ascii.fast_tab")

        BNS_mask = populations_bool(injections, "BNS", ns_max_mass=ns_max_mass)
        NSBH_mask = populations_bool(injections, "NSBH", ns_max_mass=ns_max_mass)

        allsky_BNS = allsky[BNS_mask].to_pandas()
        allsky_NSBH = allsky[NSBH_mask].to_pandas()

        real_BNS = run_realizations(
            allsky_BNS, Number_BNS[run_name], d_max, n_realizations, seed
        )
        real_NSBH = run_realizations(
            allsky_NSBH, Number_NSBH[run_name], d_max, n_realizations, seed
        )

        def stats(r):
            return {
                "realizations": r,
                "mean": float(np.mean(r)),
                "median": float(np.percentile(r, 50)),
                "p5": float(np.percentile(r, 5)),
                "p95": float(np.percentile(r, 95)),
            }

        results[run_name] = {"BNS": stats(real_BNS), "NSBH": stats(real_NSBH)}

        if verbose:
            _print_summary(run_name, results[run_name])

    return results, d_max


def _print_summary(run_name, run_results):
    """Print a formatted summary for one detector network."""
    sep = "=" * 50
    print(sep)
    print(f"  Run : {run_name}")
    print(sep)
    for pop, s in run_results.items():
        lo = s["median"] - s["p5"]
        hi = s["p95"] - s["median"]
        print(f"  {pop}")
        print(f"    Mean   : {s['mean']:.1f}")
        print(f"    Median : {s['median']:.0f}  (+{hi:.0f} / -{lo:.0f})  [90% CI]")
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(
    results,
    run_names=("HL", "HLV"),
    telescope_name="telescope",
    outdir="./",
    filename="ndet_BNS_NSBH.pdf",
    show=True,
):
    """
    Produce the cumulative histogram figure.

    Parameters
    ----------
    results : dict
        Output of :func:`run_analysis`.
    run_names : list of str
        Ordered list of runs to plot (must match keys in ``results``).
    outdir : str or None
        Directory to save the figure. If ``None``, the figure is not saved.
    filename : str
        Output file name (default: ``ndet_ZTF_BNS_NSBH.pdf``).
    show : bool
        Call ``plt.show()`` at the end (default: True).

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Use rc_context to fully isolate rendering settings from gwpy or any
    # other package that sets text.usetex=True globally. The context manager
    # restores the original rcParams when it exits, so nothing leaks out.

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(1, len(run_names))
    axes = [fig.add_subplot(gs[i]) for i in range(len(run_names))]

    # fig.suptitle(
    #     f"KN detectability  -- {telescope_name}", fontsize=28, fontweight="bold", y=1.02
    # )

    bins = np.arange(0, 30, 1)

    for ax, run_name in zip(axes, run_names):
        s_BNS = results[run_name]["BNS"]
        s_NSBH = results[run_name]["NSBH"]

        ax.hist(
            s_BNS["realizations"],
            bins=bins,
            density=True,
            cumulative=True,
            histtype="step",
            linestyle="--",
            color=COLOR_BNS,
            linewidth=4,
        )
        ax.hist(
            s_NSBH["realizations"],
            bins=bins,
            density=True,
            cumulative=True,
            histtype="step",
            linestyle=":",
            color=COLOR_NSBH,
            linewidth=4,
        )

        # ── Annotations ──────────────────────────────────────────────────
        bbox_NSBH = dict(
            facecolor="white",
            alpha=0.8,
            edgecolor=COLOR_NSBH,
            linestyle=":",
            linewidth=3,
        )
        bbox_BNS = dict(
            facecolor="white",
            alpha=0.8,
            edgecolor=COLOR_BNS,
            linestyle="--",
            linewidth=2.5,
        )

        if run_name == "HL":
            ax.text(-3.5, 0.69, "NSBH", color="k", fontsize=24, bbox=bbox_NSBH)
            ax.text(-4.2, 0.60, f"<N>={s_NSBH['mean']:.1f}", fontsize=24)
            ax.text(3.6, 0.20, "BNS", color="k", fontsize=24, bbox=bbox_BNS)
            ax.text(2.2, 0.12, f"<N>={s_BNS['mean']:.1f}", fontsize=24)
        else:
            ax.text(-3.2, 0.70, "NSBH", color="k", fontsize=24, bbox=bbox_NSBH)
            ax.text(-4.2, 0.61, f"<N>={s_NSBH['mean']:.1f}", fontsize=24)
            ax.text(4.9, 0.30, "BNS", color="k", fontsize=24, bbox=bbox_BNS)
            ax.text(3.5, 0.20, f"<N>={s_BNS['mean']:.1f}", fontsize=24)

        # ── Panel title ───────────────────────────────────────────────────
        ax.text(
            10,
            0.8,
            run_name,
            color="navy",
            fontweight="bold",
            fontsize=30,
        )

        # ── Axes formatting ───────────────────────────────────────────────
        ax.tick_params(axis="both", labelsize=18, width=2)
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        ax.set_xlabel("Number of events", size=27)

        xlim = (-4.8, 20) if run_name == "HL" else (-5, 20)
        ax.set_xlim(*xlim)
        # xlim = (0, 8) if run_name == "HL" else (0, 10)
        # ax.set_xlim(*xlim)

    axes[0].set_ylabel(
        "Cumulative probability density",
        size=27,
    )

    fig.tight_layout()

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        savepath = os.path.join(outdir, filename)
        fig.savefig(savepath)
        print(f"Figure saved ==> {savepath}")

    if show:
        plt.show()

    return fig


# # ============================================================================
# # Telescope configurations
# # ============================================================================

# TELESCOPES = {
#     "GOTO": {"mlim": 20.0, "Mabs": -16},
#     "ZTF": {"mlim": 22.0, "Mabs": -16},
#     "Vera C. Rubin": {"mlim": 25.7, "Mabs": -16},
# }


# # -- Shared config -------------------------------------------------------------
# COMMON = dict(
#     datapath="../data/runs/IR1",
#     run_names=["HL", "HLV"],
#     Number_BNS={"HL": 1, "HLV": 2},
#     Number_NSBH={"HL": 2, "HLV": 2},
#     ns_max_mass=3.0,
#     n_realizations=100_000,
#     seed=42,
#     verbose=True,
# )

# # Run the analysis
# for telescope, cfg in TELESCOPES.items():
#     if telescope == "ZTF":
#         results, d_max = run_analysis(
#             **COMMON,
#             **cfg,
#             telescope_name=telescope,
#         )
#         plot_results(
#             results,
#             run_names=COMMON["run_names"],
#             telescope_name=telescope,
#             outdir="./",
#             show=True,
#         )
