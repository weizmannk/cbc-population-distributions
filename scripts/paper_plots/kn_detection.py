# coding: utf-8
"""
---------------------------------------------------------------------------------------------------
                                    ABOUT
@createdOn      : February 2026
@description    : Module for estimating the number of gravitational-wave compact binary
                  coalescence (CBC) events whose kilonova (KN) counterpart is detectable
                  by optical telescopes, given a GW170817-like luminosity.
                  Covers IR1 (HL, HLV). O5 is not wired up yet.

                  Import run_analysis/plot_results from a notebook (they take no
                  module-level state), or run this file directly to reproduce the
                  three-telescope comparison at the bottom.

                  Statistical model: for each run and source class, the expected
                  number of GW detections is lambda = R_i * V_s * T_obs (merger-rate
                  density x sensitive volume x observing duration). The number of
                  those detections whose *true* distance also falls inside a given
                  telescope's KN horizon is then lambda * f_hor, where f_hor is the
                  fraction of the detected catalogue within that horizon. Both
                  quantities carry the same Poisson-lognormal uncertainty already
                  used for the GW detection counts elsewhere in this project, so the
                  KN counts inherit it directly instead of being computed from a
                  fixed, rounded-to-the-nearest-integer detection count.
"""

import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import Distance
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
from astropy.table import Table, join
from ligo.skymap.util import sqlite
from scipy import stats

#: Anchor on this file's own location, not the caller's cwd. This is what
#: lets the same defaults work whether the module is run directly
#: (`python kn_detection.py`, cwd = paper_plots/) or imported from a
#: notebook elsewhere in the tree (cwd = notebooks/, or Colab's repo root).
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
DATA_DIR = _REPO_ROOT / "data"
RUNS_DIR = DATA_DIR / "runs"
OUTPUT_DIR = _SCRIPT_DIR.parent / "outputs" / "kn_detection"

#: poisson_lognormal_rate_quantiles is shared with detection_rate.ipynb and
#: population_stats.py, so the KN counts use exactly the same quantile
#: convention as every other detection-count estimate in this project.
sys.path.insert(0, str(_SCRIPT_DIR.parent))
from poisson_rate_utils import poisson_lognormal_rate_quantiles  # noqa: E402

# ---------------------------------------------------------------------------
# Plotting defaults
# ---------------------------------------------------------------------------
matplotlib.rcParams["xtick.labelsize"] = 12.0
matplotlib.rcParams["ytick.labelsize"] = 12.0
matplotlib.rcParams["legend.fontsize"] = 18
matplotlib.rcParams["axes.titlesize"] = 18

COLOR_BNS = "crimson"  # Binary Neutron Star
COLOR_NSBH = "steelblue"  # Neutron Star - Black Hole

#: Source-class mass boundary, and the H5 population-sample file used to
#: split the combined merger rate into per-class rates (see
#: rates_table_for_model below). One file per model, same convention as
#: scripts/paper_plots/population_stats.py.
NS_MAX_MASS = 3.0
_CBC_SAMPLES_PATH = {
    "fullpop": DATA_DIR / "raw" / "fullpop_grid.h5",
    "pixelpop": DATA_DIR / "raw" / "pixelpop.h5",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def populations_bool(table, pop, ns_max_mass=NS_MAX_MASS):
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


def telescope_distance_limit(Mabs=-16, mlim=22):
    """
    Compute the maximum luminosity distance at which a telescope can detect a KN.

    Parameters
    ----------
    Mabs : float
        Absolute magnitude of the KN (default: -16, GW170817-like).
    mlim : float
        Telescope limiting magnitude (default: 22, ZTF clear sky / 300 s exposures).

    Returns
    -------
    float
        Distance limit in Mpc.
    """
    distmod = mlim - Mabs
    d = Distance(distmod=distmod, unit=u.Mpc)
    return d.value


# ---------------------------------------------------------------------------
# Merger-rate model: combined GWTC-5.0 rate, split into per-class rates with
# their own log-normal (mu, sigma), same construction as
# scripts/paper_plots/population_stats.py so both scripts report detection
# counts on an identical statistical footing.
# ---------------------------------------------------------------------------

_STANDARD_90PCT_INTERVAL = np.diff(stats.norm.interval(0.9))[0]


def _combined_rate_row(label):
    """Read the (lower_5, median, upper_95) combined merger rate for one
    population-model label from data/derived/rate_summary.csv, in
    Gpc^-3 yr^-1. Not yet split by source class.
    """
    rate_summary = Table.read(
        DATA_DIR / "derived" / "rate_summary.csv", format="ascii.csv"
    )
    (row,) = rate_summary[rate_summary["label"] == label]
    return float(row["lower_5"]), float(row["median"]), float(row["upper_95"])


def rates_table_for_model(model, ns_max_mass=NS_MAX_MASS):
    """
    Per-class merger-rate table for one population model, with a log-normal
    (mu, sigma) fit to each class's (lower, mid, upper) rate.

    The combined rate (all classes together) is split into BNS/NSBH/BBH by
    each class's fraction of the model's own simulated mass samples, so the
    resulting per-class rates stay self-consistent with what was actually
    simulated for this run, rather than importing an independently
    normalized per-class rate from elsewhere.

    Parameters
    ----------
    model : str
        ``'fullpop'`` or ``'pixelpop'``.
    ns_max_mass : float
        Maximum neutron-star mass in solar masses.

    Returns
    -------
    astropy.table.Table
        One row per population (``BNS``, ``NSBH``, ``BBH``), with columns
        ``mass_fraction``, ``lower``, ``mid``, ``upper`` (Gpc^-3 yr^-1),
        ``mu`` and ``sigma`` (log-normal parameters of ``mid``).
    """
    label = {"fullpop": "GWTC-5.0 FullPop", "pixelpop": "GWTC-5.0 PixelPop"}[model]
    lower, mid, upper = _combined_rate_row(label)

    cbc = Table.read(_CBC_SAMPLES_PATH[model])
    m1, m2 = cbc["mass1"], cbc["mass2"]
    mass_fraction = np.asarray(
        [
            np.sum((m1 < ns_max_mass) & (m2 < ns_max_mass)),
            np.sum((m1 >= ns_max_mass) & (m2 < ns_max_mass)),
            np.sum((m1 >= ns_max_mass) & (m2 >= ns_max_mass)),
        ]
    ) / len(cbc)

    table = Table(
        {
            "population": ["BNS", "NSBH", "BBH"],
            "mass_fraction": mass_fraction,
            "lower": lower * mass_fraction,
            "mid": mid * mass_fraction,
            "upper": upper * mass_fraction,
        }
    )
    table["mu"] = np.log(table["mid"])
    # The mass_fraction scaling cancels in this difference (it multiplies
    # both upper and lower alike), so sigma is really the shared,
    # class-independent width of the combined-rate posterior -- but it is
    # computed per row here for clarity and to keep this table
    # self-contained.
    table["sigma"] = (
        np.log(table["upper"]) - np.log(table["lower"])
    ) / _STANDARD_90PCT_INTERVAL
    return table


def load_run_catalog(run_name, model, datapath=RUNS_DIR, ns_max_mass=NS_MAX_MASS):
    """
    Load one run's detected-event catalogue and simulated rate density,
    split by source class.

    Parameters
    ----------
    run_name : str
        Run folder name under ``datapath``, e.g. ``'IR1HL'``.
    model : str
        ``'fullpop'`` or ``'pixelpop'``.
    datapath : str or Path
        Root path to the simulation data (``data/runs/``).
    ns_max_mass : float
        Maximum neutron-star mass in solar masses.

    Returns
    -------
    dict
        Keyed by population (``'BNS'``, ``'NSBH'``, ``'BBH'``), each value a
        dict with ``n_detected`` (int, number of events found in this run)
        and ``distance_true`` (numpy.ndarray, true injected distances in
        Mpc for those events -- the physical quantity that determines
        whether a kilonova is actually bright enough to see, as opposed to
        the GW pipeline's own post-detection distance *estimate*).
    rate_sim_gpc : float
        The simulated injection rate density (Gpc^-3 yr^-1) this run's
        catalogue was generated at, read from the run's own
        ``events.sqlite``.
    """
    path = Path(datapath) / run_name / model
    allsky = Table.read(str(path / "allsky.dat"), format="ascii.fast_tab")
    injections = Table.read(str(path / "injections.dat"), format="ascii.fast_tab")
    allsky.rename_column("coinc_event_id", "event_id")
    injections.rename_column("simulation_id", "event_id")
    table = join(allsky, injections)

    with sqlite.open(str(path / "events.sqlite"), "r") as db:
        ((result,),) = db.execute(
            "SELECT comment FROM process WHERE program = ?", ("bayestar-inject",)
        )
        rate_sim_gpc = u.Quantity(result).to_value(u.Gpc**-3 * u.yr**-1)

    # z_at_value does a per-element root-find and is the expensive part of
    # populations_bool -- computed once here and reused for all three
    # classes, instead of calling populations_bool three times and paying
    # for the same redshift solve on the same catalogue three times over.
    z = z_at_value(cosmo.luminosity_distance, table["distance"] * u.Mpc).to_value(
        u.dimensionless_unscaled
    )
    zp1 = z + 1
    source_mass1 = table["mass1"] / zp1
    source_mass2 = table["mass2"] / zp1
    masks = {
        "BNS": (source_mass1 < ns_max_mass) & (source_mass2 < ns_max_mass),
        "NSBH": (source_mass1 >= ns_max_mass) & (source_mass2 < ns_max_mass),
        "BBH": (source_mass1 >= ns_max_mass) & (source_mass2 >= ns_max_mass),
    }

    catalog = {}
    for pop, mask in masks.items():
        catalog[pop] = {
            "n_detected": int(np.sum(mask)),
            "distance_true": np.asarray(table["distance"][mask]),
        }
    return catalog, rate_sim_gpc


# ---------------------------------------------------------------------------
# Expected KN-accessible counts
# ---------------------------------------------------------------------------


def kn_expected(distance_true_mpc, lam, d_max_mpc, sigma, quantiles=(0.05, 0.5, 0.95)):
    """
    Expected number of KN-accessible events, lambda * f_hor, with a
    Poisson-lognormal credible interval.

    Thinning a Poisson process by a constant fraction f_hor yields another
    Poisson process at the scaled rate lambda * f_hor; if the parent rate
    carries a log-normal prior with scale sigma, the thinned rate's prior is
    log-normal with the *same* sigma, mean shifted by log(f_hor). So this
    reuses exactly the rate uncertainty already inferred for the GW
    detection count, rather than introducing a separate one.

    Parameters
    ----------
    distance_true_mpc : numpy.ndarray
        True injected distance (Mpc) of every GW-detected event of this
        source class and run.
    lam : float
        Expected number of GW detections for this class and run
        (lambda = R_i * V_s * T_obs).
    d_max_mpc : float
        Telescope KN detection horizon, in Mpc.
    sigma : float
        Log-normal scale of the merger-rate posterior for this class.
    quantiles : tuple of float
        Credible-interval quantiles to evaluate (default: 5/50/95%).

    Returns
    -------
    dict
        ``f_hor`` (fraction of the catalogue within the horizon), ``mean``
        (lambda * f_hor), and ``lo``/``mid``/``hi`` (the requested
        quantiles, floored/rounded/ceiled to integers).
    """
    distance_true_mpc = np.asarray(distance_true_mpc)
    if len(distance_true_mpc) == 0:
        f_hor = 0.0
    else:
        f_hor = float(np.mean(distance_true_mpc < d_max_mpc))

    if f_hor <= 0 or lam <= 0:
        return dict(f_hor=f_hor, mean=0.0, lo=0, mid=0, hi=0)

    mu_kn = np.log(lam) + np.log(f_hor)
    lo, mid, hi = poisson_lognormal_rate_quantiles(list(quantiles), mu_kn, sigma)
    return dict(
        f_hor=f_hor,
        mean=lam * f_hor,
        lo=int(np.floor(lo)),
        mid=int(np.round(mid)),
        hi=int(np.ceil(hi)),
    )


def sample_kn_counts(lam, f_hor, sigma, n_realizations=100_000, seed=42):
    """
    Draw samples from the same Poisson-lognormal model used by
    :func:`kn_expected`, for the cumulative-histogram figure in
    :func:`plot_results`. This is a visualization aid only: the reported
    quantiles come from the closed-form calculation in ``kn_expected``, not
    from these samples.

    Parameters
    ----------
    lam : float
        Expected number of GW detections for this class and run.
    f_hor : float
        Fraction of the detected catalogue within the telescope's horizon.
    sigma : float
        Log-normal scale of the merger-rate posterior for this class.
    n_realizations : int
        Number of samples to draw (default: 100 000).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray of int
    """
    rng = np.random.default_rng(seed)
    if f_hor <= 0 or lam <= 0:
        return np.zeros(n_realizations, dtype=int)
    mu_kn = np.log(lam) + np.log(f_hor)
    rate_draws = rng.lognormal(mean=mu_kn, sigma=sigma, size=n_realizations)
    return rng.poisson(rate_draws)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def run_analysis(
    datapath=RUNS_DIR,
    run_names=("IR1HL", "IR1HLV"),
    model="fullpop",
    run_durations=None,
    ns_max_mass=NS_MAX_MASS,
    Mabs=-16,
    mlim=22,
    n_realizations=100_000,
    seed=42,
    telescope_name="telescope",
    verbose=True,
):
    """
    Run the full KN detectability analysis for one or several GW network runs.

    For each run and source class, the expected number of GW detections is
    lambda = R_i * V_s * T_obs, computed live from the run's own simulated
    catalogue (R_i from data/derived/rate_summary.csv split by mass
    fraction, V_s from the number of detected events over the simulated
    injection rate, T_obs from ``run_durations``). The expected number of
    KN-accessible events is then lambda * f_hor (see :func:`kn_expected`).

    Parameters
    ----------
    datapath : str or Path
        Root path to the simulation data, i.e. ``data/runs/`` (contains one
        folder per run, e.g. ``IR1HL/``, each with a ``fullpop/``/``pixelpop/``
        subfolder).
    run_names : list of str
        Run folder names under ``datapath``, e.g. ``['IR1HL', 'IR1HLV']``.
    model : str
        Population model subfolder to read from each run, ``'fullpop'`` or
        ``'pixelpop'`` (default: ``'fullpop'``).
    run_durations : dict
        Observing duration in years per run, e.g. ``{'IR1HL': 0.5}``.
        Defaults to six months (0.5 yr) for every run in ``run_names``: IR1
        is a planned campaign with no calendar history to derive a duration
        from, unlike O4a/O4b.
    ns_max_mass : float
        Maximum NS mass in solar masses (default: 3.0).
    Mabs : float
        KN absolute magnitude (default: -16).
    mlim : float
        Telescope limiting magnitude (default: 22, ZTF).
    n_realizations : int
        Monte-Carlo draws used only for the histogram figure (default: 100 000).
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
        containing the expected counts, credible interval, and sampled
        realizations (for plotting) for each.
    """
    if run_durations is None:
        run_durations = {run_name: 0.5 for run_name in run_names}

    d_max = telescope_distance_limit(Mabs=Mabs, mlim=mlim)

    if verbose:
        print(f"{telescope_name} KN detection horizon : {d_max:.1f} Mpc")
        print(f"  (m_lim={mlim}, M_abs={Mabs})\n")

    rates_table = rates_table_for_model(model, ns_max_mass=ns_max_mass)

    results = {}

    for run_name in run_names:
        catalog, rate_sim_gpc = load_run_catalog(
            run_name, model, datapath=datapath, ns_max_mass=ns_max_mass
        )
        t_obs = run_durations[run_name]

        run_results = {}
        # BBH is skipped: a kilonova is only expected from a merger with at
        # least one neutron star.
        for pop in ["BNS", "NSBH"]:
            (rate_row,) = rates_table[rates_table["population"] == pop]
            n_detected = catalog[pop]["n_detected"]
            # rate_row["mid"] is the *class-scaled* rate (combined rate x
            # mass_fraction). The simulated injection rate must be scaled
            # the same way before dividing, or mass_fraction ends up
            # applied twice on one side of the product and not at all on
            # the other.
            rate_sim_scaled = rate_sim_gpc * float(rate_row["mass_fraction"])
            sensitive_volume_gpc3 = n_detected / rate_sim_scaled
            lam = float(rate_row["mid"]) * sensitive_volume_gpc3 * t_obs

            expected = kn_expected(
                catalog[pop]["distance_true"],
                lam,
                d_max,
                float(rate_row["sigma"]),
            )
            realizations = sample_kn_counts(
                lam, expected["f_hor"], float(rate_row["sigma"]), n_realizations, seed
            )

            run_results[pop] = {
                "lam": lam,
                "f_hor": expected["f_hor"],
                "mean": expected["mean"],
                "median": float(expected["mid"]),
                "p5": float(expected["lo"]),
                "p95": float(expected["hi"]),
                "realizations": realizations,
            }

        results[run_name] = run_results
        if verbose:
            _print_summary(run_name, run_results)

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
        print(f"    Expected GW x horizon fraction (lambda*f_hor) : {s['mean']:.2f}")
        print(f"    Median : {s['median']:.0f}  (+{hi:.0f} / -{lo:.0f})  [90% CI]")
    print()


def _network_label(run_name):
    """Display label for a run folder name: 'IR1HL' -> 'HL'. Falls back to
    the run name unchanged if it doesn't start with 'IR1', so this stays
    generic for any future campaign passed via run_names."""
    return run_name[3:] if run_name.startswith("IR1") else run_name


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(
    results,
    run_names=("IR1HL", "IR1HLV"),
    telescope_name="telescope",
    outdir=str(OUTPUT_DIR),
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
        Output file name (default: ``ndet_BNS_NSBH.pdf``).
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

    bins = np.arange(0, 30, 1)

    for ax, run_name in zip(axes, run_names):
        network = _network_label(run_name)
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

        if network == "HL":
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
            network,
            color="navy",
            fontweight="bold",
            fontsize=30,
        )

        # ── Axes formatting ───────────────────────────────────────────────
        ax.tick_params(axis="both", labelsize=18, width=2)
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        ax.set_xlabel("Number of events", size=27)

        xlim = (-4.8, 20) if network == "HL" else (-5, 20)
        ax.set_xlim(*xlim)

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


# ---------------------------------------------------------------------------
# Standalone entry point. Guarded so importing this module from a notebook
# (`from paper_plots.kn_detection import run_analysis, plot_results`, as
# quick_start.ipynb does) only pulls in the functions above and never
# triggers this three-telescope run as a side effect.
# ---------------------------------------------------------------------------

TELESCOPES = {
    "GOTO": {"mlim": 20.0, "Mabs": -16},
    "ZTF": {"mlim": 22.0, "Mabs": -16},
    "Vera C. Rubin": {"mlim": 25.7, "Mabs": -16},
}

# -- Shared config, FullPop over IR1 -------------------------------------------
COMMON = dict(
    datapath=RUNS_DIR,
    run_names=["IR1HL", "IR1HLV"],
    model="fullpop",
    run_durations={"IR1HL": 0.5, "IR1HLV": 0.5},
    ns_max_mass=NS_MAX_MASS,
    n_realizations=100_000,
    seed=42,
    verbose=True,
)

if __name__ == "__main__":
    for telescope, cfg in TELESCOPES.items():
        results, d_max = run_analysis(
            **COMMON,
            **cfg,
            telescope_name=telescope,
        )
        plot_results(
            results,
            run_names=COMMON["run_names"],
            telescope_name=telescope,
            outdir=str(OUTPUT_DIR),
            filename=f"ndet_BNS_NSBH_{telescope.replace(' ', '_')}.pdf",
            show=False,
        )
