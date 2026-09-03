# Simulated O4a/O4b predictions (FullPop, PixelPop) vs real O4 events.
#
# Split out of population_stats.py: that script's run_names now covers
# only IR1+O5 (forward-looking, matches the paper's O5-projections
# framing), so it no longer loads O4a/O4b at all. This script keeps the
# "simulated vs real" comparison alive on its own, independent of that
# run list.
#
# Two different real-data sources, on purpose:
#   - area(90)/vol(90): from data/archive/public-alerts.dat, the
#     low-latency public alerts. No other source in this repo carries
#     sky-localization info for real O4 events.
#   - distance: from data/raw/GWTC-5.0_from_O1-to-O4b.csv, the final
#     catalog's luminosity_distance column, not the alert's. Catalog
#     values come from full offline parameter estimation; alert-time
#     distances are a quick preliminary estimate and get superseded once
#     an event makes it into the catalog. Using the catalog value here is
#     strictly more accurate, not just a different source for its own
#     sake.

from pathlib import Path

import numpy as np
import seaborn
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
from astropy.table import Table, join
from ligo.skymap.util import sqlite
from matplotlib import pyplot as plt
from scipy import stats

#: Anchor on this file's own location, not the caller's cwd.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
DATA_DIR = _REPO_ROOT / "data"
RUNS_DIR = DATA_DIR / "runs"
OUTPUT_DIR = _SCRIPT_DIR.parent / "outputs" / "o4_alert_comparison"

ns_max_mass = 3.0  # Same boundary as population_stats.py.
pops = ["BNS", "NSBH", "BBH"]
fieldnames = ["area(90)", "vol(90)", "distance"]
fieldlabels = [
    "90% cred. area (deg²)",
    "90% cred. comoving volume (10⁶ Mpc³)",
    "Luminosity distance (Mpc)",
]
classification_colors = seaborn.color_palette(n_colors=len(pops))


def load_simulated_o4b(model_name, ns_max_mass=ns_max_mass):
    """Load O4b's simulated allsky+injections table for one model
    ("fullpop"/"pixelpop"), split into BNS/NSBH/BBH by source-frame mass.
    Same logic as population_stats.py's main loading loop, just scoped to
    a single run so it doesn't need that script's full run_names list.
    """
    path = RUNS_DIR / "O4b" / model_name
    allsky = Table.read(str(path / "allsky.dat"), format="ascii.fast_tab")
    injections = Table.read(str(path / "injections.dat"), format="ascii.fast_tab")
    allsky.rename_column("coinc_event_id", "event_id")
    injections.rename_column("simulation_id", "event_id")
    table = join(allsky, injections)

    for colname in ["searched_vol", "vol(20)", "vol(50)", "vol(90)"]:
        table[colname] *= 1e-6

    with sqlite.open(str(path / "events.sqlite"), "r") as db:
        ((result,),) = db.execute(
            "SELECT ifos FROM process WHERE program = ?", ("bayestar-realize-coincs",)
        )
        table.meta["network"] = result.replace("1", "").replace(",", "")

    z = z_at_value(cosmo.luminosity_distance, table["distance"] * u.Mpc).to_value(
        u.dimensionless_unscaled
    )
    zp1 = z + 1
    source_mass1 = table["mass1"] / zp1
    source_mass2 = table["mass2"] / zp1

    split = {}
    split["BNS"] = table[
        (source_mass1 < ns_max_mass) & (source_mass2 < ns_max_mass)
    ].copy()
    split["NSBH"] = table[
        (source_mass1 >= ns_max_mass) & (source_mass2 < ns_max_mass)
    ].copy()
    split["BBH"] = table[
        (source_mass1 >= ns_max_mass) & (source_mass2 >= ns_max_mass)
    ].copy()
    return split


def load_real_o4_alerts():
    """Real O4 public alerts (area(90), vol(90)), by classification.
    Same file/columns population_stats.py used for its o4-comparison
    figure.
    """
    o4_data = Table.read(
        str(DATA_DIR / "archive" / "public-alerts.dat"), format="ascii"
    )
    o4_data["vol(90)"] *= 1e-6
    groups = o4_data.group_by(o4_data["classification"]).groups
    return dict(zip(groups.keys, groups))


def load_real_o4_catalog_distances(ns_max_mass=ns_max_mass, far_threshold=1.0):
    """Real O4a+O4b event distances from the GWTC-5.0 catalog CSV (final,
    offline-PE values -- see module docstring for why this differs from
    public-alerts.dat's own distance column).

    Filtering matches the O4a/O4b "Observed" counts already verified
    elsewhere this session: FAR < far_threshold, catalog tag selects the
    run (GWTC-4.1 -> O4a, GWTC-5.0 -> O4b), events with no published
    component masses excluded (can't classify), and GW230518_125908
    excluded from O4a specifically (engineering run, not a real O4a
    detection -- see CLAUDE.md).
    """
    df = Table.read(str(DATA_DIR / "raw" / "GWTC-5.0_from_O1-to-O4b.csv")).to_pandas()
    df = df[df["far"] < far_threshold]
    df = df[df["name"] != "GW230518_125908"]
    df = df.dropna(subset=["mass_1_source", "mass_2_source", "luminosity_distance"])

    o4a = df[df["catalog"] == "GWTC-4.1"]
    o4b = df[df["catalog"] == "GWTC-5.0"]

    def classify(sub):
        m1, m2 = sub["mass_1_source"].to_numpy(), sub["mass_2_source"].to_numpy()
        d = sub["luminosity_distance"].to_numpy()
        return {
            "BNS": d[(m1 < ns_max_mass) & (m2 < ns_max_mass)],
            "NSBH": d[(m1 >= ns_max_mass) & (m2 < ns_max_mass)],
            "BBH": d[(m1 >= ns_max_mass) & (m2 >= ns_max_mass)],
        }

    return {"O4a": classify(o4a), "O4b": classify(o4b)}


def plot_comparison(model_name, sim_o4b, real_alerts, real_distances, outdir):
    fig, axs = plt.subplots(
        len(pops),
        len(fieldnames),
        sharex="col",
        sharey=True,
        gridspec_kw=dict(bottom=0.08, left=0.08, top=0.92, right=0.95),
        figsize=(7.3, 6),
    )

    for ax, fieldlabel in zip(axs[-1], fieldlabels):
        ax.set_xlabel(fieldlabel)
        ax.set_xscale("log")

    ax = axs[1][0]
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1])
    ax.set_ylabel("Cumulative fraction of events")

    axs[0, 0].set_xlim(1e0, 86400)
    axs[0, 1].set_xlim(1e-3, 1e4)
    axs[0, 2].set_xlim(1e1, 1e4)

    for pop, color, ax in zip(pops, classification_colors, axs[:, 0]):
        ax.text(0.05, 0.95, pop, transform=ax.transAxes, color=color, va="top")

    for ax in axs[::-1, fieldnames.index("distance")]:
        ax2 = ax.twiny()
        ax2.set_xlim(*ax.get_xlim())
        ax2.set_xscale(ax.get_xscale())
        ax2.minorticks_off()
        z = [0.01, 0.1, 1]
        ax2.set_xticks(cosmo.luminosity_distance(z).to_value(u.Mpc))
    ax2.set_xticklabels([f"$z$={_}" for _ in z])

    for ax in axs[::-1, fieldnames.index("area(90)")]:
        ax2 = ax.twiny()
        ax2.set_xlim(*ax.get_xlim())
        ax2.set_xscale(ax.get_xscale())
        ax2.minorticks_off()
        ax2.set_xticks([3, 9.6, 47])
    ax2.set_xticklabels(["DECam", "VRO", "ZTF"])
    label1, *_ = ax2.xaxis.get_ticklabels()
    label1.set_ha("right")

    for pop, color, axrow in zip(pops, classification_colors, axs):
        for fieldname, ax in zip(fieldnames, axrow):
            data = sim_o4b[pop][fieldname]
            data = data[np.isfinite(data) & (data > 0)]
            kde = stats.gaussian_kde(np.asarray(np.log(data)))
            ((std,),) = np.sqrt(kde.covariance)
            t = np.geomspace(*ax.get_xlim(), 100)
            y = (
                stats.norm(kde.dataset.ravel(), std)
                .cdf(np.log(t)[:, np.newaxis])
                .mean(1)
            )
            ax.plot(
                t,
                y,
                color=color,
                linewidth=plt.rcParams["lines.linewidth"],
                label="O4b simulation",
            )

            # Real data: catalog distance for the distance panel, alert
            # area/volume otherwise (see module docstring).
            if fieldname == "distance":
                real_data = np.concatenate(
                    [real_distances["O4a"][pop], real_distances["O4b"][pop]]
                )
                real_label = "O4a+O4b catalog"
            else:
                try:
                    real_data = real_alerts[pop][fieldname]
                    real_label = "O4 alerts"
                except KeyError:
                    continue
            if len(real_data) == 0:
                continue
            t = np.minimum(
                np.concatenate(((-np.inf,), np.sort(real_data))),
                10 * ax.get_xlim()[-1],
            )
            y = np.arange(len(real_data) + 1) / len(real_data)
            ax.plot(t, y, color="black", drawstyle="steps-post", label=real_label)

    axs[-1, -1].legend(frameon=False, fontsize=9)
    fig.align_labels()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(outdir / "o4-comparison.pdf"))
    fig.savefig(str(outdir / "o4-comparison.svg"))
    plt.close(fig)


def main():
    # Only applied when this script actually draws its own figure, not on
    # import -- population_stats.py imports load_real_o4_alerts/
    # load_real_o4_catalog_distances and must not have its own rcParams
    # (Times New Roman etc.) clobbered as a side effect of that import.
    plt.rcParams.update(
        {
            "font.size": 12,
            "legend.fontsize": 14,
            "axes.labelsize": 12,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "savefig.dpi": 300,
        }
    )

    real_alerts = load_real_o4_alerts()
    real_distances = load_real_o4_catalog_distances()

    for model_name in ["fullpop", "pixelpop"]:
        sim_o4b = load_simulated_o4b(model_name)
        outdir = OUTPUT_DIR / model_name
        plot_comparison(model_name, sim_o4b, real_alerts, real_distances, outdir)
        print(f"{model_name}: wrote {outdir / 'o4-comparison.pdf'}")


if __name__ == "__main__":
    main()
