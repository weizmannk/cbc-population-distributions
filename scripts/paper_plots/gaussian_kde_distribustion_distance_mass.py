from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
from astropy.table import Table
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from tqdm.auto import tqdm

#: Anchor on this file's own location, not the caller's cwd. Same
#: convention as population_stats.py/kn_detection.py/o4_alert_comparison.py.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
DATA_DIR = _REPO_ROOT / "data"
RUNS_DIR = DATA_DIR / "runs"
OUTPUT_DIR = _SCRIPT_DIR.parent / "outputs" / "gaussian_kde_distribustion_distance_mass"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────────
# O5a and O5c only, matching the old version's own choice of runs
# ("O5a-HLV", "O5c-HLV") -- just without the "-HLV" suffix, since the
# actual folder names carry no network suffix (renamed during the repo
# reorganisation: it's data/runs/O5a, not data/runs/O5a-HLV).
run_names = run_dirs = ["O5a", "O5c"]
MODEL = "fullpop"  # or "pixelpop"
NS_MAX_MASS = 3.0  # M_sun

# ── Read data ──────────────────────────────────────────────────────────────────
tables = {}
with tqdm(total=len(run_dirs)) as progress:
    for run_name, run_dir in zip(run_names, run_dirs):
        # "fullpop4" no longer exists -- that was the GWTC-4.0-era model
        # folder name; the current campaigns are split into fullpop/pixelpop
        # subfolders under each run (see MODEL above).
        path = RUNS_DIR / run_dir / MODEL
        injections = Table.read(str(path / "injections.dat"), format="ascii.fast_tab")
        injections.rename_column("simulation_id", "event_id")

        z = z_at_value(
            cosmo.luminosity_distance, injections["distance"] * u.Mpc
        ).to_value(u.dimensionless_unscaled)
        zp1 = z + 1

        # Store raw (linear) values — classify on source-frame masses
        tables[run_name] = {
            "distance": np.asarray(injections["distance"]),
            "mass1": np.asarray(injections["mass1"]) / zp1,
            "mass2": np.asarray(injections["mass2"]) / zp1,
        }
        progress.update()


# ── Helpers ────────────────────────────────────────────────────────────────────
def classify(mass1, mass2):
    """Classify on LINEAR source-frame masses."""
    return {
        "BNS": (mass1 < NS_MAX_MASS) & (mass2 < NS_MAX_MASS),
        "NSBH": (mass1 >= NS_MAX_MASS) & (mass2 < NS_MAX_MASS),
        "BBH": (mass1 >= NS_MAX_MASS) & (mass2 >= NS_MAX_MASS),
    }


def distance_mass_scatter(ax, distance, mass1, title=""):
    """KDE-coloured scatter of log10(distance) vs log10(mass1)."""
    distance = np.asarray(distance)
    mass1 = np.asarray(mass1)

    if len(distance) < 2:
        ax.set_title(title + " (no data)", fontname="Times New Roman", fontsize=17)
        return None

    log_dist = np.log10(distance)
    log_mass1 = mass1  # np.log10(mass1)

    xy = np.vstack([log_dist, log_mass1])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    log_dist, log_mass1, z = log_dist[idx], log_mass1[idx], z[idx]

    sc = ax.scatter(log_dist, log_mass1, c=z, s=25, cmap="plasma")

    ax.set_xlabel(
        r"$\log_{10}(d_L / \mathrm{Mpc})$",
        fontname="Times New Roman",
        fontsize=20,
        fontweight="bold",
    )
    ax.set_ylabel(
        r"$m_1 [M_\odot]$", fontname="Times New Roman", fontsize=20, fontweight="bold"
    )
    ax.set_title(
        title,
        fontname="Times New Roman",
        fontsize=18,
        fontweight="bold",
        pad=12,
        color="black",
    )

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontsize(18)

    return sc


# ── Figure : one row per run, one column per category ─────────────────────────
categories = ["BNS", "NSBH", "BBH"]
n_runs = len(run_names)
n_cats = len(categories)

plt.close("all")
fig, axs = plt.subplots(n_runs, n_cats, figsize=(7 * n_cats, 6 * n_runs))

if n_runs == 1:
    axs = axs[np.newaxis, :]

for i, run_name in enumerate(run_names):
    dist = tables[run_name]["distance"]  # linear
    m1 = tables[run_name]["mass1"]  # linear source-frame
    m2 = tables[run_name]["mass2"]  # linear source-frame

    masks = classify(m1, m2)  # classify in linear space

    for j, cat in enumerate(categories):
        ax = axs[i, j]
        mask = masks[cat]
        sc = distance_mass_scatter(
            ax, dist[mask], m1[mask], title=rf"{run_name} -- {cat}  (N={mask.sum():,})"
        )

        if sc is not None:
            cbar = fig.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label(
                "Event density",
                fontname="Times New Roman",
                fontsize=20,
                fontweight="bold",
            )

plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.2)
plt.savefig(str(OUTPUT_DIR / "distance_mass_scatter.pdf"), dpi=300, bbox_inches="tight")
plt.savefig(str(OUTPUT_DIR / "distance_mass_scatter.png"), dpi=300, bbox_inches="tight")
plt.close()
