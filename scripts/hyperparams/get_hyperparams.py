"""Build data/derived/hyperparams_map.csv and the joint mass-rate grid
caches from the GWTC-4.0/GWTC-5.0 source files. Missing sources are
fetched automatically (LIGO DCC for GWTC-4.0, Zenodo for GWTC-5.0). Run
directly to (re)build the cache; downstream code only reads the small
cached outputs, never the raw files."""

import json
import os
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.utils.data import download_file
from bilby.core.result import read_in_result

#: Anchor paths on this file's own location, not the caller's cwd.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = REPO_ROOT / "data"
DERIVED_DIR = DATA_DIR / "derived"

# --- Remote sources -----------------------------------------------------
ZENODO_RECORD_GWTC5 = "20292639"
#: Only available inside this archive, not as an individual download.
ZENODO_ARCHIVE_GWTC5 = "popsummary_files.tar.gz"
ZENODO_FILENAME_GWTC5 = (
    "production_1_mass_NotchFilterBinnedPairingMassDistribution_"
    "redshift_powerlaw_mag_iid_spin_magnitude_gaussian_tilt_"
    "iid_spin_orientation_popsummary.h5"
)

# GWTC-4.0: AllCBC_FullPop.h5 (from Zenodo, inside analyses_AllCBC.tar) is
# the source actually used to build the published rate table. Its
# per-category rate_bns/rate_nsbh/rate_bbh/rate_full columns match the
# paper's Table 2 at the MAP row (rate_full = 130.28 vs 130 published).
# The DCC bilby Result file below has the same hyperparameter posterior
# (log_likelihood matches row-for-row) but a *different*, non-reproducible
# `rate` column. compute_rate_posterior() in gwpopulation_pipe draws it
# from a random gamma distribution, so two independent post-processing
# runs of the same posterior don't agree, and it never had rate_bns/
# rate_nsbh/rate_bbh/rate_full at all. Kept only as a fallback/for tests.
DCC_BASE_URL = "https://dcc.ligo.org/LIGO-T2500311/public"
DCC_FILENAME_GWTC4 = (
    "baseline5_widesigmachi2_mass_NotchFilterBinnedPairingMassDistribution_"
    "redshift_powerlaw_mag_iid_spin_magnitude_gaussian_tilt_"
    "iid_spin_orientation_result.hdf5"
)

ZENODO_RECORD_GWTC4 = "16911563"
ZENODO_ARCHIVE_GWTC4 = "analyses_AllCBC.tar"
ZENODO_FILENAME_GWTC4 = "AllCBC_FullPop.h5"

#: PixelPop's popsummary file, from the same GWTC-5.0 archive as FullPop's.
#: Unlike FullPop, it carries no scalar `rate` column. The rate lives in
#: the `joint_pixelpop_rate` grid and has to be integrated (see
#: compute_rate_summary_pixelpop).
ZENODO_MEMBER_PIXELPOP = "all_cbc_varcut1_popsummary.h5"
ZENODO_FILENAME_PIXELPOP = "pixelpop_popsummary.h5"


def _verify_size(path: Path, expected_size: int, source_desc: str) -> None:
    """Raise OSError (and delete the file) if its size doesn't match
    expected_size. Catches truncated/incomplete downloads."""
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        path.unlink()
        raise OSError(
            f"Incomplete download of {source_desc}: expected {expected_size} "
            f"bytes, got {actual_size}. Deleted the truncated file at {path}, "
            f"re-run to retry."
        )


def download_from_zenodo(record_id: str, filename: str, dest_path: Path) -> Path:
    """
    Download `filename` from a public Zenodo record via the official API
    (queries the record's file list rather than guessing a download URL),
    unless it already exists at `dest_path`.
    """
    dest_path = Path(dest_path)
    if dest_path.exists():
        return dest_path

    api_url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(api_url, timeout=30) as response:
        record = json.load(response)

    matches = [f for f in record["files"] if f["key"] == filename]
    if not matches:
        available = [f["key"] for f in record["files"]]
        raise FileNotFoundError(
            f"{filename!r} not found in Zenodo record {record_id}. "
            f"Available files: {available}"
        )

    download_url = matches[0]["links"]["self"]
    expected_size = matches[0]["size"]
    print(f"Downloading {filename} from Zenodo record {record_id} ...")
    cached_path = Path(download_file(download_url, cache=True, show_progress=True))
    _verify_size(cached_path, expected_size, f"{filename!r} (Zenodo {record_id})")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    cached_path.replace(dest_path)
    print(f"Saved to {dest_path}")
    return dest_path


def download_and_extract_from_zenodo_tarball(
    record_id: str, archive_name: str, member_filename: str, dest_path: Path
) -> Path:
    """Download `archive_name` (a .tar or .tar.gz, compression is
    auto-detected) from a public Zenodo record and extract only the member
    ending in `member_filename` to `dest_path`. No-op if `dest_path` already
    exists. The archive itself can be several GB, so this is meant as a
    one-time step, not a routine call."""
    dest_path = Path(dest_path)
    if dest_path.exists():
        return dest_path

    api_url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(api_url, timeout=30) as response:
        record = json.load(response)

    matches = [f for f in record["files"] if f["key"] == archive_name]
    if not matches:
        available = [f["key"] for f in record["files"]]
        raise FileNotFoundError(
            f"{archive_name!r} not found in Zenodo record {record_id}. "
            f"Available files: {available}"
        )

    download_url = matches[0]["links"]["self"]
    expected_size = matches[0]["size"]
    size_gb = expected_size / 1e9
    print(
        f"Downloading {archive_name} ({size_gb:.1f} GB) from Zenodo record {record_id} ..."
    )
    cached_path = Path(download_file(download_url, cache=True, show_progress=True))
    _verify_size(cached_path, expected_size, f"{archive_name!r} (Zenodo {record_id})")

    print(f"Looking for a member ending in {member_filename!r} ...")
    with tarfile.open(cached_path) as tar:
        member_matches = [
            m for m in tar.getmembers() if m.name.endswith(member_filename)
        ]
        if not member_matches:
            raise FileNotFoundError(
                f"No member ending in {member_filename!r} inside {archive_name}"
            )
        member = member_matches[0]
        print(f"Extracting {member.name} ...")
        tar.extract(member, path=dest_path.parent, filter="data")
        extracted_path = dest_path.parent / member.name

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    extracted_path.replace(dest_path)
    print(f"Saved to {dest_path}")
    return dest_path


def download_from_dcc(filename: str, dest_path: Path) -> Path:
    """
    Download `filename` from the public LIGO DCC directory for T2500311,
    unless it already exists at `dest_path`.
    """
    dest_path = Path(dest_path)
    if dest_path.exists():
        return dest_path

    url = f"{DCC_BASE_URL}/{filename}"
    head_request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(head_request, timeout=30) as response:
        expected_size = int(response.headers["Content-Length"])

    print(f"Downloading {filename} from LIGO DCC (T2500311) ...")
    cached_path = Path(download_file(url, cache=True, show_progress=True))
    _verify_size(cached_path, expected_size, f"{filename!r} (LIGO DCC T2500311)")

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    cached_path.replace(dest_path)
    print(f"Saved to {dest_path}")
    return dest_path


def to_rst(df, title="Hyperparameters of the FullPop-4.0 model"):
    widths = [max(len(str(x)) for x in df[col]) for col in df.columns]
    widths = [max(w, len(col)) for w, col in zip(widths, df.columns)]

    def hline(sep="-"):
        return "+" + "+".join(sep * (w + 2) for w in widths) + "+"

    def row(cells):
        return (
            "|" + "|".join(f" {str(c).ljust(w)} " for c, w in zip(cells, widths)) + "|"
        )

    # build the grid table body (no directive yet)
    body = [hline("-"), row(df.columns), hline("=")]
    for _, r in df.iterrows():
        body.append(row(r))
        body.append(hline())

    # indent body so it belongs to the table directive
    indented = ["   " + line for line in body]  # 3 spaces is conventional

    lines = [f".. table:: {title}", ""]
    lines.extend(indented)
    return "\n".join(lines)


# --- in  LaTeX ---
def to_latex(df, caption="Hyperparameters of the FullPop-4.0 model"):
    out = []
    out.append(r"\begin{table}[ht]")
    out.append(r"\centering")
    out.append(rf"\caption{{{caption}}}")
    out.append(r"\begin{tabular}{lll}")
    out.append(r"\hline")
    out.append(r"Parameter & Description & Value\\")
    out.append(r"\hline")
    for _, row in df.iterrows():
        out.append(rf"{row['Parameter']} & {row['Description']} & {row['Value']} \\")
    out.append(r"\hline")
    out.append(r"\end{tabular}")
    out.append(r"\end{table}")
    return "\n".join(out)


# --- paper-style LaTeX table (booktabs, sectioned) -----------------------
def to_latex_paper(
    row: pd.Series,
    caption: str = "Hyperparameters of the FullPop-4.0 Distribution Model",
    label: str = "tab:hyperparams",
) -> str:
    """Build the sectioned booktabs-style hyperparameter table used in the
    paper (Mass Distribution / Pairing Function / Spin Distribution), from
    one row of `row` (a MAP hyperparameter Series, e.g.
    hyperparams_map.loc["GWTC-5.0 (popsummary)"])."""

    def v(key, decimals):
        return f"{row[key]:.{decimals}f}"

    return "\n".join(
        [
            r"% !TEX root = ../main.tex",
            r"\begin{table*}[htb]",
            r"\renewcommand\arraystretch{1.08}",
            r"\setlength{\tabcolsep}{16pt}",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\begin{tabular}{llr}",
            r"\toprule",
            r"\textbf{Parameter} & \textbf{Description} & \textbf{Posterior Value} \\",
            r"\midrule",
            r"\multicolumn{3}{c}{\textit{Mass Distribution}} \\",
            r"\midrule",
            rf"$m_{{\mathrm{{min,NS}}}}$ & Minimum neutron star mass ($M_\odot$) & ${v('NSmin', 2)}$ \\",
            rf"$m_{{\mathrm{{max,NS}}}} \equiv \gamma_{{\mathrm{{low}},1}}$ & Maximum neutron star mass ($M_\odot$) & ${v('NSmax', 2)}$ \\",
            rf"$m_{{\mathrm{{min,BH}}}} \equiv \gamma_{{\mathrm{{high}},1}}$ & Minimum black hole mass ($M_\odot$) & ${v('BHmin', 2)}$ \\",
            rf"$\gamma_{{\mathrm{{low}},2}}$  & Lower boundary of pair-instability gap ($M_\odot$) & ${v('UPPERmin', 2)}$ \\",
            rf"$\gamma_{{\mathrm{{high}},2}}$ & Upper boundary of pair-instability gap ($M_\odot$) & ${v('UPPERmax', 2)}$ \\",
            rf"$m_{{\mathrm{{max,BH}}}}$ & Maximum black hole mass ($M_\odot$) & ${v('BHmax', 1)}$ \\",
            r"\midrule",
            rf"$\alpha_1$ & Power-law exponent for masses below $m_{{\mathrm{{max,NS}}}}$ & ${v('alpha_1', 2)}$ \\",
            rf"$\alpha_{{\mathrm{{dip}}}}$ & Power-law exponent within the NS–BH mass gap & ${v('alpha_dip', 2)}$ \\",
            rf"$\alpha_2$ & Power-law exponent for masses above $m_{{\mathrm{{min,BH}}}}$ & ${v('alpha_2', 2)}$ \\",
            r"\midrule",
            rf"$\mu_{{\mathrm{{peak}},1}}$ & Mean of primary Gaussian peak ($M_\odot$) & ${v('mu1', 2)}$ \\",
            rf"$\sigma_{{\mathrm{{peak}},1}}$ & Std. dev. of primary Gaussian peak ($M_\odot$) & ${v('sig1', 2)}$ \\",
            rf"$c_1$ & Mixing fraction of primary Gaussian peak & ${v('mix1', 2)}$ \\",
            rf"$\mu_{{\mathrm{{peak}},2}}$ & Mean of secondary Gaussian peak ($M_\odot$) & ${v('mu2', 2)}$ \\",
            rf"$\sigma_{{\mathrm{{peak}},2}}$ & Std. dev. of secondary Gaussian peak ($M_\odot$) & ${v('sig2', 2)}$ \\",
            rf"$c_2$ & Mixing fraction of secondary Gaussian peak & ${v('mix2', 2)}$ \\",
            r"\midrule",
            rf"$A_1$ & Depth of primary mass gap suppression & ${v('A', 3)}$ \\",
            rf"$A_2$ & Depth of pair-instability gap suppression & ${v('A2', 3)}$ \\",
            "",
            r"\midrule",
            rf"$\eta_0$ & Sharpness of low-mass truncation & ${v('n0', 0)}$ \\",
            rf"$\eta_1$ & Sharpness at $m_{{\mathrm{{max,NS}}}}$ & ${v('n1', 0)}$ \\",
            rf"$\eta_2$ & Sharpness at $m_{{\mathrm{{min,BH}}}}$ & ${v('n2', 0)}$ \\",
            rf"$\eta_3$ & Sharpness at $\gamma_{{\mathrm{{low}},2}}$ & ${v('n3', 0)}$ \\",
            rf"$\eta_4$ & Sharpness at $\gamma_{{\mathrm{{high}},2}}$ & ${v('n4', 0)}$ \\",
            rf"$\eta_5$ & Sharpness of high-mass truncation & ${v('n5', 2)}$ \\",
            "",
            r"\midrule",
            r"\multicolumn{3}{c}{\textit{Pairing Function}} \\",
            r"\midrule",
            rf"$m_{{\mathrm{{break}}}}$ & Pairing function break mass ($M_\odot$) & ${v('mbreak', 1)}$ \\",
            rf"$\beta_1$ & Pairing power-law index for $m_2 < m_{{\mathrm{{break}}}}$ & ${v('beta_pair_1', 2)}$ \\",
            rf"$\beta_2$ & Pairing power-law index for $m_2 \geq m_{{\mathrm{{break}}}}$ & ${v('beta_pair_2', 2)}$ \\",
            "",
            r"\midrule",
            r"\multicolumn{3}{c}{\textit{Spin Distribution}} \\",
            r"\midrule",
            rf"$\mu_{{\chi}}$ & Mean of spin magnitude Gaussian component & ${v('mu_chi', 3)}$ \\",
            rf"$\sigma_{{\chi}}$ & Std. dev. of spin magnitude Gaussian component & ${v('sigma_chi', 2)}$ \\",
            rf"$a_{{\mathrm{{max}}}}$ & Maximum spin magnitude & ${v('amax', 0)}$ \\",
            r"\midrule",
            rf"$\xi_{{\mathrm{{spin}}}}$ & Fraction of BHs in preferentially aligned component & ${v('xi_spin', 2)}$ \\",
            rf"$\sigma_{{\mathrm{{spin}}}}$ & Width of preferentially aligned component & ${v('sigma_spin', 3)}$ \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )


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


def load_hyperparams_map(path: str) -> pd.Series:
    """Load the MAP hyperparameter sample from a FullPop result file,
    dispatching on format: popsummary (via PopulationResult) or bilby
    Result. Only the hyperparameter table is loaded, not the rate grids."""
    try:
        from popsummary.popresult import PopulationResult

        popresult = PopulationResult(path)
        df = pd.DataFrame(
            popresult.get_hyperparameter_samples(),
            columns=popresult.get_metadata("hyperparameters"),
        )
        return df.iloc[(df.log_likelihood + df.log_prior).idxmax()]
    except (KeyError, OSError):
        hyperparams = read_in_result(path)
        return _get_map_sample(hyperparams)


def extract_hyperparams_map(
    sources: dict[str, str], cache_csv: str, force: bool = False
) -> pd.DataFrame:
    """
    Return a DataFrame of MAP hyperparameters, one row per label in
    `sources` (label -> file path). Labels already present in `cache_csv`
    are reused instead of reopening their (possibly multi-GB) source file;
    pass force=True to recompute everything.
    """
    cached = pd.DataFrame()
    if not force and os.path.exists(cache_csv):
        cached = pd.read_csv(cache_csv, index_col=0)
        sources = {
            label: path for label, path in sources.items() if label not in cached.index
        }

    if not sources:
        return cached

    rows = {label: load_hyperparams_map(path) for label, path in sources.items()}
    new_df = pd.DataFrame(rows).T
    df = pd.concat([cached, new_df]) if not cached.empty else new_df
    df.index.name = "label"
    os.makedirs(os.path.dirname(cache_csv) or ".", exist_ok=True)
    df.to_csv(cache_csv)
    return df


#: The three small joint-mass rate grids present in both catalogues'
#: popsummary files (600x600 each). Excludes
#: primary_mass_secondary_mass_joint_full_posterior, which only GWTC-5.0
#: carries and is ~7 GB once loaded.
JOINT_GRID_NAMES = (
    "primary_mass_secondary_mass_joint_median",
    "primary_mass_secondary_mass_joint_ppd",
    "primary_mass_secondary_mass_joint_uncertainty",
)


def load_joint_grids(
    path: str, grid_names: tuple[str, ...] = JOINT_GRID_NAMES
) -> dict[str, tuple]:
    """
    Load the small 2D joint-mass rate grids from a popsummary file.

    Returns {grid_name: (m1_positions, m2_positions, rates)}. Grid names
    that don't exist in this particular file are silently skipped (e.g. a
    future catalogue might not ship all three). Only ever touches the
    named grids, never primary_mass_secondary_mass_joint_full_posterior.
    """
    from popsummary.popresult import PopulationResult

    popresult = PopulationResult(path)
    grids = {}
    for name in grid_names:
        try:
            (m1_pos, m2_pos), rates = popresult.get_rates_on_grids(name)
        except KeyError:
            continue
        grids[name] = (
            np.asarray(m1_pos).ravel(),
            np.asarray(m2_pos).ravel(),
            np.asarray(rates),
        )
    return grids


def extract_joint_grids(
    sources: dict[str, str],
    cache_dir: str,
    grid_names: tuple[str, ...] = JOINT_GRID_NAMES,
    force: bool = False,
) -> dict[str, str]:
    """
    Save the small joint-mass grids for each label in `sources` (label ->
    popsummary file path) to one compressed .npz per label under
    `cache_dir`. Skips labels whose .npz already exists unless force=True.
    Returns {label: npz_path}.
    """
    os.makedirs(cache_dir, exist_ok=True)
    safe = str.maketrans({c: "_" for c in " ()."})
    out_paths = {}
    for label, path in sources.items():
        npz_path = os.path.join(cache_dir, f"{label.translate(safe)}_grids.npz")
        out_paths[label] = npz_path
        if not force and os.path.exists(npz_path):
            continue
        grids = load_joint_grids(path, grid_names)
        arrays = {}
        for name, (m1_pos, m2_pos, rates) in grids.items():
            arrays[f"{name}__m1"] = m1_pos
            arrays[f"{name}__m2"] = m2_pos
            arrays[f"{name}__rates"] = rates
        np.savez_compressed(npz_path, **arrays)
    return out_paths


def compute_rate_summary_fullpop(path: str, rate_column: str = "rate") -> dict:
    """Summarise a FullPop-style scalar rate column, giving the median and
    5%/95% credible interval (the convention GWTC-5.0 itself uses, since
    its published rates come from `np.median` over the posterior, not a
    single MAP sample, see compute_rate_summary_pixelpop for why
    PixelPop has no MAP/ML point at all here), plus the MAP point for
    reference (matches GWTC-4.0's older, MAP-based convention)."""
    from popsummary.popresult import PopulationResult

    popresult = PopulationResult(path)
    df = pd.DataFrame(
        popresult.get_hyperparameter_samples(),
        columns=popresult.get_metadata("hyperparameters"),
    )
    rate = df[rate_column]
    lower_5, median, upper_95 = np.percentile(rate, [5, 50, 95])
    map_row = df.iloc[(df.log_likelihood + df.log_prior).idxmax()]
    return {
        "map": map_row[rate_column],
        "median": median,
        "lower_5": lower_5,
        "upper_95": upper_95,
    }


def compute_rate_summary_pixelpop(
    path: str, grid_name: str = "joint_pixelpop_rate"
) -> dict:
    """Summarise PixelPop's total rate by integrating its per-sample 2D
    joint mass-rate grid over the whole domain (log_mass_1 x log_mass_2).

    PixelPop's popsummary file has no scalar `rate` column at all. The
    rate only exists as a density on this grid. The grid's `positions`
    array (n_edges x n_edges) is one larger per axis than its `rates`
    array (n_bins x n_bins = (n_edges-1) x (n_edges-1)). `positions` are
    bin edges, `rates` are the density at bin centres. Reshaping
    `rates` as (n_samples, n_bins, n_bins) and summing weighted by the
    (uniform) bin widths reproduces the file's own 1D marginals to
    within 0.01%, verified before trusting this integration.

    Also unlike FullPop, there is no `log_prior` column here (PixelPop's
    binned-GP smoothing hyperparameters aren't sampled with an
    informative prior in the same sense), so there is no true MAP, only
    a maximum-likelihood (ML) point, returned as `best_fit_ml`.
    """
    import h5py

    with h5py.File(path, "r") as f:
        g = f["posterior/rates_on_grids"]
        edges = np.unique(g["log_mass_1/positions"][:])
        n_bins = len(edges) - 1
        d_logm = np.diff(edges).mean()

        rates = g[f"{grid_name}/rates"][:]
        n_samples = rates.shape[0]
        rates_3d = rates.reshape(n_samples, n_bins, n_bins)
        total_rate = rates_3d.sum(axis=(1, 2)) * d_logm * d_logm

        names = list(f.attrs["hyperparameters"])
        idx = {n: i for i, n in enumerate(names)}
        log_likelihood = f["posterior/hyperparameter_samples"][:, idx["log_likelihood"]]

    best_i = np.argmax(log_likelihood)
    lower_5, median, upper_95 = np.percentile(total_rate, [5, 50, 95])
    return {
        "best_fit_ml": total_rate[best_i],
        "median": median,
        "lower_5": lower_5,
        "upper_95": upper_95,
    }


#: label -> (path, kind, rate_column). kind picks which compute_rate_summary_*
#: function to use; rate_column is only meaningful for kind="fullpop".
RATE_SOURCES = {
    "GWTC-4.0 FullPop": (
        str(DATA_DIR / "raw" / ZENODO_FILENAME_GWTC4),
        "fullpop",
        "rate_full",
    ),
    "GWTC-5.0 FullPop": (
        str(DATA_DIR / "raw" / ZENODO_FILENAME_GWTC5),
        "fullpop",
        "rate",
    ),
    "GWTC-5.0 PixelPop": (
        str(DATA_DIR / "raw" / ZENODO_FILENAME_PIXELPOP),
        "pixelpop",
        None,
    ),
}


#: Per-class merger rates as published in GWTC-5.0's results paper
#: (https://arxiv.org/abs/2605.27226), Table 2, FullPop and PixelPop
#: rows. Static citations, not derived from any local file, kept here so
#: they exist in exactly one place instead of being copy-pasted into every
#: script that needs them (fullpop_stats.py, detection_rate.ipynb, ...).
PUBLISHED_RATES_TABLE2 = [
    {
        "catalog": "GWTC-5.0 FullPop",
        "population": "BNS",
        "lower": 15.4,
        "mid": 59.3,
        "upper": 154.7,
    },
    {
        "catalog": "GWTC-5.0 FullPop",
        "population": "NSBH",
        "lower": 6.7,
        "mid": 14.2,
        "upper": 26.2,
    },
    {
        "catalog": "GWTC-5.0 FullPop",
        "population": "BBH",
        "lower": 27.5,
        "mid": 36.0,
        "upper": 47.1,
    },
    {
        "catalog": "GWTC-5.0 PixelPop",
        "population": "BNS",
        "lower": 5.2,
        "mid": 23.4,
        "upper": 78.1,
    },
    {
        "catalog": "GWTC-5.0 PixelPop",
        "population": "NSBH",
        "lower": 7.0,
        "mid": 15.9,
        "upper": 32.8,
    },
    {
        "catalog": "GWTC-5.0 PixelPop",
        "population": "BBH",
        "lower": 28.4,
        "mid": 37.5,
        "upper": 49.4,
    },
]


def write_published_rates_table2(cache_csv: str) -> pd.DataFrame:
    """Write PUBLISHED_RATES_TABLE2 to `cache_csv`. Always overwrites,
    since these are static citations, not the output of a slow
    computation, so there's no cache to preserve across runs."""
    df = pd.DataFrame(PUBLISHED_RATES_TABLE2)
    os.makedirs(os.path.dirname(cache_csv) or ".", exist_ok=True)
    df.to_csv(cache_csv, index=False)
    return df


def extract_rate_summary(
    rate_sources: dict[str, tuple[str, str, str | None]],
    cache_csv: str,
    force: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame of rate summaries, one row per label in
    `rate_sources`. Labels already present in `cache_csv` are reused
    unless force=True."""
    cached = pd.DataFrame()
    if not force and os.path.exists(cache_csv):
        cached = pd.read_csv(cache_csv, index_col=0)
        rate_sources = {
            label: v for label, v in rate_sources.items() if label not in cached.index
        }

    if not rate_sources:
        return cached

    rows = {}
    for label, (path, kind, rate_column) in rate_sources.items():
        if kind == "fullpop":
            rows[label] = compute_rate_summary_fullpop(path, rate_column)
        elif kind == "pixelpop":
            rows[label] = compute_rate_summary_pixelpop(path)
        else:
            raise ValueError(f"unknown kind {kind!r} for label {label!r}")

    new_df = pd.DataFrame(rows).T
    df = pd.concat([cached, new_df]) if not cached.empty else new_df
    df.index.name = "label"
    os.makedirs(os.path.dirname(cache_csv) or ".", exist_ok=True)
    df.to_csv(cache_csv)
    return df


PARAMS_INFO = {
    "alpha_1": (
        r":math:`\alpha_1`",
        r"Power-law exponent for masses below :math:`m_{\mathrm{max,NS}}`",
    ),
    "alpha_2": (
        r":math:`\alpha_2`",
        r"Power-law exponent for masses above :math:`m_{\mathrm{min,BH}}`",
    ),
    "alpha_dip": (r":math:`\alpha_d`", r"Power-law exponent within the NS–BH mass gap"),
    "NSmin": (
        r":math:`m_{\mathrm{min,NS}}`",
        r"Minimum neutron star mass (:math:`M_\odot`)",
    ),
    "NSmax": (
        r":math:`\gamma_{\mathrm{low},1}`",
        r"Maximum neutron star mass (:math:`M_\odot`)",
    ),
    "BHmin": (
        r":math:`\gamma_{\mathrm{high},1}`",
        r"Minimum black hole mass (:math:`M_\odot`)",
    ),
    "BHmax": (
        r":math:`m_{\mathrm{max,BH}}`",
        r"Maximum black hole mass (:math:`M_\odot`)",
    ),
    "A": (
        r":math:`\mathrm{A}`",
        r"Depth of primary mass gap suppression",
    ),
    "UPPERmin": (
        r":math:`\gamma_{\mathrm{low},2}`",
        r"Lower boundary of pair-instability gap (:math:`M_\odot`)",
    ),
    "UPPERmax": (
        r":math:`\gamma_{\mathrm{high},2}`",
        r"Upper boundary of pair-instability gap (:math:`M_\odot`)",
    ),
    "mu1": (
        r":math:`\mu_{\mathrm{peak},1}`",
        r"Mean of primary Gaussian peak (:math:`M_\odot`)",
    ),
    "sig1": (
        r":math:`\sigma_{\mathrm{peak},1}`",
        r"Std. dev. of primary Gaussian peak (:math:`M_\odot`)",
    ),
    "mix1": (r":math:`\mathrm{c}_1`", r"Mixing fraction of primary Gaussian peak"),
    "mu2": (
        r":math:`\mu_{\mathrm{peak},2}`",
        r"Mean of secondary Gaussian peak (:math:`M_\odot`)",
    ),
    "sig2": (
        r":math:`\sigma_{\mathrm{peak},2}`",
        r"Std. dev. of secondary Gaussian peak (:math:`M_\odot`)",
    ),
    "mix2": (
        r":math:`\mathrm{c}_2`",
        r"Mixing fraction of secondary Gaussian peak",
    ),
    "absolute_mmin": (r":math:`m_\mathrm{abs,min}`", r"Absolute minimum truncation"),
    "absolute_mmax": (r":math:`m_\mathrm{abs,max}`", r"Absolute maximum truncation"),
    "n0": (
        r":math:`\eta_0`",
        r"Sharpness of low-mass truncation",
    ),
    "n5": (
        r":math:`\eta_5`",
        r"Sharpness of high-mass truncation",
    ),
    "n1": (
        r":math:`\eta_1`",
        r"Sharpness at :math:`m_{\mathrm{max,NS}}`",
    ),
    "n2": (
        r":math:`\eta_2`",
        r"Sharpness at :math:`m_{\mathrm{min,BH}}`",
    ),
    "n3": (
        r":math:`\eta_3`",
        r"Sharpness at :math:`\gamma_{\mathrm{low},2}`",
    ),
    "n4": (
        r":math:`\eta_4`",
        r"Sharpness at :math:`\gamma_{\mathrm{high},2}`",
    ),
}


# label -> source file, downloaded into data/raw/ under its official name
# if missing. Add an entry here (e.g. a future GWTC-6.0 file) and only
# that one gets fetched/loaded on the next run.
SOURCES = {
    "GWTC-4.0 (popsummary)": str(DATA_DIR / "raw" / ZENODO_FILENAME_GWTC4),
    "GWTC-5.0 (popsummary)": str(DATA_DIR / "raw" / ZENODO_FILENAME_GWTC5),
}


def main(force: bool = False) -> None:
    """Rebuild data/derived/ from data/raw/, fetching whatever's missing
    from DCC/Zenodo. force=False (default) reuses existing cache entries;
    force=True re-extracts everything from data/raw/, downloading first
    if needed."""
    download_and_extract_from_zenodo_tarball(
        ZENODO_RECORD_GWTC4,
        ZENODO_ARCHIVE_GWTC4,
        ZENODO_FILENAME_GWTC4,
        Path(SOURCES["GWTC-4.0 (popsummary)"]),
    )
    download_and_extract_from_zenodo_tarball(
        ZENODO_RECORD_GWTC5,
        ZENODO_ARCHIVE_GWTC5,
        ZENODO_FILENAME_GWTC5,
        Path(SOURCES["GWTC-5.0 (popsummary)"]),
    )
    download_and_extract_from_zenodo_tarball(
        ZENODO_RECORD_GWTC5,
        ZENODO_ARCHIVE_GWTC5,
        ZENODO_MEMBER_PIXELPOP,
        Path(RATE_SOURCES["GWTC-5.0 PixelPop"][0]),
    )

    hyperparams_csv = DERIVED_DIR / "hyperparams_map.csv"
    hyperparams_map = extract_hyperparams_map(
        SOURCES, cache_csv=str(hyperparams_csv), force=force
    )
    print(f"Hyperparameters MAP cache: {hyperparams_csv}")

    rate_summary_csv = DERIVED_DIR / "rate_summary.csv"
    extract_rate_summary(RATE_SOURCES, cache_csv=str(rate_summary_csv), force=force)
    print(f"Rate summary cache: {rate_summary_csv}")

    published_rates_csv = DERIVED_DIR / "published_rates_table2.csv"
    write_published_rates_table2(str(published_rates_csv))
    print(f"Published Table 2 rates cache: {published_rates_csv}")

    # Joint mass-rate grids. Every current SOURCES entry is a popsummary
    # file, so all of them carry grids.
    grid_cache_paths = extract_joint_grids(
        SOURCES, cache_dir=str(DERIVED_DIR), force=force
    )
    for label, npz_path in grid_cache_paths.items():
        print(f"Joint grids cache [{label}]: {npz_path}")

    # Table below is built from the GWTC-5.0 MAP.
    maxp_samp = hyperparams_map.loc["GWTC-5.0 (popsummary)"]

    rows = []
    for key, (param, desc) in PARAMS_INFO.items():
        if key in maxp_samp:
            val = maxp_samp[key]
            rows.append((f"{param}", desc, f"{val:.3g}"))

    df = pd.DataFrame(rows, columns=["Parameter", "Description", "Value"])

    title = "Hyperparameters of the FullPop model (GWTC-5.0)"

    # Saved next to this script, wherever it's invoked from.
    with open(SCRIPT_DIR / "hyperparams_table.rst", "w") as f:
        f.write(to_rst(df, title=title))

    with open(SCRIPT_DIR / "hyperparams_table.tex", "w") as f:
        f.write(to_latex(df, caption=title))

    # Paper-style (booktabs, sectioned) table, built straight from the MAP
    # row rather than the simplified `df` above.
    with open(SCRIPT_DIR / "hyperparams_table_gwtc5.tex", "w") as f:
        f.write(
            to_latex_paper(
                maxp_samp,
                caption="Hyperparameters of the FullPop Distribution Model (GWTC-5.0)",
            )
        )


if __name__ == "__main__":
    main()
