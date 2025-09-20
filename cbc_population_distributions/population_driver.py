"""
CBC Population Driver
=====================

This script provides an interface to generate gravitational-wave (GW) event samples
using max a posteriori (MAP) parameters from population inference. It implements the
Broken Power Law + Two Peaks (Gaussian peaks) mass distribution with a power-law
redshift evolution, the default population model of GWTC-4.

The script extracts MAP hyperparameters from ``gwpopulation``-style result files and draws
simulated CBC events under the GWTC-4 population model. Samples are generated in chunks for
memory safety and saved in JSON and HDF5 formats.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table
from bilby.core.result import read_in_result
from bilby.hyper.model import Model
from gwpopulation.models.redshift import PowerLawRedshift
from tqdm import tqdm

from .mass import (
    matter_matters_pairing,
    matter_matters_primary_secondary_independent,
)
from .population_sampler import draw_true_values
from .spin import (
    iid_spin_magnitude_gaussian,
    iid_spin_orientation_gaussian_isotropic,
)

# Configure logger
logger = logging.getLogger(__name__)


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


# ---------------- Drawing ----------------
def sample_max_post(
    hyperparams_file: str,
    outdir: str = "outdir",
    n_samples: int = 1_000_000,
    chunk_size: int = 1_000,
    absolute_mmin: float = 0.5,
    absolute_mmax: float = 350.0,
    z_max: float = 2.3,
    pairing: bool = True,
) -> dict[str, str]:
    """
    Draw samples from MAP hyperparameters of the Broken
    Power Law + Two Peaks population model from GWTC-4.

    Parameters
    ----------
    hyperparams_file : str
        Path to the Bilby result file (.hdf5) with GW hyperparameters posterior samples.
    outdir : str
        Output directory.
    n_samples : int
        Total number of samples to generate.
    chunk_size : int
        Number of events per chunk (for incremental writes).
    absolute_mmin, absolute_mmax : float
        Absolute component-mass bounds.
    z_max : float
        Maximum redshift for PowerLawRedshift.
    pairing : bool
        If True, use the pairing mass model; if False, independent primary/secondary.

    Returns
    -------
    dict[str, str]
        Paths to the final JSON and HDF5 files.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    # --- outputs ---
    out_path = Path(outdir).absolute()
    out_path.mkdir(parents=True, exist_ok=True)
    label = Path(hyperparams_file).stem
    events_prefix = out_path / f"{label}_events_baseline5"

    # --- MAP hyperparameters ---
    # Extract the Maximum a Posteriori (MAP) parameters from a Bilby result.
    # - Load the .hdf5 result file and copy the posterior samples.
    # - Compute a score:
    #     * If the prior is non-uniform, use log_likelihood + log_prior (true MAP).
    #     * If the prior is uniform (constant, as in this case), log_prior adds nothing,
    #       this means Maximum Likelihood (ML) and Maximum A Posteriori (MAP) coincide.
    # - Select the sample that maximizes this score.

    # Load "Broken Power Law + 2 Peaks model" and extract MAP sample
    hyperparams = read_in_result(hyperparams_file)
    maxp_samp = _get_map_sample(hyperparams)

    # Set minimum and maximum allowed masses for the model
    maxp_samp["absolute_mmin"] = absolute_mmin
    maxp_samp["absolute_mmax"] = absolute_mmax
    logger.info(f"[{label}] MAP hyperparameters loaded.")

    # --- model (no caching) ---
    if pairing:
        components = [
            matter_matters_pairing,
            iid_spin_orientation_gaussian_isotropic,
            iid_spin_magnitude_gaussian,
            PowerLawRedshift(z_max=z_max),
        ]

    else:
        components = [
            matter_matters_primary_secondary_independent,
            iid_spin_orientation_gaussian_isotropic,
            iid_spin_magnitude_gaussian,
            PowerLawRedshift(z_max=z_max),
        ]

    model = Model(components, cache=False)
    model.parameters.update(maxp_samp)

    # rng = np.random.default_rng(seed) if seed is not None else None

    # --- chunked sampling ---
    # We split the total number of samples into smaller "chunks" to avoid
    # memory overload and to save intermediate results to disk.
    dfs = []
    n_chunks = int(np.ceil(n_samples / chunk_size))
    logger.info(f"Sampling {n_samples} events in {n_chunks} chunks of {chunk_size}")

    # Loop over each chunk
    for counter in tqdm(range(n_chunks), desc="Simulating CBC events"):
        current_chunk = min(chunk_size, n_samples - counter * chunk_size)
        if current_chunk <= 0:
            break

        # Generate events from the population model
        events_chunk = draw_true_values(
            model=model, vt_model=None, n_samples=current_chunk
        )

        # Save the current chunk immediately as JSON
        # (this prevents memory issues and keeps partial results safe)
        events_chunk.reset_index(drop=True).to_json(
            f"{events_prefix}_{counter + 1}.json", indent=2
        )

        # Keep the chunk in memory for final concatenation
        dfs.append(events_chunk)

    # --- concatenate all chunks + save global outputs ---
    # Save final outputs
    json_file = f"{events_prefix}_all.json"
    h5_file = f"{events_prefix}_all.h5"

    events = pd.concat(dfs).reset_index(drop=True)
    events.to_json(json_file, indent=4)
    Table.from_pandas(events).write(
        h5_file, path="events", overwrite=True, format="hdf5"
    )

    # quick counts
    bns_count = (events["mass_1"] < 3).sum()
    nsbh_count = ((events["mass_2"] < 3) & (events["mass_1"] >= 3)).sum()
    bbh_count = (events["mass_2"] >= 3).sum()
    logger.info(f"BNS={bns_count}  NSBH={nsbh_count}  BBH={bbh_count}")

    logger.info(f"Saved outputs: {json_file}, {h5_file}")
    return {"json": str(json_file), "h5": str(h5_file)}


# ---------------- CLI ----------------
def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="cbc-sample",
        description="Draw CBC population samples from MAP hyperparameters (GWTC-4 distribution).",
    )
    parser.add_argument(
        "--hyperparams-file", required=True, help="Bilby result .hdf5 (MAP source)."
    )
    parser.add_argument("--outdir", default="outdir", help="Output directory.")
    parser.add_argument(
        "--n-samples", type=int, default=1_000_000, help="Number of samples to draw."
    )
    # pairing toggle
    parser.add_argument(
        "--pairing",
        dest="pairing",
        action="store_true",
        help="Use pairing model (default).",
    )
    parser.add_argument(
        "--no-pairing",
        dest="pairing",
        action="store_false",
        help="Use independent mass model.",
    )
    parser.set_defaults(pairing=True)

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000,
        help="Chunk size for incremental writes.",
    )
    parser.add_argument(
        "--absolute-mmin", type=float, default=0.5, help="Absolute min mass."
    )
    parser.add_argument(
        "--absolute-mmax", type=float, default=350.0, help="Absolute max mass."
    )
    parser.add_argument(
        "--z-max", type=float, default=2.3, help="Max redshift for PowerLawRedshift."
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("Starting CBC population sampling...")

    file = Path(args.hyperparams_file)
    if not file.exists():
        raise FileNotFoundError(f"Result file not found: {file}")

    sample_max_post(
        hyperparams_file=str(file),
        outdir=args.outdir,
        n_samples=args.n_samples,
        chunk_size=args.chunk_size,
        absolute_mmin=args.absolute_mmin,
        absolute_mmax=args.absolute_mmax,
        z_max=args.z_max,
        pairing=args.pairing,
    )


if __name__ == "__main__":
    main()
