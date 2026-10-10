"""Measure the SNR threshold equivalent to FAR < 1/yr on the LVK injections.

In each run, class and chirp-mass bin, rho_eq is the simulated network SNR
above which the same injections are kept as often as the searches found them.
SNRs are those of the injection release or, with --reference, those of the
reference curves of the simulations. The observing detectors of each injection
come from the GWOSC segments.

    python segments.py                          # once
    python measure.py                           # release SNRs
    python measure.py --reference ../../data/asd    # SNRs of the o4b_*_ref.txt curves

Outputs, in outputs/:
    rho_eq.csv       rho_eq per run, class and bin, with its bootstrap error
    variants.csv     rho_eq of the reference bins for subsets and settings:
                     each search, each network, other FAR thresholds, month,
                     dominant detector, chirp mass, redshift, noise model
    histograms.csv   found and missed injections per SNR interval (reference bins)
"""

import argparse
import csv
from pathlib import Path

import injections
import numpy as np
import segments
import sensitivity
from config import (
    CHIRP_MASS_EDGES,
    CLASSES,
    DUTY_CYCLES,
    FAR_THRESHOLD,
    INJECTIONS,
    OUT_DIR,
    REFERENCE_BINS,
    RUN_DETECTORS,
    RUNS,
)
from threshold import NOISE_DRAWS, bootstrap_error, equivalent_threshold, network_snr

MAX_PER_BIN = 400_000  # above the largest bin: every injection is used
MIN_FOUND = 30
FAR_GRID = (0.01, 0.1, 0.25, 10.0)
HISTOGRAM_EDGES = np.geomspace(4.0, 32.0, 46)
GPS_EPOCH_UTC = np.datetime64("1980-01-06T00:00:00") - np.timedelta64(18, "s")


def months(gps):
    return (
        (GPS_EPOCH_UTC + gps.astype("timedelta64[s]"))
        .astype("datetime64[M]")
        .astype(str)
    )


def to_reference(inj, run, directory):
    """Optimal SNRs for the reference curves of the simulations (O4 runs only).

    For a given source the SNR is proportional to the range of the PSD, so the
    SNR of each detector is multiplied by range(reference curve) / range(release
    PSD of the injection's month). This acts on the optimal SNR, before the
    noise is added. The O4b curves are used for O4a too, so that every O4
    threshold refers to the sensitivity assumed by the simulations. The O3
    release PSD is already typical of the run.
    """
    if run.startswith("O3"):
        return inj
    release = sensitivity.month_ranges()
    month = np.char.replace(months(inj.gps), "-", "_")
    for ifo, snr in inj.optimal_snr.items():
        reference = sensitivity.reference_range(
            directory / f"o4b_{ifo.lower()}_ref.txt"
        )
        fallback = np.mean(
            [r for (d, m), r in release.items() if d == ifo and m != "O3"]
        )
        factor = np.array([reference / release.get((ifo, m), fallback) for m in month])
        inj.optimal_snr[ifo] = snr * factor
    return inj


def duty_cycle_state(run, n, rng):
    networks = list(DUTY_CYCLES[run])
    weights = np.array(list(DUTY_CYCLES[run].values()))
    pick = rng.choice(len(networks), (n, NOISE_DRAWS), p=weights / weights.sum())
    return {
        ifo: np.isin(pick, [k for k, net in enumerate(networks) if ifo in net])
        for ifo in RUN_DETECTORS[run]
    }


def subsets(inj, state):
    """(group, name, mask) of the subsets of one bin used to test the result."""
    net = segments.network(state)
    for name in np.unique(net):
        yield "network", name, net == name
    month = months(inj.gps)
    for name in np.unique(month):
        yield "month", name, month == name
    coincident = np.isin(net, ["H1L1", "H1L1V1"])
    louder = inj.optimal_snr["H1"] > inj.optimal_snr["L1"]
    yield "dominant", "H1", coincident & louder
    yield "dominant", "L1", coincident & ~louder
    terciles = np.quantile(inj.chirp_mass_det, [0, 1 / 3, 2 / 3, 1])
    for k in range(3):
        name = f"{terciles[k]:.2f}-{terciles[k + 1]:.2f}"
        inside = (inj.chirp_mass_det >= terciles[k]) & (
            inj.chirp_mass_det <= terciles[k + 1]
        )
        yield "chirp mass", name, inside
    # Redshift from the detector- and source-frame chirp masses. Within a bin of
    # detector-frame chirp mass the signal seen by the search is the same at
    # every redshift; distance enters only through the SNR.
    source_chirp = (inj.masses[:, 0] * inj.masses[:, 1]) ** 0.6 / inj.masses.sum(
        1
    ) ** 0.2
    z = inj.chirp_mass_det / source_chirp - 1
    terciles = np.quantile(z, [0, 1 / 3, 2 / 3, 1])
    for k in range(3):
        name = f"{terciles[k]:.3f}-{terciles[k + 1]:.3f}"
        yield "redshift", name, (z >= terciles[k]) & (z <= terciles[k + 1])


def variants(inj, state, snr, found, run, far, rng):
    """rho_eq of one bin under each variant: {(group, name): (rho_eq, n_found)}."""
    out = {}

    def add(group, name, values, selected):
        if selected.sum() >= MIN_FOUND:
            out[group, name] = (
                equivalent_threshold(values, selected),
                int(selected.sum()),
            )

    for family in injections.SEARCHES:
        add("search", family, snr, inj.found(far, family))
    for value in FAR_GRID:
        add("far", f"{value:g}", snr, inj.found(value))
    for group, name, mask in subsets(inj, state):
        add(group, name, snr[mask], found[mask])
    add(
        "noise",
        f"{2 * NOISE_DRAWS} draws",
        network_snr(inj.optimal_snr, state, rng, draws=2 * NOISE_DRAWS),
        found,
    )
    add(
        "noise",
        "no peak offset",
        network_snr(inj.optimal_snr, state, rng, offset=0.0),
        found,
    )
    if run in DUTY_CYCLES:
        drawn = duty_cycle_state(run, len(inj), rng)
        add("noise", "duty cycle", network_snr(inj.optimal_snr, drawn, rng), found)
    return out


def histogram(snr, found):
    weight = np.full(snr.shape, 1.0 / snr.shape[1])
    hit = np.broadcast_to(found[:, None], snr.shape)
    n_found, _ = np.histogram(snr[hit], HISTOGRAM_EDGES, weights=weight[hit])
    n_missed, _ = np.histogram(snr[~hit], HISTOGRAM_EDGES, weights=weight[~hit])
    return n_found, n_missed


def write(rows, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_DIR / name}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--injections", default=INJECTIONS)
    parser.add_argument("--runs", nargs="+", default=list(RUNS))
    parser.add_argument("--far", type=float, default=FAR_THRESHOLD)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="directory with o4b_{h1,l1,v1}_ref.txt (ASD): express the SNRs "
        "for these curves instead of the release PSDs",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    results, variant_rows, histogram_rows = [], [], []
    print(f"{'run':4} {'class':5} {'Mc_det':>9} {'found':>13} {'rho_eq':>13}")

    for run in args.runs:
        inj = injections.load(args.injections, run)
        if args.reference:
            inj = to_reference(inj, run, args.reference)
        state = segments.observing(inj.gps, run)
        inside = np.any(list(state.values()), axis=0)
        if not inside.all():
            print(f"{run}: {np.sum(~inside)} injections outside the segments, dropped")
        inj = inj.subset(inside)
        state = {ifo: on[inside] for ifo, on in state.items()}

        for cls in CLASSES:
            for lo, hi in zip(CHIRP_MASS_EDGES[:-1], CHIRP_MASS_EDGES[1:]):
                index = np.flatnonzero(
                    (inj.source_class == cls)
                    & (inj.chirp_mass_det >= lo)
                    & (inj.chirp_mass_det < hi)
                )
                if len(index) > MAX_PER_BIN:
                    index = np.sort(rng.choice(index, MAX_PER_BIN, replace=False))
                sub = inj.subset(index)
                sub_state = {ifo: on[index] for ifo, on in state.items()}
                found = sub.found(args.far)
                if found.sum() < MIN_FOUND:
                    continue

                snr = network_snr(sub.optimal_snr, sub_state, rng)
                rho = equivalent_threshold(snr, found)
                n_found = int(found.sum())
                share = NOISE_DRAWS * n_found
                results.append(
                    dict(
                        run=run,
                        source_class=cls,
                        chirp_mass_min=lo,
                        chirp_mass_max=hi,
                        n_injections=len(sub),
                        n_found=n_found,
                        rho_eq=round(rho, 3),
                        error=round(bootstrap_error(snr, found, rng), 3),
                        found_below=round(
                            np.sum(found[:, None] & (snr < rho)) / share, 4
                        ),
                        missed_above=round(
                            np.sum(~found[:, None] & (snr >= rho)) / share, 4
                        ),
                    )
                )
                print(
                    f"{run:4} {cls:5} {lo:>5g}-{hi:<5g} {n_found:6d}/{len(sub):<6d} "
                    f"{rho:6.2f} ± {results[-1]['error']:.2f}"
                )

                if (cls, lo, hi) not in REFERENCE_BINS:
                    continue
                for (group, name), (value, count) in variants(
                    sub, sub_state, snr, found, run, args.far, rng
                ).items():
                    variant_rows.append(
                        dict(
                            run=run,
                            source_class=cls,
                            chirp_mass_min=lo,
                            chirp_mass_max=hi,
                            group=group,
                            variant=name,
                            rho_eq=round(value, 3),
                            n_found=count,
                        )
                    )
                for k, (a, b) in enumerate(zip(*histogram(snr, found))):
                    histogram_rows.append(
                        dict(
                            run=run,
                            source_class=cls,
                            chirp_mass_min=lo,
                            snr_min=round(HISTOGRAM_EDGES[k], 4),
                            snr_max=round(HISTOGRAM_EDGES[k + 1], 4),
                            found=round(a, 3),
                            missed=round(b, 3),
                        )
                    )

    write(results, "rho_eq.csv")
    write(variant_rows, "variants.csv")
    write(histogram_rows, "histograms.csv")


if __name__ == "__main__":
    main()
