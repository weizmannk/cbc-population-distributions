"""Measure the SNR threshold equivalent to FAR < 1/yr on the LVK injections.

In each run, class and chirp-mass bin, rho_eq is the simulated network SNR
above which the same injections are kept as often as the searches found them.
SNRs are the optimal SNRs of the injection release or, with --snr-scale, those
the simulations would compute for the noise curves of that table (snr_scale.py).
The observing detectors of each injection come from the GWOSC segments.

    python segments.py                          # once
    python measure.py                           # release SNRs
    python measure.py --runs O4a O4b \\
        --snr-scale outputs/snr_scale.csv       # SNRs of the simulations

Outputs, in outputs/. Every name ends in _release or _simulations after the
SNRs it was measured on, so the two tables never share a name and can sit
side by side; the first header line of each file repeats which is which.

    rho_eq_<kind>.csv   rho_eq per run, class and bin, with its bootstrap error
    rho_eq_network_<kind>.csv  the same per observing network, pooled where
                     fewer than MIN_FOUND injections are found in a network
    variants_<kind>.csv  rho_eq of the reference bins for subsets and settings:
                     each search, each network, other FAR thresholds, month,
                     dominant detector, chirp mass, redshift, noise model
    histograms_<kind>.csv  found and missed injections per SNR interval
                     (reference bins, or every bin with --all-histograms)
"""

import argparse
import csv
from pathlib import Path

import injections
import numpy as np
import segments
import snr_scale
import threshold
from config import (
    CHIRP_MASS_EDGES,
    CLASSES,
    FAR_THRESHOLD,
    INJECTIONS,
    OUT_DIR,
    REFERENCE_BINS,
    RUNS,
    provenance,
)
from threshold import (
    PEAK_OFFSET,
    bootstrap_error,
    equivalent_threshold,
    network_snr,
)

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


def rescale(inj, run, path):
    """Optimal SNRs the simulations would compute, from a table of snr_scale.py.

    Each detector's SNR is multiplied by the factor of its month and of the
    detector-frame masses of the injection, before the noise is added. A month
    without a release PSD for a detector takes the nearest month that has one.
    """
    table = snr_scale.read(path)
    month = (
        np.full(len(inj), "O3")
        if run.startswith("O3")
        else np.char.replace(months(inj.gps), "-", "_")
    )
    total_mass = inj.masses.sum(axis=1) * (1 + inj.redshift)
    mass_ratio = inj.masses[:, 1] / inj.masses[:, 0]
    for ifo, snr in inj.optimal_snr.items():
        available = sorted(m for r, d, m in table if (r, d) == (run, ifo))
        if not available:
            raise ValueError(f"{path} has no factor for {run} {ifo}")
        factor = np.empty(len(inj))
        for m in np.unique(month):
            use = m if m in available else snr_scale.nearest(m, available)
            if use != m:
                print(f"{run} {ifo} {m}: no release PSD, factor of {use}")
            selected = month == m
            factor[selected] = table[run, ifo, use](
                total_mass[selected], mass_ratio[selected]
            )
        inj.optimal_snr[ifo] = snr * factor
    return inj


def subsets(inj, state):
    """(group, name, mask) of the subsets of one bin used to test the result."""
    net = segments.network(state)
    for name in np.unique(net):
        yield "network", name, net == name
    month = months(inj.gps)
    for name in np.unique(month):
        yield "month", name, month == name
    # Months restricted to the time when both LIGO detectors observe, so that a
    # fit of rho_eq against the range is not diluted by the single-detector time.
    coincident_hl = state["H1"] & state["L1"]
    for name in np.unique(month[coincident_hl]):
        yield "month HL", name, coincident_hl & (month == name)
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
    # Within a bin of detector-frame chirp mass the signal seen by the search is
    # the same at every redshift; distance enters only through the SNR.
    z = inj.redshift
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
        f"{2 * threshold.NOISE_DRAWS} draws",
        network_snr(inj.optimal_snr, state, rng, draws=2 * threshold.NOISE_DRAWS),
        found,
    )
    add(
        "noise",
        "no peak offset",
        network_snr(inj.optimal_snr, state, rng, offset=0.0),
        found,
    )
    return out


def network_thresholds(state, snr, found, rng, rho, error):
    """rho_eq per observing network of one bin, pooled where too few are found."""
    net = segments.network(state)
    rows = []
    for name in sorted(set(net)):
        mask = net == name
        n_found = int(found[mask].sum())
        pooled = n_found < MIN_FOUND
        value, err = (
            (rho, error)
            if pooled
            else (
                equivalent_threshold(snr[mask], found[mask]),
                bootstrap_error(snr[mask], found[mask], rng),
            )
        )
        rows.append(
            dict(
                network=name,
                n_injections=int(mask.sum()),
                n_found=n_found,
                rho_eq=round(value, 3),
                error=round(err, 3),
                pooled=int(pooled),
            )
        )
    return rows


def histogram(snr, found):
    weight = np.full(snr.shape, 1.0 / snr.shape[1])
    hit = np.broadcast_to(found[:, None], snr.shape)
    n_found, _ = np.histogram(snr[hit], HISTOGRAM_EDGES, weights=weight[hit])
    n_missed, _ = np.histogram(snr[~hit], HISTOGRAM_EDGES, weights=weight[~hit])
    return n_found, n_missed


def histogram_row(run, cls, lo, k, found, missed):
    return dict(
        run=run,
        source_class=cls,
        chirp_mass_min=lo,
        snr_min=round(HISTOGRAM_EDGES[k], 4),
        snr_max=round(HISTOGRAM_EDGES[k + 1], 4),
        found=round(found, 3),
        missed=round(missed, 3),
    )


def write(rows, name, units):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / name, "w", newline="") as f:
        f.writelines(f"# {line}\n" for line in provenance(units))
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
        "--noise-draws",
        type=int,
        default=threshold.NOISE_DRAWS,
        help="noise realizations per injection [default: %(default)s]",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=threshold.BOOTSTRAP,
        help="resamples for the error on rho_eq [default: %(default)s]",
    )
    parser.add_argument(
        "--peak-offset",
        type=float,
        default=PEAK_OFFSET,
        help="peak-search excess added to each detector SNR [default: %(default)s]",
    )
    parser.add_argument(
        "--all-histograms",
        action="store_true",
        help="write histograms.csv for every bin, not only the reference bins",
    )
    parser.add_argument(
        "--snr-scale",
        type=Path,
        help="table of snr_scale.py: measure on the SNRs the simulations would "
        "compute for its noise curves, instead of the release SNRs",
    )
    args = parser.parse_args()

    # The paper's values are the defaults; lowering them gives a quick run,
    # and the command line is written at the top of every output file.
    threshold.NOISE_DRAWS = args.noise_draws
    threshold.BOOTSTRAP = args.bootstrap

    # The two tables have the same columns, so each one states its units in
    # its first header line and carries them in its name.
    if args.snr_scale:
        target = snr_scale.header(args.snr_scale, "target")
        templates = snr_scale.header(args.snr_scale, "simulations")
        units = (
            f"optimal SNRs of the simulations ({templates}) for {target}, "
            f"converted with {args.snr_scale}"
        )
        kind = "simulations"
    else:
        units = "optimal SNRs of the release, monthly reference PSDs"
        kind = "release"

    results, variant_rows, histogram_rows, network_rows = [], [], [], []
    print(f"{'run':4} {'class':5} {'Mc_det':>9} {'found':>13} {'rho_eq':>13}")

    for run in args.runs:
        inj = injections.load(args.injections, run)
        if args.snr_scale:
            inj = rescale(inj, run, args.snr_scale)
        state = segments.observing(inj.gps, run)
        inside = np.any(list(state.values()), axis=0)
        if not inside.all():
            print(f"{run}: {np.sum(~inside)} injections outside the segments, dropped")
        inj = inj.subset(inside)
        state = {ifo: on[inside] for ifo, on in state.items()}

        for class_index, cls in enumerate(CLASSES):
            for bin_index, (lo, hi) in enumerate(
                zip(CHIRP_MASS_EDGES[:-1], CHIRP_MASS_EDGES[1:])
            ):
                # One noise stream per bin, so that a bin does not depend on
                # what was measured before it. RUNS.index keeps a run's
                # thresholds the same whether --runs holds one run or all.
                rng = np.random.default_rng(
                    [args.seed, RUNS.index(run), class_index, bin_index]
                )
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

                snr = network_snr(
                    sub.optimal_snr, sub_state, rng, offset=args.peak_offset
                )
                rho = equivalent_threshold(snr, found)
                n_found = int(found.sum())
                share = threshold.NOISE_DRAWS * n_found
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

                bin_columns = dict(
                    run=run,
                    source_class=cls,
                    chirp_mass_min=lo,
                    chirp_mass_max=hi,
                )
                network_rows.extend(
                    bin_columns | row
                    for row in network_thresholds(
                        sub_state, snr, found, rng, rho, results[-1]["error"]
                    )
                )
                if args.all_histograms and (cls, lo, hi) not in REFERENCE_BINS:
                    histogram_rows.extend(
                        histogram_row(run, cls, lo, k, a, b)
                        for k, (a, b) in enumerate(zip(*histogram(snr, found)))
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
                histogram_rows.extend(
                    histogram_row(run, cls, lo, k, a, b)
                    for k, (a, b) in enumerate(zip(*histogram(snr, found)))
                )

    write(results, f"rho_eq_{kind}.csv", units)
    write(network_rows, f"rho_eq_network_{kind}.csv", units)
    write(variant_rows, f"variants_{kind}.csv", units)
    write(histogram_rows, f"histograms_{kind}.csv", units)


if __name__ == "__main__":
    main()
