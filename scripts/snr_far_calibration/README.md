# SNR threshold equivalent to a FAR cut

The observing-scenarios simulations count a signal as detected when its network
SNR exceeds 8; the LVK catalogues select candidates with FAR < 1/yr. The two
criteria do not select the same signals, and the gap depends on the mass. This
directory measures, on the LVK sensitivity injections, the network SNR threshold
`rho_eq` that selects as many signals as the FAR cut, per source class and
detector-frame chirp-mass bin, for O3a, O3b, O4a and O4b.

## Method

Classes follow the source-frame component masses: NS below 2.5 Msun, BH from 2.5
to 100 Msun, the upper end of the FullPop mass range. Injections with a heavier
component are left out. Bins are in detector-frame chirp mass, from 0.85 to 300
Msun, a factor 1.7 wide and halved below 2.2 Msun where `rho_eq` varies within a
bin in O4. `measure.py` loops over the runs, classes and bins, and for each one:

1. *found*: the smallest FAR over the searches of the run is below 1/yr
   (`injections.Injections.found`);
2. the network SNR is simulated as in `bayestar-realize-coincs`
   (`threshold.network_snr`): in each detector observing at the injection time
   (GWOSC segments, `segments.observing`), `|rho_opt + n| + 0.1` with `n` a unit
   complex Gaussian, the detector counted only above SNR 1, summed in
   quadrature, over sixteen noise realizations per injection;
3. `rho_eq` is the SNR above which the fraction of simulated values equals the
   found fraction (`threshold.equivalent_threshold`). Equivalently, the found
   injections below `rho_eq` are as many as the missed injections above it.

No population weights enter the measurement. Because the found fraction rises
gradually with SNR, the threshold reproduces the number of FAR-selected signals
exactly only for the SNR distribution of the injections. Bins with fewer than 30
found injections are skipped. A bin is subsampled above 400 000 injections, a
cap no bin reaches. The error on `rho_eq` is the standard deviation over 20000
bootstrap resamples of the injections.

Each bin draws its noise from its own stream, seeded by the run, the class and
the bin index. A bin therefore gives the same threshold whether `--runs` holds
one run or all of them, and adding a variant or a check leaves every published
threshold where it was.

The optimal SNRs are those of the injection release, or with `--snr-scale`
those the simulations would compute (next section).

## SNRs of the simulations

The release computes each optimal SNR with IMRPhenomXPHM from 16 Hz and, for
O4, with the 10% quantile in each frequency bin of the PSDs of the injection's
month (Essick et al. 2025, Sec. V.A). These PSDs are more sensitive than the
data the searches analysed: their mean BNS ranges are 173/180 Mpc (H1/L1) in
O4a and 181/192/60 Mpc (H1/L1/V1) in O4b, against 161/171/53 Mpc for the O4b
reference curves, which reproduce the published O4b sensitivity. The
simulations compute the SNR with other templates, from 25 Hz, and with the noise
curves given to them. A threshold applies only to SNRs computed the same way,
so the thresholds applied to the simulations are measured on the SNRs the
simulations would compute for the injections.

`snr_scale.py` tabulates the factor between the two,

    sigma(simulation template, noise curve) / sigma(release template, release PSD),

for each detector and month, on a grid of detector-frame total mass and mass
ratio (non-spinning, face-on templates; the grid is accurate to 0.5%).
`measure.py --snr-scale` multiplies each optimal SNR by its factor before the
noise is added. The found labels do not change. `checks/release_snr.py` checks
on face-on injections that the release side of the factor reproduces the
optimal SNRs of the release.

The noise curves given to `snr_scale.py` set what the table means.

*   To compare simulations with the counts of a run, give the curves of those
    simulations. The table then selects, on their SNR scale, the sources the
    searches found, whatever the sensitivity of the data was.
*   For forecasts, give the curves that represent the sensitivity of the run
    whose injections are measured. The table is then the threshold of that
    run's searches on SNRs computed with the sensitivity of its data, and
    applies unchanged to simulations with any noise curve: within a run the
    threshold does not follow the sensitivity (`checks/range_fit.py`).

For O4 both are the `o4b_*_ref.txt` curves, so one table serves both uses. New
thresholds need the injections, release PSDs and representative noise curves of
the run they are measured on; a new simulation needs nothing.

## Data

| Path (under `data/`) | Content |
|---|---|
| `injections/mixture-semi_o1_o2-...-clipped.hdf` | GWTC-5.0 sensitivity injections, doi:10.5281/zenodo.19500052, 1.1 GB |
| `psds-o1234ab/` | reference PSDs of that release, for its optimal SNRs, 25 MB packed |
| `segments/` | GWOSC observing segments, written by `segments.py` |
| `asd/o4b_*_ref.txt` | O4b reference curves, needed only by `snr_scale.py` |

To download them, from `data/` at the repository root:

```
mkdir -p injections && cd injections
curl -OL https://zenodo.org/records/19500052/files/mixture-semi_o1_o2-real_o3_o4a_o4b-cartesian_spins_20260410130052UTC-clipped.hdf
cd .. && curl -OL https://zenodo.org/records/19500052/files/psds-o1234ab.tar.gz && tar -xf psds-o1234ab.tar.gz
mkdir -p asd && cd asd
curl -OL https://dcc.ligo.org/T2500363-v1/public/o4b_h1_ref.txt
curl -OL https://dcc.ligo.org/T2500363-v1/public/o4b_l1_ref.txt
curl -OL https://dcc.ligo.org/T2500388-v1/public/o4b_v1_ref.txt
```

Then `python segments.py` writes `data/segments/`. The injection file holds the
O1 and O2 semi-analytic injections as well; the runs measured here come from its
GPS boundaries, so the smaller `mixture-real_o3_o4a_o4b-...` file of the same
record works too, after pointing `config.INJECTIONS` at it.

`CBCPOP_REPO` sets the repository root and `SNRFAR_OUT` the output directory.

## Usage

```
python segments.py        # once, needs gwosc
python measure.py         # outputs/rho_eq.csv, variants.csv, histograms.csv
python sensitivity.py     # outputs/release_ranges.csv
python figure.py          # outputs/snr_threshold.pdf

python snr_scale.py --runs O4a O4b -o outputs/snr_scale.csv \
    --H1 ../../data/asd/o4b_h1_ref.txt \
    --L1 ../../data/asd/o4b_l1_ref.txt \
    --V1 ../../data/asd/o4b_v1_ref.txt
SNRFAR_OUT=outputs/reference python measure.py --runs O4a O4b \
    --snr-scale outputs/snr_scale.csv
```

Requirements: numpy, scipy, h5py, matplotlib; gwosc for the segments;
ligo.skymap for `snr_scale.py`, with `SEOBNRv4ROM_v3.0.hdf5` in
`LAL_DATA_PATH` for the templates above 4 Msun.

`bash run_all.sh` runs every step and check, writes the thresholds of the
reference curves to `outputs/reference/`, and packs `outputs/` with the log
`run_all.log` into `results.zip`.

## Which table to use

Two tables carry `rho_eq` with the same columns, and the first line of each one
names the PSDs its SNRs are expressed with.

| Table | SNRs | Where it belongs |
|---|---|---|
| `outputs/rho_eq.csv` | optimal SNRs of the release, monthly PSDs | the measurement itself, as published in Table 2 of LIGO-P2600188 |
| `outputs/reference/rho_eq.csv` | SNRs of the simulations for `o4b_*_ref.txt`, through `outputs/snr_scale.csv` | applied to the simulations: O4a and O4b with their own rows, IR1 and later runs with the O4b rows (Table 4 and the IR1 forecasts) |

A simulated SNR is never compared with the release table: a threshold has to be
computed the same way as the quantity it cuts. `measure.py` refuses to write
`--snr-scale` results into `outputs/` and release results into a directory
whose name contains `reference`.

## Files

| File | Role |
|---|---|
| `config.py` | paths, runs, bins, constants and their sources |
| `injections.py` | reads one run of the injection file |
| `segments.py` | GWOSC segments and observing detectors |
| `threshold.py` | noise model, `rho_eq` and its bootstrap error |
| `measure.py` | `rho_eq` per run, class and bin, variants and SNR histograms |
| `sensitivity.py` | BNS range of the release PSDs |
| `snr_scale.py` | factors from the release SNRs to those of the simulations |
| `figure.py` | paper figure |
| `checks/coverage.py` | mass ranges per class, injections removed by the 100 Msun cut and outside the bins |
| `checks/closure.py` | recovers a planted threshold from synthetic injections |
| `checks/binning.py` | `rho_eq` in half bins and in bins shifted by half a width |
| `checks/noise_model.py` | measures the peak-search excess, `PEAK_OFFSET` (needs ligo.skymap) |
| `checks/range_fit.py` | fits `rho_eq` against the monthly BNS range within a run |
| `checks/release_snr.py` | reproduces the release SNRs of face-on injections from the release PSDs |
| `CHECKS.md` | every check and the numbers it produced, and the open points |

## Variants

`variants.csv` repeats the measurement for the reference bins (BNS 1.12–1.4,
NSBH 2.2–4, BBH 20–35 Msun) on subsets and with other settings. A variant is
written only where at least 30 injections were found, so the sparse bins and
networks are absent.

| Group | Variants |
|---|---|
| search | GstLAL, MBTA, PyCBC, cWB alone |
| far | 0.01, 0.1, 0.25, 10 per year |
| network | each detector network |
| month | each calendar month |
| dominant | H1- or L1-dominated injections (HL time) |
| chirp mass | terciles of the bin (signal duration) |
| redshift | terciles of the bin |
| noise | thirty-two draws instead of sixteen, no peak offset |

## Where the thresholds are used

`outputs/reference/rho_eq.csv`, written by

```
SNRFAR_OUT=outputs/reference python measure.py --runs O4a O4b \
    --snr-scale outputs/snr_scale.csv
```

is the table the observing-scenarios fork reads, as `thresholds/rho_eq.csv` of
`observing-scenarios-simulations/`, to weight the simulated detections by the
FAR-equivalent threshold instead of the fixed network SNR of 8. That copy drops
`found_below` and `missed_above`, which record the closure of the measurement
and are not needed downstream. Measuring the thresholds again needs the
injection file and the segments, so that fork keeps the table as data.
