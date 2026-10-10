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
   quadrature, over eight noise realizations per injection;
3. `rho_eq` is the SNR above which the fraction of simulated values equals the
   found fraction (`threshold.equivalent_threshold`). Equivalently, the found
   injections below `rho_eq` are as many as the missed injections above it.

Both fractions are taken over the same injections, so the distribution they were
drawn from cancels and no population weights are needed. Bins with fewer than 30
found injections are skipped. A bin is subsampled above 400 000 injections, a
cap no bin reaches. The error on `rho_eq` is the standard deviation over 200
bootstrap resamples of the injections.

The optimal SNRs are those of the injection release. With `--reference`, they
are rescaled by the ratio of BNS ranges to the `o4b_*_ref.txt` curves of the
simulations, before the noise is added. The O4b curves are used for O4a as well,
so that every O4 threshold refers to one sensitivity, the one the simulations
assume. The O3 release PSD is already typical of its run and is left alone.

## Data

| Path (under `data/`) | Content |
|---|---|
| `injections/mixture-semi_o1_o2-...-clipped.hdf` | GWTC-5.0 sensitivity injections, doi:10.5281/zenodo.19500052, 1.1 GB |
| `psds-o1234ab/` | reference PSDs of that release, for its optimal SNRs, 25 MB packed |
| `segments/` | GWOSC observing segments, written by `segments.py` |
| `asd/o4b_*_ref.txt` | reference curves of the simulations, needed only by `--reference` |

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
```

Requirements: numpy, h5py, matplotlib; gwosc for the segments.

`bash run_all.sh` runs every step and check, writes the thresholds of the
reference curves to `outputs/reference/`, and packs `outputs/` with the log
`run_all.log` into `results.zip`.

## Files

| File | Role |
|---|---|
| `config.py` | paths, runs, bins, constants and their sources |
| `injections.py` | reads one run of the injection file |
| `segments.py` | GWOSC segments and observing detectors |
| `threshold.py` | noise model, `rho_eq` and its bootstrap error |
| `measure.py` | `rho_eq` per run, class and bin, variants and SNR histograms |
| `sensitivity.py` | BNS range of the release PSDs |
| `figure.py` | paper figure |
| `checks/coverage.py` | mass ranges per class, injections removed by the 100 Msun cut and outside the bins |
| `checks/closure.py` | recovers a planted threshold from synthetic injections |
| `checks/binning.py` | `rho_eq` in half bins and in bins shifted by half a width |
| `checks/noise_model.py` | measures the peak-search excess, `PEAK_OFFSET` (needs ligo.skymap) |
| `checks/range_fit.py` | fits `rho_eq` against the monthly BNS range within a run |

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
| noise | sixteen draws instead of eight, no peak offset, detectors drawn from the duty cycles |

## Where the thresholds are used

`outputs/reference/rho_eq.csv`, written by

```
SNRFAR_OUT=outputs/reference python measure.py --runs O4a O4b --reference ../../data/asd
```

is the table the observing-scenarios fork reads, as `thresholds/rho_eq.csv` of
`observing-scenarios-simulations/`, to weight the simulated detections by the
FAR-equivalent threshold instead of the fixed network SNR of 8. That copy drops
`found_below` and `missed_above`, which record the closure of the measurement
and are not needed downstream. Measuring the thresholds again needs the
injection file and the segments, so that fork keeps the table as data.
