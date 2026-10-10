# Notebooks

Four notebooks, in the order they are meant to be run. Every path below is
relative to this directory. Each notebook also runs on Colab: its first cell
clones the repository and installs the dependencies, and does nothing locally.

## 1. Build the cached inputs

```
python ../hyperparams/get_hyperparams.py
```

This downloads the GWTC-4.0 result from the LIGO DCC and the GWTC-5.0
popsummary files from Zenodo (record 20292639), then writes
`data/derived/hyperparams_map.csv`, `hyperparams_median.csv`,
`rate_summary.csv` and the two mass-rate grid caches. The notebooks read those
caches only, never the multi-gigabyte source files.

## 2. Download what is not fetched automatically

| Data | Path | DOI or source | Size |
|---|---|---|---|
| GWTC-5.0 sensitivity injections (Essick et al., arXiv:2508.10638) | `data/injections/mixture-semi_o1_o2-...-clipped.hdf` | doi:10.5281/zenodo.19500052 | 1.1 GB |
| Reference PSDs of that release | `data/psds-o1234ab/` | same record, `psds-o1234ab.tar.gz` | 25 MB |
| Significant candidates, O1 to O4b | `data/raw/GWTC-5.0_from_O1-to-O4b.csv` | <https://gwosc.org/eventapi/html/GWTC-5.0/>, exported as CSV | 110 kB |
| Reference sensitivity curves of the simulations | `data/asd/o4b_*_ref.txt` | LIGO-T2500363 (H1, L1), LIGO-T2500388 (V1) | small |
| `SEOBNRv4ROM_v3.0.hdf5`, needed by `lalsimulation` | `$LAL_DATA_PATH` | doi:10.5281/zenodo.14999310 | 2 GB |
| Simulated runs, `injections.dat`, `coincs.dat`, `allsky.dat` | `data/runs/<run>/<model>/` | produced by <https://github.com/lpsinger/observing-scenarios-simulations> | |

`../snr_far_calibration/README.md` gives the download commands for the
injections, the PSDs and the reference curves, and the GWOSC segments that go
with them.

## 3. Run the notebooks

| Order | Notebook | What it does | Inputs |
|---|---|---|---|
| 1 | `quick_start.ipynb` | mass distributions of GWTC-4.0 FullPop-4.0 against GWTC-5.0 FullPop, and the steps of the pipeline | `data/derived`, `data/raw`, `data/asd`, `data/runs` |
| 2 | `cbc_pipeline_step_by_step.ipynb` | the observing-scenarios pipeline by hand, from the population samples to a sky map, one `bayestar` step per cell | population samples, PSDs, `$LAL_DATA_PATH` |
| 3 | `SNR_FAR_detection_criterion.ipynb` | SNR of the found injections against the SNR of the real candidates, per source class | injection file, candidate CSV |
| 4 | `detection_rate.ipynb` | detection rates per class and run, under the network SNR of 8 and under the FAR-equivalent thresholds, with the FullPop redshift weights | `data/derived`, `data/runs`, `../snr_far_calibration/outputs/reference/rho_eq.csv` |

`detection_rate.ipynb` comes last: it reads the caches of step 1, the simulated
runs, and the thresholds measured in `../snr_far_calibration/`. It writes
`../outputs/detection_rates_rho_eq.csv`.

## Models and papers

| Name | What it is | Reference |
|---|---|---|
| FullPop-4.0 | GWTC-4.0 full-spectrum mass model, broken power law with two Gaussian peaks and the lower mass gap | arXiv:2508.18083, Eq. (B24) |
| FullPop | GWTC-5.0 update of the same functional form, new hyperparameters | GWTC-5.0 population paper, Zenodo record 20292639 |
| PixelPop | binned Gaussian process on the joint mass distribution, default parametric models elsewhere | same record, `may26_datarelease/all_cbc_varcut1` |
| Power Law + Dip + Break | the GWTC-3 ancestor of FullPop | Farah et al. (2022), doi:10.3847/1538-4357/ac5f03 |
| `rho_eq` | network SNR threshold equivalent to FAR < 1/yr, per class and chirp-mass bin | LIGO-P2600188, `../snr_far_calibration/` |
