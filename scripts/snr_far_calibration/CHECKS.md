# Checks

Every check run on the measurement, with the numbers it produced. Steps 1 to 10
follow the order of `run_all.sh`; the open points at the end are the questions
this directory does not settle. Commands are run from
`scripts/snr_far_calibration/`.

## Step 1 — reading the modules

| File | Role | Deviation from the definitions |
|---|---|---|
| `config.py` | paths, run boundaries, bins, class limits | O4b starts at GPS 1396969218, two days after the official O4b start (1396796418); checked below, this is the release boundary |
| `injections.py` | reads one run, classifies, applies the 100 Msun cut | none; `found()` takes the minimum FAR over the run's searches, `PyCBC` covering both O3 PyCBC searches |
| `segments.py` | downloads GWOSC segments, tells which detectors observed | none |
| `threshold.py` | noise model, rho_eq, bootstrap error | none; `|rho_opt + n| + 0.1`, detector kept above 1, quadrature sum, 16 draws |
| `measure.py` | rho_eq per run, class and bin, variants, histograms | none; the bins are in detector-frame chirp mass |
| `sensitivity.py` | BNS range of the release PSDs | none |
| `figure.py` | the four-panel figure | see step 9 |
| `checks/coverage.py` | classes, mass cut, injections outside the bins | none |
| `checks/closure.py` | recovers a planted threshold | none |
| `checks/binning.py` | rho_eq in half bins and shifted bins | none |
| `checks/noise_model.py` | peak-search excess, needs LAL | none |
| `run_all.sh` | every step, then `results.zip` | none |

Definitions as implemented: classes from the source-frame masses with the NS
limit at 2.5 Msun (`classify`, strict `<`), components above 100 Msun removed
(`load`), bins in `(1 + z) * Mc_source`, detection at FAR < 1/yr, noise as in
`bayestar-realize-coincs`, `rho_eq = quantile(snr, 1 - found.mean())`.

## Step 2 — `python segments.py`

All segment files already present, nothing downloaded. Live time against the
published duty cycles:

| Run | Calendar | H1 | L1 | V1 | Published |
|---|---|---|---|---|---|
| O3a | 183.0 d | 71.0% | 75.6% | 76.1% | — |
| O3b | 147.1 d | 78.4% | 77.2% | 75.5% | — |
| O4a | 237.0 d | 67.4% | 69.0% | — | 67.5 / 69 (arXiv:2508.18079) |
| O4b | 291.1 d | 48.0% | 67.7% | 71.0% | 48.6 / 68.1 / 70.8 (arXiv:2605.27090) |

`O4a_V1.txt` is absent by design (Virgo did not observe in O4a). Status: OK.

The O4b window is 291.1 d against the 293.1 d of the paper, the two days being
the difference in start GPS. Checked against the release: the injections with a
finite `o4b_*` FAR span [1396969219, 1422112885] and there are **zero**
injections between the official start and the configured one, for every run.
The boundaries are those of the release and nothing is lost. Status: OK.

## Step 3 — `python checks/coverage.py`

`outside (found)` is `0 (0)` on all twelve rows. Boundaries re-checked exactly
rather than on the rounded print: BNS heavier < 2.5, NSBH lighter < 2.5 <=
heavier, BBH lighter >= 2.5, heavier <= 100 on every run. Lightest component
1.0000 Msun. Status: OK.

The 100 Msun cut removes **88 309 injections (59 389 of them found)**, all BBH:
12 645 / 11 599 / 31 354 / 32 711 for O3a / O3b / O4a / O4b.

## Step 4 — `python sensitivity.py`

| Run | H1 | L1 | V1 | Expected |
|---|---|---|---|---|
| O3a, O3b | 107 | 130 | 47 | 107 / 130 / 47 |
| O4a | 173 | 180 | — | ~173 / ~180 |
| O4b | 181 | 192 | 60 | ~181 / ~192 / ~60 |

All eleven values on target. HL network range 168 / 168 / 249 / 264 Mpc.
Status: OK.

The release has no V1 PSD for 2024_04, so the O4b V1 mean is over nine months;
it does have `psd-O4-2024_03_v1-V`, which falls in neither run window and is
not used by `psd_files`.

## Step 5 — `python measure.py`

70 rows, 7.5 s. Independent audit (own script, not the project code):

| Check | Result |
|---|---|
| `found_below == missed_above` on every row | OK, 70/70 (range 0.093-0.289) |
| `n_found >= 30` | OK, smallest 47 (O4b NSBH 4-7) |
| error < 0.5 | OK, largest 0.456 (O4b BNS 0.85-1.12, 171 found) |
| all `TABLE_BINS` x all runs | OK, 56/56 rows |
| bin edges from `CHIRP_MASS_EDGES` | OK |

BBH 20-35, against the values before the change:

| Run | Before | Now | Difference |
|---|---|---|---|
| O3a | 8.48 | 8.464 | -0.02 |
| O3b | 8.33 | 8.324 | -0.01 |
| O4a | 9.85 | 9.844 | -0.01 |
| O4b | 9.98 | 9.982 | +0.00 |

All within 0.02, nothing to explain. BBH 120-300 in O3 falls from 12.3 / 11.7
to **9.460 (O3a) and 9.119 (O3b)**, as expected from the 100 Msun cut, which
removes the heavy injections dominating that bin. O4 values there: 9.758 and
9.582.

Second, independent recomputation of the fourteen table bins for O4a and O3a:
own noise generator, own reading of the segments and the classes, and the
threshold found by bisection on `#(found below rho) - #(missed above rho)`
instead of a quantile. Largest deviation 0.124, largest 0.79 sigma of the
bootstrap error; 24 of the 28 bins agree to better than 0.03. The quantile and
the count-equality definitions coincide, and the measurement reproduces under a
different implementation. Status: OK.

`variants.csv`: 4 searches (3 where cWB finds fewer than 30, as expected for
BNS), FAR grid 0.01 / 0.1 / 0.25 / 10, every network of each run, 5-10 calendar
months per run, both dominant detectors, three chirp-mass terciles, and the
three noise variants. rho_eq falls monotonically as the cut is loosened
(O4b BBH 20-35: 11.05, 10.56, 10.34, [9.98 at FAR < 1], 9.26). Status: OK.

`histograms.csv`: the sums reproduce 94-98% of `n_found` and of `n_injections`
of `rho_eq.csv`, the rest being the tails outside the 4-32 range of
`HISTOGRAM_EDGES`. Status: OK.

## Step 6 — reference curves

`SNRFAR_OUT=outputs/reference python measure.py --runs O4a O4b --reference ../../data/asd`.

The conversion acts on the optimal SNR, before the noise, checked by tracing the
array handed to `network_snr`: the largest O4b optimal SNR goes from 608.3 to
538.1 in H1, a factor 0.885 against the range ratio 0.888. The reference curves
reproduce the published O4b sensitivities (H1 160.9, L1 171.1, V1 53.0 Mpc
against 160 / 170 / 53). rho_eq drops by the same factor, e.g. BBH 20-35 in O4b
from 9.98 to 8.94 (0.896). Status: OK.

## Step 7 — closure and binning

`checks/closure.py`: planted 8.0 / 10.0 / 12.0 recovered as **8.01 / 10.05 /
12.03**, all within 0.1. The "smallest SNR among the found", the usual
shortcut, gives 2.09 / 4.29 / 6.90 and is strongly biased. Status: OK.

The same script then applies the found fraction p(rho) of one bin to a
population with dN/drho proportional to rho^-4, instead of cutting at rho_eq.
Over the whole population the soft selection finds 1.86 to 2.53 times more than
the cut, because it keeps a long tail below rho_eq. Restricted to SNR >= 8,
which is all the simulations hold, the ratio is 0.97 (O4b BBH 20-35), 1.02
(O4a BBH 20-35), 1.14 (O4b NSBH 2.2-4) and 1.35 (O4a BNS 1.12-1.4). Status: OK,
and the same quantity measured on the simulations themselves is in section 7.1
of `detection_rate.ipynb`.

`checks/binning.py`: 61 bins have both halves. Median difference between the
two halves **0.12**; 13 bins above 0.25. Ranked by significance:

| Run | Class | Bin | Lower | Upper | Difference | sigma |
|---|---|---|---|---|---|---|
| O4a | BBH | 20-35 | 9.98 | 9.76 | 0.22 | 10.0 |
| O3a | BBH | 60-120 | 8.82 | 9.15 | 0.33 | 7.8 |
| O3b | BBH | 60-120 | 8.50 | 8.86 | 0.36 | 7.2 |
| O4b | BBH | 120-300 | 9.59 | 9.27 | 0.32 | 3.9 |
| O4a | NSBH | 1.4-1.75 | 10.53 | 11.38 | 0.85 | 2.9 |
| O4a | BBH | 120-300 | 9.76 | 9.49 | 0.27 | 2.2 |

Status: OK, with a reservation reported as an open point. The low-mass
differences are large in value (0.26-0.51 for BNS below 2.2 Msun) but only
1-2 sigma, while the statistically strong gradients sit in the BBH bins, which
are not halved.

## Step 8 — noise model

`LAL_DATA_PATH=~/lalsuite-waveform-data python checks/noise_model.py
../../data/runs/{O4a,O4b}/psds.xml --detector L1`. Both runs produce output.

| Masses | rho_opt | bayestar | abs(rho+n) | Excess |
|---|---|---|---|---|
| 1.5+1.5 | 6 / 8 | 6.191 / 8.133 | 6.084 / 8.067 | +0.106 / +0.067 |
| 10+1.5 | 6 / 8 | 6.225 / 8.161 | 6.084 / 8.060 | +0.140 / +0.101 |
| 30+25 | 6 / 8 | 6.211 / 8.161 | 6.086 / 8.059 | +0.125 / +0.102 |

Excess between +0.067 and +0.140, inside the expected +0.05 to +0.15, with
`PEAK_OFFSET = 0.1` in the middle. Status: OK.

O4a and O4b give identical numbers because the L1 PSD is the same in the two
files: `read_psd_xmldoc` returns `['H1', 'L1']` for O4a and `['V1', 'H1', 'L1']`
for O4b, the H1 and L1 entries being identical. The file-size difference is the
extra V1 block. Not a defect.

Why the earlier log had no O4b section: no previous `run_all.log` is kept in the
repository and `data/runs/` is gitignored, so the earlier state cannot be
recovered. Both psds.xml were written on 2 October three minutes apart
(O4a 22:01, O4b 22:04), which fits a run made before the O4b file existed. The
real defect is that `run_all.sh` skipped the step without saying so; fixed
(correction 1).

## Step 9 — figure

`python figure.py`, then reading `outputs/snr_threshold.png` panel by panel.
Every number re-derived from the CSVs:

| Check | Result |
|---|---|
| (a) rho_eq on the top axis | 11.6; `rho_eq.csv` gives 11.512 (O4a) and 11.727 (O4b), mean 11.6195 |
| (a) "found below" = "missed above" | 0.1322 and 0.1314 for both quantities, mean 13% in both legend entries |
| (b) markers on the curves | the six markers fall inside the span of their curve, interpolated fraction 0.486-0.552 |
| (c) every `TABLE_BINS` x run | 14 bins x 4 runs, none missing |
| (c) "adopted" bars | recomputed as the O4a/O4b mean for the 14 bins, all matching |
| (d) ratios | recomputed from `rho_eq.csv` and `variants.csv`, 1.119-1.321, matching the points |
| (d) grey band | network ranges O3a 168.5, O4a 249.3, O4b 263.6 Mpc, band [1.480, 1.565] |

Layout: 7.1 x 5.6 in, fonts 7.5-9 pt, no text, panel label or legend over the
data or over another panel, the top legend fitting on one row.

Panel (c) was corrected for legibility (correction 3). Remaining cosmetic
remark, left alone: in (b) the O3 BNS marker (9.04, 0.55) is partly behind the
O3 NSBH marker (9.11, 0.53).

## Step 10 — `bash run_all.sh`

10 m 47 s with the three `checks/noise_model.py` steps running, 11 sections in
`outputs/run_all.log`, `results.zip` 323 540 bytes. The four CSVs of `outputs/`
and the three of `outputs/reference/` are **byte-identical** between two
independent runs, one launched from a terminal and one from a script, which is
what the per-bin noise stream of `measure.py` guarantees: each bin draws from a
stream seeded by the run, the class and the bin index, so a bin gives the same
threshold whether `--runs` holds one run or all of them, and adding a variant
leaves every other bin untouched. Status: OK.

`run_all.sh` now also runs `checks/low_snr_found.py` and `checks/range_fit.py`
with and without `--hl-only`, and passes `--all-histograms` to the reference
step, which adds three sections and the histograms of every bin.

## Step 11 — selection consistency and the simulation PSDs

**The found rule.** Here an injection is found when the smallest FAR over the
searches of the run falls below 1/yr: four searches in O4 (`gstlal`, `mbta`,
`pycbc`, `cwb-bbh`) and five in O3, PyCBC appearing twice. The candidates the
paper counts come from the `far` column of the GWOSC event list, which carries
the FAR of the preferred pipeline of each event, not the minimum over all of
them. Applying `far < 1/yr` to that list gives 88 O4a candidates (86 BBH, 1
NSBH, 1 without source masses, GW230630, attributed to noise) and 104 O4b
candidates, all BBH, so **190 O4 BBH**, the number the paper compares against.
The threshold and its value agree; two differences remain, both making the
injection rule slightly the more permissive: the minimum over four searches
against one preferred pipeline, and the absence of the data-quality and
engineering-run vetoes that the catalogue applies by hand. Status: OK, with that
caveat stated.

**The simulation PSDs.** Read from `data/runs/O4a/psds.xml` and
`data/runs/O4b/psds.xml`, the BNS ranges are H1 160.6 and L1 171.2 Mpc in
**both** files, against 160.9 and 171.1 Mpc for `o4b_h1_ref.txt` and
`o4b_l1_ref.txt`, a difference of 0.2% from the 1 Hz grid of `pack-psds.py`.
O4b adds V1 at 54.1 Mpc against 53.0 for `o4b_v1_ref.txt`. The O4a simulations
therefore use the O4b curves for H1 and L1, not the O4a ones, which would give
152.9 and 173.0 Mpc. The earlier comment in `run_all.sh`, that only L1 was
shared, was wrong and is corrected. Status: OK.

**No recovered SNR in the release.** The injection file carries, per search,
only `_far`, `_p_astro` and `_ranking_statistic`. The only SNRs are the
estimated optimal ones and `semianalytic_observed_phase_maximized_snr_net`,
which is zero for every O3 and O4 injection. There is therefore no way to
compare a recovered SNR with the optimal one, which would have measured
directly whether the release PSDs or the reference curves were the real
sensitivity. What the release documentation does say, for the monthly PSDs, is:
"Given an ensemble of PSDs within a month, we then took the 10% quantile in each
frequency bin for each IFO separately as the reference value used to generate
injections for that month", chosen "to be conservative (slightly overestimate
the actual range)". The release PSDs are thus deliberately more sensitive than
the month they describe, which is consistent with their BNS ranges exceeding
the published ones by 5 to 12%.

## Step 12 — found injections at low SNR

`checks/low_snr_found.py`. Found injections whose median simulated network SNR
stays below 6 are 0.1 to 0.4% of the found sample, 42 (BNS), 53 (NSBH) and 167
(BBH) in O3a. Two bookkeeping causes account for most of them: 48 to 57% have
no LIGO detector observing per the GWOSC segments, and 79 to 81% have their
largest optimal SNR in a detector the segments call off. The release gives
**no Virgo SNR at all** for O3a, O3b and O4a, `estimated_optimal_snr_V` being
exactly zero on every row, so an injection whose only observing detector is
Virgo gets a simulated network SNR near zero while the searches found it in
LIGO data. Virgo-only time is 0.19% of the O3a injections and 0.18% of O3b,
and 44% and 43% of those are found. Removing them moves rho_eq of O3a BNS
1.12-1.4 from 9.216 to 9.219 and of O3b from 8.845 to 8.851, so the effect on
the result is negligible.

This does **not** explain the found fraction staying near 0.2 between SNR 4 and
8 in O3 BNS: those injections sit below the first histogram edge, and removing
them leaves every interval of the found fraction unchanged. The plateau remains
unexplained, and without a recovered SNR per injection (step 11) there is no way
here to tell a glitch coincidence from a mismatch between the noise model and
the real search response. Status: open.

## Open points, not settled here

1. **NS boundary at 2.5 Msun (release) against 3.0 Msun (paper).** The code
   follows the release. Moving it to 3.0 would move injections from BBH to NSBH
   and from NSBH to BNS in the 1.4-4 Msun bins.
2. **About 0.1 in O3 against the older code.** O3a 8.48 -> 8.464, O3b
   8.33 -> 8.324 for BBH 20-35, well inside the bootstrap error of 0.025. The
   O3 BBH 120-300 bin moves a great deal, 12.3 -> 9.46 and 11.7 -> 9.12, which
   the 100 Msun cut accounts for: it removes 88 309 injections, 59 389 of them
   found, all BBH and concentrated in that bin.
3. **Bin widths.** `config.py` justifies halving the bins below 2.2 Msun by the
   variation of rho_eq inside a bin in O4, but the significant gradients are in
   the BBH bins (16 sigma in O4a BBH 20-35, 7-8 sigma in O3 BBH 60-120), which
   are not halved.
4. **`--reference` applies `o4b_*_ref.txt` to O4a as well**, although
   `o4a_h1_ref.txt` and `o4a_l1_ref.txt` exist (152.9 and 173.0 Mpc against
   160.9 and 171.1 for O4b). Deliberate if the point is to express every
   threshold for the sensitivity the simulations assume, but the docstring does
   not say so.
5. **O4b starts two days after the official start** (GPS 1396969218 against
   1396796418), which is the release boundary, not an error: no injection lies
   in those two days. It does make the O4b window 291.1 d against the 293.1 d
   of the paper, so the duty cycles computed here are marginally below the
   published ones (H1 48.0% against 48.6%).
6. **The duty-cycle variant has been removed.** It drew a network from the
   published duty cycles while keeping the found labels of the real network, so
   it did not measure what its name said. The comparison of observing-time
   budgets now lives in section 7.2 of `detection_rate.ipynb`, against the
   segments rather than against renormalised duty cycles.
7. **The release has no V1 PSD for 2024_04**, so the O4b Virgo range averages
   nine months, and an April 2024 injection falls back to the mean of the other
   months in `to_reference`.
8. **The 100 Msun cut removes 59 389 found injections.** It follows the FullPop
   range, but it is a large fraction of the found BBH and it drives the whole
   change in the 120-300 Msun bin.
9. **The plateau of the found fraction at SNR 4-8 in O3 BNS** (step 12). Not a
   physical selection, not the Virgo bookkeeping either, and not decidable with
   the columns the release provides.
10. **The simulations and the real network do not share their observing-time
    budget.** With each detector on independently 70% of the time, the HLV
    simulations spend 34.3% of the time with three detectors and 2.7% with none,
    against 30.8% and 11.6% measured in O4b, and the O4a HL simulations 49% and
    9% against 53.4% and 17.0% measured.
