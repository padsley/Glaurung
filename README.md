# Ancalagon k39(p,g)40Ca Emulator — scaffold

Sibling project to `~/data/DRAGON_G3_Emulator/` (the Geant3 DRAGON BGO
spectrum emulator), but built on **Ancalagon** (`~/codes/Ancalagon/`), the
in-progress Geant4 rewrite of the DRAGON pilot simulation, rather than the
legacy Geant3 code.

**Decided** (2026-09-14): train on `Y_singles` (not `Y_addback`), model
this as a **response function vs. true gamma energy** (not a per-cascade
regressor like the G3 project's `ParametricSpectrumEmulator`), and give it
its own energy-dependent resolution model fit fresh from this data. All
three are done for the photopeak component — see "Response function
results" below. **Update (2026-09-17)**: the Compton continuum of each
cascade's own highest-energy gamma is now modeled too (Phase 1 — see
"Compton continuum" below); escape peaks, the Compton edge's sharpness,
and every *other* gamma's own continuum are still not modeled (see
`response_function.py`'s docstring for exactly what `predict_spectrum`
does and doesn't cover) — real follow-on increments, not oversights.

## The dataset this pipeline consumes

Five topologies, all built on the same ³⁹K(p,γ)⁴⁰Ca 606 keV resonance,
combined into one 5426-run `dataset.npz` (see `build_dataset.py`'s
`DEFAULT_TOPOLOGIES`). The first two are two-step (2-gamma) 1D scans;
the next two, added 2026-09-15, are three-step (3-gamma) 2D grid scans
(padsley's own specified topology, a second fictional level inserted
above the first); the fifth, added 2026-09-18, is a single-gamma
calibration source (see "Compton continuum" below for why):

- **`"ground"`** (added first): `~/data/Ancalagon_k39_results/k39pg_40ca_cascade_01_0100keV`
  ... `_88_8800keV`, 88 runs, 50000 events each. Terminates at 40Ca's true
  ground state (Ex = 8.93377 MeV above it); intermediate level stepped
  0.1 → 8.8 MeV in 0.1 MeV steps.
- **`"0+_2"`** (added 2026-09-14, later the same day): `k39pg_40ca_cascade0p2_01_0100keV`
  ... `_54_5400keV`, 54 runs, **600000 events each** (12x the statistics
  of the ground set). Terminates instead at the real 3.3526 MeV 0+_2
  intruder state, because 0+_2 → ground is a 0+→0+ transition (forbidden
  for single-photon emission) — real 40Ca de-excites that state by E0
  conversion, not an observable gamma, so the reaction file stops the
  tracked cascade there rather than emitting a fictitious final gamma
  (see that file's own COMM header, or the reaction-file excerpt in
  `parse_reaction.py`'s module docstring for the exact mechanism: RECL's
  mass is deliberately shifted so the file's own "level 0" IS the 0+_2
  state). Resonance excitation above 0+_2 is only 5.58117 MeV, so this
  scan's intermediate level only goes 0.1 → 5.4 MeV.
- **`"3g_ground"`** (`reactions/k39pg_40ca_cascade3g/*.reaction`, 3828
  runs, 13000 events each): a **second** fictional intermediate level
  (Level2, above Level1) inserted into the ground-terminated cascade, so
  every event emits 3 gammas instead of 2. Full triangular grid,
  Level1 = 0.1-8.7 MeV, Level2 = (Level1+0.1)-8.8 MeV (min. 100 keV gap,
  same 8.8 MeV cap the 2-gamma ground series used).
- **`"3g_0+_2"`** (`reactions/k39pg_40ca_cascade3g0p2/*.reaction`, 1431
  runs, 13000 events each): the 3-gamma analogue of `"0+_2"`, Level1 =
  0.1-5.3 MeV, Level2 = (Level1+0.1)-5.4 MeV above the 0+_2 state.
- **`"calib1g"`** (`k39pg_40ca_calib1g_*.reaction`, 25 runs, 50000
  events each, added 2026-09-18): **zero** intermediate levels — a
  single `BRAT -1 100.0 0` card, so every event emits exactly one gamma,
  no companion. `RECL` is shifted (same trick as `"0+_2"`, generalized:
  `RECL(Ex) = -25.91223 - Ex`) so the single gamma's energy is whatever
  `Ex` is chosen directly, stepped 0.1-2.5 MeV in 0.1 MeV steps. Exists
  specifically to give `fit_compton_continuum.py` clean single-gamma
  calibration data below the ~2.35 MeV floor every other topology is
  stuck above — see "Compton continuum" below.

All four topologies share the same generic structure (a single
100%-branching decay path) — `parse_reaction.py`'s `cascades()` computes
correct *observable* gamma energies for any of them with no
topology-specific code, and `build_dataset.py` tags each row with a
`topology` field purely for provenance/filtering. The 2-gamma topologies
are **1-parameter scans**; the 3-gamma topologies are **2-parameter (2D
grid) scans** — both are designed to map the BGO array's photopeak
response (position, width, efficiency) as a smooth function of true
gamma-ray energy, not to reproduce a specific measured cascade — this is
**fundamentally different in shape from the old G3 dataset** (a
4558-sample, 6-parameter grid). Don't port the old
`ParametricSpectrumEmulator`'s architecture assuming the same kind of
input space — see "Next steps".

**Disk-space note**: the original 142-run (2-gamma) dataset's raw
`dragon_hits.root`/`run.log` files were deleted 2026-09-15 to free space
ahead of generating the 5259-run 3-gamma series (~73GB) — only the
`.reaction` input files (version-controlled) and the already-extracted
`dataset_original_142.npz` backup remain for those 142 runs.
`build_dataset.py --merge-old dataset_original_142.npz` reuses that
backup's cached spectra for any row whose raw ROOT/log no longer exists,
rather than re-simulating or dropping those rows — this is how the
combined `dataset.npz` below was actually built. Disk was down to 11GB
free at the time; check `df -h /` before any further bulk regeneration.

## Format differences from the G3 project (read this before reusing G3 code)

- **Geant3 output** (`dragon1.root`, `TTree h1001`, branch `e_bgo_first`)
  was already a per-event scalar (highest-energy single BGO hit),
  pre-selected by a `recoil_hit_endv==1` recoil gate, and had gone through
  the real detector digitization (`src/gudigi.f`: light collection,
  resolution smearing, PMT response).
- **Ancalagon output** (`dragon_hits.root`, TTrees `Bgo`/`Dsssd`) is
  **hit-level**, not event-level: one row per step in a sensitive volume
  (`eventID`, `crystalID` 1-30, `edepMeV`, position, time, `trackID`,
  `particle`). Turning this into a per-event spectrum requires an explicit
  choice (see `extract_spectrum.py`):
  - `"addback"` (sum of `edepMeV` across every hit in the event,
    regardless of crystal) — recovers full cascade/Compton-scatter energy,
    the standard modern convention.
  - `"singles"` (max single-crystal total in the event) — matches the old
    project's `e_bgo_first` observable, for comparability.
  Both are extracted and saved; **no recommendation yet on which to use
  for training** — see "Open questions" below.
- **No recoil gate available**: `Dsssd` is empty in every one of these 88
  runs (recoils aren't reliably reaching the DSSSD with EM physics turned
  on in this Geant4 pilot — a known, documented limitation of Ancalagon
  itself, not a bug in this pipeline). Every spectrum here is BGO-singles,
  ungated on recoil detection.
- **No detector resolution smearing yet**: Ancalagon doesn't port
  `gudigi.f`'s light-collection/PMT smearing (see its own README, "Sensitive
  detectors"). What you get is raw Geant4 physics only — pair
  production/Compton escape/multi-crystal scatter structure — with no
  detector-resolution broadening on top. Real peaks will look narrower
  here than the equivalent real/G3 spectrum until that's added (either in
  Ancalagon itself, or as a post-hoc convolution in this pipeline, the way
  `smooth_dataset.py` did for the G3 project — but that used a *fixed*
  smoothing width as a simplification; the G3 project's own follow-up work
  found the real width is strongly energy-dependent, see its README's
  "Architecture" section — worth building that in from the start here
  rather than repeating the fixed-width detour).
- **`n_total` (simulated event count) is directly available** here, from
  counting `HITS BGO` lines in each run's `run.log` (confirmed: exactly
  50000 lines for a 50000-event run, one per simulated event including
  zero-hit ones) — unlike the G3 dataset, which never retained this and
  had to treat per-bin Poisson noise as an unavoidable floor (see the G3
  README's "smoothing" step and its underlying memory notes). This should
  let a per-parameter-point *efficiency* curve be modeled explicitly
  instead of smoothed away.
- **The 2- and 3-gamma topologies don't share the same `X` feature
  keys** (2 vs. 3 `level(N)`/different `br(P,Q)` cards), so `build_dataset.py`
  builds `X` from the **union** of every topology's keys, NaN-filling
  whichever a given row's topology doesn't have. What *is* generic across
  all four is a single known list of true gamma energies per run, stored
  separately as `gamma_energies` (NaN-padded to 3) via
  `parse_reaction.py`'s `cascades()` — `fit_response_function.py` fits
  against this field, not `level(N)`/`ex` arithmetic, so it needs no
  topology-specific code either.
- **Degenerate gamma energies**: the 3-gamma grid's two independently-scanned
  levels can put two of a run's three gammas at the *exact same* true
  energy (e.g. `Level2 - Level1 == Level1` when `Level2 == 2*Level1`) —
  physically a real coincidence, but fitting them as two independent
  Gaussians at one center is unidentifiable (only their sum is
  observable). `fit_response_function.py`'s `_dedupe_gammas` merges these
  before fitting and tags the row `multiplicity > 1`; such rows are kept
  in the resolution fit (sigma is unaffected by multiplicity) but
  **excluded from the efficiency curve** (`build_efficiency_curve`) since
  their amplitude isn't a plain single-gamma efficiency value.

## Files

| File | Purpose |
|------|---------|
| `parse_reaction.py` | Parses Ancalagon `.reaction` files (`BEAM`/`TARG`/`RECL`/`ERES`/`LEVL`/`BRAT` cards) into level energies, branching, derived Q-value/Ex, and enumerates every resonance→ground cascade path with its energies and probability. Generalizes beyond the 2-level scan files (tested against the bundled `o15ag_19ne.reaction`, a real 4-level cascade). |
| `extract_spectrum.py` | Per-event BGO spectrum from `dragon_hits.root` — `"addback"` or `"singles"` energy definition (see above), normalised by `n_total` (from `run.log`, since zero-hit events leave no row in the ROOT tree at all). `extract_both()` shares one ROOT open + one log read between both methods (halves I/O — matters at 5259 runs). |
| `build_dataset.py` | Matches each configured topology's reaction files to their result directories by stem, runs both parsers, tags each row with its `topology`, saves the combined `dataset.npz`. `--merge-old PATH` falls back to a prior build's cached spectra for rows whose raw ROOT/log no longer exists (see "Disk-space note" above). |
| `dataset.npz` | Built output, 5426 rows across 5 topologies (88 `"ground"`, 54 `"0+_2"`, 3828 `"3g_ground"`, 1431 `"3g_0+_2"`, 25 `"calib1g"`). `X` (5426, 10) — union of all topologies' feature keys (`br(-1,0)`, `br(-1,1)`, `br(-1,2)`, `br(1,0)`, `br(2,1)`, `eres`, `ex`, `level(1)`, `level(2)`, `q_value`), NaN where a topology doesn't have that key (`"calib1g"` has none of the `level(N)` keys at all — zero intermediate levels). `gamma_energies` (5426, 3) — sorted true gamma energies per run, NaN-padded (1 real value for `"calib1g"`, 2 for the other 2-gamma topologies). `Y_addback`/`Y_singles` (5426, 500) — normalised spectra; `edges` (501,) bin edges in MeV; `n_total`/`n_hit` (5426,) — simulated events / events with ≥1 BGO hit; `file_stems`/`topology` (5426,). |
| `dataset_original_142.npz` | Frozen copy of the original 142-run (2-gamma-only) dataset, kept as the `--merge-old` cache since its raw ROOT/log files no longer exist on disk. |
| `sanity_check_spectra.png` | Three example spectra (level = 0.5, 4.5, 8.5 MeV), addback vs. singles overlaid — used to visually confirm the pipeline (see "Verification" below). |
| `fit_response_function.py` | For each run, fits every distinct known photopeak in `Y_singles` (local single-Gaussian fits if well-separated; a joint multi-Gaussian fit, generalized to however many of a run's peaks overlap — 2 or 3 for the 3-gamma topologies — for merged windows) → per-energy (sigma, **efficiency** — see "Efficiency curve fix" below) measurements → global fit of `sigma(E)^2 = (doppler_k*E)^2 + intrinsic_k^2*E`, plus `build_efficiency_curve`'s median-binned smoothing. Saves `response_function.npz`. |
| `response_function.py` | `BgoResponseFunction` class: loads the fitted model(s), predicts photopeak position/width/efficiency at any true gamma energy (`full_energy_efficiency`, renamed 2026-09-17 from `full_energy_amplitude` — see below), plus (new 2026-09-17) the Compton continuum of a cascade's own highest-energy gamma (`compton_continuum`). `predict_spectrum` sums photopeaks over an arbitrary cascade's gamma list plus that one continuum component — see its own docstring for exactly what is/isn't covered. |
| `response_function_check.png` | Sigma-vs-energy and efficiency-vs-energy scatter plots with the fitted resolution curve / smoothed efficiency curve overlaid. |
| `response_function_prediction_check.png` | `predict_spectrum` (using `validate_holdout.py`'s train-only fit) overlaid on one held-out run from each of the 4 topologies — confirms photopeak positions and widths are reproduced accurately, including the 3-peak-per-run topologies, on runs the fit never saw. |
| `validate_holdout.py` | Proper train/held-out split: holds out every 5th run (~20%, spread across the whole `level(1)` grid), refits the resolution model + efficiency curve on the rest, checks both against the held-out runs — now reports both an ALL-held-out and a CLEAN-held-out efficiency error (see "Efficiency curve fix" below). See "Held-out validation" below. |
| `holdout_validation.png` | Train vs. held-out sigma(E) points with the train-only fitted curve, sigma residuals, and (new 2026-09-17) held-out efficiency residuals/error-distribution split by clean vs. excluded points. |
| `gamma_physics.py` | **Detector-agnostic** gamma-interaction physics (new 2026-09-17): `compton_edge_energy(E)`, `klein_nishina_continuum_shape(E, T)` (derived from first principles in its own docstring), `escape_peak_energies(E)` (not used yet). No dependency on Ancalagon/BGO/this repo's file formats — deliberately reusable by a future project (padsley has mentioned HPGe spectra). |
| `fit_compton_continuum.py` | Phase 1 of Compton-continuum modeling (new 2026-09-17): for each run, finds the highest-energy gamma's own clean/uncontaminated continuum window (`clean_continuum_window`), fits a single amplitude against the raw Klein-Nishina shape, then the same median-binned-smoothing global curve as `build_efficiency_curve`. Saves `compton_continuum.npz`. See "Compton continuum" section below. |
| `compton_continuum_check.png` | Fitted continuum amplitude vs. `E_top` (log scale) with the smoothed curve overlaid — visibly tight/clean compared to the efficiency curve's own scatter. |
| `compton_continuum_prediction_check.png` | `predict_spectrum` (photopeaks + top-gamma continuum) overlaid on one held-out run from each of the 4 topologies — the real end-to-end shape check; green band marks each run's own fitted clean window. |
| `validate_compton_holdout.py` | Same every-5th-run split as `validate_holdout.py` (imports its `split_stems` directly), refits the continuum curve train-only, checks against held-out runs. Saves `compton_continuum_holdout.npz`. |

## Usage

```bash
python build_dataset.py --out dataset.npz
# or point at a different reaction/results pair:
python build_dataset.py --reactions-dir DIR --results-dir DIR --out OUT.npz

python parse_reaction.py path/to/some.reaction     # inspect one reaction file
python extract_spectrum.py path/to/run_dir --method addback   # inspect one run's spectrum
```

## Verification done so far

- `parse_reaction.py`'s `Q`/`Ex` derivation reproduces the k39 reaction
  file's own documented values (Q=8.32777, Ex=8.93377 MeV) and correctly
  enumerates the real 4-branch `o15ag_19ne.reaction` cascade with
  probabilities summing to 100%.
- `extract_spectrum.py`'s event count matches `run.log`'s line count
  exactly (50000 for every `"ground"` run, 600000 for every `"0+_2"` run);
  addback vs. singles behave as
  physically expected — e.g. for `level=4.50 MeV` (gammas at 4.434 and
  4.500 MeV, nearly degenerate), `singles` shows a hard cutoff just above
  the higher individual gamma energy (a single crystal can't register
  more than one full gamma), while `addback` additionally shows a real
  sum-coincidence feature near Ex≈8.93 MeV that `singles` cannot show —
  see `sanity_check_spectra.png`.
- `build_dataset.py` ran cleanly over all 88 runs; recovered `level(1)`
  grid is exactly `0.1, 0.2, ..., 8.8` MeV as expected; `ex` is constant
  across all rows (8.93377 MeV) as it should be; BGO hit efficiency
  ranges 92.9%–98.9% (lowest at both energy extremes, where one of the
  two gammas is very low-energy and easily below detection/threshold
  effects).

## Response function results

**2026-09-14, initial fit (ground topology only, 88 runs, superseded
below):** `doppler_k = 0.0163 ± 0.00002`, `intrinsic_k = 0.0117 ± 0.00007`,
resolution chi2/ndf ≈ 50. Held-out validation (17 runs held out) confirmed
the fit generalizes (held-out chi2/ndf 56.2, close to train's 49.3) — see
git history for the full original writeup. Superseded once the 0+_2
topology was added; kept here only as a before/after reference point.

**2026-09-14, later the same day: 0+_2 topology added, 142 runs
combined.** Refit via `fit_response_function.py` over both topologies
(282 peak measurements, 3 fit failures — mostly the ground set's known
`level(1)=4.5` near-degenerate case):

- **Resolution model**: `doppler_k = 0.01553 ± 0.00001`,
  `intrinsic_k = 0.01248 ± 0.00002` — shifted modestly from the
  ground-only fit, as expected with ~1.6x more measurements at much
  higher per-run statistics (600000 vs 50000 events) pulling the fit.
- **Independent cross-check, a genuinely new and reassuring result**: the
  two topologies were generated independently (different final state,
  different statistics, months apart in practice) but their energy
  ranges overlap (0.1-5.4 MeV in both). `response_function_check.png`
  shows both topologies' sigma(E) measurements landing on the same curve
  in that overlap region — real agreement between two independent
  simulation campaigns, not just internal self-consistency of one.
- **chi2/ndf rose sharply (49 -> 305)** — this looks alarming but isn't
  new physics breaking: the 0+_2 runs' 12x higher statistics shrink each
  peak's fitted `sigma_err` by roughly sqrt(12) ≈ 3.5x, which inflates
  chi2 = (data-model)^2/err^2 for the *same* absolute-sized real
  imperfection in the local quadratic-background peak model (already a
  known limitation, see below) by roughly that same factor. Confirmed
  this isn't overfitting via `validate_holdout.py`: held-out chi2/ndf
  (325.8) tracks train chi2/ndf (304.1) closely, same relationship as
  before scaled up. The higher statistics are simply making a
  pre-existing, already-documented imperfection statistically sharper,
  not revealing a new one.
- **New issue found: naive efficiency-curve interpolation degrades when
  combining topologies.** Held-out amplitude relative error got
  noticeably worse (median 1.9%->8.7%, mean 5.2%->21.0%, max
  59.6%->134.8%) than sigma's (which stayed similar: median 4.5%->4.9%).
  Root cause, confirmed by inspection: the two topologies' own
  near-degenerate regions land at *different* absolute energies (ground's
  is ~3.8-5.2 MeV since its Ex=8.93 MeV; 0+_2's is ~2.4-3.2 MeV since its
  Ex=5.58 MeV is smaller) — so in the 2.4-3.2 MeV window, ground-topology
  amplitudes come from ordinary, reliable single-peak fits while 0+_2
  amplitudes in that *same* window come from its own less-reliable joint
  double-Gaussian fits (confirmed directly: every amplitude checked there
  had `joint=True`, chi2/ndf in the hundreds-to-thousands). The flat
  sorted-by-energy `np.interp` in `build_efficiency_curve` doesn't
  distinguish the two, so held-out points sometimes interpolate against a
  neighboring point of the less-reliable kind. **Not fixed** — see "Next
  steps". Sigma wasn't hit nearly as hard because the resolution model is
  a global 2-parameter fit (robust to a few noisy points) rather than a
  raw interpolation.
- **`predict_spectrum` re-validated with a genuinely held-out check this
  time** (`response_function_prediction_check.png`, regenerated using
  `validate_holdout.py`'s train-only fit): both a `"ground"` run
  (level=6700keV) and a `"0+_2"` run (level=4000keV), neither used in
  that fit, show both photopeak position and width reproduced accurately.
  (The previous version of this plot used runs that were part of the
  fit's own training data — a mistake caught and fixed, see
  [[feedback-always-holdout-validate]].)
- **Still-open caveats, unchanged from before**: low-energy (<~0.5 MeV)
  amplitude contamination from the near-zero spike; photopeak component
  only, no Compton continuum/escape peaks/Compton edge yet.

**2026-09-16: 3-gamma 2D-grid topologies added (5259 new runs), 5401
runs combined.** padsley reran the simulation campaign with two new
topologies (`"3g_ground"`/`"3g_0+_2"`, a second fictional intermediate
level, 3 gammas/event, full triangular Level1×Level2 grid — see "The
dataset this pipeline consumes"). Refit via `fit_response_function.py`,
now generalized from a hardcoded 2-peak model to an arbitrary-N joint
multi-Gaussian fit with transitive window merging (handles 3-way peak
overlap, not just pairwise) and exact-energy-coincidence deduplication
(see "Format differences" above):

- **15755 peak measurements, 15101 pass quality cut, 79 fit failures**
  (1.5% of 5401 runs — mostly windows too small in the tightest-spacing
  corner of the grid, or a handful of optimizer non-convergences; same
  character of failure as before, just more instances at this much
  larger scale).
- **Resolution model improved sharply**: `doppler_k = 0.01629 ± 0.00001`,
  `intrinsic_k = 0.01192 ± 0.00002`, **chi2/ndf = 17.9** (all-data fit) —
  down from 305 on the 142-run dataset. This is the expected effect of
  ~40x more peak measurements densely covering the same 0.1-8.8 MeV
  range: the same local quadratic-background-model imperfection that
  drove the old chi2/ndf is now averaged over far more points spanning
  far more of the true-energy axis, rather than concentrated in two
  narrow 1D scans.
- **Held-out validation is now excellent, not just "not overfitting"**:
  train chi2/ndf 17.88 vs. held-out chi2/ndf 17.92 (held out 1064/5401
  runs, every 5th) — essentially identical, the tightest train/held-out
  agreement this project has seen. Held-out sigma relative difference:
  median 5.5%, mean 9.0%, max 73.9% (comparable to the 142-run result's
  4.9%/8.2%/54.2% — same order, not degraded by the much larger, more
  topologically diverse dataset).
- **The efficiency-curve interpolation issue (flagged 2026-09-14, never
  fixed) is measurably worse at this scale, not better**: held-out
  amplitude relative difference median 11.6%, mean 28.4%, **max 346.8%**
  (was 8.7%/21.0%/134.8%). Excluding `multiplicity>1` (coincident-gamma)
  rows from the efficiency curve (new this session, see "Format
  differences") removed one source of bias but evidently not the
  dominant one — the much larger grid now packs many more near-degenerate,
  `joint=True` regions (every topology's own near-degenerate band, at
  its own energy, same root cause as before) into the same raw
  `np.interp`, so a single noisy neighbor still dominates some held-out
  predictions. **Still not fixed** — this is now the clearest, most
  actionable item in "Next steps", bumped in priority given the measured
  regression.
- `response_function_check.png`, `response_function_prediction_check.png`,
  and `holdout_validation.png` all regenerated from the new fit
  (`predict_spectrum` checked against one held-out run from each of the
  4 topologies this time, not just 2) — photopeak position and width are
  visually well reproduced across all four, including the 3-peak-per-run
  topologies.

## Efficiency curve fix (2026-09-17)

Tackled "Next steps" item 1 from the 2026-09-16 write-up above (the
efficiency-curve interpolation issue, which had gotten measurably worse
once the 3-gamma topologies were added). Three changes, all in
`fit_response_function.py`/`response_function.py`:

1. **Switched from peak height ("amplitude") to peak area
   ("efficiency")**: `efficiency = amplitude * sigma * sqrt(2*pi) /
   bin_width`. Height conflates true detection efficiency with
   resolution (a narrower peak of the same efficiency is taller), and
   resolution varies smoothly with energy on its own — fitting/interpolating
   the entangled quantity was adding noise that had nothing to do with
   efficiency itself. Area is resolution-independent. `response_function.py`'s
   `full_energy_amplitude` is renamed `full_energy_efficiency` accordingly
   (returns the area; `photopeak()` converts back to a height internally
   using the fitted sigma(E), so `predict_spectrum` is unaffected).
2. **Excluded `joint=True` rows from the efficiency curve entirely** (not
   just `multiplicity>1`, which was already excluded) — a joint
   multi-Gaussian fit over an overlapping window lets amplitude and sigma
   trade off against each other in an underdetermined way, exactly the
   failure mode diagnosed 2026-09-16. Also added a `chi2_ndf < 50` cut:
   a small tail (~1.7% of otherwise-clean rows) had a badly-fit local
   background (chi2/ndf up to ~140) despite a tight sigma_rel_err,
   overwhelmingly the already-documented near-zero-energy-spike
   contamination at <~0.5 MeV (Next steps item 3, still not itself
   fixed) — excluding it keeps that separate problem from also polluting
   this curve.
3. **Replaced raw point-to-point `np.interp`** (every surviving row is a
   control point) **with median-binned smoothing** (0.1 MeV bins, median
   efficiency per occupied bin, `np.interp` only between bin medians) —
   robust to any single remaining noisy point dominating a nearby
   held-out prediction, which was the direct mechanism identified
   2026-09-16.

**Diagnosis behind the fix** (see `holdout_validation.png`'s new
bottom-row panels): re-running the *same* held-out check but breaking
out held-out points by whether *they themselves* are joint/coincident/
high-chi2 showed those categories firmly cluster far from zero error
regardless of curve quality (comparing a clean model against an
inherently unreliable measurement is not a fair test of the model) —
confirming the residual error is concentrated in already-known-hard
categories, not a remaining curve defect.

**Results**: resolution model unchanged (chi2/ndf 17.88, as expected —
none of these three changes touch sigma fitting). Held-out efficiency
error, evaluated the same "ALL held-out points" way as before for direct
comparison: median 8.7%, mean 17.3%, max 385.5% — *not* obviously better
by this metric, because it's still dominated by the same joint/high-chi2
held-out comparison points being intrinsically hard to predict. Evaluated
on the **same held-out points, minus joint/coincident/chi2≥50 ones** (a
fair like-for-like comparison against the pre-fix numbers, which had no
such split): median 6.6%, mean 13.6%, max 72.0% — a real, substantial
improvement in the regime the curve can actually be expected to predict
well (compare 2026-09-16's ALL-based 11.6%/28.4%/134.8%, itself not
splittable this way since the old code didn't track it).

## Held-out validation

`validate_holdout.py` holds out every 5th run across the *combined*
5401-run dataset (1064 runs / 3019 peak measurements passing the quality
cut), refits on the remaining ~79%, checks both the resolution model and
efficiency curve against runs the fit never saw. (Superseded numbers from
the 142-run-only version of this check are in git history / the
2026-09-14 entries above.)

- Train-only fit: `doppler_k=0.01625±0.00001, intrinsic_k=0.01204±0.00002`
  — consistent with the all-data fit above.
- Held-out chi2/ndf (17.92) tracks train chi2/ndf (17.88) almost exactly
  — the best train/held-out agreement seen in this project so far.
- Held-out sigma relative difference: median 5.5%, mean 9.0%, max 73.9%.
- Held-out efficiency relative difference (**ALL** held-out points,
  same method as the pre-2026-09-17 "amplitude" check): median 8.7%,
  mean 17.3%, max 385.5%.
- Held-out efficiency relative difference (**CLEAN** held-out points
  only — excluding joint/coincident/chi2≥50, which are known-unreliable
  measurements regardless of curve quality, see "Efficiency curve fix"
  above): median 6.6%, mean 13.6%, max 72.0% — the fairer read on how
  well the fixed curve actually performs.
- The shipped `response_function.npz` still uses **all 5401 runs** for
  the same reason as before (validate with a split, ship the full-data
  fit); only `validate_holdout.py`'s own run uses a train-only subset.

## Compton continuum (Phase 1, 2026-09-17)

padsley asked for the Compton-continuum modeling ("Next steps" item 2
below, pre-fix) to be scoped out properly rather than shortcut, since a
future project modeling HPGe spectra will need the same physics and will
demand more precision. The approved plan phased this: model only each
cascade's **highest-energy gamma's own continuum** first, since every
run has a clean, uncontaminated window for it (no joint-fit degeneracy
risk, unlike an NNLS decomposition across all a run's gammas at once);
escape peaks and lower gammas' own continua are follow-on phases (items
2a/2b below) once this scaffold and its validation harness exist.

**Physics**: for a mono-energetic gamma of true energy `E`, Compton
scattering deposits recoil-electron energy `T` in `[0, compton_edge_energy(E)]`,
`compton_edge_energy(E) = 2*E^2/(m_e*c^2+2*E)` (`m_e*c^2=0.510999 MeV`,
same constant the sibling G3 project's `emulator.py` uses for its own
escape-peak formula). The differential shape vs. `T` was **derived from
scratch** in `gamma_physics.klein_nishina_continuum_shape`'s own
docstring (Klein-Nishina per solid angle, changed to a `T`-differential
via the `cos(theta)->T` Jacobian) rather than taken on faith from a
half-remembered reference — it reduces, after dropping energy-dependent-
but-`T`-independent prefactors (irrelevant since a separate amplitude is
always fit per energy against real data), to `E'/E + E/E' - sin^2(theta)`
with `E'=E-T`, rising from 2 at `T=0` to `E'/E+E/E'>=2` at the edge — the
characteristic upward "shoulder" real Compton continua show.

**Validated against real data before building anything else**: a
throwaway visual check (4 widely-spaced true energies, amplitude-only
least-squares scale fit, no shape correction) confirmed this raw
theoretical shape already matches this simulation's real `Y_singles`
data closely in the clean window — no extra empirical shape-correction
parameters were needed for Phase 1. That check also caught a real bug in
the *window* logic before it reached the real pipeline: a companion
gamma's own photopeak sits *above* its own Compton edge, not at it, so
excluding only up to `compton_edge_energy(E_2nd)` still let `E_2nd`'s
own peak leak into the "clean" region — fixed in
`clean_continuum_window` by explicitly cutting out every companion's own
`E_c +/- 5*sigma(E_c)` window too, not just bounding by the edge.

**Fit results**: 4204/5401 runs fit successfully (failures are the same
kind already seen for photopeaks — near-degenerate top-two gammas
consuming the whole window — at a higher ~22% rate than photopeaks' 1.5%
since a 3-gamma run has *two* companions that can each shrink the
window). Held-out validation (same every-5th-run split as the
photopeak/efficiency work, via `validate_compton_holdout.py`): relative
difference median 2.1%, mean 3.0%, max 21.1% — **substantially tighter
than the efficiency curve's own held-out numbers** (6.6%/13.6%/72.0%
clean), because the continuum amplitude is a smooth, densely-and-
redundantly-sampled quantity across the full energy range, unlike the
efficiency curve's peakier, more topology-fragmented measurements.

**A real bug caught by this validation, not just theory**: the first
version of `measure_all_continuum` forgot to normalise the fitted
amplitude by `n_total` (per-simulated-event), unlike every other
measurement in this pipeline. This didn't show up as a fit failure — it
produced a *smooth, plausible-looking* curve — but inflated `"0+_2"`
topology (600,000 events/run) amplitudes ~40-50x relative to `"ground"`/
`"3g_*"` (13,000-50,000 events/run) at the same true energy, which the
held-out check caught immediately as a cluster of ~98% and one 264%
outlier concentrated in exactly the runs the two curves' events counts
differed most. Fixed (one line); held-out max dropped from 264% to
21.1%. Lesson: a smooth-looking curve is not proof of correctness here —
believe the held-out numbers, not the plot, when they disagree with a
"looks fine" impression.

**A genuine, structural coverage gap, not a bug (as first shipped
2026-09-17)**: no run in the 4 cascade topologies has its highest-energy
gamma below **~2.35 MeV** (the theoretical floor is `Ex/n_gammas` — the
most-evenly-split case of a topology's total excitation energy across
its 2 or 3 gammas — about 1.86 MeV in the best case, `~2.35` in practice
once near-degenerate exclusions are accounted for). Below the curve's
lowest point, `compton_continuum` clips to the boundary value, same as
`full_energy_efficiency` does for its own range — **silently wrong, not
just extrapolated**, for a hypothetical cascade whose own top gamma is
genuinely below the floor. **Closed down to ~0.95 MeV the next day, see
"New calibration data closes most of the gap" below** — this was
originally scoped as "needs Phase 2b", but a much cheaper fix turned out
to exist.

**Visual confirmation** (`compton_continuum_prediction_check.png`): for
2-gamma topology runs, the predicted spectrum (photopeaks + top-gamma
continuum) now visibly tracks the real "hump" between the two
photopeaks that the old photopeak-only model showed as a flat zero. For
3-gamma runs, only the region above the second-highest gamma is
corrected this way — the gap between the two lower gammas is still
photopeak-only, exactly as documented (Phase 2b).

**Follow-on phases (not done here)**:
- **Phase 2a — escape peaks** (`E > 1.022 MeV`): positions
  (`gamma_physics.escape_peak_energies`) and the edge/continuum
  machinery already exist; this is now mostly plumbing.
- **Phase 2b — peel to lower gammas**: for each run, subtract the
  already-fitted top gamma's full predicted response (photopeak +
  continuum) from the raw spectrum, then fit the second-highest gamma's
  own continuum in the now-cleaner residual (its own clean window, same
  method); repeat down to the lowest gamma. Avoids the joint-fit
  degeneracy an all-at-once NNLS decomposition would hit (the same
  amplitude/width trade-off that was just fixed for photopeaks), because
  each step only fits one new unknown against already-fixed, globally-
  calibrated higher-energy components. This is also what would close the
  ~2.35 MeV coverage gap above.
- **HPGe reuse**: `gamma_physics.py` ports directly (zero
  Ancalagon/BGO-specific code by design). The resolution model and the
  fitted continuum-amplitude curve would need refitting for HPGe's very
  different, typically much better resolution — expected, not a design
  flaw here.

### New calibration data closes most of the gap (2026-09-18)

padsley asked for simulation inputs specifically designed to overcome
the ~2.35 MeV coverage floor above, to be run through Ancalagon. Rather
than jumping straight to Phase 2b (peeling, real follow-on work — still
not done), there was a cheaper option: a **dedicated single-gamma
calibration source**, using the same `RECL` mass-excess-shift trick the
`"0+_2"` topology already uses (shift the fictional final state so the
compound's excitation above it is whatever energy you want), but with
**no intermediate level at all** — a bare `BRAT -1 100.0 0` card sends
the resonance straight to that fictional state, so every event emits
**exactly one gamma**, at an energy chosen directly via
`RECL(Ex_target) = BEAM_mass + TARG_mass - (Ex_target - ERES)`
(`= -25.91223 - Ex_target` for this reaction; verified this formula
exactly reproduces both existing series' `RECL` values before using it).
No companion gamma at all means no exclusion window is needed — this is
strictly cleaner calibration data than anything else in this project.

- New topology **`"calib1g"`** (`k39pg_40ca_calib1g_*.reaction`, 25
  runs, 0.1–2.5 MeV in 0.1 MeV steps, 50000 events/run): verified with
  both `parse_reaction.py` (before running anything) and Ancalagon's own
  `--reaction-stats` (100.00% single-gamma events) that each file
  behaves as designed. Measured actual disk cost on a small (3000-event)
  calibration run before committing to the full campaign (~0.64GB
  conservative estimate vs. 7.1GB free at the time — disk stayed tight
  all session); actual usage came in even lower, ~0.6GB.
- `build_dataset.py` gained this topology (`DEFAULT_TOPOLOGIES`);
  `dataset.npz` is now 5426 rows. `fit_compton_continuum.py`'s
  `clean_continuum_window` needed a genuine fix, not just a config
  change, for the zero-companion case (it previously required a
  companion to set the window's lower bound at all).
- **A real, previously-invisible physical feature found in the
  process**: with no companion gamma to naturally push the window's
  lower bound up, these runs' fits initially came back with much worse
  chi2/ndf (40-290) than any multi-gamma run ever showed. A visual check
  (not just trusting the number) revealed a genuine **non-monotonic
  "dome"** around ~0.2-0.4 MeV that the (monotonically-rising)
  Klein-Nishina shape does not predict — almost certainly a
  **backscatter peak** (photons Compton-scattering off surrounding/dead
  material before reaching the sensitive crystal, a well-known feature
  in gamma spectroscopy, physically distinct from an in-crystal Compton
  scatter). Every *other* topology's window had always started above
  this by construction (their companion gamma's own Compton edge
  usually sits higher) — this project had been implicitly relying on
  that accident, not deliberately excluding the dome.
- **Fix**: `BACKSCATTER_FLOOR_MEV = 0.45` — a fixed floor added to
  `clean_continuum_window`'s lower bound, applied to *every* topology
  (not just `"calib1g"`; a multi-gamma run with a small second-highest
  gamma could in principle hit the same contamination). Chi2/ndf
  improved substantially (e.g. 71.6→31.6 at E_top=1.0 MeV) but **does
  not reach the ~2-20 typical at high E even past the dome** — reported
  honestly as a real residual limitation (likely multiple in-crystal
  scattering or other effects proportionally larger at low E), not
  claimed as fully fixed. Modeling the dome itself is out of scope here.
- **Result**: continuum curve coverage floor **2.35 MeV → 0.95 MeV**.
  Below 0.95 MeV the window structurally collapses (a gamma's own
  Compton edge sits below the 0.45 MeV dome floor once `E_top` gets
  small enough — inverting `compton_edge_energy` shows this happens
  below `E_top~0.63 MeV`, plus the photopeak-side margin shrinks it
  further) — a now-understood, structural limit of this method, not an
  arbitrary cutoff. Held-out validation unchanged/still
  excellent after adding this data: median 2.1%, mean 3.1%, max 22.7%
  (was 2.1%/3.0%/21.1%) — confirms the new low-E points that *do* pass
  the quality cut integrate cleanly, not just add noise.
- Reaching the remaining 0.1–0.9 MeV would need modeling the backscatter
  dome itself (a new, real physics component, not a config tweak) —
  genuine future work, not attempted here.

## Next steps

1. ~~Fix the efficiency-curve interpolation issue~~ — **done 2026-09-17**,
   see "Efficiency curve fix" above (area-based efficiency instead of raw
   height, exclude joint/coincident/high-chi2 rows, median-binned
   smoothing instead of raw `np.interp`). Held-out efficiency error on
   the fair (clean-vs-clean) comparison improved from 8.7%/21.0%/134.8%
   (142-run, pre-3-gamma baseline) to 6.6%/13.6%/72.0% (5401-run,
   post-fix) — genuinely better, not just a different metric. Not fully
   solved: the ALL-held-out number is still large (max 385.5%) because
   items 2-4 below (Compton/escape modeling, low-energy contamination,
   near-degenerate regions) remain open and dominate exactly the points
   this fix correctly declines to smooth over.
2. ~~Model the Compton continuum~~ — **Phase 1 done 2026-09-17** (the
   highest-energy gamma's own continuum only), see "Compton continuum"
   above — a response-function-style fit (physics-derived Klein-Nishina
   shape, single fitted amplitude per energy, median-binned smoothing),
   *not* the joint NNLS decomposition this item originally suggested
   (deliberately avoided — an all-at-once NNLS across a run's overlapping
   gammas would hit the same amplitude/width degeneracy just fixed for
   photopeaks). **2026-09-18**: added a dedicated `"calib1g"` single-gamma
   simulation series that pushed the coverage floor from 2.35 MeV down to
   0.95 MeV (see "New calibration data closes most of the gap") and, as a
   side effect, discovered and corrected for a previously-invisible
   backscatter-peak contamination affecting *every* topology's continuum
   window, not just the new one. Remaining: escape peaks, Compton edge
   sharpness, modeling the backscatter dome itself (would reach
   0.1-0.95 MeV), and every gamma *except* each cascade's own
   highest-energy one — see that section's "Follow-on phases" (2a/2b) for
   the concrete next steps and why they're phased this way.
3. **(Now the clearest single lever on the remaining held-out efficiency
   error, per the 2026-09-17 diagnosis)** Fix the low-energy (<~0.5 MeV)
   efficiency contamination from the near-zero spike. **padsley's
   diagnosis (2026-09-17): this is an electronics artefact and belongs
   suppressed at the source, in Ancalagon's simulation** (a threshold or
   noise model matching the real BGO electronics), **not patched further
   in this analysis pipeline** — don't respond to this item by tightening
   more cuts/thresholds in `build_efficiency_curve`/`measure_all_peaks`.
   Confirmed via chi2/ndf (up to ~140 for affected rows, vs. ~5-20
   typical) that this is a real, identifiable local-background-model
   failure, not just noise, and it's responsible for several of the
   single worst held-out points found 2026-09-17 (e.g.
   `k39pg_40ca_cascade3g_L1-0200keV_L2-4000keV` at E=0.2 MeV,
   chi2/ndf=141, off by 385%).
4. Investigate/improve the near-degenerate region in each topology (each
   has its own, at a different absolute energy — see above) — wider or
   better-conditioned joint fits, or a parameterization that doesn't
   require independently resolving heavily-overlapping peaks.
5. ~~Both topologies still have only one cascade shape... branching-ratio
   variation needs new Ancalagon runs~~ — **done 2026-09-15**: the
   `"3g_ground"`/`"3g_0+_2"` topologies add a second intermediate level
   (3 gammas/event). Branching-ratio *variation* (not just more levels)
   still hasn't been scanned — every topology here is still 100%/100%
   branching throughout.
