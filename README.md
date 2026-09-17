# Ancalagon k39(p,g)40Ca Emulator — scaffold

Sibling project to `~/data/DRAGON_G3_Emulator/` (the Geant3 DRAGON BGO
spectrum emulator), but built on **Ancalagon** (`~/codes/Ancalagon/`), the
in-progress Geant4 rewrite of the DRAGON pilot simulation, rather than the
legacy Geant3 code.

**Decided** (2026-09-14): train on `Y_singles` (not `Y_addback`), model
this as a **response function vs. true gamma energy** (not a per-cascade
regressor like the G3 project's `ParametricSpectrumEmulator`), and give it
its own energy-dependent resolution model fit fresh from this data. All
three are now done for the photopeak component — see "Response function
results" below. Compton continuum / escape peaks are explicitly **not**
modeled yet (see `response_function.py`'s docstring) — that's the next
real increment, not an oversight.

## The dataset this pipeline consumes

Four cascade topologies, all scans of the same ³⁹K(p,γ)⁴⁰Ca 606 keV
resonance, combined into one 5401-run `dataset.npz` (see
`build_dataset.py`'s `DEFAULT_TOPOLOGIES`). The first two are two-step
(2-gamma) 1D scans; the last two, added 2026-09-15, are three-step
(3-gamma) 2D grid scans (padsley's own specified topology, a second
fictional level inserted above the first):

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
| `dataset.npz` | Built output, 5401 rows across 4 topologies (88 `"ground"`, 54 `"0+_2"`, 3828 `"3g_ground"`, 1431 `"3g_0+_2"`). `X` (5401, 9) — union of all topologies' feature keys (`br(-1,1)`, `br(-1,2)`, `br(1,0)`, `br(2,1)`, `eres`, `ex`, `level(1)`, `level(2)`, `q_value`), NaN where a topology doesn't have that key. `gamma_energies` (5401, 3) — sorted true gamma energies per run, NaN-padded (2 real values for the 2-gamma topologies). `Y_addback`/`Y_singles` (5401, 500) — normalised spectra; `edges` (501,) bin edges in MeV; `n_total`/`n_hit` (5401,) — simulated events / events with ≥1 BGO hit; `file_stems`/`topology` (5401,). |
| `dataset_original_142.npz` | Frozen copy of the original 142-run (2-gamma-only) dataset, kept as the `--merge-old` cache since its raw ROOT/log files no longer exist on disk. |
| `sanity_check_spectra.png` | Three example spectra (level = 0.5, 4.5, 8.5 MeV), addback vs. singles overlaid — used to visually confirm the pipeline (see "Verification" below). |
| `fit_response_function.py` | For each run, fits every distinct known photopeak in `Y_singles` (local single-Gaussian fits if well-separated; a joint multi-Gaussian fit, generalized to however many of a run's peaks overlap — 2 or 3 for the 3-gamma topologies — for merged windows) → per-energy (sigma, amplitude) measurements → global fit of `sigma(E)^2 = (doppler_k*E)^2 + intrinsic_k^2*E`. Saves `response_function.npz`. |
| `response_function.py` | `BgoResponseFunction` class: loads the fitted model, predicts photopeak position/width/amplitude at any true gamma energy, and sums photopeaks over an arbitrary cascade's gamma list (`predict_spectrum`). Photopeak component only — see its own docstring. |
| `response_function_check.png` | Sigma-vs-energy and amplitude-vs-energy scatter plots with the fitted resolution curve overlaid. |
| `response_function_prediction_check.png` | `predict_spectrum` (using `validate_holdout.py`'s train-only fit) overlaid on one held-out run from each of the 4 topologies — confirms photopeak positions and widths are reproduced accurately, including the 3-peak-per-run topologies, on runs the fit never saw. |
| `validate_holdout.py` | Proper train/held-out split: holds out every 5th run (~20%, spread across the whole `level(1)` grid), refits the resolution model + efficiency curve on the rest, checks both against the held-out runs. See "Held-out validation" below. |
| `holdout_validation.png` | Train vs. held-out sigma(E) points with the train-only fitted curve, plus held-out relative-difference residuals. |

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
- Held-out amplitude relative difference: median 11.6%, mean 28.4%, max
  **346.8%** — worse than the 142-run check (8.7%/21.0%/134.8%); see the
  2026-09-16 write-up above for why, and "Next steps" item 1.
- The shipped `response_function.npz` still uses **all 5401 runs** for
  the same reason as before (validate with a split, ship the full-data
  fit); only `validate_holdout.py`'s own run uses a train-only subset.

## Next steps

1. **(Priority raised 2026-09-16 — measurably worse now, not just
   unfixed)** Fix the efficiency-curve interpolation issue — e.g. exclude
   `joint=True` amplitude measurements from `build_efficiency_curve`
   entirely (not just `multiplicity>1`, already done), or fit a smooth
   parametric efficiency(E) model instead of raw `np.interp` so a single
   noisy neighbor can't dominate a nearby held-out prediction.
2. Model the Compton continuum + escape peaks/Compton edge, most likely
   via a joint NNLS-style decomposition across all runs simultaneously
   (same spirit as the G3 project's approach, but here every basis
   function's position is known analytically in advance, no PCA needed).
   The much larger 3-gamma dataset should help constrain this further
   than the 0+_2 topology alone would have.
3. Fix the low-energy (<~0.5 MeV) amplitude contamination from the
   near-zero spike (unchanged from before).
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
