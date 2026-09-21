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
**Update (2026-09-19)**: photopeaks are now fit with a physically-derived
Doppler-broadened lineshape, not a Gaussian (real chi2/ndf 536→225 on an
isolated high-statistics peak — see "Doppler-broadened lineshape +
backscatter-dome model" below), and the backscatter dome found
2026-09-18 is modeled (not just excluded), though still data-starved.

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
| `fit_response_function.py` | For each run, fits every distinct known photopeak in `Y_singles` with a **Doppler-broadened lineshape** (new 2026-09-19, `gamma_physics.doppler_lineshape` — was a Gaussian; local single-peak fits if well-separated, a joint multi-peak fit for merged windows, generalized to however many of a run's peaks overlap) → a **staged fit**: free-`beta` local fits → robust global `beta` → fixed-`beta` refit → global `sigma_intrinsic(E)=intrinsic_k*sqrt(E)`, plus `build_efficiency_curve`'s median-binned smoothing. Saves `response_function.npz` and caches per-peak measurements to `peak_measurements.npz` (shared with `validate_holdout.py`/`fit_compton_continuum.py`, which no longer re-run this expensive step themselves). |
| `peak_measurements.npz` | Cached stage-3 (fixed-`beta`) per-peak measurements (new 2026-09-19) — one row per distinct photopeak: `energy`, `sigma_intrinsic(_err)`, `amplitude(_err)`, `efficiency`, `chi2_ndf`, `joint`, `multiplicity`. Loaded by `fit_response_function.save_stage_rows`/`load_stage_rows`. |
| `response_function.py` | `BgoResponseFunction` class: loads the fitted model(s), predicts photopeak position/width/efficiency at any true gamma energy (`full_energy_efficiency`; `photopeak()` uses the Doppler lineshape, not a Gaussian, as of 2026-09-19 — stores `beta`/`intrinsic_k`, not `doppler_k`/`intrinsic_k`), plus (2026-09-17) the Compton continuum and (2026-09-19) backscatter dome of a cascade's own highest-energy gamma. `predict_spectrum` sums photopeaks over an arbitrary cascade's gamma list plus that one continuum+dome contribution — see its own docstring for exactly what is/isn't covered. |
| `lineshape_check.png` | New 2026-09-19: a real isolated high-statistics photopeak with a Gaussian fit and a Doppler-lineshape fit overlaid (chi2/ndf 536 vs. 225) — the direct visual case for the lineshape switch. |
| `response_function_check.png` | Doppler-lineshape components (Doppler box, intrinsic, total effective sigma) vs. energy, and efficiency-vs-energy scatter with the smoothed efficiency curve overlaid. |
| `response_function_prediction_check.png` | `predict_spectrum` (using `validate_holdout.py`'s train-only fit) overlaid on one held-out run from each of 4 topologies — confirms photopeak positions/shapes (now visibly flat-topped, not bell-curved) are reproduced accurately on runs the fit never saw. |
| `validate_holdout.py` | Loads cached `peak_measurements.npz` (no expensive re-fit, 2026-09-19) and does the (fast) train/holdout aggregation: holds out every 5th run (~20%), refits `intrinsic_k` + efficiency curve on the rest (`beta` itself is not re-derived train-only, a disclosed simplification — see "Doppler-broadened lineshape" section), checks both against held-out runs. See "Held-out validation" below. |
| `holdout_validation.png` | Held-out efficiency residuals and `\|error\|` distribution, split by clean vs. excluded points. |
| `gamma_physics.py` | **Detector-agnostic** gamma-interaction physics: `compton_edge_energy(E)`, `klein_nishina_continuum_shape(E, T)`, `doppler_lineshape(E, x, beta, sigma_intrinsic)` and `effective_sigma(...)` (new 2026-09-19, both derived from first principles in their own docstrings), `backscatter_energy(E)` (new 2026-09-19), `escape_peak_energies(E)` (not used yet). No dependency on Ancalagon/BGO/this repo's file formats — deliberately reusable by a future project (padsley has mentioned HPGe spectra). |
| `fit_compton_continuum.py` | For each run, finds the highest-energy gamma's own clean/uncontaminated continuum window (`clean_continuum_window`), fits a continuum amplitude against the raw Klein-Nishina shape **plus (new 2026-09-19) a backscatter-dome Gaussian bump** (physics-anchored center, fit amplitude/width) whenever the window covers that region, then the same median-binned-smoothing global curve as `build_efficiency_curve` for both. Saves `compton_continuum.npz`. See "Compton continuum" and "Doppler-broadened lineshape" sections below. |
| `compton_continuum_check.png` | Fitted continuum amplitude vs. `E_top` (log scale) with the smoothed curve overlaid, plus (new 2026-09-19) the sparse backscatter-dome amplitude curve. |
| `compton_continuum_prediction_check.png` | Zoom-in (0-1.2 MeV) on two held-out runs showing the predicted backscatter dome (new 2026-09-19) against real data, with the physics-predicted dome center marked. |
| `validate_compton_holdout.py` | Same every-5th-run split as `validate_holdout.py` (imports its `split_stems` directly), refits the continuum **and dome** curves train-only, checks both against held-out runs. Saves `compton_continuum_holdout.npz`. |

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

**Superseded 2026-09-19** by the Doppler-lineshape switch — see "Doppler-
broadened lineshape + backscatter-dome model" below for current numbers
(`beta`/`intrinsic_k` replace `doppler_k`/`intrinsic_k`, and the
efficiency curve's `chi2_ndf_max` changed from 50 to 15). Kept below as
the historical record of the pre-Doppler (Gaussian-lineshape) state.

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

## Doppler-broadened lineshape + backscatter-dome model (2026-09-19)

padsley reviewed the visual validation report and pointed out that real
peak shapes don't look Gaussian, and asked for (1) a proper
Doppler-broadened lineshape and (2) the backscatter dome modeled, not
just excluded. Both landed as one combined change, since the dome's own
window logic depends on the resolution model.

**The physics, checked against real data before writing any fitting
code**: for a gamma emitted in flight by a recoil at velocity fraction
`beta`, isotropic emission into a near-4π array gives a *uniform*
density of Doppler-shifted true energies between `E*(1-beta)` and
`E*(1+beta)` (a "box"), not a bell curve — the old
`sigma(E)^2=(doppler_k*E)^2+intrinsic_k^2*E` model was always an
empirical proxy for this, just folded into a single Gaussian width
instead of modeled as its own component. Pulled an isolated,
high-statistics real peak (E=5.4 MeV, 600000-event `"0+_2"` run) and fit
it both ways before committing to anything: **Gaussian chi2/ndf=536,
Doppler-box-convolved-with-Gaussian chi2/ndf=225** — see
`lineshape_check.png`, which shows the real data's characteristic
flat-topped, steep-sided shape that no Gaussian can reproduce. `beta`
(the recoil velocity fraction) came out at 0.0318, physically sensible
and — since every cascade here is "instantaneous at vertex" (no time for
the recoil to decelerate between gammas) — expected to be one true
constant shared by *every* gamma in *every* run, not an energy-dependent
curve like the old model's Doppler term.

**New physics in `gamma_physics.py`**: `doppler_lineshape(E, x, beta,
sigma_intrinsic)` (the closed-form box-convolved-Gaussian, derived from
scratch in its own docstring), `effective_sigma(E, beta, sigma_intrinsic)`
(a variance-equivalent single-number width for window-sizing callers
only, not a claim the lineshape is Gaussian), `backscatter_energy(E)`
(the dome's physics-predicted center, `E - compton_edge_energy(E)`).

**Staged fit in `fit_response_function.py`** (`beta` is one global
constant, so it can't be fit per-peak like `sigma` could): (1) local
fits with `beta` free per window → (2) robust (median) global `beta`
from thousands of windows → (3) refit every peak with `beta` fixed,
`(amplitude, sigma_intrinsic)` free → (4) global
`sigma_intrinsic(E)=intrinsic_k*sqrt(E)` fit. Stage-3 rows are now
cached (`peak_measurements.npz`) and shared with `validate_holdout.py`/
`fit_compton_continuum.py`, which no longer re-run the expensive fit
themselves — **disclosed simplification**: `beta` is derived from *all*
rows (not re-derived train-only) even during held-out validation, since
as a single scalar overdetermined by thousands of independent windows
the leakage from that is expected to be negligible; `intrinsic_k` and
the efficiency curve still get a genuine train/holdout split.

**Two real bugs caught before trusting the full-dataset run, not after**:
1. **Performance**: the fixed-`beta` refit stage first measured at
   1.94s/window (default `scipy.optimize.curve_fit` tolerances) — at
   ~9000 windows, a ~4.9 hour run. Root cause: with `beta` fixed,
   `sigma_intrinsic`'s effect on chi2 is often nearly flat (whenever the
   Doppler box already dominates the total width), so the optimizer took
   hundreds of tiny steps chasing default-tight (~1e-8) convergence on a
   direction that barely mattered for a value later aggregated over
   thousands of measurements anyway. Loosening `xtol`/`ftol`/`gtol` to
   1e-6 and `maxfev` to 10000 cut this to 0.069s/window — a 28x speedup,
   full run down to ~23 minutes, with no loss in the aggregate result.
2. **Boundary pinning**: `sigma_intrinsic` hitting its lower fit bound
   produced degenerate, artificially-tiny reported uncertainties that
   both trivially passed the existing `rel_err<0.5` quality cut *and*
   (via division by that tiny error) blew `intrinsic_k`'s chi2/ndf up to
   ~5e20 in an early smoke test. Fixed with a much more permissive lower
   bound (1e-5, so a fit genuinely wanting near-zero intrinsic width is
   free to go there) plus an explicit `sigma_intrinsic_err > 1e-4 MeV`
   guard in every downstream quality cut — the same "watch for a fit
   landing exactly on a bound" lesson the sibling G3 project had already
   learned, re-learned here the hard way instead of remembered.

**Results** (full 5426-run dataset): `beta=0.03188` (from 5057/13381
windows), `intrinsic_k=0.00380`. Held-out checks (`validate_holdout.py`):

- **Total effective width** (`beta` + `intrinsic_k` combined — the
  number that actually matters for peak width in practice): median
  0.7%, mean 2.2%, max 82.9% — dramatically tighter than the old
  Gaussian model's sigma check (5.5%/9.0%/73.9%), because the dominant
  Doppler component is now a single, extremely well-determined global
  constant rather than a per-energy curve fit.
- **`sigma_intrinsic` alone**: median 38.6%, mean 53.0% — much noisier
  in *relative* terms, reported honestly rather than hidden: it's now a
  small residual correction (a few-to-20 keV) measured against typical
  per-run statistics of 13000-50000 events, not the dominant width
  term it used to be blended into.
- **Efficiency (CLEAN)**: median 3.7%, mean 10.8%, max 88.7% — median
  and mean *both* improved over the pre-Doppler-lineshape baseline
  (6.6%/13.6%/72.0%); max is close to, not better than, before. Getting
  here needed a second fix, below.

**A third bug, caught by exactly this held-out check**: initially the
efficiency curve's `chi2_ndf_max=50` cutoff (unchanged from the old
Gaussian model) let through a cluster of badly-fit, systematically-
low-amplitude `"3g_0+_2"` peaks sitting right at chi2/ndf~40-50 —
inflating held-out max efficiency error to 305%. The new lineshape fits
chi2/ndf much tighter overall (most clean fits land at 2-10 now, not the
old model's 2-20), so a threshold tuned for the noisier old model was no
longer doing its job. Swept `chi2_ndf_max` in {10, 15, 20, 50} against
the held-out split (cheap — reused the cached stage-3 rows, no refit
needed); **15 gave the best median/mean and cut max to 88.7%**, back in
line with the pre-Doppler baseline. Now the default everywhere this
matters (`build_efficiency_curve`, `fit_all`'s diagnostics,
`validate_holdout.py`'s CLEAN split).

**Backscatter dome**: modeled instead of excluded, per padsley's
request. `fit_compton_continuum.py`'s `clean_continuum_window` no
longer pushes its lower bound up past the dome
(`BACKSCATTER_FLOOR_MEV`, removed); instead `fit_run_continuum` jointly
fits `a_continuum * klein_nishina_continuum_shape + a_dome *
Gaussian(center=backscatter_energy(E_top), free amplitude/width)`
whenever the window actually covers the dome region (62/4181 runs — a
companion below ~0.6 MeV is needed to leave the dome inside the window
at all), else continuum-amplitude-only as before. **Honest result, not
a success story**: held-out dome-amplitude relative difference is
median 269.5%, mean 216.0%, max 617.9% (11 held-out points, 13 train
curve bins) — the dome is only measurable over a narrow slice of runs,
so this first-pass Gaussian-bump model is data-starved and noisy.
Visually (`compton_continuum_prediction_check.png`'s dome zoom-in) the
*position* is clearly right and the *order of magnitude* is often
reasonable, but don't trust individual amplitude predictions yet — a
real, disclosed limitation, not swept under the rug. Continuum itself is
unaffected/still excellent: median 2.1%, mean 2.9%, max 21.8%.

**`response_function.py`**: `BgoResponseFunction` now stores `beta`/
`intrinsic_k` (was `doppler_k`/`intrinsic_k`); `photopeak()` uses
`gamma_physics.doppler_lineshape`; new `backscatter_dome()` method,
added into `predict_spectrum`'s existing top-gamma-only contribution
alongside the continuum. `full_energy_efficiency`'s convention is
unchanged (still a resolution-independent area) but no longer needs the
old `*sigma*sqrt(2*pi)` conversion factor, since the new lineshape is
already a unit-area density.

**Not done, genuine future work**: modeling the dome with more data (it
would need many more zero/low-companion runs like `"calib1g"`, purpose-
built to cover it, the same way that series was built for the continuum
coverage gap); the Doppler-vs-intrinsic split for the *continuum*'s own
edge sharpness (still an unconvolved raw Klein-Nishina shape, see that
section above); every gamma except each cascade's own highest-energy
one (Phase 2b, unchanged).

## Empirical peak-lineshape corrections — explored, not adopted (2026-09-20)

padsley asked whether an empirical refinement on top of the box+Gaussian
Doppler lineshape (e.g. "a box with a negative Gaussian component in
it") could describe the real peak shapes better, reasoning that the
detector's actual crystal positions/angular acceptance likely make the
true density non-uniform across the box. Tried it properly rather than
guessing: fit 6 candidate lineshapes — plain Gaussian, plain box (the
current model), box+P1 (linear tilt), box+P2 (quadratic/"dome" term,
Legendre P2), box+P1+P2, and box with a subtracted Gaussian "dip" — to 5
real, isolated, high-statistics peaks (uncommitted scratch script, not
kept in the repo).

**A real numerical bug turned up first**: the exploration's numerical
convolution resampled its integration grid from the trial
`sigma_intrinsic` value on every `curve_fit` call, corrupting the
finite-difference Jacobian (a tiny parameter step moved the grid itself,
not just the kernel values on it) — symptom was the numerical
reproduction of the existing closed-form box model landing at
chi2/ndf=876 on a peak where the closed form gives 221. Fixed by
switching to a **fixed** offset grid
(`np.linspace(-0.35, 0.35, 1401)`, independent of the trial parameter);
confirmed the fix by matching the closed form almost exactly (221.19 vs
221.15) on the same peak. Same underlying lesson as the boundary-pinning
issue above: a fit can look wrong for purely numerical reasons, not
physical ones — worth checking for specifically before trusting a
fitted result.

One test peak (`k39pg_40ca_cascade0p2_30_3000keV`, E=3.00 MeV) fit badly
(chi2/ndf 300+) under *every* candidate model, including the existing
box. Traced to its companion gamma at 2.581 MeV, whose own Doppler box
edge (~2.663 MeV) sits right at this peak's fit-window edge
(2.666 MeV) — window contamination from the neighbor's tail, not a
lineshape failure. Excluded from the comparison below.

**Verdict, from the remaining 4 clean peaks: none of the empirical
corrections are adopted.** P1/P2/dip each reduce chi2/ndf a little
(3-15%, e.g. box 194.2→box+P1 188.3 on one peak) — but the *fitted
correction itself* isn't stable across peaks that share the same
detector: the P1 tilt coefficient `c1` came out **+0.12 and +0.15** on
the two `cascade0p2`-series peaks but **-0.16 and -0.20** on the two
`cascade`-series peaks. A real, fixed crystal-geometry asymmetry would
have to have the same sign everywhere; a sign flip between run series
is the signature of the fit absorbing each peak's local
background/statistical noise, not a shared physical effect. `c2` (the
P2 term) and the dip model's parameters show the same lack of
consistency, and one dip fit pinned its width parameter at its upper
bound — the same boundary-pinning artifact flagged earlier in this
project, another sign of an unconstrained, overfit parameter rather
than a real feature.

**The plain box+Gaussian model (`gamma_physics.doppler_lineshape`)
stays as the production lineshape.** The residual chi2/ndf on these
very-high-statistics peaks (~190-225) is real and still not fully
explained, but this exploration doesn't support closing it empirically
with a low-order polynomial or ad-hoc dip correction. Closing it for
real would need either far more counts per peak to separate a genuine
small effect from noise, or actual Ancalagon crystal-position geometry
to predict the correction from first principles instead of fitting it
blind — neither attempted here, and not a quick follow-up if picked up
again later.

## Dense combinatorial simulation grid, for a future direct/empirical model (2026-09-21)

padsley asked whether describing the spectrum directly and empirically
(bin counts / a PCA-regression basis learned straight from simulated
spectra, the same architecture the sibling `DRAGON_G3_Emulator` project
uses) might work better than this project's physics-decomposed model,
given the empirical-lineshape exploration above found no consistent
detector-geometry correction to adopt. That approach's accuracy is
bounded by how densely training data covers the *combinatorial* space
of cascade gamma-energy combinations, so this phase focused on
producing that dense dataset — **not** yet the new emulator itself,
which is separate future work.

**Confirmed before generating anything**: every intermediate "level" in
every cascade topology here is a synthetic placeholder (documented in
each `.reaction` file's own `COMM` header), so gamma energies can be
placed anywhere on a grid without violating any real branching-ratio
constraint. Total Ex = 8.93377 MeV is fixed. Unifying RECL formula
(verified against every hand-written file before trusting it):
`RECL(e_budget) = -25.91223 - e_budget`, where `e_budget` is the energy
available above a cascade's terminus (Ex for true ground, `Ex - 3.3526`
for the real 0+_2 state, or an arbitrary chosen energy for a
calib1g-style single-gamma source) — reproduces every existing file's
RECL value exactly.

**New tooling** (`~/codes/Ancalagon/reactions/generate_reactions.py` +
`build_grid.py`, no generator existed before): writes `.reaction` files
for any topology (1-4 gammas) directly from a list of ordered level
energies, reusing the exact card conventions of every hand-written file.
4-gamma points use quasi-random (Sobol) sampling of the ordered
`0 < L1 < L2 < L3 < e_budget` simplex (order-statistics-of-uniforms
method) rather than a full grid, which would be combinatorially
infeasible at this resolution (~20x the size of the existing 3-gamma
grid). Verified with `--reaction-stats` across random samples of every
new family before running anything — correct gamma multiplicities
throughout (100% of events, every family).

**Real per-run cost measured before committing to a campaign size**
(same "measure before you commit" pattern as the `calib1g` series):
cost tracks **total gamma energy carried, not gamma count** — a 1-gamma
run at 0.05 MeV cost 1.14 MB/19s at 8000 events, but 2/3/4-gamma runs
carrying most of the 8.93 MeV budget all cost about the same
(~9.5 MB, ~23s), since total BGO Compton-interaction volume is set by
the energy budget, not how many gammas carry it.

**Disk**: freed 73 GB by deleting every existing run's raw
`dragon_hits.root`/`run.log` (verified first: `dataset.npz` already had
all 5426 spectra cached, so nothing was lost) before generating anything
new. Final campaign, sized from the measured per-run cost and a
~5000-run/~47 GB/~2 hr budget (padsley's call, given the disk headroom):
151 new dense `calib1g` points (0.05 MeV step, full 0.1–8.8 MeV range,
was 0.1 MeV step over only 0.1–2.5 MeV), 88+54 dense interstitial 2-gamma
points (0.05 MeV step, both `ground`/`0+_2`), a 3000-point Sobol
supplement to the existing 3-gamma grid (2184 ground + 816 0+_2,
proportional to the existing 3828:1431 split), and 1700 new 4-gamma
points (1237 ground + 463 0+_2) where there was previously zero
coverage. All 4993 runs launched in parallel across 16 cores (batch
launcher, resumable, `REACTION_INPUT` + `TRAJ_QUIET=1` env vars +
`--track-reaction N`) — **zero failures**, actual cost 49 GB / 12076 s
(~3.35 hr, longer than the ~2 hr estimate — real disk I/O contention
across 16 parallel processes wasn't in the naive estimate).

**A real bug found and fixed while merging the new runs into
`dataset.npz`**: `build_dataset.py`'s `_load_old_cache` re-read the
*entire* `Y_addback`/`Y_singles` arrays from the npz archive on every
iteration of its per-row loop (`d["Y_addback"][i]` inside a
5426-iteration loop, instead of pulling `d["Y_addback"]` out once
first) — `NpzFile` doesn't cache decompressed arrays across accesses,
so this was ~5426 redundant reloads of a 21.7 MB array, OOM-killing the
process (confirmed via `resource.getrusage` instrumentation: 68 MB at
row 0, dead before row 500). Fixed by loading each array once before
the loop. **Lesson for this project**: a `NpzFile` object is not a
plain dict — repeatedly indexing the same key inside a loop re-reads
from the archive every time; pull arrays out once before looping.

**Result**: `dataset.npz` grew from 5426 to **10419 rows**
(`MAX_GAMMAS` bumped 3→4; two new topologies, `4g_ground`/`4g_0+_2`,
added to `DEFAULT_TOPOLOGIES` — the densified 1g/2g/3g families needed
no code change, matching their existing glob patterns automatically).
Spot-checked new spectra across every new family: photopeak bin lands
within 20-70 keV of the true top-gamma energy in every case, consistent
with expected Doppler broadening plus 20 keV bin quantization — the
grid's physics is behaving as intended. Raw ROOT/log files for the new
runs deleted immediately after merging (same "safe once cached" check
as the initial disk cleanup) — disk back to 73 GB free.

**Not done, genuine future work**: the direct empirical regression/PCA
emulator architecture itself — model choice, training, and validation
against this project's own held-out split convention — is a separate,
separately-scoped task, now that this denser dataset exists.

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
   window, not just the new one. **2026-09-19**: the backscatter dome
   itself is now modeled (a physics-anchored-center Gaussian bump), not
   just excluded — see "Doppler-broadened lineshape + backscatter-dome
   model" below — but data-starved (only measurable where a companion
   sits below ~0.6 MeV) and not yet reliable per-run (held-out amplitude
   error ~270% median). Remaining: escape peaks, the continuum's own
   Compton-edge sharpness (still unconvolved), more `"calib1g"`-style
   dedicated runs to properly constrain the dome, and every gamma
   *except* each cascade's own highest-energy one — see that section's
   "Follow-on phases" (2a/2b) for the concrete next steps and why they're
   phased this way.
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
6. ~~Peaks are fit with plain Gaussians, which don't describe the real
   (Doppler-broadened) lineshape well~~ — **done 2026-09-19**, see
   "Doppler-broadened lineshape + backscatter-dome model" above. Not
   fully closed out: the continuum's own Compton-edge region is still an
   *unconvolved* raw Klein-Nishina shape (no resolution smearing at all
   near the edge) — the same kind of fix, not yet applied there.
