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

Two cascade topologies, both two-step scans of the same ³⁹K(p,γ)⁴⁰Ca
606 keV resonance, combined into one 142-run `dataset.npz` (see
`build_dataset.py`'s `DEFAULT_TOPOLOGIES`):

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

Both topologies share the same LEVL/BRAT structure (one intermediate
level, 100/100% branching) — `parse_reaction.py`'s generic level-difference
logic computes correct *observable* gamma energies for both with no
topology-specific code, and `build_dataset.py` tags each row with a
`topology` field purely for provenance/filtering. Every run is a
**1-parameter scan** designed to map the BGO array's photopeak response
(position, width, efficiency) as a smooth function of true gamma-ray
energy, not to reproduce a specific measured cascade — this is
**fundamentally different in shape from the old G3 dataset** (a
4558-sample, 6-parameter grid). Don't port the old
`ParametricSpectrumEmulator`'s architecture assuming the same kind of
input space — see "Next steps".

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

## Files

| File | Purpose |
|------|---------|
| `parse_reaction.py` | Parses Ancalagon `.reaction` files (`BEAM`/`TARG`/`RECL`/`ERES`/`LEVL`/`BRAT` cards) into level energies, branching, derived Q-value/Ex, and enumerates every resonance→ground cascade path with its energies and probability. Generalizes beyond the 2-level scan files (tested against the bundled `o15ag_19ne.reaction`, a real 4-level cascade). |
| `extract_spectrum.py` | Per-event BGO spectrum from `dragon_hits.root` — `"addback"` or `"singles"` energy definition (see above), normalised by `n_total` (from `run.log`, since zero-hit events leave no row in the ROOT tree at all). |
| `build_dataset.py` | Matches each configured topology's reaction files to their result directories by stem, runs both parsers, tags each row with its `topology`, saves the combined `dataset.npz`. |
| `dataset.npz` | Built output: `X` (142, 6) — `br(-1,1)`, `br(1,0)`, `eres`, `ex`, `level(1)`, `q_value` (`level(1)` varies within each topology; `ex`/`q_value` are constant *within* a topology but differ *between* them — 8.93377 MeV for `"ground"`, 5.58117 MeV for `"0+_2"`); `Y_addback`/`Y_singles` (142, 500) — normalised spectra; `edges` (501,) bin edges in MeV; `n_total`/`n_hit` (142,) — simulated events / events with ≥1 BGO hit (`n_total` is 50000 for `"ground"` rows, 600000 for `"0+_2"` rows); `file_stems`/`topology` (142,). |
| `sanity_check_spectra.png` | Three example spectra (level = 0.5, 4.5, 8.5 MeV), addback vs. singles overlaid — used to visually confirm the pipeline (see "Verification" below). |
| `fit_response_function.py` | For each run, fits both known photopeaks in `Y_singles` (local single-Gaussian fits if well-separated, joint double-Gaussian if the two peaks' windows overlap — happens for `level(1))` roughly 3.8–5.2 MeV) → per-energy (sigma, amplitude) measurements → global fit of `sigma(E)^2 = (doppler_k*E)^2 + intrinsic_k^2*E`. Saves `response_function.npz`. |
| `response_function.py` | `BgoResponseFunction` class: loads the fitted model, predicts photopeak position/width/amplitude at any true gamma energy, and sums photopeaks over an arbitrary cascade's gamma list (`predict_spectrum`). Photopeak component only — see its own docstring. |
| `response_function_check.png` | Sigma-vs-energy and amplitude-vs-energy scatter plots with the fitted resolution curve overlaid. |
| `response_function_prediction_check.png` | `predict_spectrum` overlaid on two real spectra — confirms both photopeak positions and widths are reproduced accurately across the whole energy range. **Caveat**: the two runs shown here (`level=2000keV`/`7000keV`) were part of the fit's own training data, not a held-out check — see `validate_holdout.py` below for the real generalization test. |
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

## Held-out validation

`validate_holdout.py` holds out every 5th run across the *combined*
142-run dataset (28 runs / 56 peak measurements: 10 from `"0+_2"`, 18
from `"ground"`), refits on the remaining 113 runs, checks both the
resolution model and efficiency curve against runs the fit never saw.

- Train-only fit: `doppler_k=0.01543±0.00001, intrinsic_k=0.01280±0.00003`
  — consistent with the all-data fit above.
- Held-out chi2/ndf (325.8) tracks train chi2/ndf (304.1) — not
  overfitting, same conclusion as before just at the new, higher
  chi2/ndf scale (see "chi2/ndf rose sharply" above for why that scale
  moved).
- Held-out sigma relative difference: median 4.9%, mean 8.2%, max 54.2%
  — essentially unchanged from the ground-only result.
- Held-out amplitude relative difference: median 8.7%, mean 21.0%, max
  134.8% — **materially worse than before** (was 1.9%/5.2%/59.6%); this
  is the efficiency-interpolation issue above showing up quantitatively
  in the held-out check, not a new held-out-specific problem — a real,
  now-measured degradation worth fixing before trusting
  `full_energy_amplitude()` between ~2-3.5 MeV.
- The shipped `response_function.npz` still uses **all 142 runs** for
  the same reason as before (validate with a split, ship the full-data
  fit); only `validate_holdout.py`'s own run uses a train-only subset.

## Next steps

1. Fix the efficiency-curve interpolation issue above — e.g. exclude
   `joint=True` amplitude measurements from `build_efficiency_curve`
   entirely (they're already known less-reliable), or fit a smooth
   parametric efficiency(E) model instead of raw `np.interp` so a single
   noisy neighbor can't dominate a nearby held-out prediction.
2. Model the Compton continuum + escape peaks/Compton edge, most likely
   via a joint NNLS-style decomposition across all runs simultaneously
   (same spirit as the G3 project's approach, but here every basis
   function's position is known analytically in advance, no PCA needed).
   The 0+_2 topology's much higher statistics should help constrain this.
3. Fix the low-energy (<~0.5 MeV) amplitude contamination from the
   near-zero spike (unchanged from before).
4. Investigate/improve the near-degenerate region in each topology (each
   has its own, at a different absolute energy — see above) — wider or
   better-conditioned joint fits, or a parameterization that doesn't
   require independently resolving two heavily-overlapping peaks.
5. Both topologies still have only one cascade shape (single intermediate
   level, 100/100% branching) — branching-ratio variation needs new
   Ancalagon runs, not just more of either current scan.
