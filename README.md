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

`~/data/Ancalagon_k39_results/k39pg_40ca_cascade_01_0100keV` ...
`_88_8800keV` (88 completed Geant4 runs, 50000 events each), driven by
matching reaction files in `~/codes/Ancalagon/reactions/`. Each run is
the same ³⁹K(p,γ)⁴⁰Ca compound state (Ex = 8.93377 MeV, at the 606 keV
resonance) decaying through **one fictional intermediate level**, stepped
0.1 → 8.8 MeV in 0.1 MeV increments (file `NN` puts it at `NN × 0.1`
MeV) — a two-step cascade, 100% branching at each step, no branching-ratio
variation. This is **fundamentally different in shape from the old G3
dataset**: that one was a 4558-sample, 6-parameter grid (3 level energies
+ 3 branching ratios, all varying); this one is a dense **1-parameter
scan** designed to map the BGO array's photopeak response (position,
width, efficiency, escape/Compton structure) as a smooth function of true
gamma-ray energy, not to reproduce a specific measured cascade. Don't
port the old `ParametricSpectrumEmulator`'s architecture assuming the same
kind of input space — see "Next steps".

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
| `build_dataset.py` | Matches every `k39pg_40ca_cascade_*.reaction` file to its result directory by stem, runs both parsers, saves `dataset.npz`. |
| `dataset.npz` | Built output: `X` (88, 6) — `br(-1,1)`, `br(1,0)`, `eres`, `ex`, `level(1)`, `q_value` (only `level(1)` actually varies across rows; the rest are dataset-wide constants, kept for a future mixed-topology dataset); `Y_addback`/`Y_singles` (88, 500) — normalised spectra; `edges` (501,) bin edges in MeV; `n_total`/`n_hit` (88,) — simulated events / events with ≥1 BGO hit; `file_stems` (88,). |
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
  exactly (50000 for every run so far); addback vs. singles behave as
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

## Response function results (2026-09-14)

Fit via `fit_response_function.py` over all 88 runs (each contributing 2
known photopeak energies, 174 total, 1 fit failure — see caveats):

- **Resolution model**: `doppler_k = 0.0163 ± 0.00002`,
  `intrinsic_k = 0.0117 ± 0.00007` (same functional form as the G3
  project, refit fresh — different geometry/physics, coefficients are not
  expected to match and don't: G3 had `doppler_k=0.0210, intrinsic_k=0.0142`).
  `response_function_check.png`'s left panel shows an almost perfectly
  linear sigma-vs-E trend (Doppler-dominated above ~0.5 MeV, same
  qualitative finding as the G3 project), with measured points tracking
  the fitted curve closely outside the near-degenerate region below.
- **`predict_spectrum` validated visually**
  (`response_function_prediction_check.png`): both photopeak position and
  width are reproduced essentially exactly across a wide energy range
  (checked at 2.0/6.93 MeV and 1.93/7.0 MeV) — confirms the resolution
  model generalizes, not just fits its own training points.
- **Known caveats, not yet fixed**:
  - One run (`level(1)=4.5`, the two gammas only 66 keV apart — the
    closest-approach point in this scan) failed to converge; the
    near-degenerate region generally (roughly 3.8–5.2 MeV) has visibly
    larger scatter/error bars in `response_function_check.png` — a joint
    double-Gaussian fit is inherently harder to constrain when the two
    peaks strongly overlap. Not fixed; excluded via the `sigma_rel_err`
    quality cut in the global resolution fit.
  - The efficiency/amplitude curve (`response_function_check.png`, right
    panel) has a real, physical downward trend (photoelectric absorption
    less likely as E grows) but the lowest 2-3 energy points (~0.1-0.3
    MeV) are **not reliable**: a large near-zero-energy spike in the raw
    spectrum (partial-deposit events, not a physics peak) sits right next
    to the real photopeak there and leaks into the local quadratic
    background fit, suppressing the fitted amplitude — confirmed by
    inspecting the raw bin values directly (a real peak is visible at
    ~0.09-0.11 MeV once you look past the spike at ~0.01 MeV). Not fixed;
    `full_energy_amplitude()` will under-report efficiency below ~0.3-0.5
    MeV until this is addressed (e.g. exclude the near-zero region
    explicitly from the local background fit, or model it as its own
    component).
  - **Photopeak component only**: Compton continuum, escape peaks, and
    Compton edges are not modeled (`predict_spectrum` will look like the
    orange curves in `response_function_prediction_check.png`, not the
    full blue spectrum). This needs a genuinely joint fit across runs
    (each run's continuum is itself a superposition from both its
    gammas), which is real, separate follow-up work.

## Held-out validation (2026-09-14 follow-up)

The result above was originally fit on **all 88 runs at once** — no
train/test split — so the "prediction check" plot only showed the model
reproducing runs it had itself been fit on, not genuine generalization.
`validate_holdout.py` redoes this properly: holds out every 5th run on
the `level(1)` grid (17 runs / 33 peak measurements, spread across the
whole 0.1–8.8 MeV range), refits `doppler_k`/`intrinsic_k` and the
efficiency interpolation on the remaining 70 runs only, then checks both
against the runs the fit never saw.

- **Fitted parameters barely move**: train-only
  `doppler_k=0.01628±0.00003, intrinsic_k=0.01153±0.00008` vs.
  all-data `doppler_k=0.0163, intrinsic_k=0.0117` — consistent to well
  within 1-sigma.
- **Held-out chi2/ndf (56.2) is close to train chi2/ndf (49.3)** — if the
  2-parameter resolution model were overfitting the training points, held-out
  chi2 would be much worse than train chi2; it isn't. The elevated chi2/ndf
  itself (~50, not ~1) is consistent with the already-documented per-peak
  local-background-model imperfection, not with overfitting — it affects
  train and held-out points equally.
- **Held-out relative differences**: sigma median 4.5% / mean 6.4% / max
  53.3%; amplitude median 1.9% / mean 5.2% / max 59.6% — small typical
  errors, with the max outliers traced to the *already-documented* problem
  points, not new ones: the 53.3% sigma outlier is `E=4.334 MeV` (the
  `level(1)=4.6` run's low gamma, inside the known near-degenerate
  ~3.8-5.2 MeV region), and the 59.6% amplitude outlier is `E=0.334 MeV`
  (the `level(1)=8.6` run's low gamma, below ~0.5 MeV — the known
  near-zero-spike contamination zone). See `holdout_validation.png`.
- **Conclusion**: the resolution model and efficiency curve generalize
  about as well on unseen energies as they fit their own training points
  — the fit is not overfitting the 88-point grid. The shipped
  `response_function.npz`/`response_function.py` still use **all 88
  runs** (standard practice: validate with a holdout split, then ship the
  full-data fit for best accuracy) — only `validate_holdout.py`'s own
  run uses a train-only subset, and only for this check.

## Next steps

1. Model the Compton continuum + escape peaks/Compton edge, most likely
   via a joint NNLS-style decomposition across all 88 runs simultaneously
   (same spirit as the G3 project's approach, but here every basis
   function's position is known analytically in advance, no PCA needed).
2. Fix the low-energy (<~0.5 MeV) amplitude contamination from the
   near-zero spike (see above) before trusting the efficiency curve there.
3. Investigate/improve the near-degenerate region (~3.8-5.2 MeV) — wider
   or better-conditioned joint fits, or a different parameterization that
   doesn't require independently resolving two heavily-overlapping peaks.
4. Only one cascade topology exists so far (single intermediate level,
   100/100% branching) — if this response function is to feed into a
   future full-cascade emulator (once Ancalagon has branching-ratio
   variation and a working recoil gate), that needs new Ancalagon runs,
   not just more of the current scan.
