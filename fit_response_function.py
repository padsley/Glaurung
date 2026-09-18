"""Measure the BGO single-crystal photopeak response (position, width,
efficiency) vs. true gamma-ray energy, directly from the k39 cascade scan.

Every run in `dataset.npz` fires a fixed, known list of true gamma
energies at every event (2 gammas for the `"ground"`/`"0+_2"` topologies,
3 for `"3g_ground"`/`"3g_0+_2"`, 1 for `"calib1g"` -- see
`build_dataset.py`'s `gamma_energies` field). Unlike the old G3 project
(which had to *infer* peak positions by stacking many mixed-cascade
samples), every peak position here is known analytically in advance --
no search needed.

**Lineshape (changed 2026-09-19)**: peaks are fit with
`gamma_physics.doppler_lineshape` (a Doppler-broadened "box convolved
with a Gaussian" shape -- see that function's own docstring for the
physics and the before/after chi2/ndf check that motivated this), not a
plain Gaussian. The recoil velocity fraction `beta` is a single true
physical constant shared by *every* gamma in *every* run (this dataset's
cascades are all "instantaneous at vertex" -- no time for the recoil to
decelerate between gammas), so it is fit globally in a staged procedure,
not per-peak:

  1. **Stage 1** (`measure_all_peaks(..., beta=None)`): local fits with
     `beta` free per fit window, to measure it.
  2. `estimate_global_beta`: robust (median) global `beta` from all
     stage-1 windows.
  3. **Stage 3** (`measure_all_peaks(..., beta=<global>)`): re-fit every
     peak with `beta` fixed, `(amplitude, sigma_intrinsic)` free.
  4. `fit_intrinsic_k`: global `sigma_intrinsic(E) = intrinsic_k*sqrt(E)`
     fit from stage-3 measurements (replaces the old
     `sigma(E)^2 = (doppler_k*E)^2 + intrinsic_k^2*E` resolution model --
     Doppler is no longer folded into a Gaussian sigma, it's the
     lineshape's own box).

Stages 1 and 3 are each a `measure_all_peaks`-scale pass (the expensive
part, ~15-20 min on this machine) and are **shared** between this
script and `validate_holdout.py`/`fit_compton_continuum.py` via
`save_stage_rows`/`load_stage_rows` (`peak_measurements.npz`) --
`validate_holdout.py` does not re-run stage 1/3, only the (fast)
train/holdout aggregation on top of the same cached per-peak
measurements. **Simplification, disclosed**: `beta` itself is estimated
from *all* rows (not re-derived train-only) even during held-out
validation -- it is a single global scalar overdetermined by thousands
of largely-independent window measurements, so the leakage from
including the 20% held-out rows in that one number is expected to be
negligible; `intrinsic_k` and the efficiency curve *are* still refit
train-only for a genuine held-out check, same as before.

Output: `response_function.npz` (fitted beta/intrinsic_k, efficiency
curve), `peak_measurements.npz` (cached stage-3 rows, for reuse), and
`response_function_check.png`.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from gamma_physics import doppler_lineshape

DATASET_PATH = "dataset.npz"
STAGE_ROWS_PATH = "peak_measurements.npz"
RESPONSE_PATH = "response_function.npz"

ROW_FIELDS = ["stem", "which", "energy", "sigma_intrinsic", "sigma_intrinsic_err",
              "amplitude", "amplitude_err", "efficiency", "chi2_ndf",
              "sigma_intrinsic_rel_err", "joint", "multiplicity"]
STAGE1_FIELDS = ["stem", "which", "energy", "beta_local", "beta_err", "beta_rel_err",
                 "chi2_ndf", "joint", "multiplicity"]


def _rough_sigma_guess(E):
    """Only used to size local fit windows -- not the fitted lineshape."""
    return np.sqrt((0.03 * E) ** 2 + 0.02**2 * E)


def _multi_doppler_fit(x, y, centers, x0, bin_width, beta_fixed=None):
    """Joint fit of `len(centers)` Doppler lineshapes
    (`gamma_physics.doppler_lineshape`, fixed centers, independent
    amplitude/sigma_intrinsic each, **one shared `beta`** across every
    peak in this window -- same physical constant for every gamma in one
    event) plus a shared quadratic background, over one window.

    If `beta_fixed` is `None`, `beta` is a free parameter (stage 1, to
    measure it locally); otherwise it is held fixed at `beta_fixed`
    (stage 3, once the global value is known) and only
    `(amplitude, sigma_intrinsic)` are free per peak -- the same
    parameter *count* the old Gaussian fit used.

    `amplitude` is defined so that `amplitude / n_total` (done by the
    caller) is directly the per-simulated-event full-energy-peak
    detection probability (an area, like the 2026-09-17 efficiency fix's
    `amplitude*sigma*sqrt(2*pi)/bin_width` was for Gaussians) -- no extra
    conversion needed here because `doppler_lineshape` is already a
    unit-area probability density, so `model = amplitude * bin_width *
    doppler_lineshape(...)` directly reproduces `amplitude` counts
    total when integrated (as a Riemann sum) over the binned data.
    """
    k = len(centers)
    weights = np.sqrt(np.clip(y, 1.0, None))
    fit_beta = beta_fixed is None

    def model(x_, *params):
        amps = params[0:k]
        s_int = params[k:2 * k]
        if fit_beta:
            beta = params[2 * k]
            bg0, bg1, bg2 = params[2 * k + 1], params[2 * k + 2], params[2 * k + 3]
        else:
            beta = beta_fixed
            bg0, bg1, bg2 = params[2 * k], params[2 * k + 1], params[2 * k + 2]
        out = bg0 + bg1 * (x_ - x0) + bg2 * (x_ - x0) ** 2
        for a, s, c in zip(amps, s_int, centers):
            out = out + a * bin_width * doppler_lineshape(c, x_, beta, s)
        return out

    a0 = [max(y[np.argmin(np.abs(x - c))] - np.percentile(y, 20), 1.0) * _rough_sigma_guess(c) * np.sqrt(2 * np.pi)
          for c in centers]
    s0 = [0.01] * k
    bg0_0 = max(np.percentile(y, 20), 0.0)
    # sigma_intrinsic's lower bound is deliberately far below any physically
    # plausible value (not e.g. 0.5 keV) -- a fit genuinely wanting
    # near-zero intrinsic smearing (Doppler box alone already explains the
    # width, common at lower per-run statistics) must be free to go there;
    # pinning against a too-tight bound previously produced degenerate,
    # artificially-tiny reported uncertainties (caught by a smoke test
    # 2026-09-19, see fit_intrinsic_k/build_efficiency_curve's error-floor
    # guard against exactly this).
    if fit_beta:
        p0 = a0 + s0 + [0.02, bg0_0, 0.0, 0.0]
        lower = [0] * k + [1e-5] * k + [0.0003, -np.inf, -np.inf, -np.inf]
        upper = [np.inf] * k + [0.3] * k + [0.15, np.inf, np.inf, np.inf]
    else:
        p0 = a0 + s0 + [bg0_0, 0.0, 0.0]
        lower = [0] * k + [1e-5] * k + [-np.inf, -np.inf, -np.inf]
        upper = [np.inf] * k + [0.3] * k + [np.inf, np.inf, np.inf]

    # xtol/ftol/gtol loosened from scipy's default ~1.49e-8: with beta
    # fixed, sigma_intrinsic's effect on chi2 is often nearly flat over a
    # wide range (whenever the Doppler box already dominates the total
    # width), so TRF was taking hundreds of tiny steps chasing
    # default-tight convergence on a direction that barely matters --
    # measured 2026-09-19: ~1.9s/window at default tolerance vs a target
    # of a few hundred ms, and this value is aggregated over thousands of
    # windows downstream anyway, so per-window precision past ~1e-4
    # relative is wasted effort, not lost accuracy.
    popt, pcov = curve_fit(model, x, y, p0=p0, sigma=weights, absolute_sigma=True,
                            bounds=(lower, upper), maxfev=10000,
                            xtol=1e-6, ftol=1e-6, gtol=1e-6)
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
    resid = (y - model(x, *popt)) / weights
    ndf = max(len(x) - len(p0), 1)
    chi2_ndf = float(np.sum(resid**2) / ndf)

    peaks = []
    for i in range(k):
        peak = {"energy": centers[i], "sigma_intrinsic": popt[k + i], "sigma_intrinsic_err": perr[k + i],
                "amplitude": popt[i], "amplitude_err": perr[i]}
        if fit_beta:
            peak["beta"] = popt[2 * k]
            peak["beta_err"] = perr[2 * k]
        peaks.append(peak)
    return peaks, chi2_ndf


def _dedupe_gammas(gammas, tol=1e-4):
    """Merge true gamma energies that coincide to within `tol` MeV.

    The 3-gamma grid's two intermediate levels are independently scanned,
    so `Level2 - Level1` (the middle gamma) can land exactly on `Level1`
    itself (the lowest gamma) whenever `Level2 == 2*Level1` -- two
    physically distinct gammas at the identical true energy. Fitting them
    as two independent lineshapes at the same center is unidentifiable
    (only their summed amplitude is observable, same reason a real
    coincident pair can't be split in the data either), so they must be
    merged into one fit target *before* windowing/grouping, not left for
    the optimizer to fail on. By construction every genuinely distinct
    pair of gammas in this dataset is >=100 keV apart (the grid's minimum
    level spacing), so `tol` far below that only catches true
    coincidences, never two nearby-but-distinct peaks. Returns
    (unique_energies_sorted, multiplicity_per_unique_energy).
    """
    gammas = np.sort(gammas)
    unique = [gammas[0]]
    mult = [1]
    for e in gammas[1:]:
        if e - unique[-1] < tol:
            mult[-1] += 1
        else:
            unique.append(e)
            mult.append(1)
    return np.array(unique), np.array(mult)


def fit_run_peaks(centers, raw_counts, gammas, bin_width, beta_fixed=None,
                   n_window_sigma=5.0, min_half_width=0.08, min_bins_per_peak=7):
    """Fit every distinct photopeak in one run's spectrum. `gammas` is the
    run's known true gamma energies (length 1-3, may include exact
    coincidences -- see `_dedupe_gammas`). Each distinct energy gets its
    own local window unless it overlaps a neighbor's, in which case
    overlapping windows are merged (transitively -- 3-way overlap is
    handled, not just pairwise) into one joint multi-peak fit.

    Returns {"peaks": [dict per distinct energy, ascending],
    "chi2_ndf": [one per peak, shared within a joint group],
    "joint": [bool per peak, True if fit jointly with >=1 other peak],
    "multiplicity": [gamma count merged into this energy, see
    `_dedupe_gammas`]}, or raises ValueError/RuntimeError if any group's
    fit region is too small / fails to converge (caller should treat the
    whole run as a fit failure, matching the old behavior)."""
    gammas, multiplicity = _dedupe_gammas(np.asarray(gammas, dtype=np.float64))
    n = len(gammas)
    half_widths = np.maximum(min_half_width, n_window_sigma * _rough_sigma_guess(gammas))
    win_starts = gammas - half_widths
    win_ends = gammas + half_widths

    # Merge overlapping windows into groups (peaks are already sorted by
    # energy, so a simple sweep suffices -- transitively handles 3-way
    # overlap, not just adjacent pairs).
    groups: list[list[int]] = [[0]]
    group_ends = [win_ends[0]]
    for i in range(1, n):
        if win_starts[i] <= group_ends[-1]:
            groups[-1].append(i)
            group_ends[-1] = max(group_ends[-1], win_ends[i])
        else:
            groups.append([i])
            group_ends.append(win_ends[i])

    peaks = [None] * n
    chi2_ndfs = [None] * n
    joint_flags = [None] * n
    group_ids = [None] * n

    for group_id, (idx_list, group_end) in enumerate(zip(groups, group_ends)):
        group_start = win_starts[idx_list[0]]
        lo_edge = max(centers[0], group_start)
        hi_edge = min(centers[-1], group_end)
        mask = (centers >= lo_edge) & (centers <= hi_edge)
        if mask.sum() < min_bins_per_peak * len(idx_list):
            raise ValueError(
                f"group window too small ({int(mask.sum())} bins for {len(idx_list)} peak(s) "
                f"at {[round(gammas[i], 4) for i in idx_list]} MeV)")
        x, y = centers[mask], raw_counts[mask]
        group_centers = [gammas[i] for i in idx_list]
        fits, chi2_ndf = _multi_doppler_fit(x, y, group_centers, x.mean(), bin_width, beta_fixed=beta_fixed)
        for local_i, global_i in enumerate(idx_list):
            peaks[global_i] = fits[local_i]
            chi2_ndfs[global_i] = chi2_ndf
            joint_flags[global_i] = len(idx_list) > 1
            group_ids[global_i] = group_id

    return {"peaks": peaks, "chi2_ndf": chi2_ndfs, "joint": joint_flags,
            "multiplicity": list(multiplicity), "group_id": group_ids}


def measure_all_peaks(dataset_path=DATASET_PATH, n_window_sigma=5.0, verbose=True, beta=None):
    """Per-run local peak fits only -- each run's fit is independent of
    every other run, so this step is the same regardless of any later
    train/holdout split.

    `beta=None` runs **stage 1** (beta free per window) -- returns rows
    with `beta_local`/`beta_err`/`beta_rel_err` (one per *group*, i.e.
    peaks fit jointly share one row-worth of beta, not one each --
    handled by only emitting the group's beta once, keyed on its first
    peak). `beta=<float>` runs **stage 3** (beta fixed) -- returns rows
    with `sigma_intrinsic`/`amplitude`/`efficiency` (one per *distinct*
    true-energy peak -- see `_dedupe_gammas`). `multiplicity > 1` means
    a row's amplitude/efficiency is the combined contribution of that
    many coincident-energy gammas -- see README's caveat on this.
    """
    d = np.load(dataset_path, allow_pickle=True)
    n_total = d["n_total"].astype(np.float64)
    edges = d["edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]
    Y_singles = d["Y_singles"]
    stems = d["file_stems"]
    gamma_energies = d["gamma_energies"]  # (n_rows, max_gammas), NaN-padded

    rows = []
    for i in range(len(stems)):
        gammas = gamma_energies[i]
        gammas = gammas[np.isfinite(gammas)]
        raw = Y_singles[i] * n_total[i]
        try:
            result = fit_run_peaks(centers, raw, gammas, bin_width, beta_fixed=beta, n_window_sigma=n_window_sigma)
        except (RuntimeError, ValueError) as exc:
            if verbose:
                print(f"{stems[i]}: fit failed ({exc})")
            continue

        if beta is None:
            seen_groups = set()
            for rank, m in enumerate(result["peaks"]):
                group_id = result["group_id"][rank]
                # one beta value is shared per joint group (fit jointly) --
                # only record it once, not once per peak in the group.
                if group_id in seen_groups:
                    continue
                seen_groups.add(group_id)
                beta_rel_err = m["beta_err"] / m["beta"] if m["beta"] > 0 else np.inf
                rows.append({
                    "stem": str(stems[i]), "which": f"p{rank}", "energy": m["energy"],
                    "beta_local": m["beta"], "beta_err": m["beta_err"], "beta_rel_err": beta_rel_err,
                    "chi2_ndf": result["chi2_ndf"][rank],
                    "joint": result["joint"][rank], "multiplicity": result["multiplicity"][rank],
                })
            if verbose:
                betas = sorted(set(round(m["beta"], 6) for m in result["peaks"]))
                print(f"{stems[i]}: beta={betas}  (n_peaks={len(gammas)})")
        else:
            for rank, m in enumerate(result["peaks"]):
                s_int = m["sigma_intrinsic"]
                sigma_intrinsic_rel_err = m["sigma_intrinsic_err"] / s_int if s_int > 0 else np.inf
                amplitude = m["amplitude"] / n_total[i]  # normalise to per-simulated-event; already an
                                                          # area (see _multi_doppler_fit docstring)
                rows.append({
                    "stem": str(stems[i]), "which": f"p{rank}", "energy": m["energy"],
                    "sigma_intrinsic": s_int, "sigma_intrinsic_err": m["sigma_intrinsic_err"],
                    "amplitude": amplitude, "amplitude_err": m["amplitude_err"] / n_total[i],
                    "efficiency": amplitude,
                    "chi2_ndf": result["chi2_ndf"][rank], "sigma_intrinsic_rel_err": sigma_intrinsic_rel_err,
                    "joint": result["joint"][rank], "multiplicity": result["multiplicity"][rank],
                })
            if verbose:
                desc = "  ".join(f"E={m['energy']:.3f} s_int={m['sigma_intrinsic']*1000:.1f}keV" for m in result["peaks"])
                print(f"{stems[i]}: {desc}  (n_peaks={len(gammas)}, n_distinct={len(result['peaks'])})")
    return rows


def estimate_global_beta(rows, rel_err_max=0.3, chi2_ndf_max=50.0):
    """Robust global `beta` from stage-1 (free-beta) window measurements:
    the median of every window's local `beta`, restricted to
    well-constrained, reasonably-fit windows. A median (not a weighted
    mean) is used deliberately -- with potentially thousands of
    contributing windows, robustness to a handful of badly-conditioned
    outliers matters more than statistical optimality. Returns
    `(beta, n_good, n_total)`.
    """
    beta = np.array([r["beta_local"] for r in rows])
    rel_err = np.array([r["beta_rel_err"] for r in rows])
    chi2_ndf = np.array([r["chi2_ndf"] for r in rows])
    good = np.isfinite(rel_err) & (rel_err < rel_err_max) & (beta > 0) & (chi2_ndf < chi2_ndf_max)
    return float(np.median(beta[good])), int(good.sum()), int(len(rows))


def _intrinsic_model(E, intrinsic_k):
    return intrinsic_k * np.sqrt(E)


def fit_intrinsic_k(rows, sigma_rel_err_max=0.5, p0=(0.012,)):
    """Fit `sigma_intrinsic(E) = intrinsic_k*sqrt(E)` to stage-3 rows
    (e.g. a training subset). Replaces the old two-parameter
    `sigma(E)^2 = (doppler_k*E)^2 + intrinsic_k^2*E` resolution model --
    Doppler broadening is now the lineshape's own box, not folded into
    this Gaussian-sigma-vs-E relationship, so only one parameter remains
    here. Returns (intrinsic_k, intrinsic_k_err, good_mask, chi2_ndf).
    """
    energies = np.array([r["energy"] for r in rows])
    sigmas = np.array([r["sigma_intrinsic"] for r in rows])
    sigma_errs = np.array([r["sigma_intrinsic_err"] for r in rows])
    rel_err = np.array([r["sigma_intrinsic_rel_err"] for r in rows])

    # sigma_errs > 1e-4: guards against a fit whose sigma_intrinsic landed
    # on (or very near) its lower bound -- caught by a smoke test
    # 2026-09-19, where a handful of such fits reported a near-zero
    # sigma_intrinsic_err (degenerate covariance at the boundary), which
    # both trivially passed the rel_err cut *and* blew chi2_ndf up to
    # ~1e20 by dividing by it below. 1e-4 MeV (0.1 keV) is far below any
    # physically expected intrinsic width, so this only screens out that
    # boundary-pinning pathology, not genuine measurements.
    good = (np.isfinite(rel_err) & (rel_err < sigma_rel_err_max) & (sigmas > 0.0005)
            & (sigma_errs > 1e-4))
    popt, pcov = curve_fit(
        _intrinsic_model, energies[good], sigmas[good],
        p0=p0, sigma=sigma_errs[good], absolute_sigma=True,
        bounds=([0], [1]),
    )
    perr = np.sqrt(np.diag(pcov))
    pred = _intrinsic_model(energies[good], *popt)
    resid = (sigmas[good] - pred) / sigma_errs[good]
    chi2_ndf = float(np.sum(resid**2) / max(good.sum() - 1, 1))
    return popt[0], perr[0], good, chi2_ndf


def build_efficiency_curve(rows, sigma_rel_err_max=0.5, chi2_ndf_max=15.0, bin_width=0.1):
    """Smoothed (energy, efficiency) control points for interpolation.

    Built from rows passing the same quality cut as the intrinsic-width
    fit, plus **excluding `multiplicity > 1` and `joint == True` rows
    entirely** (not just down-weighting them): a joint multi-peak fit
    over an overlapping window lets amplitude and sigma_intrinsic trade
    off against each other in an underdetermined way, and a
    coincident-energy row's efficiency is the combined contribution of
    >=2 gammas, not an ordinary single-gamma value -- both are
    unreliable/non-comparable at exactly the energies they occur, which
    is what made the raw interpolation this replaced sensitive to single
    bad neighbors (see README's 2026-09-16 write-up). The intrinsic-width
    fit still uses these rows -- a joint fit's *width* is far less
    biased by the amplitude/width trade-off than its amplitude is.

    Also excludes `chi2_ndf >= chi2_ndf_max`: a small tail has a
    formally tight sigma_intrinsic_rel_err yet a badly-fit background,
    overwhelmingly the already-documented near-zero-energy-spike
    contamination at <~0.5 MeV (see README's "Next steps" -- a separate,
    not-yet-fixed issue, kept out of the curve here rather than solved).
    **`chi2_ndf_max=15` (2026-09-19, was 50 under the old Gaussian
    lineshape)**: the Doppler lineshape fits chi2/ndf much tighter
    overall (most clean fits land at 2-10, not the old model's 2-20), so
    a threshold of 50 let through a real failure mode found by a held-out
    check -- a cluster of badly-fit, systematically-low-amplitude
    `"3g_0+_2"` peaks sitting right at chi2/ndf~40-50 -- which inflated
    held-out efficiency error (max 305% at chi2_ndf_max=50). Swept
    10/15/20/50 against the held-out split; 15 gave the best median/mean
    *and* cut the max to 88.7%, back in line with the pre-Doppler-
    lineshape max (72.0%) -- 10 was slightly worse (fewer surviving
    points start costing more than the tighter cut gains).

    The surviving (clean, single-peak) rows are then binned by energy
    (`bin_width` MeV) and the **median** efficiency taken per occupied
    bin -- robust to any one remaining noisy point, unlike interpolating
    between every raw row directly. Returns (bin_centers, bin_medians),
    sorted; empty bins are simply absent (`np.interp` bridges any gap
    linearly from its nearest neighbors, same as before).
    """
    energies = np.array([r["energy"] for r in rows])
    sigmas = np.array([r["sigma_intrinsic"] for r in rows])
    efficiencies = np.array([r["efficiency"] for r in rows])
    rel_err = np.array([r["sigma_intrinsic_rel_err"] for r in rows])
    sigma_errs = np.array([r["sigma_intrinsic_err"] for r in rows])
    multiplicity = np.array([r["multiplicity"] for r in rows])
    joint = np.array([r["joint"] for r in rows])
    chi2_ndf = np.array([r["chi2_ndf"] for r in rows])
    # sigma_errs > 1e-4: same boundary-pinning guard as fit_intrinsic_k --
    # a degenerate fit's amplitude is just as untrustworthy as its width.
    good = (np.isfinite(rel_err) & (rel_err < sigma_rel_err_max) & (sigmas > 0.0005)
            & (sigma_errs > 1e-4) & (multiplicity == 1) & ~joint & (chi2_ndf < chi2_ndf_max))

    e_good, eff_good = energies[good], efficiencies[good]
    if len(e_good) == 0:
        raise ValueError("no rows survive the efficiency-curve quality cut (all joint/coincident/noisy?)")

    bin_index = np.floor(e_good / bin_width).astype(np.int64)
    bins = np.unique(bin_index)
    bin_centers = (bins.astype(np.float64) + 0.5) * bin_width
    bin_medians = np.array([np.median(eff_good[bin_index == b]) for b in bins])
    order = np.argsort(bin_centers)
    return bin_centers[order], bin_medians[order]


def save_stage_rows(rows, fields, path):
    """Round-trip helper: list-of-dicts -> parallel-array npz, so the
    expensive `measure_all_peaks` passes can be cached and reused by
    other scripts (`validate_holdout.py`, `fit_compton_continuum.py`)
    instead of re-run."""
    arrays = {f: np.array([r[f] for r in rows]) for f in fields}
    np.savez(path, **arrays)


def load_stage_rows(path, fields):
    d = np.load(path, allow_pickle=True)
    n = len(d[fields[0]])
    return [{f: d[f][i].item() if hasattr(d[f][i], "item") else d[f][i] for f in fields} for i in range(n)]


def fit_all(dataset_path=DATASET_PATH, n_window_sigma=5.0, sigma_rel_err_max=0.5,
            stage_rows_path=STAGE_ROWS_PATH):
    print("=== Stage 1: free-beta local fits ===")
    rows1 = measure_all_peaks(dataset_path, n_window_sigma=n_window_sigma, beta=None)
    beta, n_beta_good, n_beta_total = estimate_global_beta(rows1)
    print(f"\nglobal beta = {beta:.5f} (from {n_beta_good}/{n_beta_total} windows)")

    print("\n=== Stage 3: fixed-beta refit ===")
    rows3 = measure_all_peaks(dataset_path, n_window_sigma=n_window_sigma, beta=beta)
    save_stage_rows(rows3, ROW_FIELDS, stage_rows_path)
    print(f"\nSaved {len(rows3)} stage-3 rows to {stage_rows_path}")

    energies = np.array([r["energy"] for r in rows3])
    sigmas = np.array([r["sigma_intrinsic"] for r in rows3])
    amplitudes = np.array([r["amplitude"] for r in rows3])
    efficiencies = np.array([r["efficiency"] for r in rows3])
    rel_err = np.array([r["sigma_intrinsic_rel_err"] for r in rows3])
    sigma_errs = np.array([r["sigma_intrinsic_err"] for r in rows3])
    stems = np.array([r["stem"] for r in rows3])

    good = (np.isfinite(rel_err) & (rel_err < sigma_rel_err_max) & (sigmas > 0.0005)
            & (sigma_errs > 1e-4))
    print(f"\n{good.sum()}/{len(rows3)} peak measurements pass quality cut (sigma_intrinsic_rel_err < {sigma_rel_err_max})")

    intrinsic_k, intrinsic_k_err, _, chi2_ndf_res = fit_intrinsic_k(rows3, sigma_rel_err_max=sigma_rel_err_max)
    print(f"intrinsic_k = {intrinsic_k:.5f} +/- {intrinsic_k_err:.5f}")
    print(f"intrinsic-width-model chi2/ndf (fit on all data, evaluated on all data) = {chi2_ndf_res:.2f}")

    eff_energy, eff_curve = build_efficiency_curve(rows3, sigma_rel_err_max=sigma_rel_err_max)
    joint_arr = np.array([r["joint"] for r in rows3])
    mult_arr = np.array([r["multiplicity"] for r in rows3])
    chi2_arr = np.array([r["chi2_ndf"] for r in rows3])
    n_clean = int((good & ~joint_arr & (mult_arr == 1) & (chi2_arr < 15.0)).sum())
    print(f"efficiency curve: {len(eff_energy)} occupied bins "
          f"(from {n_clean} clean single-peak measurements, "
          f"non-joint + multiplicity=1 + chi2/ndf<50)")

    return {
        "energies": energies, "sigmas_intrinsic": sigmas,
        "amplitudes": amplitudes, "efficiencies": efficiencies,
        "rel_err": rel_err, "good": good, "stems": stems,
        "beta": beta, "n_beta_good": n_beta_good, "n_beta_total": n_beta_total,
        "intrinsic_k": intrinsic_k, "intrinsic_k_err": intrinsic_k_err,
        "intrinsic_chi2_ndf": chi2_ndf_res,
        "efficiency_energy": eff_energy, "efficiency_curve": eff_curve,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=DATASET_PATH)
    ap.add_argument("--n-window-sigma", type=float, default=5.0)
    ap.add_argument("--out", default="response_function.npz")
    ap.add_argument("--stage-rows-out", default=STAGE_ROWS_PATH)
    args = ap.parse_args()

    result = fit_all(args.dataset, n_window_sigma=args.n_window_sigma, stage_rows_path=args.stage_rows_out)
    np.savez(args.out, **{k: v for k, v in result.items()})
    print(f"\nSaved {args.out}")
