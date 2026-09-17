"""Measure the BGO single-crystal photopeak response (position, width,
efficiency) vs. true gamma-ray energy, directly from the k39 cascade scan.

Every run in `dataset.npz` fires a fixed, known list of true gamma
energies at every event (2 gammas for the `"ground"`/`"0+_2"` topologies,
3 for `"3g_ground"`/`"3g_0+_2"` -- see `build_dataset.py`'s
`gamma_energies` field). Unlike the old G3 project (which had to *infer*
peak positions by stacking many mixed-cascade samples), every peak
position here is known analytically in advance -- no search needed. This
script fits every photopeak in each run's `Y_singles` spectrum: peaks
whose local windows don't overlap get an independent single-Gaussian +
quadratic-background fit each; peaks whose windows *do* overlap (this
happens for both 2- and 3-gamma runs whenever the level spacing is small
-- the 3-gamma grid's minimum spacing is 100 keV, so 2- or even 3-way
overlap both occur) get one joint multi-Gaussian fit over the merged
window instead. This gives up to ~3 (E, sigma, amplitude) measurements
per run, then fits the global energy-dependent resolution model

    sigma(E)^2 = (doppler_k * E)^2 + intrinsic_k^2 * E

(same functional form as the G3 project, refit fresh for Ancalagon --
different geometry/physics, no reason to assume the same coefficients),
plus a smooth (interpolated) full-energy-peak efficiency curve vs. E.

Output: `response_function.npz` (fitted doppler_k/intrinsic_k, per-peak
measurement table, efficiency curve) and `response_function_check.png`.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

DATASET_PATH = "dataset.npz"


def _rough_sigma_guess(E):
    """Only used to size local fit windows -- not the fitted resolution model."""
    return np.sqrt((0.03 * E) ** 2 + 0.02**2 * E)


def _multi_gaussian_fit(x, y, centers, x0):
    """Joint fit of `len(centers)` Gaussians (fixed centers, independent
    amplitude/sigma each) plus a shared quadratic background, over one
    window. `len(centers) == 1` is just an ordinary single-peak fit --
    this one function replaces the old separate single-/double-Gaussian
    fitters, generalized to however many known peaks fall in a merged
    window (up to 3 here)."""
    k = len(centers)
    weights = np.sqrt(np.clip(y, 1.0, None))

    def model(x_, *params):
        amps = params[0:k]
        sigmas = params[k:2 * k]
        bg0, bg1, bg2 = params[2 * k], params[2 * k + 1], params[2 * k + 2]
        out = bg0 + bg1 * (x_ - x0) + bg2 * (x_ - x0) ** 2
        for a, s, c in zip(amps, sigmas, centers):
            out = out + a * np.exp(-0.5 * ((x_ - c) / s) ** 2)
        return out

    a0 = [max(y[np.argmin(np.abs(x - c))] - np.percentile(y, 20), 1.0) for c in centers]
    bg0_0 = max(np.percentile(y, 20), 0.0)
    p0 = a0 + [0.05] * k + [bg0_0, 0.0, 0.0]
    lower = [0] * k + [0.003] * k + [-np.inf, -np.inf, -np.inf]
    upper = [np.inf] * k + [0.6] * k + [np.inf, np.inf, np.inf]
    popt, pcov = curve_fit(model, x, y, p0=p0, sigma=weights, absolute_sigma=True,
                            bounds=(lower, upper), maxfev=60000)
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
    resid = (y - model(x, *popt)) / weights
    ndf = max(len(x) - len(p0), 1)
    chi2_ndf = float(np.sum(resid**2) / ndf)

    peaks = [
        {"energy": centers[i], "sigma": popt[k + i], "sigma_err": perr[k + i],
         "amplitude": popt[i], "amplitude_err": perr[i]}
        for i in range(k)
    ]
    return peaks, chi2_ndf


def _dedupe_gammas(gammas, tol=1e-4):
    """Merge true gamma energies that coincide to within `tol` MeV.

    The 3-gamma grid's two intermediate levels are independently scanned,
    so `Level2 - Level1` (the middle gamma) can land exactly on `Level1`
    itself (the lowest gamma) whenever `Level2 == 2*Level1` -- two
    physically distinct gammas at the identical true energy. Fitting them
    as two independent Gaussians at the same center is unidentifiable
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


def fit_run_peaks(centers, raw_counts, gammas, n_window_sigma=5.0,
                   min_half_width=0.08, min_bins_per_peak=7):
    """Fit every distinct photopeak in one run's spectrum. `gammas` is the
    run's known true gamma energies (length 2 or 3, may include exact
    coincidences -- see `_dedupe_gammas`). Each distinct energy gets its
    own local window unless it overlaps a neighbor's, in which case
    overlapping windows are merged (transitively -- 3-way overlap is
    handled, not just pairwise) into one joint multi-Gaussian fit.

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

    for idx_list, group_end in zip(groups, group_ends):
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
        fits, chi2_ndf = _multi_gaussian_fit(x, y, group_centers, x.mean())
        for local_i, global_i in enumerate(idx_list):
            peaks[global_i] = fits[local_i]
            chi2_ndfs[global_i] = chi2_ndf
            joint_flags[global_i] = len(idx_list) > 1

    return {"peaks": peaks, "chi2_ndf": chi2_ndfs, "joint": joint_flags, "multiplicity": list(multiplicity)}


def _resolution_model(E, doppler_k, intrinsic_k):
    return np.sqrt((doppler_k * E) ** 2 + intrinsic_k**2 * E)


def measure_all_peaks(dataset_path=DATASET_PATH, n_window_sigma=5.0, verbose=True):
    """Per-run local peak fits only -- each run's fit is independent of
    every other run, so this step is the same regardless of any later
    train/holdout split. Returns a list of row dicts (one per *distinct*
    true-energy peak per successfully-fit run -- see `_dedupe_gammas` for
    why two gammas can collapse into one row): stem, which, energy,
    sigma, sigma_err, amplitude, amplitude_err, chi2_ndf, sigma_rel_err,
    joint, multiplicity. `multiplicity > 1` means this row's amplitude is
    the combined contribution of that many coincident-energy gammas, not
    directly comparable to an ordinary single-gamma amplitude at a nearby
    energy -- see README's caveat on this.
    """
    d = np.load(dataset_path, allow_pickle=True)
    n_total = d["n_total"].astype(np.float64)
    edges = d["edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    Y_singles = d["Y_singles"]
    stems = d["file_stems"]
    gamma_energies = d["gamma_energies"]  # (n_rows, max_gammas), NaN-padded

    rows = []
    for i in range(len(stems)):
        gammas = gamma_energies[i]
        gammas = gammas[np.isfinite(gammas)]
        raw = Y_singles[i] * n_total[i]
        try:
            result = fit_run_peaks(centers, raw, gammas, n_window_sigma=n_window_sigma)
        except (RuntimeError, ValueError) as exc:
            if verbose:
                print(f"{stems[i]}: fit failed ({exc})")
            continue
        for rank, m in enumerate(result["peaks"]):
            sigma_rel_err = m["sigma_err"] / m["sigma"] if m["sigma"] > 0 else np.inf
            rows.append({
                "stem": str(stems[i]), "which": f"p{rank}", "energy": m["energy"],
                "sigma": m["sigma"], "sigma_err": m["sigma_err"],
                "amplitude": m["amplitude"] / n_total[i],  # normalise to per-simulated-event
                "amplitude_err": m["amplitude_err"] / n_total[i],
                "chi2_ndf": result["chi2_ndf"][rank], "sigma_rel_err": sigma_rel_err,
                "joint": result["joint"][rank], "multiplicity": result["multiplicity"][rank],
            })
        if verbose:
            desc = "  ".join(f"E={m['energy']:.3f} sigma={m['sigma']*1000:.1f}keV" for m in result["peaks"])
            print(f"{stems[i]}: {desc}  (n_peaks={len(gammas)}, n_distinct={len(result['peaks'])})")
    return rows


def fit_resolution_model(rows, sigma_rel_err_max=0.5, p0=(0.02, 0.015)):
    """Fit sigma(E)^2 = (doppler_k*E)^2 + intrinsic_k^2*E to the given
    rows (e.g. a training subset). Returns (doppler_k, intrinsic_k,
    doppler_k_err, intrinsic_k_err, good_mask, chi2_ndf)."""
    energies = np.array([r["energy"] for r in rows])
    sigmas = np.array([r["sigma"] for r in rows])
    sigma_errs = np.array([r["sigma_err"] for r in rows])
    rel_err = np.array([r["sigma_rel_err"] for r in rows])

    good = np.isfinite(rel_err) & (rel_err < sigma_rel_err_max) & (sigmas > 0.002)
    popt, pcov = curve_fit(
        _resolution_model, energies[good], sigmas[good],
        p0=p0, sigma=sigma_errs[good], absolute_sigma=True,
        bounds=([0, 0], [1, 1]),
    )
    perr = np.sqrt(np.diag(pcov))
    pred = _resolution_model(energies[good], *popt)
    resid = (sigmas[good] - pred) / sigma_errs[good]
    chi2_ndf = float(np.sum(resid**2) / max(good.sum() - 2, 1))
    return popt[0], popt[1], perr[0], perr[1], good, chi2_ndf


def build_efficiency_curve(rows, sigma_rel_err_max=0.5):
    """Sorted (energy, amplitude) arrays for interpolation, from rows
    passing the same quality cut as the resolution fit. Also drops
    `multiplicity > 1` rows (see `_dedupe_gammas`) -- their amplitude is
    the combined contribution of >=2 coincident-energy gammas, not an
    ordinary single-gamma full-energy-peak probability, so mixing them
    into this curve would bias it at exactly those energies (unlike
    sigma, which the resolution fit still uses them for -- coincident
    gammas don't bias a *width* measurement)."""
    energies = np.array([r["energy"] for r in rows])
    sigmas = np.array([r["sigma"] for r in rows])
    amplitudes = np.array([r["amplitude"] for r in rows])
    rel_err = np.array([r["sigma_rel_err"] for r in rows])
    multiplicity = np.array([r["multiplicity"] for r in rows])
    good = np.isfinite(rel_err) & (rel_err < sigma_rel_err_max) & (sigmas > 0.002) & (multiplicity == 1)
    order = np.argsort(energies[good])
    return energies[good][order], amplitudes[good][order]


def fit_all(dataset_path=DATASET_PATH, n_window_sigma=5.0, sigma_rel_err_max=0.5):
    rows = measure_all_peaks(dataset_path, n_window_sigma=n_window_sigma)

    energies = np.array([r["energy"] for r in rows])
    sigmas = np.array([r["sigma"] for r in rows])
    sigma_errs = np.array([r["sigma_err"] for r in rows])
    amplitudes = np.array([r["amplitude"] for r in rows])
    rel_err = np.array([r["sigma_rel_err"] for r in rows])
    stems = np.array([r["stem"] for r in rows])

    good = np.isfinite(rel_err) & (rel_err < sigma_rel_err_max) & (sigmas > 0.002)
    print(f"\n{good.sum()}/{len(rows)} peak measurements pass quality cut (sigma_rel_err < {sigma_rel_err_max})")

    doppler_k, intrinsic_k, doppler_k_err, intrinsic_k_err, _, chi2_ndf_res = fit_resolution_model(
        rows, sigma_rel_err_max=sigma_rel_err_max)
    print(f"doppler_k = {doppler_k:.5f} +/- {doppler_k_err:.5f}")
    print(f"intrinsic_k = {intrinsic_k:.5f} +/- {intrinsic_k_err:.5f}")
    print(f"resolution-model chi2/ndf (fit on all data, evaluated on all data) = {chi2_ndf_res:.2f}")

    eff_energy, eff_amplitude = build_efficiency_curve(rows, sigma_rel_err_max=sigma_rel_err_max)

    return {
        "energies": energies, "sigmas": sigmas, "sigma_errs": sigma_errs,
        "amplitudes": amplitudes, "rel_err": rel_err, "good": good, "stems": stems,
        "doppler_k": doppler_k, "intrinsic_k": intrinsic_k,
        "doppler_k_err": doppler_k_err, "intrinsic_k_err": intrinsic_k_err,
        "resolution_chi2_ndf": chi2_ndf_res,
        "efficiency_energy": eff_energy, "efficiency_amplitude": eff_amplitude,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=DATASET_PATH)
    ap.add_argument("--n-window-sigma", type=float, default=5.0)
    ap.add_argument("--out", default="response_function.npz")
    args = ap.parse_args()

    result = fit_all(args.dataset, n_window_sigma=args.n_window_sigma)
    np.savez(args.out, **{k: v for k, v in result.items()})
    print(f"\nSaved {args.out}")
