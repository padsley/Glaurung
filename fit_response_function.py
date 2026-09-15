"""Measure the BGO single-crystal photopeak response (position, width,
efficiency) vs. true gamma-ray energy, directly from the k39 cascade scan.

Each of the 88 runs in `dataset.npz` fires two gammas of known energy
(E_lo, E_hi = sorted(level(1), ex - level(1))) at every event. Unlike the
old G3 project (which had to *infer* peak positions by stacking many
mixed-cascade samples), every peak position here is known analytically in
advance -- no search needed. This script fits both photopeaks in each
run's `Y_singles` spectrum simultaneously (a local double-Gaussian +
linear-background fit, at fixed centers E_lo/E_hi, independent widths),
giving up to 176 (E, sigma, amplitude) measurements spanning ~0.1-8.8 MeV,
then fits the global energy-dependent resolution model

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


def _single_gaussian_fit(x, y, center, x0):
    weights = np.sqrt(np.clip(y, 1.0, None))

    def model(x_, a, s, bg0, bg1, bg2):
        return a * np.exp(-0.5 * ((x_ - center) / s) ** 2) + bg0 + bg1 * (x_ - x0) + bg2 * (x_ - x0) ** 2

    a0 = max(y[np.argmin(np.abs(x - center))] - np.percentile(y, 20), 1.0)
    bg0_0 = max(np.percentile(y, 20), 0.0)
    p0 = [a0, 0.05, bg0_0, 0.0, 0.0]
    bounds = ([0, 0.003, -np.inf, -np.inf, -np.inf], [np.inf, 0.6, np.inf, np.inf, np.inf])
    popt, pcov = curve_fit(model, x, y, p0=p0, sigma=weights, absolute_sigma=True,
                            bounds=bounds, maxfev=40000)
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
    resid = (y - model(x, *popt)) / weights
    chi2_ndf = float(np.sum(resid**2) / max(len(x) - 5, 1))
    return {"energy": center, "sigma": popt[1], "sigma_err": perr[1],
            "amplitude": popt[0], "amplitude_err": perr[0]}, chi2_ndf


def _double_gaussian_fit(x, y, c_lo, c_hi, x0):
    weights = np.sqrt(np.clip(y, 1.0, None))

    def model(x_, a_lo, s_lo, a_hi, s_hi, bg0, bg1, bg2):
        g_lo = a_lo * np.exp(-0.5 * ((x_ - c_lo) / s_lo) ** 2)
        g_hi = a_hi * np.exp(-0.5 * ((x_ - c_hi) / s_hi) ** 2)
        return g_lo + g_hi + bg0 + bg1 * (x_ - x0) + bg2 * (x_ - x0) ** 2

    i_lo = np.argmin(np.abs(x - c_lo))
    i_hi = np.argmin(np.abs(x - c_hi))
    a_lo0 = max(y[i_lo] - np.percentile(y, 20), 1.0)
    a_hi0 = max(y[i_hi] - np.percentile(y, 20), 1.0)
    bg0_0 = max(np.percentile(y, 20), 0.0)
    p0 = [a_lo0, 0.05, a_hi0, 0.05, bg0_0, 0.0, 0.0]
    bounds = ([0, 0.003, 0, 0.003, -np.inf, -np.inf, -np.inf],
              [np.inf, 0.6, np.inf, 0.6, np.inf, np.inf, np.inf])
    popt, pcov = curve_fit(model, x, y, p0=p0, sigma=weights, absolute_sigma=True,
                            bounds=bounds, maxfev=40000)
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
    resid = (y - model(x, *popt)) / weights
    chi2_ndf = float(np.sum(resid**2) / max(len(x) - 7, 1))
    lo = {"energy": c_lo, "sigma": popt[1], "sigma_err": perr[1],
          "amplitude": popt[0], "amplitude_err": perr[0]}
    hi = {"energy": c_hi, "sigma": popt[3], "sigma_err": perr[3],
          "amplitude": popt[2], "amplitude_err": perr[2]}
    return lo, hi, chi2_ndf


def fit_run_peaks(centers, raw_counts, e_lo, e_hi, n_window_sigma=5.0,
                   min_half_width=0.08, min_bins=7):
    """Fit both photopeaks in one run's spectrum, each in its own LOCAL
    window (not spanning the whole gap between them) unless the two
    windows overlap, in which case a joint double-Gaussian fit is used
    for the merged region. Returns a dict with per-peak (energy, sigma,
    sigma_err, amplitude, amplitude_err) for 'lo' and 'hi', plus a
    chi2/ndf per peak (shared for the two if a joint fit was used), or
    None if a fit region has too few bins / fails to converge.
    """
    hw_lo = max(min_half_width, n_window_sigma * _rough_sigma_guess(e_lo))
    hw_hi = max(min_half_width, n_window_sigma * _rough_sigma_guess(e_hi))
    win_lo = (e_lo - hw_lo, e_lo + hw_lo)
    win_hi = (e_hi - hw_hi, e_hi + hw_hi)

    if win_lo[1] >= win_hi[0]:  # overlapping -> joint fit
        lo_edge = max(centers[0], win_lo[0])
        hi_edge = min(centers[-1], win_hi[1])
        mask = (centers >= lo_edge) & (centers <= hi_edge)
        if mask.sum() < min_bins:
            raise ValueError(f"joint window too small ({int(mask.sum())} bins)")
        x, y = centers[mask], raw_counts[mask]
        lo, hi, chi2_ndf = _double_gaussian_fit(x, y, e_lo, e_hi, x.mean())
        return {"lo": lo, "hi": hi, "chi2_ndf_lo": chi2_ndf, "chi2_ndf_hi": chi2_ndf,
                "n_bins": int(mask.sum()), "joint": True}
    else:
        mask_lo = (centers >= max(centers[0], win_lo[0])) & (centers <= win_lo[1])
        mask_hi = (centers >= win_hi[0]) & (centers <= min(centers[-1], win_hi[1]))
        if mask_lo.sum() < min_bins or mask_hi.sum() < min_bins:
            raise ValueError(f"lo/hi window too small ({int(mask_lo.sum())}/{int(mask_hi.sum())} bins)")
        lo, chi2_lo = _single_gaussian_fit(centers[mask_lo], raw_counts[mask_lo], e_lo,
                                            centers[mask_lo].mean())
        hi, chi2_hi = _single_gaussian_fit(centers[mask_hi], raw_counts[mask_hi], e_hi,
                                            centers[mask_hi].mean())
        return {"lo": lo, "hi": hi, "chi2_ndf_lo": chi2_lo, "chi2_ndf_hi": chi2_hi,
                "n_bins": int(mask_lo.sum() + mask_hi.sum()), "joint": False}


def _resolution_model(E, doppler_k, intrinsic_k):
    return np.sqrt((doppler_k * E) ** 2 + intrinsic_k**2 * E)


def measure_all_peaks(dataset_path=DATASET_PATH, n_window_sigma=5.0, verbose=True):
    """Per-run local peak fits only -- each run's fit is independent of
    every other run, so this step is the same regardless of any later
    train/holdout split. Returns a list of row dicts (one per lo/hi peak
    per successfully-fit run): stem, which, energy, sigma, sigma_err,
    amplitude, amplitude_err, chi2_ndf, sigma_rel_err, joint.
    """
    d = np.load(dataset_path)
    labels = list(d["labels"])
    lvl = d["X"][:, labels.index("level(1)")]
    ex = d["X"][:, labels.index("ex")]
    n_total = d["n_total"].astype(np.float64)
    edges = d["edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    Y_singles = d["Y_singles"]
    stems = d["file_stems"]

    e_lo_all = np.minimum(lvl, ex - lvl)
    e_hi_all = np.maximum(lvl, ex - lvl)

    rows = []
    for i in range(len(stems)):
        raw = Y_singles[i] * n_total[i]
        try:
            result = fit_run_peaks(centers, raw, e_lo_all[i], e_hi_all[i], n_window_sigma=n_window_sigma)
        except (RuntimeError, ValueError) as exc:
            if verbose:
                print(f"{stems[i]}: fit failed ({exc})")
            continue
        for which in ("lo", "hi"):
            m = result[which]
            sigma_rel_err = m["sigma_err"] / m["sigma"] if m["sigma"] > 0 else np.inf
            rows.append({
                "stem": str(stems[i]), "which": which, "energy": m["energy"],
                "sigma": m["sigma"], "sigma_err": m["sigma_err"],
                "amplitude": m["amplitude"] / n_total[i],  # normalise to per-simulated-event
                "amplitude_err": m["amplitude_err"] / n_total[i],
                "chi2_ndf": result[f"chi2_ndf_{which}"], "sigma_rel_err": sigma_rel_err,
                "joint": result["joint"],
            })
        if verbose:
            print(f"{stems[i]}: E_lo={result['lo']['energy']:.3f} sigma_lo={result['lo']['sigma']*1000:.1f}keV  "
                  f"E_hi={result['hi']['energy']:.3f} sigma_hi={result['hi']['sigma']*1000:.1f}keV  "
                  f"chi2/ndf(lo/hi)={result['chi2_ndf_lo']:.2f}/{result['chi2_ndf_hi']:.2f} joint={result['joint']}")
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
    passing the same quality cut as the resolution fit."""
    energies = np.array([r["energy"] for r in rows])
    sigmas = np.array([r["sigma"] for r in rows])
    amplitudes = np.array([r["amplitude"] for r in rows])
    rel_err = np.array([r["sigma_rel_err"] for r in rows])
    good = np.isfinite(rel_err) & (rel_err < sigma_rel_err_max) & (sigmas > 0.002)
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
