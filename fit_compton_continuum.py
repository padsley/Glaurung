"""Fit the Compton-continuum amplitude of each run's *highest-energy*
gamma vs. true energy (Phase 1 of the Compton-continuum plan -- see
README's "Compton continuum" section and the approved plan this
implements). Escape peaks and lower-energy gammas' own continua are
future phases, not attempted here.

Every run's highest gamma (`E_top`) has a clean, uncontaminated energy
window for its own Compton continuum: above the second-highest gamma's
Compton edge (`gamma_physics.compton_edge_energy`), below `E_top`'s own
photopeak, and with every companion gamma's own photopeak (which sits
*above* its own Compton edge, not at it -- an easy mistake, see
`clean_continuum_window`'s docstring) cut out of the middle if it falls
inside that range. Within that window, `gamma_physics.klein_nishina_continuum_shape`
matches the real (simulated) data well with just a single free
amplitude -- confirmed by a throwaway visual check (4 widely-spaced test
runs, amplitude-only least-squares fit) before writing this pipeline; no
extra shape-correction parameters were needed.

This mirrors `fit_response_function.py`'s per-run-then-global structure
exactly: local per-run fit -> per-energy measurement -> global
median-binned smooth curve -> same train/holdout split for validation.

Output: `compton_continuum.npz` (per-run amplitude measurements, plus
the smoothed `continuum_energy`/`continuum_curve` control points).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from gamma_physics import compton_edge_energy, klein_nishina_continuum_shape

DATASET_PATH = "dataset.npz"
RESPONSE_PATH = "response_function.npz"


def _resolution_sigma(E, doppler_k, intrinsic_k):
    E = np.asarray(E, dtype=np.float64)
    return np.sqrt((doppler_k * E) ** 2 + intrinsic_k**2 * E)


def clean_continuum_window(gammas, doppler_k, intrinsic_k, n_sigma=5.0, margin=0.03):
    """The energy sub-intervals (MeV) where `E_top`'s own Compton
    continuum can be measured free of contamination from any other
    gamma in the same run.

    Starts from `(compton_edge_energy(E_2nd) + margin, min(compton_edge_energy(E_top),
    E_top - n_sigma*sigma(E_top)) - margin)` (`E_2nd` = the second-highest
    gamma, whose own continuum reaches furthest of all the companions,
    since `compton_edge_energy` is monotonic in E) then **cuts out every
    companion gamma's own photopeak** (`E_c +/- n_sigma*sigma(E_c)`) from
    that range wherever it overlaps -- a companion's photopeak sits at
    its own energy `E_c`, which is *always above* `compton_edge_energy(E_c)`,
    so it is not already excluded by the starting lower bound alone (this
    was caught by a visual check against real data before trusting the
    window logic: `E_2nd`'s own peak was initially found to leak into the
    supposedly-clean region).

    Returns `(intervals, e_top)`, `intervals` a list of `(lo, hi)` tuples
    (possibly more than one, e.g. below and above a companion's peak that
    sits mid-window); raises `ValueError` if nothing usable survives
    (near-degenerate `E_top`/`E_2nd`, same failure mode already handled
    for photopeaks in `fit_response_function.py`).
    """
    gammas = np.sort(np.asarray(gammas, dtype=np.float64))
    e_top = gammas[-1]
    companions = gammas[:-1]
    if len(companions) == 0:
        raise ValueError("run has no companion gamma to bound the continuum window")
    e_2nd = companions[-1]

    t_ce_2nd = compton_edge_energy(e_2nd)
    t_ce_top = compton_edge_energy(e_top)
    sigma_top = _resolution_sigma(e_top, doppler_k, intrinsic_k)

    lo = t_ce_2nd + margin
    hi = min(t_ce_top, e_top - n_sigma * sigma_top) - margin
    if hi <= lo:
        raise ValueError(f"window empty/too narrow ({lo:.4f}, {hi:.4f}) MeV")

    intervals = [(lo, hi)]
    for e_c in companions:
        sigma_c = _resolution_sigma(e_c, doppler_k, intrinsic_k)
        p_lo, p_hi = e_c - n_sigma * sigma_c, e_c + n_sigma * sigma_c
        next_intervals = []
        for a, b in intervals:
            if p_hi <= a or p_lo >= b:
                next_intervals.append((a, b))
                continue
            if p_lo > a:
                next_intervals.append((a, p_lo))
            if p_hi < b:
                next_intervals.append((p_hi, b))
        intervals = [(a, b) for a, b in next_intervals if b > a]

    if not intervals:
        raise ValueError("companion photopeak exclusions consumed the whole window")
    return intervals, e_top


def fit_run_continuum(centers, raw_counts, gammas, doppler_k, intrinsic_k,
                       n_sigma=5.0, margin=0.03, min_bins=15):
    """Single-amplitude weighted least-squares fit of
    `a * klein_nishina_continuum_shape(e_top, x)` against one run's
    `Y_singles` data, restricted to `clean_continuum_window`'s
    intervals. Returns a dict with `e_top`, `amplitude`, `amplitude_err`,
    `chi2_ndf`, `n_bins`; raises ValueError/RuntimeError on failure
    (too-small window or non-convergence), same as the photopeak fitter.
    """
    intervals, e_top = clean_continuum_window(gammas, doppler_k, intrinsic_k, n_sigma, margin)
    mask = np.zeros_like(centers, dtype=bool)
    for lo, hi in intervals:
        mask |= (centers >= lo) & (centers <= hi)
    if mask.sum() < min_bins:
        raise ValueError(f"clean window has too few bins ({int(mask.sum())})")

    x, y = centers[mask], raw_counts[mask]
    weights = np.sqrt(np.clip(y, 1.0, None))
    shape = klein_nishina_continuum_shape(e_top, x)

    def model(x_, a):
        return a * klein_nishina_continuum_shape(e_top, x_)

    a0 = max(np.sum(y * shape) / max(np.sum(shape**2), 1e-12), 1.0)
    popt, pcov = curve_fit(model, x, y, p0=[a0], sigma=weights, absolute_sigma=True,
                            bounds=([0.0], [np.inf]), maxfev=20000)
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
    resid = (y - model(x, *popt)) / weights
    chi2_ndf = float(np.sum(resid**2) / max(len(x) - 1, 1))
    return {"e_top": e_top, "amplitude": float(popt[0]), "amplitude_err": float(perr[0]),
            "chi2_ndf": chi2_ndf, "n_bins": int(mask.sum())}


def measure_all_continuum(dataset_path=DATASET_PATH, response_path=RESPONSE_PATH,
                           n_sigma=5.0, verbose=True):
    """Per-run continuum-amplitude measurements, one row per
    successfully-fit run (keyed by each run's own `E_top`, not
    necessarily distinct across runs). Mirrors
    `fit_response_function.measure_all_peaks`'s structure/failure
    handling."""
    d = np.load(dataset_path, allow_pickle=True)
    rf = np.load(response_path)
    doppler_k, intrinsic_k = float(rf["doppler_k"]), float(rf["intrinsic_k"])

    n_total = d["n_total"].astype(np.float64)
    edges = d["edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    Y_singles = d["Y_singles"]
    stems = d["file_stems"]
    gamma_energies = d["gamma_energies"]

    rows = []
    for i in range(len(stems)):
        gammas = gamma_energies[i]
        gammas = gammas[np.isfinite(gammas)]
        raw = Y_singles[i] * n_total[i]
        try:
            result = fit_run_continuum(centers, raw, gammas, doppler_k, intrinsic_k, n_sigma=n_sigma)
        except (RuntimeError, ValueError) as exc:
            if verbose:
                print(f"{stems[i]}: fit failed ({exc})")
            continue
        amplitude = result["amplitude"] / n_total[i]  # normalise to per-simulated-event, same
                                                        # convention as measure_all_peaks -- the
                                                        # raw fit was against Y_singles*n_total
                                                        # counts, so different runs' n_total
                                                        # (13000/50000/600000) aren't comparable
                                                        # before this division.
        amplitude_err = result["amplitude_err"] / n_total[i]
        rel_err = amplitude_err / amplitude if amplitude > 0 else np.inf
        rows.append({
            "stem": str(stems[i]), "energy": result["e_top"],
            "amplitude": amplitude, "amplitude_err": amplitude_err,
            "rel_err": rel_err, "chi2_ndf": result["chi2_ndf"], "n_bins": result["n_bins"],
        })
        if verbose:
            print(f"{stems[i]}: E_top={result['e_top']:.3f} amplitude={amplitude:.4g} "
                  f"+/- {amplitude_err:.2g} chi2/ndf={result['chi2_ndf']:.2f} "
                  f"n_bins={result['n_bins']}")
    return rows, doppler_k, intrinsic_k


def build_continuum_curve(rows, rel_err_max=0.5, chi2_ndf_max=50.0, bin_width=0.1):
    """Median-binned smoothed (energy, amplitude) control points, same
    method as `fit_response_function.build_efficiency_curve` (quality
    cuts on relative error and chi2/ndf, then a per-bin median so a
    single noisy run can't dominate a nearby prediction)."""
    energies = np.array([r["energy"] for r in rows])
    amplitudes = np.array([r["amplitude"] for r in rows])
    rel_err = np.array([r["rel_err"] for r in rows])
    chi2_ndf = np.array([r["chi2_ndf"] for r in rows])
    good = np.isfinite(rel_err) & (rel_err < rel_err_max) & (chi2_ndf < chi2_ndf_max)

    e_good, a_good = energies[good], amplitudes[good]
    if len(e_good) == 0:
        raise ValueError("no rows survive the continuum-curve quality cut")

    bin_index = np.floor(e_good / bin_width).astype(np.int64)
    bins = np.unique(bin_index)
    bin_centers = (bins.astype(np.float64) + 0.5) * bin_width
    bin_medians = np.array([np.median(a_good[bin_index == b]) for b in bins])
    order = np.argsort(bin_centers)
    return bin_centers[order], bin_medians[order]


def fit_all(dataset_path=DATASET_PATH, response_path=RESPONSE_PATH, n_sigma=5.0, rel_err_max=0.5):
    rows, doppler_k, intrinsic_k = measure_all_continuum(dataset_path, response_path, n_sigma=n_sigma)

    energies = np.array([r["energy"] for r in rows])
    amplitudes = np.array([r["amplitude"] for r in rows])
    rel_err = np.array([r["rel_err"] for r in rows])
    chi2_ndf = np.array([r["chi2_ndf"] for r in rows])
    stems = np.array([r["stem"] for r in rows])

    good = np.isfinite(rel_err) & (rel_err < rel_err_max)
    print(f"\n{good.sum()}/{len(rows)} continuum measurements pass the rel_err<{rel_err_max} cut")

    continuum_energy, continuum_curve = build_continuum_curve(rows, rel_err_max=rel_err_max)
    print(f"continuum curve: {len(continuum_energy)} occupied bins")

    return {
        "energies": energies, "amplitudes": amplitudes, "rel_err": rel_err,
        "chi2_ndf": chi2_ndf, "stems": stems, "good": good,
        "continuum_energy": continuum_energy, "continuum_curve": continuum_curve,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=DATASET_PATH)
    ap.add_argument("--response", default=RESPONSE_PATH)
    ap.add_argument("--n-sigma", type=float, default=5.0)
    ap.add_argument("--out", default="compton_continuum.npz")
    args = ap.parse_args()

    result = fit_all(args.dataset, args.response, n_sigma=args.n_sigma)
    np.savez(args.out, **result)
    print(f"\nSaved {args.out}")
