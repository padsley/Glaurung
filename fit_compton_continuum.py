"""Fit the Compton-continuum amplitude (and, where the window covers it,
the backscatter-dome amplitude/width) of each run's *highest-energy*
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
amplitude -- confirmed by a throwaway visual check (4 widely-spaced,
high-E test runs, amplitude-only least-squares fit) before writing this
pipeline; no extra shape-correction parameters were needed *there*.

**Backscatter dome, modeled not excluded (2026-09-19)**: adding the
`"calib1g"` single-gamma calibration series (2026-09-18, no companion
gamma, so nothing was naturally excluding the low-energy end of the
window) exposed a real, distinct feature around ~0.18-0.26 MeV -- a
non-monotonic "dome" the raw Klein-Nishina shape does not predict (which
is monotonic rising from T=0 to the edge), almost certainly photons
Compton-scattering off surrounding/dead material before entering the
sensitive crystal (a "backscatter peak", well known in gamma
spectroscopy, physically distinct from a single in-crystal Compton
scatter). Every *other* topology's window happened to already exclude
this by construction (their companion gamma's own Compton edge usually
sits above it) -- but not always (a companion below ~0.6 MeV leaves the
dome inside the window), so this can contaminate any topology, not just
the new one. Was first handled (2026-09-18) by just excluding it
(`BACKSCATTER_FLOOR_MEV`); now modeled instead, per padsley's request --
`gamma_physics.backscatter_energy(E_top)` gives the dome's physics-
predicted center (photon energy after a theta=180 in-material scatter),
fixed (not fit); amplitude and width are fit jointly with the continuum
amplitude whenever the window actually covers the dome region (see
`fit_run_continuum`'s `include_dome` check) -- otherwise (dome outside
the window, the common case for most runs whose companion sits above
~0.6 MeV) the fit is continuum-amplitude-only, as before.

This mirrors `fit_response_function.py`'s per-run-then-global structure
exactly: local per-run fit -> per-energy measurement -> global
median-binned smooth curve -> same train/holdout split for validation.

Output: `compton_continuum.npz` (per-run amplitude/dome measurements,
plus the smoothed `continuum_energy`/`continuum_curve` and
`dome_energy`/`dome_curve`/`dome_sigma_curve` control points).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from gamma_physics import backscatter_energy, compton_edge_energy, effective_sigma, klein_nishina_continuum_shape

DATASET_PATH = "dataset.npz"
RESPONSE_PATH = "response_function.npz"


def clean_continuum_window(gammas, beta, intrinsic_k, n_sigma=5.0, margin=0.03):
    """The energy sub-intervals (MeV) where `E_top`'s own Compton
    continuum (and, where it falls inside, the backscatter dome) can be
    measured free of contamination from any other gamma in the same run.

    Starts from `(compton_edge_energy(E_2nd) + margin, min(compton_edge_energy(E_top),
    E_top - n_sigma*effective_sigma(E_top)) - margin)` (`E_2nd` = the
    second-highest gamma, whose own continuum reaches furthest of all the
    companions, since `compton_edge_energy` is monotonic in E) then
    **cuts out every companion gamma's own photopeak**
    (`E_c +/- n_sigma*effective_sigma(E_c)`) from that range wherever it
    overlaps -- a companion's photopeak sits at its own energy `E_c`,
    which is *always above* `compton_edge_energy(E_c)`, so it is not
    already excluded by the starting lower bound alone (this was caught
    by a visual check against real data before trusting the window
    logic: `E_2nd`'s own peak was initially found to leak into the
    supposedly-clean region).

    Returns `(intervals, e_top)`, `intervals` a list of `(lo, hi)` tuples
    (possibly more than one, e.g. below and above a companion's peak that
    sits mid-window); raises `ValueError` if nothing usable survives
    (near-degenerate `E_top`/`E_2nd`, same failure mode already handled
    for photopeaks in `fit_response_function.py`).

    **Zero-companion runs** (the `"calib1g"` single-gamma calibration
    series): there is no `E_2nd` to set the lower bound, so the window
    simply starts from `margin` (just above zero) instead -- the *entire*
    spectrum below the photopeak is this one gamma's own uncontaminated
    response (including the backscatter dome, if present -- see
    `fit_run_continuum`).
    """
    gammas = np.sort(np.asarray(gammas, dtype=np.float64))
    e_top = gammas[-1]
    companions = gammas[:-1]

    t_ce_top = compton_edge_energy(e_top)
    sigma_top = effective_sigma(e_top, beta, intrinsic_k * np.sqrt(e_top))

    lo = margin if len(companions) == 0 else compton_edge_energy(companions[-1]) + margin
    hi = min(t_ce_top, e_top - n_sigma * sigma_top) - margin
    if hi <= lo:
        raise ValueError(f"window empty/too narrow ({lo:.4f}, {hi:.4f}) MeV")

    intervals = [(lo, hi)]
    for e_c in companions:
        sigma_c = effective_sigma(e_c, beta, intrinsic_k * np.sqrt(e_c))
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


def fit_run_continuum(centers, raw_counts, gammas, beta, intrinsic_k,
                       n_sigma=5.0, margin=0.03, min_bins=15, dome_min_bins=8):
    """Weighted least-squares fit against one run's `Y_singles` data,
    restricted to `clean_continuum_window`'s intervals: `a_continuum *
    klein_nishina_continuum_shape(e_top, x)`, plus -- when the window
    actually covers the backscatter-dome region (`>= dome_min_bins` bins
    within +/-0.15 MeV of `backscatter_energy(e_top)`) -- a Gaussian bump
    `a_dome * exp(-0.5*((x-dome_center)/dome_sigma)^2)` with `dome_center`
    fixed at `backscatter_energy(e_top)` and `(a_dome, dome_sigma)` free.

    Returns a dict with `e_top`, `amplitude`, `amplitude_err`, `chi2_ndf`,
    `n_bins`, `has_dome`, and (only if `has_dome`) `dome_amplitude`,
    `dome_amplitude_err`, `dome_sigma`, `dome_sigma_err`. Raises
    ValueError/RuntimeError on failure (too-small window or
    non-convergence), same as the photopeak fitter.
    """
    intervals, e_top = clean_continuum_window(gammas, beta, intrinsic_k, n_sigma, margin)
    mask = np.zeros_like(centers, dtype=bool)
    for lo, hi in intervals:
        mask |= (centers >= lo) & (centers <= hi)
    if mask.sum() < min_bins:
        raise ValueError(f"clean window has too few bins ({int(mask.sum())})")

    x, y = centers[mask], raw_counts[mask]
    weights = np.sqrt(np.clip(y, 1.0, None))
    shape_cont = klein_nishina_continuum_shape(e_top, x)
    dome_center = backscatter_energy(e_top)
    near_dome = np.abs(x - dome_center) < 0.15
    include_dome = int(near_dome.sum()) >= dome_min_bins

    if include_dome:
        def model(x_, a_cont, a_dome, dome_sigma):
            return (a_cont * klein_nishina_continuum_shape(e_top, x_)
                     + a_dome * np.exp(-0.5 * ((x_ - dome_center) / dome_sigma) ** 2))

        a_cont0 = max(np.sum(y * shape_cont) / max(np.sum(shape_cont**2), 1e-12), 1.0)
        excess = y[near_dome] - a_cont0 * shape_cont[near_dome]
        a_dome0 = max(float(excess.max()), 1.0)
        p0 = [a_cont0, a_dome0, 0.05]
        popt, pcov = curve_fit(model, x, y, p0=p0, sigma=weights, absolute_sigma=True,
                                bounds=([0.0, 0.0, 0.01], [np.inf, np.inf, 0.2]), maxfev=20000)
        perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
        resid = (y - model(x, *popt)) / weights
        chi2_ndf = float(np.sum(resid**2) / max(len(x) - 3, 1))
        return {"e_top": e_top, "amplitude": float(popt[0]), "amplitude_err": float(perr[0]),
                "dome_amplitude": float(popt[1]), "dome_amplitude_err": float(perr[1]),
                "dome_sigma": float(popt[2]), "dome_sigma_err": float(perr[2]),
                "chi2_ndf": chi2_ndf, "n_bins": int(mask.sum()), "has_dome": True}

    def model1(x_, a):
        return a * klein_nishina_continuum_shape(e_top, x_)

    a0 = max(np.sum(y * shape_cont) / max(np.sum(shape_cont**2), 1e-12), 1.0)
    popt, pcov = curve_fit(model1, x, y, p0=[a0], sigma=weights, absolute_sigma=True,
                            bounds=([0.0], [np.inf]), maxfev=20000)
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
    resid = (y - model1(x, *popt)) / weights
    chi2_ndf = float(np.sum(resid**2) / max(len(x) - 1, 1))
    return {"e_top": e_top, "amplitude": float(popt[0]), "amplitude_err": float(perr[0]),
            "dome_amplitude": None, "dome_amplitude_err": None, "dome_sigma": None, "dome_sigma_err": None,
            "chi2_ndf": chi2_ndf, "n_bins": int(mask.sum()), "has_dome": False}


def measure_all_continuum(dataset_path=DATASET_PATH, response_path=RESPONSE_PATH,
                           n_sigma=5.0, verbose=True):
    """Per-run continuum-amplitude (and, where covered, dome) measurements,
    one row per successfully-fit run (keyed by each run's own `E_top`, not
    necessarily distinct across runs). Mirrors
    `fit_response_function.measure_all_peaks`'s structure/failure
    handling."""
    d = np.load(dataset_path, allow_pickle=True)
    rf = np.load(response_path)
    beta, intrinsic_k = float(rf["beta"]), float(rf["intrinsic_k"])

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
            result = fit_run_continuum(centers, raw, gammas, beta, intrinsic_k, n_sigma=n_sigma)
        except (RuntimeError, ValueError) as exc:
            if verbose:
                print(f"{stems[i]}: fit failed ({exc})")
            continue
        # normalise to per-simulated-event, same convention as measure_all_peaks -- the
        # raw fit was against Y_singles*n_total counts, so different runs' n_total
        # (13000/50000/600000) aren't comparable before this division.
        amplitude = result["amplitude"] / n_total[i]
        amplitude_err = result["amplitude_err"] / n_total[i]
        rel_err = amplitude_err / amplitude if amplitude > 0 else np.inf
        row = {
            "stem": str(stems[i]), "energy": result["e_top"],
            "amplitude": amplitude, "amplitude_err": amplitude_err,
            "rel_err": rel_err, "chi2_ndf": result["chi2_ndf"], "n_bins": result["n_bins"],
            "has_dome": result["has_dome"],
        }
        if result["has_dome"]:
            dome_amplitude = result["dome_amplitude"] / n_total[i]
            dome_amplitude_err = result["dome_amplitude_err"] / n_total[i]
            row["dome_amplitude"] = dome_amplitude
            row["dome_amplitude_err"] = dome_amplitude_err
            row["dome_rel_err"] = dome_amplitude_err / dome_amplitude if dome_amplitude > 0 else np.inf
            row["dome_sigma"] = result["dome_sigma"]
            row["dome_sigma_err"] = result["dome_sigma_err"]
        else:
            row.update({"dome_amplitude": np.nan, "dome_amplitude_err": np.nan, "dome_rel_err": np.inf,
                        "dome_sigma": np.nan, "dome_sigma_err": np.nan})
        rows.append(row)
        if verbose:
            dome_desc = (f" dome_amp={row['dome_amplitude']:.4g} dome_sigma={row['dome_sigma']*1000:.1f}keV"
                         if result["has_dome"] else "")
            print(f"{stems[i]}: E_top={result['e_top']:.3f} amplitude={amplitude:.4g} "
                  f"+/- {amplitude_err:.2g} chi2/ndf={result['chi2_ndf']:.2f} "
                  f"n_bins={result['n_bins']}{dome_desc}")
    return rows, beta, intrinsic_k


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


def build_dome_curve(rows, rel_err_max=0.5, chi2_ndf_max=50.0, bin_width=0.2):
    """Median-binned smoothed `(energy, dome_amplitude, dome_sigma)`
    control points, same method as `build_continuum_curve` -- restricted
    to rows where the window actually covered the dome
    (`has_dome=True`). Wider `bin_width` than the continuum curve's
    (0.2 vs 0.1 MeV) since the dome is only ever measured over a narrow
    `E_top` range (wherever a companion sits below ~0.6 MeV) -- fewer,
    less energy-diverse measurements than the continuum's full 0.95-8.85
    MeV coverage."""
    has_dome = np.array([r["has_dome"] for r in rows])
    rows = [r for r, h in zip(rows, has_dome) if h]
    if not rows:
        raise ValueError("no rows have a dome measurement (window never covered the dome region)")

    energies = np.array([r["energy"] for r in rows])
    dome_amp = np.array([r["dome_amplitude"] for r in rows])
    dome_sigma = np.array([r["dome_sigma"] for r in rows])
    rel_err = np.array([r["dome_rel_err"] for r in rows])
    chi2_ndf = np.array([r["chi2_ndf"] for r in rows])
    good = (np.isfinite(rel_err) & (rel_err < rel_err_max) & (chi2_ndf < chi2_ndf_max)
            & (dome_sigma > 0.011) & (dome_sigma < 0.19))  # away from its own fit bounds

    e_good, a_good, s_good = energies[good], dome_amp[good], dome_sigma[good]
    if len(e_good) == 0:
        raise ValueError("no rows survive the dome-curve quality cut")

    bin_index = np.floor(e_good / bin_width).astype(np.int64)
    bins = np.unique(bin_index)
    bin_centers = (bins.astype(np.float64) + 0.5) * bin_width
    bin_amp_medians = np.array([np.median(a_good[bin_index == b]) for b in bins])
    bin_sigma_medians = np.array([np.median(s_good[bin_index == b]) for b in bins])
    order = np.argsort(bin_centers)
    return bin_centers[order], bin_amp_medians[order], bin_sigma_medians[order]


def fit_all(dataset_path=DATASET_PATH, response_path=RESPONSE_PATH, n_sigma=5.0, rel_err_max=0.5):
    rows, beta, intrinsic_k = measure_all_continuum(dataset_path, response_path, n_sigma=n_sigma)

    energies = np.array([r["energy"] for r in rows])
    amplitudes = np.array([r["amplitude"] for r in rows])
    rel_err = np.array([r["rel_err"] for r in rows])
    chi2_ndf = np.array([r["chi2_ndf"] for r in rows])
    stems = np.array([r["stem"] for r in rows])
    has_dome = np.array([r["has_dome"] for r in rows])

    good = np.isfinite(rel_err) & (rel_err < rel_err_max)
    print(f"\n{good.sum()}/{len(rows)} continuum measurements pass the rel_err<{rel_err_max} cut "
          f"({has_dome.sum()} include a dome fit)")

    continuum_energy, continuum_curve = build_continuum_curve(rows, rel_err_max=rel_err_max)
    print(f"continuum curve: {len(continuum_energy)} occupied bins")

    dome_energy = dome_curve = dome_sigma_curve = np.array([])
    try:
        dome_energy, dome_curve, dome_sigma_curve = build_dome_curve(rows, rel_err_max=rel_err_max)
        print(f"dome curve: {len(dome_energy)} occupied bins")
    except ValueError as exc:
        print(f"dome curve: not built ({exc})")

    return {
        "energies": energies, "amplitudes": amplitudes, "rel_err": rel_err,
        "chi2_ndf": chi2_ndf, "stems": stems, "good": good, "has_dome": has_dome,
        "continuum_energy": continuum_energy, "continuum_curve": continuum_curve,
        "dome_energy": dome_energy, "dome_curve": dome_curve, "dome_sigma_curve": dome_sigma_curve,
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
