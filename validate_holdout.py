"""Held-out validation of the response-function fit in
`fit_response_function.py`.

Unlike the earlier version of this script, this does **not** re-run the
expensive per-peak fits (`measure_all_peaks`) -- it loads the already-
computed stage-3 rows from `peak_measurements.npz` (`fit_all`'s cache,
shared with `fit_compton_continuum.py`) and only redoes the fast
train/holdout aggregation on top of them: hold out every 5th run (~20%,
spread across the full grid to test interpolation, not extrapolation),
fit `intrinsic_k` and the efficiency curve on the remaining ~80% only,
then check both against the held-out runs those two fits never saw.

**Simplification, disclosed** (see `fit_response_function.py`'s module
docstring for the full reasoning): `beta` itself is *not* re-derived
train-only here -- it was already estimated from all rows when
`peak_measurements.npz` was built, and as a single global scalar
overdetermined by thousands of window measurements, the leakage from
that is expected to be negligible. Only `intrinsic_k` and the efficiency
curve get a genuine held-out check.
"""
from __future__ import annotations

import numpy as np

from fit_response_function import (
    DATASET_PATH, ROW_FIELDS, STAGE_ROWS_PATH, RESPONSE_PATH,
    _intrinsic_model, build_efficiency_curve, fit_intrinsic_k, load_stage_rows,
)


def split_stems(all_stems, every_n=5, offset=4):
    """Hold out every `every_n`-th run (by sorted level(1)/stem order),
    spread across the whole grid rather than a contiguous block."""
    unique_stems = sorted(set(all_stems))
    holdout = set(unique_stems[offset::every_n])
    train = set(unique_stems) - holdout
    return train, holdout


def run_validation(stage_rows_path=STAGE_ROWS_PATH, response_path=RESPONSE_PATH,
                    every_n=5, offset=4, sigma_rel_err_max=0.5):
    rows = load_stage_rows(stage_rows_path, ROW_FIELDS)
    beta = float(np.load(response_path)["beta"])
    all_stems = [r["stem"] for r in rows]
    train_stems, holdout_stems = split_stems(all_stems, every_n=every_n, offset=offset)
    print(f"{len(train_stems)} train runs, {len(holdout_stems)} held-out runs "
          f"(every {every_n}th, offset {offset})")
    print("held-out stems:", sorted(holdout_stems))

    train_rows = [r for r in rows if r["stem"] in train_stems]
    holdout_rows = [r for r in rows if r["stem"] in holdout_stems]

    # --- intrinsic-width model: fit on train only (beta is fixed, see module docstring) ---
    intrinsic_k, ik_err, train_good, train_chi2 = fit_intrinsic_k(
        train_rows, sigma_rel_err_max=sigma_rel_err_max)
    print(f"\n[trained on {len(train_rows)} train rows; beta={beta:.5f} fixed, from all rows]")
    print(f"intrinsic_k = {intrinsic_k:.5f} +/- {ik_err:.5f}")
    print(f"train chi2/ndf (fit on train, evaluated on train) = {train_chi2:.2f}")

    ho_energy = np.array([r["energy"] for r in holdout_rows])
    ho_sigma = np.array([r["sigma_intrinsic"] for r in holdout_rows])
    ho_sigma_err = np.array([r["sigma_intrinsic_err"] for r in holdout_rows])
    ho_rel_err = np.array([r["sigma_intrinsic_rel_err"] for r in holdout_rows])
    ho_good = np.isfinite(ho_rel_err) & (ho_rel_err < sigma_rel_err_max) & (ho_sigma > 0.0005) & (ho_sigma_err > 1e-4)

    ho_pred = _intrinsic_model(ho_energy[ho_good], intrinsic_k)
    ho_resid = (ho_sigma[ho_good] - ho_pred) / ho_sigma_err[ho_good]
    ho_chi2 = float(np.sum(ho_resid**2) / max(ho_good.sum() - 1, 1))
    ho_rel_diff = (ho_sigma[ho_good] - ho_pred) / ho_sigma[ho_good]
    print(f"HELD-OUT chi2/ndf (fit on train, evaluated on {ho_good.sum()} held-out points) = {ho_chi2:.2f}")
    print(f"HELD-OUT sigma_intrinsic relative difference: median={np.median(np.abs(ho_rel_diff)) * 100:.1f}%, "
          f"mean={np.mean(np.abs(ho_rel_diff)) * 100:.1f}%, max={np.max(np.abs(ho_rel_diff)) * 100:.1f}%")

    # --- efficiency curve: build interpolation on train only ---
    eff_energy, eff_curve = build_efficiency_curve(train_rows, sigma_rel_err_max=sigma_rel_err_max)
    ho_eff = np.array([r["efficiency"] for r in holdout_rows])
    ho_eff_pred = np.interp(ho_energy[ho_good], eff_energy, eff_curve)
    ho_eff_rel_diff = (ho_eff[ho_good] - ho_eff_pred) / ho_eff[ho_good]
    print(f"HELD-OUT efficiency relative difference (ALL {ho_good.sum()} held-out points passing the "
          f"sigma-only cut): median={np.median(np.abs(ho_eff_rel_diff)) * 100:.1f}%, "
          f"mean={np.mean(np.abs(ho_eff_rel_diff)) * 100:.1f}%, max={np.max(np.abs(ho_eff_rel_diff)) * 100:.1f}%")

    # Same numbers, but also excluding held-out points that are themselves
    # joint/coincident/high-chi2 (the same categories build_efficiency_curve
    # excludes from the *training* side) -- these are known-unreliable
    # measurements to compare against regardless of curve quality (see
    # README's 2026-09-16 "efficiency curve" write-up), so the ALL number
    # above is a deliberately pessimistic upper bound, not evidence the
    # curve itself is that inaccurate.
    ho_joint = np.array([r["joint"] for r in holdout_rows])[ho_good]
    ho_mult = np.array([r["multiplicity"] for r in holdout_rows])[ho_good]
    ho_chi2ndf = np.array([r["chi2_ndf"] for r in holdout_rows])[ho_good]
    ho_clean = ~ho_joint & (ho_mult == 1) & (ho_chi2ndf < 15.0)
    clean_rel_diff = ho_eff_rel_diff[ho_clean]
    print(f"HELD-OUT efficiency relative difference (CLEAN {ho_clean.sum()} of those, excluding "
          f"joint/coincident/high-chi2 held-out points): "
          f"median={np.median(np.abs(clean_rel_diff)) * 100:.1f}%, "
          f"mean={np.mean(np.abs(clean_rel_diff)) * 100:.1f}%, max={np.max(np.abs(clean_rel_diff)) * 100:.1f}%")

    return {
        "beta": beta, "intrinsic_k": intrinsic_k,
        "train_chi2_ndf": train_chi2, "holdout_chi2_ndf": ho_chi2,
        "holdout_energy": ho_energy[ho_good], "holdout_sigma_intrinsic": ho_sigma[ho_good],
        "holdout_sigma_intrinsic_pred": ho_pred, "holdout_sigma_intrinsic_rel_diff": ho_rel_diff,
        "holdout_efficiency": ho_eff[ho_good], "holdout_efficiency_pred": ho_eff_pred,
        "holdout_efficiency_rel_diff": ho_eff_rel_diff, "holdout_efficiency_clean_mask": ho_clean,
        "eff_energy": eff_energy, "eff_curve": eff_curve,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-rows", default=STAGE_ROWS_PATH)
    ap.add_argument("--response", default=RESPONSE_PATH)
    ap.add_argument("--every-n", type=int, default=5)
    ap.add_argument("--offset", type=int, default=4)
    ap.add_argument("--out", default="holdout_validation.npz")
    args = ap.parse_args()

    result = run_validation(args.stage_rows, args.response, every_n=args.every_n, offset=args.offset)
    np.savez(args.out, **result)
    print(f"\nSaved {args.out}")
