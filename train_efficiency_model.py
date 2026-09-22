"""Phase A of the dense-grid-informed empirical model (see README's
"Cascade-context efficiency/sigma_intrinsic model" section): tests
whether `efficiency`/`sigma_intrinsic` depend on more than a peak's own
energy -- the current model (`fit_intrinsic_k`, `build_efficiency_curve`
in `fit_response_function.py`) fits each as a smooth curve of energy
only, independently per gamma, structurally blind to any cross-gamma
effect (multiplicity, coincidence summing, etc.) even if one is real.

Trains a `HistGradientBoostingRegressor` (the same architecture the
sibling DRAGON_G3_Emulator project already found beats every neural-net
variant it tried) on each peak's FULL cascade context -- its own energy,
every other gamma's energy in the same run, and multiplicity -- and
compares held-out accuracy against the existing marginal curves on the
*same* rows, *same* split, so the numbers are directly comparable to
this project's own README.

Pre-declared success criterion: a meaningful win is a clear median/mean
improvement over the marginal-curve baseline below, not "similar or
slightly better". If this doesn't clearly beat the baseline, that's the
answer -- cross-gamma context isn't the missing piece.
"""
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from fit_response_function import (
    DATASET_PATH, ROW_FIELDS, STAGE_ROWS_PATH, _intrinsic_model,
    build_efficiency_curve, fit_intrinsic_k, load_stage_rows,
)
from validate_holdout import split_stems

MAX_GAMMAS = 4  # keep in sync with build_dataset.py
MODEL_PATH = "efficiency_model.pkl"


def quality_mask(rows, sigma_rel_err_max=0.5, chi2_ndf_max=15.0):
    """Same guard `build_efficiency_curve` uses (single-peak, non-joint,
    tight chi2) -- used for BOTH targets here (a stricter cut than
    `fit_intrinsic_k` alone permits) since this model predicts
    [efficiency, sigma_intrinsic] jointly from one shared feature row and
    needs one shared, honestly-comparable row set to do that on."""
    rel_err = np.array([r["sigma_intrinsic_rel_err"] for r in rows])
    sigmas = np.array([r["sigma_intrinsic"] for r in rows])
    sigma_errs = np.array([r["sigma_intrinsic_err"] for r in rows])
    multiplicity = np.array([r["multiplicity"] for r in rows])
    joint = np.array([r["joint"] for r in rows])
    chi2_ndf = np.array([r["chi2_ndf"] for r in rows])
    return (np.isfinite(rel_err) & (rel_err < sigma_rel_err_max) & (sigmas > 0.0005)
            & (sigma_errs > 1e-4) & (multiplicity == 1) & ~joint & (chi2_ndf < chi2_ndf_max))


def feature_row(E, gammas, max_gammas=MAX_GAMMAS):
    """[this_gamma_energy, other gamma energies sorted descending (NaN-
    padded to max_gammas-1), multiplicity] for ONE gamma of true energy
    `E` within a cascade whose full (possibly NaN-padded) gamma-energy
    list is `gammas`. Single source of truth for this feature convention
    -- `response_function.py` imports this directly so training and
    inference can never drift apart. HistGradientBoostingRegressor
    handles NaN natively (missing-value-aware splits), no imputation."""
    others = sorted(
        (g for g in gammas if np.isfinite(g) and abs(g - E) > 1e-6), reverse=True
    )[: max_gammas - 1]
    others = others + [np.nan] * (max_gammas - 1 - len(others))
    mult = int(np.sum(np.isfinite(gammas)))
    return [E, *others, mult]


def build_features(rows, stem_to_gammas, max_gammas=MAX_GAMMAS):
    return np.array(
        [feature_row(r["energy"], stem_to_gammas[r["stem"]], max_gammas) for r in rows],
        dtype=np.float64,
    )


def relative_pct_error(y_true, y_pred):
    return 100.0 * np.abs(y_pred - y_true) / np.abs(y_true)


def make_regressor():
    return MultiOutputRegressor(HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=20, random_state=0,
    ))


def main():
    d = np.load(DATASET_PATH, allow_pickle=True)
    stem_to_gammas = {str(s): g for s, g in zip(d["file_stems"], d["gamma_energies"])}

    rows = load_stage_rows(STAGE_ROWS_PATH, ROW_FIELDS)
    good = quality_mask(rows)
    rows = [r for r, g in zip(rows, good) if g]
    print(f"{len(rows)} rows pass quality cut (of {len(good)} total in {STAGE_ROWS_PATH})")

    all_stems = [r["stem"] for r in rows]
    train_stems, holdout_stems = split_stems(all_stems, every_n=5, offset=4)
    train_rows = [r for r in rows if r["stem"] in train_stems]
    holdout_rows = [r for r in rows if r["stem"] in holdout_stems]
    print(f"{len(train_rows)} train rows, {len(holdout_rows)} held-out rows "
          f"({len(train_stems)}/{len(holdout_stems)} runs)")

    X_train = build_features(train_rows, stem_to_gammas)
    X_hold = build_features(holdout_rows, stem_to_gammas)
    y_train = np.array([[r["efficiency"], r["sigma_intrinsic"]] for r in train_rows])
    y_hold = np.array([[r["efficiency"], r["sigma_intrinsic"]] for r in holdout_rows])

    reg = make_regressor()
    reg.fit(X_train, y_train)
    pred_hold = reg.predict(X_hold)

    eff_err = relative_pct_error(y_hold[:, 0], pred_hold[:, 0])
    sig_err = relative_pct_error(y_hold[:, 1], pred_hold[:, 1])
    print("\n=== NEW: cascade-context regressor, held-out ===")
    print(f"efficiency:       median={np.median(eff_err):.1f}%  mean={np.mean(eff_err):.1f}%  max={np.max(eff_err):.1f}%")
    print(f"sigma_intrinsic:  median={np.median(sig_err):.1f}%  mean={np.mean(sig_err):.1f}%  max={np.max(sig_err):.1f}%")

    # --- baseline: current marginal energy-only curves, fit on the SAME
    # train rows, checked against the SAME held-out rows ---
    intrinsic_k, _, _, _ = fit_intrinsic_k(train_rows)
    eff_energy, eff_curve = build_efficiency_curve(train_rows)
    e_hold = np.array([r["energy"] for r in holdout_rows])
    pred_eff_curve = np.interp(e_hold, eff_energy, eff_curve)
    pred_sig_curve = _intrinsic_model(e_hold, intrinsic_k)

    eff_err_curve = relative_pct_error(y_hold[:, 0], pred_eff_curve)
    sig_err_curve = relative_pct_error(y_hold[:, 1], pred_sig_curve)
    print("\n=== BASELINE: marginal energy-only curves, SAME held-out rows ===")
    print(f"efficiency:       median={np.median(eff_err_curve):.1f}%  mean={np.mean(eff_err_curve):.1f}%  max={np.max(eff_err_curve):.1f}%")
    print(f"sigma_intrinsic:  median={np.median(sig_err_curve):.1f}%  mean={np.mean(sig_err_curve):.1f}%  max={np.max(sig_err_curve):.1f}%")

    # --- ship a model trained on ALL rows (train convention this project
    # already uses elsewhere: validate with a split, ship the full fit) ---
    X_all = build_features(rows, stem_to_gammas)
    y_all = np.array([[r["efficiency"], r["sigma_intrinsic"]] for r in rows])
    reg_full = make_regressor()
    reg_full.fit(X_all, y_all)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": reg_full, "max_gammas": MAX_GAMMAS}, f)
    print(f"\nSaved {MODEL_PATH} (trained on all {len(rows)} quality-passing rows)")


if __name__ == "__main__":
    main()
