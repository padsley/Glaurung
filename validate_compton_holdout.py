"""Held-out validation of the Compton-continuum fit in
`fit_compton_continuum.py`, mirroring `validate_holdout.py`'s approach
for the photopeak resolution/efficiency fit: hold out every 5th run
(same `split_stems`, so results are directly comparable run-for-run),
refit the continuum curve on the rest, check against runs the fit never
saw.
"""
from __future__ import annotations

import numpy as np

from fit_compton_continuum import DATASET_PATH, RESPONSE_PATH, build_continuum_curve, measure_all_continuum
from validate_holdout import split_stems


def run_validation(dataset_path=DATASET_PATH, response_path=RESPONSE_PATH, every_n=5, offset=4, rel_err_max=0.5):
    rows, doppler_k, intrinsic_k = measure_all_continuum(dataset_path, response_path, verbose=False)
    all_stems = [r["stem"] for r in rows]
    train_stems, holdout_stems = split_stems(all_stems, every_n=every_n, offset=offset)
    print(f"{len(train_stems)} train runs, {len(holdout_stems)} held-out runs (every {every_n}th, offset {offset})")

    train_rows = [r for r in rows if r["stem"] in train_stems]
    holdout_rows = [r for r in rows if r["stem"] in holdout_stems]

    continuum_energy, continuum_curve = build_continuum_curve(train_rows, rel_err_max=rel_err_max)
    print(f"[trained on {len(train_rows)} train rows] continuum curve: {len(continuum_energy)} occupied bins")

    ho_energy = np.array([r["energy"] for r in holdout_rows])
    ho_amp = np.array([r["amplitude"] for r in holdout_rows])
    ho_rel_err = np.array([r["rel_err"] for r in holdout_rows])
    ho_chi2 = np.array([r["chi2_ndf"] for r in holdout_rows])
    ho_good = np.isfinite(ho_rel_err) & (ho_rel_err < rel_err_max)

    ho_pred = np.interp(ho_energy[ho_good], continuum_energy, continuum_curve)
    ho_rel_diff = (ho_amp[ho_good] - ho_pred) / ho_amp[ho_good]
    print(f"HELD-OUT continuum-amplitude relative difference (ALL {ho_good.sum()} held-out points): "
          f"median={np.median(np.abs(ho_rel_diff)) * 100:.1f}%, "
          f"mean={np.mean(np.abs(ho_rel_diff)) * 100:.1f}%, max={np.max(np.abs(ho_rel_diff)) * 100:.1f}%")

    ho_clean = ho_chi2[ho_good] < 50.0
    clean_diff = ho_rel_diff[ho_clean]
    print(f"HELD-OUT continuum-amplitude relative difference (CLEAN {ho_clean.sum()} of those, chi2/ndf<50): "
          f"median={np.median(np.abs(clean_diff)) * 100:.1f}%, "
          f"mean={np.mean(np.abs(clean_diff)) * 100:.1f}%, max={np.max(np.abs(clean_diff)) * 100:.1f}%")

    return {
        "holdout_energy": ho_energy[ho_good], "holdout_amplitude": ho_amp[ho_good],
        "holdout_amplitude_pred": ho_pred, "holdout_amplitude_rel_diff": ho_rel_diff,
        "holdout_clean_mask": ho_clean,
        "continuum_energy": continuum_energy, "continuum_curve": continuum_curve,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=DATASET_PATH)
    ap.add_argument("--response", default=RESPONSE_PATH)
    ap.add_argument("--every-n", type=int, default=5)
    ap.add_argument("--offset", type=int, default=4)
    ap.add_argument("--out", default="compton_continuum_holdout.npz")
    args = ap.parse_args()

    result = run_validation(args.dataset, args.response, every_n=args.every_n, offset=args.offset)
    np.savez(args.out, **result)
    print(f"\nSaved {args.out}")
