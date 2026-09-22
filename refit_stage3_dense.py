"""One-off: refit stage-3 (fixed-beta) peaks against the post-dense-grid
dataset.npz (10419 rows), replacing the stale peak_measurements.npz
(Sep 18, pre-dense-grid, 5426 rows only). beta is NOT re-derived --
already extremely well-determined from thousands of old windows and
Phase A's plan explicitly keeps it fixed; only stage 3 (fixed-beta
per-peak refit) needs to see the new runs."""
import numpy as np

from fit_response_function import ROW_FIELDS, STAGE_ROWS_PATH, DATASET_PATH, measure_all_peaks, save_stage_rows

beta = float(np.load("response_function.npz")["beta"])
print(f"using fixed beta={beta:.6f}")

rows3 = measure_all_peaks(DATASET_PATH, n_window_sigma=5.0, beta=beta)
save_stage_rows(rows3, ROW_FIELDS, STAGE_ROWS_PATH)
print(f"\nSaved {len(rows3)} stage-3 rows to {STAGE_ROWS_PATH}")
