"""Build a training dataset from Ancalagon k39(p,g)40Ca cascade-scan runs.

Matches each reaction file (in `~/codes/Ancalagon/reactions/` by default)
to its result directory of the same stem (in
`~/data/Ancalagon_k39_results/` by default, each containing
`dragon_hits.root` + `run.log`), extracts a BGO spectrum from each, and
saves everything to a single `.npz`.

Four cascade topologies are combined by default (see `DEFAULT_TOPOLOGIES`):
- `"ground"` (`k39pg_40ca_cascade_*.reaction`, 88 runs, 50000 events each):
  two-step (1 intermediate level, 2 gammas) cascade terminating at 40Ca's
  true ground state.
- `"0+_2"` (`k39pg_40ca_cascade0p2_*.reaction`, 54 runs, 600000 events
  each): two-step cascade terminating at the real 3.3526 MeV 0+_2 state
  instead (0+_2 -> ground is E0-conversion, not gamma-observable, so the
  reaction file's own "level 0" IS the 0+_2 state -- see that file's own
  COMM header). `"ground"` and `"0+_2"` share the same LEVL/BRAT structure
  (one intermediate level, 100/100% branching).
- `"3g_ground"` (`k39pg_40ca_cascade3g/*.reaction`, 3828 runs, 13000
  events each, added 2026-09-15): three-step (2 intermediate levels, 3
  gammas) cascade, a full triangular Level1/Level2 grid, terminating at
  true ground.
- `"3g_0+_2"` (`k39pg_40ca_cascade3g0p2/*.reaction`, 1431 runs, same event
  count, same date): the three-step analogue of `"0+_2"`.

The two- and three-gamma topologies do **not** share the same LEVL/BRAT
structure (2 vs. 3 gammas per event, different `level(N)`/`br(P,Q)` keys),
so the flat `X` feature matrix is a **union** of every topology's keys,
NaN-filled where a given row's topology doesn't have that key -- don't
assume every column is populated for every row, check `topology` first.
What every topology *does* share, generically (via `parse_reaction.py`'s
`ReactionFile.cascades()`, which needs no per-topology code) is a single
100%-branching decay path with N known gamma energies -- these are what
`fit_response_function.py` actually fits against, stored here as
`gamma_energies` (shape (n_rows, max_gammas), ascending, NaN-padded).

Usage:
    python build_dataset.py --out dataset.npz
    python build_dataset.py --reactions-dir DIR --results-dir DIR --out OUT.npz
    # reuse cached spectra for rows whose raw ROOT/log has since been
    # deleted (e.g. the original 142-run scan, removed 2026-09-15 to free
    # disk space) instead of skipping them:
    python build_dataset.py --merge-old dataset_original_142.npz --out dataset.npz
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from extract_spectrum import extract_both
from parse_reaction import find_reaction_files, parse_reaction_file

DEFAULT_REACTIONS_DIR = os.path.expanduser("~/codes/Ancalagon/reactions")
DEFAULT_RESULTS_DIR = os.path.expanduser("~/data/Ancalagon_k39_results")
MAX_GAMMAS = 3
# (topology label, glob pattern relative to reactions_dir) -- order matters
# only for output ordering.
DEFAULT_TOPOLOGIES = [
    ("ground", "k39pg_40ca_cascade_*.reaction"),
    ("0+_2", "k39pg_40ca_cascade0p2_*.reaction"),
    ("3g_ground", "k39pg_40ca_cascade3g/*.reaction"),
    ("3g_0+_2", "k39pg_40ca_cascade3g0p2/*.reaction"),
]


def _gamma_energies(rf, max_gammas: int = MAX_GAMMAS) -> np.ndarray:
    """Sorted true gamma energies (MeV) for a reaction file's single decay
    path, NaN-padded to `max_gammas`. Every topology here is a single
    100%-branching cascade (verified by `parse_reaction.py`'s own
    `--reaction-stats`, see the emulator README), so `cascades()` always
    returns exactly one path -- more than one would mean a topology this
    pipeline doesn't yet support."""
    cascades = rf.cascades()
    if len(cascades) != 1:
        raise ValueError(
            f"{rf.path}: expected a single 100%-branching cascade, got {len(cascades)} paths "
            "-- this pipeline's response-function fit assumes one fixed gamma list per run")
    energies = sorted(step[2] for step in cascades[0]["steps"])
    if len(energies) > max_gammas:
        raise ValueError(f"{rf.path}: {len(energies)} gammas > max_gammas={max_gammas}, raise the constant")
    padded = np.full(max_gammas, np.nan)
    padded[:len(energies)] = energies
    return padded


def _load_old_cache(path: str) -> dict[str, dict]:
    """Load a previously-built dataset.npz whose raw ROOT/run.log may no
    longer exist on disk, keyed by file stem, so its already-extracted
    spectra can be reused instead of re-extracted."""
    d = np.load(path, allow_pickle=True)
    cache = {}
    for i, stem in enumerate(d["file_stems"]):
        cache[str(stem)] = {
            "Y_addback": d["Y_addback"][i],
            "Y_singles": d["Y_singles"][i],
            "n_total": int(d["n_total"][i]),
            "n_hit": int(d["n_hit"][i]),
        }
    return cache


def build_dataset(
    reactions_dir: str = DEFAULT_REACTIONS_DIR,
    results_dir: str = DEFAULT_RESULTS_DIR,
    topologies: list[tuple[str, str]] = DEFAULT_TOPOLOGIES,
    emin: float = 0.0,
    emax: float = 10.0,
    nbins: int = 500,
    old_cache: dict[str, dict] | None = None,
) -> dict:
    rows = []
    for topology, pattern in topologies:
        reaction_files = find_reaction_files(reactions_dir, pattern)
        if not reaction_files:
            print(f"warning: no reaction files matching {pattern!r} in {reactions_dir} (topology {topology!r})")
            continue
        for stem, reaction_path in reaction_files.items():
            run_dir = os.path.join(results_dir, stem)
            root_path = os.path.join(run_dir, "dragon_hits.root")
            log_path = os.path.join(run_dir, "run.log")
            have_fresh = os.path.exists(root_path) and os.path.exists(log_path)
            have_cached = old_cache is not None and stem in old_cache
            if not (have_fresh or have_cached):
                print(f"skipping {stem}: missing dragon_hits.root/run.log in {run_dir} (and no cached fallback)")
                continue
            rows.append((stem, topology, reaction_path, root_path, log_path, have_fresh))

    if not rows:
        raise FileNotFoundError(
            f"no reaction files (of any configured topology) had a matching result directory under {results_dir}")

    rows.sort(key=lambda r: (r[1], r[0]))

    feature_keys: set[str] = set()
    parsed = []  # (stem, topology, feats, gammas, root_path, log_path, have_fresh)
    for stem, topology, reaction_path, root_path, log_path, have_fresh in rows:
        rf = parse_reaction_file(reaction_path)
        feats = rf.feature_dict()
        feature_keys |= set(feats.keys())
        gammas = _gamma_energies(rf)
        parsed.append((stem, topology, feats, gammas, root_path, log_path, have_fresh))

    feature_keys_sorted = sorted(feature_keys)

    X_rows, Y_addback_rows, Y_singles_rows = [], [], []
    n_total_list, n_hit_list, stems, topology_list, gamma_rows = [], [], [], [], []
    edges = np.linspace(emin, emax, nbins + 1)

    n_cached = 0
    for stem, topology, feats, gammas, root_path, log_path, have_fresh in parsed:
        X_rows.append([feats.get(k, np.nan) for k in feature_keys_sorted])
        gamma_rows.append(gammas)

        if have_fresh:
            both = extract_both(root_path, log_path, emin=emin, emax=emax, nbins=nbins)
            addback, singles = both["addback"], both["singles"]
        else:
            cached = old_cache[stem]
            addback = {"counts": cached["Y_addback"], "n_total": cached["n_total"], "n_hit": cached["n_hit"]}
            singles = {"counts": cached["Y_singles"]}
            n_cached += 1

        Y_addback_rows.append(addback["counts"])
        Y_singles_rows.append(singles["counts"])
        n_total_list.append(addback["n_total"])
        n_hit_list.append(addback["n_hit"])
        stems.append(stem)
        topology_list.append(topology)
        source = "cached" if not have_fresh else "fresh"
        print(f"[{topology}] {stem} ({source}): n_total={addback['n_total']} n_hit={addback['n_hit']} "
              f"({100 * addback['n_hit'] / addback['n_total']:.1f}%)")

    if n_cached:
        print(f"\n{n_cached}/{len(parsed)} rows came from --merge-old cache (raw ROOT/log no longer on disk)")

    return {
        "X": np.array(X_rows, dtype=np.float64),
        "labels": np.array(feature_keys_sorted),
        "Y_addback": np.array(Y_addback_rows, dtype=np.float64),
        "Y_singles": np.array(Y_singles_rows, dtype=np.float64),
        "edges": edges,
        "n_total": np.array(n_total_list, dtype=np.int64),
        "n_hit": np.array(n_hit_list, dtype=np.int64),
        "file_stems": np.array(stems),
        "topology": np.array(topology_list),
        "gamma_energies": np.array(gamma_rows, dtype=np.float64),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reactions-dir", default=DEFAULT_REACTIONS_DIR)
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--pattern", default=None,
                     help="single glob pattern, overriding the default topology list "
                          "(topology tag becomes 'unknown' for all matches)")
    ap.add_argument("--merge-old", default=None,
                     help="path to a previously-built dataset .npz to fall back on, keyed by file "
                          "stem, for any row whose raw dragon_hits.root/run.log no longer exists on disk")
    ap.add_argument("--emin", type=float, default=0.0)
    ap.add_argument("--emax", type=float, default=10.0)
    ap.add_argument("--nbins", type=int, default=500)
    ap.add_argument("--out", default="dataset.npz")
    args = ap.parse_args()

    topologies = [("unknown", args.pattern)] if args.pattern else DEFAULT_TOPOLOGIES
    old_cache = _load_old_cache(args.merge_old) if args.merge_old else None

    data = build_dataset(
        reactions_dir=args.reactions_dir,
        results_dir=args.results_dir,
        topologies=topologies,
        emin=args.emin,
        emax=args.emax,
        nbins=args.nbins,
        old_cache=old_cache,
    )
    np.savez(args.out, **data)
    print(f"\nSaved {len(data['file_stems'])} samples to {args.out}")
    print(f"X shape {data['X'].shape}, labels: {list(data['labels'])}")
    print(f"Y_addback shape {data['Y_addback'].shape}, Y_singles shape {data['Y_singles'].shape}")
    print(f"gamma_energies shape {data['gamma_energies'].shape}")
    topologies_seen, counts = np.unique(data["topology"], return_counts=True)
    print(f"topologies: {dict(zip(topologies_seen.tolist(), counts.tolist()))}")
