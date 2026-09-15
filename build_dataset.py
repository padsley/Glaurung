"""Build a training dataset from Ancalagon k39(p,g)40Ca cascade-scan runs.

Matches each reaction file (in `~/codes/Ancalagon/reactions/` by default)
to its result directory of the same stem (in
`~/data/Ancalagon_k39_results/` by default, each containing
`dragon_hits.root` + `run.log`), extracts a BGO spectrum from each, and
saves everything to a single `.npz`.

Two cascade topologies are combined by default (see `DEFAULT_TOPOLOGIES`):
- `"ground"` (`k39pg_40ca_cascade_*.reaction`, 88 runs, 50000 events each):
  two-step cascade terminating at 40Ca's true ground state.
- `"0+_2"` (`k39pg_40ca_cascade0p2_*.reaction`, 54 runs, 600000 events
  each): two-step cascade terminating at the real 3.3526 MeV 0+_2 state
  instead (0+_2 -> ground is E0-conversion, not gamma-observable, so the
  reaction file's own "level 0" IS the 0+_2 state -- see that file's own
  COMM header). Both topologies share the same LEVL/BRAT structure (one
  intermediate level, 100/100% branching), so `parse_reaction.py`'s
  generic level-difference logic gives correct *observable* gamma
  energies for both without any topology-specific code -- what differs
  per-row is only the (correctly, automatically computed) `ex`/`q_value`
  values, and a `topology` tag is carried through purely for provenance/
  filtering, not because the physics needs it.

Usage:
    python build_dataset.py --out dataset.npz
    python build_dataset.py --reactions-dir DIR --results-dir DIR --out OUT.npz
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from extract_spectrum import extract_spectrum
from parse_reaction import find_reaction_files, parse_reaction_file

DEFAULT_REACTIONS_DIR = os.path.expanduser("~/codes/Ancalagon/reactions")
DEFAULT_RESULTS_DIR = os.path.expanduser("~/data/Ancalagon_k39_results")
# (topology label, glob pattern) -- order matters only for output ordering.
DEFAULT_TOPOLOGIES = [
    ("ground", "k39pg_40ca_cascade_*.reaction"),
    ("0+_2", "k39pg_40ca_cascade0p2_*.reaction"),
]


def build_dataset(
    reactions_dir: str = DEFAULT_REACTIONS_DIR,
    results_dir: str = DEFAULT_RESULTS_DIR,
    topologies: list[tuple[str, str]] = DEFAULT_TOPOLOGIES,
    emin: float = 0.0,
    emax: float = 10.0,
    nbins: int = 500,
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
            if not (os.path.exists(root_path) and os.path.exists(log_path)):
                print(f"skipping {stem}: missing dragon_hits.root/run.log in {run_dir}")
                continue
            rows.append((stem, topology, reaction_path, root_path, log_path))

    if not rows:
        raise FileNotFoundError(
            f"no reaction files (of any configured topology) had a matching result directory under {results_dir}")

    rows.sort(key=lambda r: (r[1], r[0]))

    feature_keys: list[str] | None = None
    X_rows, Y_addback_rows, Y_singles_rows = [], [], []
    n_total_list, n_hit_list, stems, topology_list = [], [], [], []
    edges = None

    for stem, topology, reaction_path, root_path, log_path in rows:
        rf = parse_reaction_file(reaction_path)
        feats = rf.feature_dict()
        if feature_keys is None:
            feature_keys = sorted(feats.keys())
        elif sorted(feats.keys()) != feature_keys:
            missing = set(feature_keys) - set(feats.keys())
            extra = set(feats.keys()) - set(feature_keys)
            raise ValueError(
                f"{stem}: parameter set differs from earlier files "
                f"(missing {missing}, extra {extra}) -- every configured topology must "
                "share the same LEVL/BRAT structure (same levels/branches, just "
                "different energies) for this flat feature representation to make sense."
            )
        X_rows.append([feats[k] for k in feature_keys])

        addback = extract_spectrum(root_path, log_path, method="addback", emin=emin, emax=emax, nbins=nbins)
        singles = extract_spectrum(root_path, log_path, method="singles", emin=emin, emax=emax, nbins=nbins)
        if edges is None:
            edges = addback["edges"]

        Y_addback_rows.append(addback["counts"])
        Y_singles_rows.append(singles["counts"])
        n_total_list.append(addback["n_total"])
        n_hit_list.append(addback["n_hit"])
        stems.append(stem)
        topology_list.append(topology)
        print(f"[{topology}] {stem}: n_total={addback['n_total']} n_hit={addback['n_hit']} "
              f"({100 * addback['n_hit'] / addback['n_total']:.1f}%)")

    return {
        "X": np.array(X_rows, dtype=np.float64),
        "labels": np.array(feature_keys),
        "Y_addback": np.array(Y_addback_rows, dtype=np.float64),
        "Y_singles": np.array(Y_singles_rows, dtype=np.float64),
        "edges": edges,
        "n_total": np.array(n_total_list, dtype=np.int64),
        "n_hit": np.array(n_hit_list, dtype=np.int64),
        "file_stems": np.array(stems),
        "topology": np.array(topology_list),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reactions-dir", default=DEFAULT_REACTIONS_DIR)
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--pattern", default=None,
                     help="single glob pattern, overriding the default topology list "
                          "(topology tag becomes 'unknown' for all matches)")
    ap.add_argument("--emin", type=float, default=0.0)
    ap.add_argument("--emax", type=float, default=10.0)
    ap.add_argument("--nbins", type=int, default=500)
    ap.add_argument("--out", default="dataset.npz")
    args = ap.parse_args()

    topologies = [("unknown", args.pattern)] if args.pattern else DEFAULT_TOPOLOGIES

    data = build_dataset(
        reactions_dir=args.reactions_dir,
        results_dir=args.results_dir,
        topologies=topologies,
        emin=args.emin,
        emax=args.emax,
        nbins=args.nbins,
    )
    np.savez(args.out, **data)
    print(f"\nSaved {len(data['file_stems'])} samples to {args.out}")
    print(f"X shape {data['X'].shape}, labels: {list(data['labels'])}")
    print(f"Y_addback shape {data['Y_addback'].shape}, Y_singles shape {data['Y_singles'].shape}")
    topologies_seen, counts = np.unique(data["topology"], return_counts=True)
    print(f"topologies: {dict(zip(topologies_seen.tolist(), counts.tolist()))}")
