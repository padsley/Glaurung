"""Extract a per-event BGO energy-deposit spectrum from an Ancalagon
(Geant4) `dragon_hits.root` file.

Format (see `~/codes/Ancalagon/README.md`, "Sensitive detectors" / "ROOT
output"): the `Bgo` TTree has **one row per hit** (not per event) --
`eventID`, `crystalID` (1-30), `edepMeV`, position, time, `trackID`,
`particle`. A single event can span many rows: multiple steps in one
crystal, and/or multiple crystals if the gamma Compton-scatters between
them. Rows are only written for events with at least one BGO step --
zero-hit events are simply absent from the tree (there is no `edep=0`
placeholder row), so **the tree's row/event count alone cannot give you
the simulated-event total** -- see `get_n_simulated events` below.

Two per-event energy definitions are supported:
  - "addback" (default): sum of edepMeV across every hit in the event,
    regardless of crystal -- recovers the full cascade energy even when
    a gamma Compton-scatters between crystals. This is the standard
    modern BGO "add-back" analysis convention.
  - "singles": the single crystal with the largest total deposited
    energy in that event (summed within that one crystal only) -- matches
    the older DRAGON Geant3 dataset's `e_bgo_first` observable (see
    `~/data/DRAGON_G3_Emulator/`), which used only the highest-energy
    single hit, not add-back. Kept for comparability with that project;
    not necessarily the better physical choice for a fresh analysis.

**No recoil gate is available here** (unlike the Geant3 dataset's
`recoil_hit_endv==1`): the `Dsssd` tree is currently empty for every
run in this dataset (recoils aren't reliably reaching the DSSSD with
EM physics turned on in this Geant4 pilot -- a known, documented
limitation, not a bug in this extractor). Spectra extracted here are
therefore BGO-singles-run spectra, ungated on recoil detection.
"""
from __future__ import annotations

import os

import numpy as np
import uproot


def get_n_simulated_events(run_log_path: str) -> int:
    """Count simulated events from the run's verbose log.

    `EventAction` prints one `HITS BGO ...` line per simulated event
    (confirmed: exactly 50000 lines for a 50000-event run, including
    zero-hit events printed as `n=0`) -- this is the only record of the
    true denominator, since zero-hit events leave no trace in the ROOT
    file itself.
    """
    n = 0
    with open(run_log_path, "rb") as fh:
        for line in fh:
            if line.startswith(b"HITS BGO"):
                n += 1
    return n


def _read_bgo_hits(root_path: str):
    """One ROOT open, all three branches -- shared by both energy
    definitions so a caller wanting both (`extract_both` below) doesn't
    pay for two separate file opens."""
    with uproot.open(root_path) as f:
        tree = f["Bgo"]
        event_id = tree["eventID"].array(library="np")
        edep = tree["edepMeV"].array(library="np")
        crystal_id = tree["crystalID"].array(library="np")
    return event_id, edep, crystal_id


def _addback_energies(event_id: np.ndarray, edep: np.ndarray) -> np.ndarray:
    if len(event_id) == 0:
        return np.array([], dtype=np.float64)
    order = np.argsort(event_id, kind="stable")
    eid_sorted = event_id[order]
    edep_sorted = edep[order]
    totals = np.add.reduceat(edep_sorted, np.r_[0, np.flatnonzero(np.diff(eid_sorted)) + 1])
    return totals


def _singles_energies(event_id: np.ndarray, edep: np.ndarray, crystal_id: np.ndarray) -> np.ndarray:
    if len(event_id) == 0:
        return np.array([], dtype=np.float64)
    # sum within (event, crystal), then take the max crystal per event
    key = event_id.astype(np.int64) * 1000 + crystal_id.astype(np.int64)  # crystalID is 1-30
    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    edep_sorted = edep[order]
    group_starts = np.r_[0, np.flatnonzero(np.diff(key_sorted)) + 1]
    per_crystal_totals = np.add.reduceat(edep_sorted, group_starts)
    per_crystal_event = key_sorted[group_starts] // 1000

    order2 = np.argsort(per_crystal_event, kind="stable")
    eid2_sorted = per_crystal_event[order2]
    tot2_sorted = per_crystal_totals[order2]
    ev_starts = np.r_[0, np.flatnonzero(np.diff(eid2_sorted)) + 1]
    ev_ends = np.r_[ev_starts[1:], len(eid2_sorted)]
    singles = np.array([tot2_sorted[s:e].max() for s, e in zip(ev_starts, ev_ends)])
    return singles


def get_event_energies(root_path: str, method: str = "addback") -> np.ndarray:
    """Per-event total BGO energy deposit (MeV), one entry per event that
    had >=1 hit (zero-hit events are not represented -- pad with the
    known `n_total - len(result)` zeros yourself if you need every
    simulated event, e.g. when histogramming for an efficiency-correct
    spectrum -- `extract_spectrum` below does this).
    """
    if method not in ("addback", "singles"):
        raise ValueError(f"method must be 'addback' or 'singles', got {method!r}")

    event_id, edep, crystal_id = _read_bgo_hits(root_path)
    if method == "addback":
        return _addback_energies(event_id, edep)
    return _singles_energies(event_id, edep, crystal_id)


def extract_spectrum(
    root_path: str,
    run_log_path: str,
    method: str = "addback",
    emin: float = 0.0,
    emax: float = 10.0,
    nbins: int = 500,
) -> dict:
    """Build a normalised (counts-per-simulated-event) BGO spectrum.

    Returns {"counts": (nbins,), "edges": (nbins+1,), "n_total": int,
    "n_hit": int} -- `n_total` is every simulated event (from the run
    log, see `get_n_simulated_events`), `n_hit` is how many had >=1 BGO
    hit at all (regardless of energy). `counts` sums to
    `n_hit_above_threshold / n_total` if you want an overall BGO
    efficiency figure, not to 1.
    """
    energies = get_event_energies(root_path, method=method)
    n_total = get_n_simulated_events(run_log_path)
    n_hit = len(energies)

    edges = np.linspace(emin, emax, nbins + 1)
    hist, _ = np.histogram(energies, bins=edges)
    counts = hist.astype(np.float64) / n_total

    return {"counts": counts, "edges": edges, "n_total": n_total, "n_hit": n_hit}


def extract_both(
    root_path: str,
    run_log_path: str,
    emin: float = 0.0,
    emax: float = 10.0,
    nbins: int = 500,
) -> dict:
    """Same as calling `extract_spectrum` for both methods, but with one
    ROOT open and one run.log read shared between them -- halves the I/O
    for callers (e.g. `build_dataset.py`) that always want both.
    Returns {"addback": {...}, "singles": {...}}, each shaped like
    `extract_spectrum`'s return value.
    """
    event_id, edep, crystal_id = _read_bgo_hits(root_path)
    n_total = get_n_simulated_events(run_log_path)
    edges = np.linspace(emin, emax, nbins + 1)

    out = {}
    for method, energies in (
        ("addback", _addback_energies(event_id, edep)),
        ("singles", _singles_energies(event_id, edep, crystal_id)),
    ):
        hist, _ = np.histogram(energies, bins=edges)
        out[method] = {
            "counts": hist.astype(np.float64) / n_total,
            "edges": edges,
            "n_total": n_total,
            "n_hit": len(energies),
        }
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="directory containing dragon_hits.root and run.log")
    ap.add_argument("--method", choices=["addback", "singles"], default="addback")
    ap.add_argument("--emin", type=float, default=0.0)
    ap.add_argument("--emax", type=float, default=10.0)
    ap.add_argument("--nbins", type=int, default=500)
    args = ap.parse_args()

    root_path = os.path.join(args.run_dir, "dragon_hits.root")
    log_path = os.path.join(args.run_dir, "run.log")
    result = extract_spectrum(root_path, log_path, method=args.method, emin=args.emin, emax=args.emax, nbins=args.nbins)
    print(f"n_total={result['n_total']}  n_hit={result['n_hit']} ({100 * result['n_hit'] / result['n_total']:.2f}%)")
    print(f"sum(counts)={result['counts'].sum():.4f} (fraction of simulated events landing in [{args.emin},{args.emax}] MeV)")
    nz = np.flatnonzero(result["counts"])
    if len(nz):
        centers = 0.5 * (result["edges"][:-1] + result["edges"][1:])
        peak_bin = nz[np.argmax(result["counts"][nz])]
        print(f"largest bin: {centers[peak_bin]:.3f} MeV, {result['counts'][peak_bin]:.5f} counts/event")
