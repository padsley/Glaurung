"""Parser for Ancalagon (Geant4) `.reaction` config files.

These are the run-time reaction-specification files consumed by
`dragon_g4_pilot` (see `~/codes/Ancalagon/README.md`, "Reaction config
file"): a small namelist-style card format --

    COMM  free-text comment line, ignored
    BEAM  Z A massExcessMeV               # projectile
    TARG  Z A massExcessMeV               # target (at rest)
    RECL  Z A massExcessMeV chargeState    # recoil
    ERES  EresCM_MeV                       # CM resonance energy above threshold
    RWID  widthCM_MeV                      # optional, resonance width
    BKIN  beamEnergyMeV                    # optional, beam-energy-loss vertex energy
    RTUN  scale                            # optional, precomputed retune scale
    LEVL  levelID  energyMeV  lifetimeS    # one line per excited level
    BRAT  fromLevel  percent  toLevel      # one line per branch (fromLevel=-1 is the resonance)
    SENT  end-of-file marker

Trailing `#`-prefixed inline comments are stripped from every card line.
Level id -1 is the compound/resonance state; level id 0 is the ground
state; these never appear in a LEVL card (their energies are derived,
not given).
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field


@dataclass
class ReactionFile:
    path: str
    beam: tuple          # (Z, A, mass_excess_MeV)
    targ: tuple
    recl: tuple          # (Z, A, mass_excess_MeV, charge_state)
    eres: float          # CM resonance energy above threshold, MeV
    rwid: float | None = None
    bkin: float | None = None
    rtun: float | None = None
    levels: dict = field(default_factory=dict)     # {level_id: (energy_MeV, lifetime_s)}
    branches: list = field(default_factory=list)   # [(from_level, percent, to_level), ...]

    @property
    def q_value_mev(self) -> float:
        """Q = massExcess(beam) + massExcess(target) - massExcess(recoil).

        Valid because A_beam + A_target == A_recoil (mass-number-conserving
        capture reaction), so the A*amu terms in each mass cancel.
        """
        return self.beam[2] + self.targ[2] - self.recl[2]

    @property
    def ex_mev(self) -> float:
        """Compound-nucleus excitation energy above ground: Q + ERES."""
        return self.q_value_mev + self.eres

    def level_energy(self, level_id: int) -> float:
        """Excitation energy (MeV) of `level_id`; -1 is the resonance, 0 is ground."""
        if level_id == -1:
            return self.ex_mev
        if level_id == 0:
            return 0.0
        return self.levels[level_id][0]

    def cascades(self) -> list[dict]:
        """Enumerate every resonance->...->ground decay path.

        Returns a list of dicts, one per complete cascade path:
            {
              "steps": [(from_level, to_level, gamma_energy_MeV), ...],
              "probability": float,   # fraction (0-1), product of each step's branch fraction
            }
        Branch percentages at a given `fromLevel` are used as given (not
        renormalised here) -- matches how small pilot cascades in this
        dataset are written (single branch = 100%). For a level with
        multiple branches that don't sum to 100, the raw BRAT percentages
        are still used as the per-step probability weight.
        """
        by_from: dict[int, list[tuple[int, float]]] = {}
        for frm, pct, to in self.branches:
            by_from.setdefault(frm, []).append((to, pct))

        paths: list[dict] = []

        def walk(level_id: int, steps: list, prob: float):
            if level_id == 0:
                paths.append({"steps": list(steps), "probability": prob})
                return
            for to_level, pct in by_from.get(level_id, []):
                e_from = self.level_energy(level_id)
                e_to = self.level_energy(to_level)
                gamma_e = e_from - e_to
                steps.append((level_id, to_level, gamma_e))
                walk(to_level, steps, prob * pct / 100.0)
                steps.pop()

        walk(-1, [], 1.0)
        return paths

    def feature_dict(self) -> dict:
        """Flat, ML-friendly parameter dict.

        Follows the same naming convention as the old Geant3 project's
        `parse_input.py` (`level(N)`, `br(from,to)`) for continuity, plus
        this project's own derived quantities (`eres`, `ex`, `q_value`).
        """
        out = {"eres": self.eres, "q_value": self.q_value_mev, "ex": self.ex_mev}
        for level_id, (energy, _lifetime) in sorted(self.levels.items()):
            out[f"level({level_id})"] = energy
        for frm, pct, to in self.branches:
            out[f"br({frm},{to})"] = pct
        return out


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].rstrip()


def parse_reaction_file(path: str) -> ReactionFile:
    beam = targ = recl = None
    eres = rwid = bkin = rtun = None
    levels: dict[int, tuple[float, float]] = {}
    branches: list[tuple[int, float, int]] = []

    with open(path) as fh:
        for raw_line in fh:
            line = _strip_comment(raw_line).strip()
            if not line or line.startswith("COMM"):
                continue
            parts = line.split()
            card = parts[0]
            if card == "BEAM":
                beam = (int(parts[1]), int(parts[2]), float(parts[3]))
            elif card == "TARG":
                targ = (int(parts[1]), int(parts[2]), float(parts[3]))
            elif card == "RECL":
                recl = (int(parts[1]), int(parts[2]), float(parts[3]), int(parts[4]))
            elif card == "ERES":
                eres = float(parts[1])
            elif card == "RWID":
                rwid = float(parts[1])
            elif card == "BKIN":
                bkin = float(parts[1])
            elif card == "RTUN":
                rtun = float(parts[1])
            elif card == "LEVL":
                level_id = int(parts[1])
                levels[level_id] = (float(parts[2]), float(parts[3]))
            elif card == "BRAT":
                branches.append((int(parts[1]), float(parts[2]), int(parts[3])))
            elif card == "SENT":
                break
            else:
                raise ValueError(f"{path}: unrecognised card {card!r}")

    if beam is None or targ is None or recl is None or eres is None:
        raise ValueError(f"{path}: missing one of BEAM/TARG/RECL/ERES")

    return ReactionFile(
        path=path, beam=beam, targ=targ, recl=recl, eres=eres,
        rwid=rwid, bkin=bkin, rtun=rtun, levels=levels, branches=branches,
    )


def find_reaction_files(reactions_dir: str, pattern: str = "*.reaction") -> dict[str, str]:
    """Return {stem: path} for every reaction file matching `pattern`."""
    out = {}
    for p in sorted(glob.glob(os.path.join(reactions_dir, pattern))):
        stem = os.path.splitext(os.path.basename(p))[0]
        out[stem] = p
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reaction_file")
    args = ap.parse_args()

    rf = parse_reaction_file(args.reaction_file)
    print(f"Q = {rf.q_value_mev:.5f} MeV, Ex = {rf.ex_mev:.5f} MeV")
    print("levels:", rf.levels)
    print("branches:", rf.branches)
    print("features:", rf.feature_dict())
    print("cascades:")
    for c in rf.cascades():
        gammas = ", ".join(f"{lvl_from}->{lvl_to}: {e:.5f} MeV" for lvl_from, lvl_to, e in c["steps"])
        print(f"  P={100 * c['probability']:.3f}%  {gammas}")
