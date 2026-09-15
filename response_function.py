"""BGO single-crystal ("singles") response function for the Ancalagon
k39(p,g)40Ca pilot, fitted from `fit_response_function.py`'s per-energy
measurements (see `response_function.npz`).

This is a **response-function model**, not a per-cascade regressor (see
`README.md`'s "Open questions" / the follow-up memory note for why that
framing fits this dataset better than porting the G3 project's
`ParametricSpectrumEmulator`): given any true gamma-ray energy, it
predicts the *shape* (position + energy-dependent width) and *amplitude*
(full-energy-peak detection probability) of that gamma's contribution to
a BGO singles spectrum. A full cascade's predicted spectrum is just the
sum of `photopeak(E)` over its gammas -- see `predict_spectrum`.

**Scope of this v1**: photopeak component only (Gaussian, energy-dependent
width). It does NOT yet model the Compton continuum, escape peaks, or
Compton edge -- those need a genuinely joint multi-run deconvolution
(each run's spectrum is a superposition of two gammas' full response, and
the continuum from one overlaps the other), which is real follow-on work,
not done here. Don't expect `predict_spectrum` to reproduce a full
spectrum shape yet -- it reproduces the photopeaks, which is what the
resolution-model fit actually measured.
"""
from __future__ import annotations

import numpy as np


class BgoResponseFunction:
    def __init__(self, doppler_k: float, intrinsic_k: float,
                 efficiency_energy: np.ndarray, efficiency_amplitude: np.ndarray):
        self.doppler_k = doppler_k
        self.intrinsic_k = intrinsic_k
        self.efficiency_energy = np.asarray(efficiency_energy)
        self.efficiency_amplitude = np.asarray(efficiency_amplitude)

    @classmethod
    def load(cls, path: str = "response_function.npz") -> "BgoResponseFunction":
        d = np.load(path)
        return cls(
            doppler_k=float(d["doppler_k"]), intrinsic_k=float(d["intrinsic_k"]),
            efficiency_energy=d["efficiency_energy"], efficiency_amplitude=d["efficiency_amplitude"],
        )

    def sigma(self, E: np.ndarray) -> np.ndarray:
        """Predicted photopeak sigma (MeV) at true gamma energy E (MeV)."""
        E = np.asarray(E, dtype=np.float64)
        return np.sqrt((self.doppler_k * E) ** 2 + self.intrinsic_k**2 * E)

    def full_energy_amplitude(self, E: np.ndarray) -> np.ndarray:
        """Interpolated per-simulated-event full-energy-peak amplitude at
        energy E (MeV) -- clipped to the measured range's boundary values
        outside [0.13, 8.83] MeV (no extrapolation model yet). Below
        ~0.3-0.5 MeV this is known to be contaminated by a large
        near-zero-energy spike in the raw data (partial-deposit/threshold
        events) leaking into the local peak fit's background -- see
        README's "Verification" section -- treat amplitudes there as
        unreliable, not just low.
        """
        return np.interp(E, self.efficiency_energy, self.efficiency_amplitude)

    def photopeak(self, E: float, energies: np.ndarray) -> np.ndarray:
        """Predicted photopeak contribution (counts per simulated event
        per bin) at bin centers `energies` (MeV), for one gamma of true
        energy E (MeV)."""
        sigma = self.sigma(E)
        amplitude = self.full_energy_amplitude(E)
        bin_width = energies[1] - energies[0] if len(energies) > 1 else 1.0
        # amplitude was fit as a peak height (counts/bin, already per-event-normalised);
        # reproduce it as a height-normalised Gaussian, not an area-normalised one.
        return amplitude * np.exp(-0.5 * ((energies - E) / sigma) ** 2)

    def predict_spectrum(self, gamma_energies: list[float], edges: np.ndarray) -> np.ndarray:
        """Sum of photopeak contributions for a cascade emitting the given
        list of true gamma energies (MeV). Photopeak component only --
        see module docstring."""
        centers = 0.5 * (edges[:-1] + edges[1:])
        total = np.zeros_like(centers)
        for E in gamma_energies:
            total += self.photopeak(E, centers)
        return total


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--response", default="response_function.npz")
    ap.add_argument("--energies", type=float, nargs="+", required=True,
                     help="true gamma energies (MeV) to predict a combined spectrum for")
    args = ap.parse_args()

    rf = BgoResponseFunction.load(args.response)
    for E in args.energies:
        print(f"E={E:.4f} MeV: sigma={1000*rf.sigma(E):.1f} keV, "
              f"amplitude={rf.full_energy_amplitude(E):.5f} counts/event")
