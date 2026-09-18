"""BGO single-crystal ("singles") response function for the Ancalagon
k39(p,g)40Ca pilot, fitted from `fit_response_function.py`'s per-energy
measurements (see `response_function.npz`).

This is a **response-function model**, not a per-cascade regressor (see
`README.md`'s "Open questions" / the follow-up memory note for why that
framing fits this dataset better than porting the G3 project's
`ParametricSpectrumEmulator`): given any true gamma-ray energy, it
predicts the *shape* (position + energy-dependent width) and
*efficiency* (full-energy-peak detection probability, i.e. peak area,
not height -- see `full_energy_efficiency`'s docstring) of that gamma's
contribution to a BGO singles spectrum. A full cascade's predicted
spectrum is the sum of `photopeak(E)` over its gammas, plus (2026-09-17,
Phase 1 of the Compton-continuum work) the highest-energy gamma's own
Compton continuum -- see `predict_spectrum`.

**Scope of this version**: photopeaks (all gammas) plus **the Compton
continuum of each cascade's own highest-energy gamma only** -- see
`compton_continuum`'s docstring for exactly why (every run has a clean,
uncontaminated fit window for its top gamma; lower gammas' own continua
are hidden under the top gamma's and need a "peel from highest to
lowest" decomposition, real follow-on work, not done here -- see
README's "Compton continuum" section). Escape peaks and the Compton edge
sharpness are also not modeled yet. Don't expect `predict_spectrum` to
reproduce a full spectrum shape for anything below a cascade's own
second-highest gamma -- that part is still photopeak-only.
"""
from __future__ import annotations

import numpy as np

from gamma_physics import compton_edge_energy, klein_nishina_continuum_shape


class BgoResponseFunction:
    def __init__(self, doppler_k: float, intrinsic_k: float,
                 efficiency_energy: np.ndarray, efficiency_curve: np.ndarray,
                 continuum_energy: np.ndarray | None = None,
                 continuum_curve: np.ndarray | None = None):
        self.doppler_k = doppler_k
        self.intrinsic_k = intrinsic_k
        self.efficiency_energy = np.asarray(efficiency_energy)
        self.efficiency_curve = np.asarray(efficiency_curve)
        # Optional: absent means predict_spectrum falls back to
        # photopeak-only (pre-2026-09-17 behavior), e.g. when only
        # response_function.npz (no compton_continuum.npz) is available.
        self.continuum_energy = None if continuum_energy is None else np.asarray(continuum_energy)
        self.continuum_curve = None if continuum_curve is None else np.asarray(continuum_curve)

    @classmethod
    def load(cls, path: str = "response_function.npz",
              continuum_path: str = "compton_continuum.npz") -> "BgoResponseFunction":
        d = np.load(path)
        continuum_energy = continuum_curve = None
        try:
            c = np.load(continuum_path)
            continuum_energy, continuum_curve = c["continuum_energy"], c["continuum_curve"]
        except FileNotFoundError:
            pass
        return cls(
            doppler_k=float(d["doppler_k"]), intrinsic_k=float(d["intrinsic_k"]),
            efficiency_energy=d["efficiency_energy"], efficiency_curve=d["efficiency_curve"],
            continuum_energy=continuum_energy, continuum_curve=continuum_curve,
        )

    def sigma(self, E: np.ndarray) -> np.ndarray:
        """Predicted photopeak sigma (MeV) at true gamma energy E (MeV)."""
        E = np.asarray(E, dtype=np.float64)
        return np.sqrt((self.doppler_k * E) ** 2 + self.intrinsic_k**2 * E)

    def full_energy_efficiency(self, E: np.ndarray) -> np.ndarray:
        """Interpolated per-simulated-event full-energy-peak detection
        probability at energy E (MeV) -- the fitted Gaussian peak's
        *area* (`amplitude * sigma * sqrt(2*pi) / bin_width`), not its
        height, so it's a genuine, resolution-independent efficiency
        (see `fit_response_function.py`'s `measure_all_peaks` docstring
        for why that distinction matters -- height alone conflated
        efficiency with resolution and made the curve noisy). Built from
        a robust median-per-energy-bin smoothing (`build_efficiency_curve`),
        clipped to the measured range's boundary values outside
        [0.13, 8.83] MeV (no extrapolation model yet). Below ~0.3-0.5 MeV
        this is known to be contaminated by a large near-zero-energy
        spike in the raw data (partial-deposit/threshold events) leaking
        into the local peak fit's background -- see README's
        "Verification" section -- treat efficiencies there as unreliable,
        not just low.
        """
        return np.interp(E, self.efficiency_energy, self.efficiency_curve)

    def photopeak(self, E: float, energies: np.ndarray) -> np.ndarray:
        """Predicted photopeak contribution (counts per simulated event
        per bin) at bin centers `energies` (MeV), for one gamma of true
        energy E (MeV)."""
        sigma = self.sigma(E)
        efficiency = self.full_energy_efficiency(E)
        bin_width = energies[1] - energies[0] if len(energies) > 1 else 1.0
        # efficiency is an *area* (bin_width-normalised integrated counts/event);
        # convert back to the height of the height-normalised Gaussian this
        # spectrum's bins actually store, i.e. invert the Gaussian-area formula.
        height = efficiency * bin_width / (sigma * np.sqrt(2 * np.pi))
        return height * np.exp(-0.5 * ((energies - E) / sigma) ** 2)

    def compton_continuum(self, E: float, energies: np.ndarray) -> np.ndarray:
        """Predicted Compton-continuum contribution (counts per simulated
        event per bin) at bin centers `energies` (MeV), for one gamma of
        true energy `E` (MeV) that is the **highest-energy gamma in its
        cascade** -- see `predict_spectrum` and the module docstring for
        why this only applies to the top gamma (Phase 1 scope).

        Shape from `gamma_physics.klein_nishina_continuum_shape` (the raw
        Klein-Nishina Compton-electron-energy shape -- confirmed to match
        this simulation's real data well with just a single free
        amplitude, no extra shape correction, by a visual check across 4
        widely-spaced true energies before this was fit), scaled by
        `continuum_curve` (interpolated vs. `E`, same median-binned-
        smoothing method as `efficiency_curve`, fit only from each run's
        own clean uncontaminated continuum window -- see
        `fit_compton_continuum.py`). Zero outside `[0, compton_edge_energy(E)]`.

        **Only calibrated for `E >= ~2.35 MeV`**: no run in this dataset
        has its highest-energy gamma below that (every topology's gammas
        are constrained to sum to a fixed total `Ex`, so when 2 or 3
        gammas are shared out as evenly as possible the *smallest
        possible* top-gamma energy is `Ex/n_gammas` -- about 1.86 MeV in
        the best case here, `~2.35` in practice once near-degenerate
        exclusions are accounted for). Below the curve's lowest measured
        point this clips to that boundary value like `full_energy_efficiency`
        does -- silently wrong, not just extrapolated, for a cascade whose
        own top gamma is genuinely below ~2.35 MeV. If `continuum_curve`
        is `None` (no `compton_continuum.npz` loaded), returns zeros.
        """
        if self.continuum_energy is None:
            return np.zeros_like(np.asarray(energies, dtype=np.float64))
        amplitude = np.interp(E, self.continuum_energy, self.continuum_curve)
        shape = klein_nishina_continuum_shape(E, energies)
        return amplitude * shape

    def predict_spectrum(self, gamma_energies: list[float], edges: np.ndarray) -> np.ndarray:
        """Sum of photopeak contributions for every gamma in a cascade,
        plus the Compton continuum of the cascade's own **highest-energy**
        gamma only (Phase 1 -- see module docstring; lower gammas'
        continua are not modeled, so the predicted spectrum is
        photopeak-only below the second-highest gamma)."""
        centers = 0.5 * (edges[:-1] + edges[1:])
        total = np.zeros_like(centers)
        for E in gamma_energies:
            total += self.photopeak(E, centers)
        if len(gamma_energies) > 0:
            e_top = max(gamma_energies)
            continuum = self.compton_continuum(e_top, centers)
            total += np.where(centers <= compton_edge_energy(e_top), continuum, 0.0)
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
        continuum_note = ""
        if rf.continuum_energy is not None:
            continuum_note = (f", continuum_amplitude(if top gamma)="
                               f"{np.interp(E, rf.continuum_energy, rf.continuum_curve):.4g}")
        print(f"E={E:.4f} MeV: sigma={1000*rf.sigma(E):.1f} keV, "
              f"efficiency={rf.full_energy_efficiency(E):.5f} counts/event{continuum_note}")
