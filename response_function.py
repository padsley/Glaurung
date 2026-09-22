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
Compton continuum and backscatter dome -- see `predict_spectrum`.

**Lineshape (changed 2026-09-19)**: `photopeak` uses
`gamma_physics.doppler_lineshape` (a Doppler-broadened "box convolved
with a Gaussian" shape), not a plain Gaussian -- see that function's own
docstring for the physics and `fit_response_function.py`'s module
docstring for the before/after chi2/ndf check that motivated the switch
(551 -> 218 on a real isolated peak). `beta` (the recoil velocity
fraction, a single physical constant) replaces `doppler_k`;
`intrinsic_k` now describes *only* the non-Doppler (light-collection)
part of the width.

**Scope of this version**: photopeaks (all gammas, Doppler lineshape)
plus, for each cascade's own **highest-energy gamma only**: Compton
continuum and (2026-09-19) its backscatter dome -- see
`compton_continuum`/`backscatter_dome`'s docstrings for exactly why this
is top-gamma-only (every run has a clean, uncontaminated fit window for
its top gamma; lower gammas' own continua are hidden under the top
gamma's and need a "peel from highest to lowest" decomposition, real
follow-on work, not done here -- see README's "Compton continuum"
section). Escape peaks and the Compton edge's sharpness are still not
modeled. Don't expect `predict_spectrum` to reproduce a full spectrum
shape for anything below a cascade's own second-highest gamma -- that
part is still photopeak-only.
"""
from __future__ import annotations

import pickle

import numpy as np

from gamma_physics import backscatter_energy, compton_edge_energy, doppler_lineshape, klein_nishina_continuum_shape


class BgoResponseFunction:
    def __init__(self, beta: float, intrinsic_k: float,
                 efficiency_energy: np.ndarray, efficiency_curve: np.ndarray,
                 continuum_energy: np.ndarray | None = None,
                 continuum_curve: np.ndarray | None = None,
                 dome_energy: np.ndarray | None = None,
                 dome_curve: np.ndarray | None = None,
                 dome_sigma_curve: np.ndarray | None = None,
                 efficiency_model=None, efficiency_model_max_gammas: int = 4):
        self.beta = beta
        self.intrinsic_k = intrinsic_k
        self.efficiency_energy = np.asarray(efficiency_energy)
        self.efficiency_curve = np.asarray(efficiency_curve)
        # Optional (Phase A, 2026-09-22): a HistGradientBoostingRegressor
        # predicting [efficiency, sigma_intrinsic] from a gamma's full
        # cascade context (see train_efficiency_model.py) instead of from
        # its own energy alone. Absent by default -- the curve-based
        # model above stays the default/fallback, not replaced outright;
        # only used by predict_spectrum's cascade-aware path below, never
        # by the single-energy photopeak()/sigma_intrinsic()/
        # full_energy_efficiency() methods (those have no cascade context
        # to give it and keep their existing curve-based behavior).
        self.efficiency_model = efficiency_model
        self.efficiency_model_max_gammas = efficiency_model_max_gammas
        # Optional: absent means predict_spectrum falls back to
        # photopeak-only (pre-2026-09-17 behavior), e.g. when only
        # response_function.npz (no compton_continuum.npz) is available.
        self.continuum_energy = None if continuum_energy is None else np.asarray(continuum_energy)
        self.continuum_curve = None if continuum_curve is None else np.asarray(continuum_curve)
        # Optional: absent means compton_continuum's region includes no
        # backscatter-dome bump (pre-2026-09-19 behavior).
        self.dome_energy = None if dome_energy is None else np.asarray(dome_energy)
        self.dome_curve = None if dome_curve is None else np.asarray(dome_curve)
        self.dome_sigma_curve = None if dome_sigma_curve is None else np.asarray(dome_sigma_curve)

    @classmethod
    def load(cls, path: str = "response_function.npz",
              continuum_path: str = "compton_continuum.npz",
              efficiency_model_path: str | None = "efficiency_model.pkl") -> "BgoResponseFunction":
        d = np.load(path)
        continuum_energy = continuum_curve = None
        dome_energy = dome_curve = dome_sigma_curve = None
        try:
            c = np.load(continuum_path)
            continuum_energy, continuum_curve = c["continuum_energy"], c["continuum_curve"]
            if "dome_energy" in c.files:
                dome_energy, dome_curve = c["dome_energy"], c["dome_curve"]
                dome_sigma_curve = c["dome_sigma_curve"]
        except FileNotFoundError:
            pass
        efficiency_model, efficiency_model_max_gammas = None, 4
        if efficiency_model_path is not None:
            try:
                with open(efficiency_model_path, "rb") as f:
                    saved = pickle.load(f)
                efficiency_model = saved["model"]
                efficiency_model_max_gammas = saved["max_gammas"]
            except FileNotFoundError:
                pass
        return cls(
            beta=float(d["beta"]), intrinsic_k=float(d["intrinsic_k"]),
            efficiency_energy=d["efficiency_energy"], efficiency_curve=d["efficiency_curve"],
            continuum_energy=continuum_energy, continuum_curve=continuum_curve,
            dome_energy=dome_energy, dome_curve=dome_curve, dome_sigma_curve=dome_sigma_curve,
            efficiency_model=efficiency_model, efficiency_model_max_gammas=efficiency_model_max_gammas,
        )

    def sigma_intrinsic(self, E: np.ndarray) -> np.ndarray:
        """Predicted *intrinsic* (non-Doppler, light-collection) photopeak
        width (MeV) at true gamma energy E (MeV) -- one component of
        `photopeak`'s Doppler lineshape, not a claim that the peak itself
        is Gaussian with this width (it isn't, see `photopeak`)."""
        E = np.asarray(E, dtype=np.float64)
        return self.intrinsic_k * np.sqrt(E)

    def effective_sigma(self, E: np.ndarray) -> np.ndarray:
        """Variance-equivalent total photopeak width (MeV) -- a single-
        number stand-in for "how wide is this peak", for window-sizing
        callers only (e.g. `fit_compton_continuum.py`), not the lineshape
        itself. See `gamma_physics.effective_sigma`."""
        E = np.asarray(E, dtype=np.float64)
        return np.sqrt((self.beta * E) ** 2 / 3.0 + self.sigma_intrinsic(E) ** 2)

    def full_energy_efficiency(self, E: np.ndarray) -> np.ndarray:
        """Interpolated per-simulated-event full-energy-peak detection
        probability at energy E (MeV) -- the fitted lineshape's *area*
        (a resolution-independent full-energy-peak efficiency; see
        `fit_response_function.py`'s `_multi_doppler_fit` docstring for
        why the new lineshape needs no extra area-conversion factor, unlike
        the old Gaussian-height convention). Built from a robust
        median-per-energy-bin smoothing (`build_efficiency_curve`),
        clipped to the measured range's boundary values (no extrapolation
        model yet). Below ~0.3-0.5 MeV this is known to be contaminated by
        a large near-zero-energy spike in the raw data (partial-deposit/
        threshold events) leaking into the local peak fit's background --
        see README's "Verification" section -- treat efficiencies there
        as unreliable, not just low.
        """
        return np.interp(E, self.efficiency_energy, self.efficiency_curve)

    def photopeak(self, E: float, energies: np.ndarray) -> np.ndarray:
        """Predicted photopeak contribution (counts per simulated event
        per bin) at bin centers `energies` (MeV), for one gamma of true
        energy E (MeV). Doppler-broadened lineshape (see module
        docstring), not a Gaussian."""
        sigma_intrinsic = self.sigma_intrinsic(E)
        efficiency = self.full_energy_efficiency(E)
        bin_width = energies[1] - energies[0] if len(energies) > 1 else 1.0
        # efficiency is already an area (see full_energy_efficiency's docstring);
        # doppler_lineshape is a unit-area density, so scaling by
        # efficiency*bin_width directly reproduces per-bin counts/event.
        return efficiency * bin_width * doppler_lineshape(E, energies, self.beta, sigma_intrinsic)

    def _photopeaks_for_cascade(self, gamma_energies: list[float], energies: np.ndarray) -> np.ndarray:
        """Sum of photopeak contributions for every gamma in a cascade.
        Uses `efficiency_model` (if loaded) to predict each gamma's
        `(efficiency, sigma_intrinsic)` jointly from its full cascade
        context; falls back to the per-energy curves (`photopeak`) if no
        model was loaded -- identical output to summing `photopeak(E,
        energies)` over `gamma_energies` in that case."""
        bin_width = energies[1] - energies[0] if len(energies) > 1 else 1.0
        total = np.zeros_like(energies, dtype=np.float64)
        if self.efficiency_model is None:
            for E in gamma_energies:
                total += self.photopeak(E, energies)
            return total

        from train_efficiency_model import feature_row  # local import: optional dependency

        gammas_arr = np.array(gamma_energies, dtype=np.float64)
        X = np.array([
            feature_row(E, gammas_arr, max_gammas=self.efficiency_model_max_gammas)
            for E in gamma_energies
        ])
        pred = self.efficiency_model.predict(X)  # columns: [efficiency, sigma_intrinsic]
        for E, (efficiency, sigma_intrinsic) in zip(gamma_energies, pred):
            total += efficiency * bin_width * doppler_lineshape(E, energies, self.beta, sigma_intrinsic)
        return total

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

        **Only calibrated for `E >= ~0.95 MeV`** (was ~2.35 MeV before
        the 2026-09-18 `"calib1g"` single-gamma calibration series):
        below the curve's lowest measured point this clips to that
        boundary value like `full_energy_efficiency` does -- silently
        wrong, not just extrapolated, for a cascade whose own top gamma
        is genuinely below that floor. If `continuum_curve` is `None` (no
        `compton_continuum.npz` loaded), returns zeros.
        """
        if self.continuum_energy is None:
            return np.zeros_like(np.asarray(energies, dtype=np.float64))
        amplitude = np.interp(E, self.continuum_energy, self.continuum_curve)
        shape = klein_nishina_continuum_shape(E, energies)
        return amplitude * shape

    def backscatter_dome(self, E: float, energies: np.ndarray) -> np.ndarray:
        """Predicted backscatter-dome contribution (counts per simulated
        event per bin) at bin centers `energies` (MeV), for one gamma of
        true energy `E` (MeV) that is the cascade's own **highest-energy**
        gamma (same Phase-1 scope as `compton_continuum` -- found and
        fit alongside it, see `fit_compton_continuum.py`).

        A Gaussian bump centered at `gamma_physics.backscatter_energy(E)`
        (physics-anchored, not fit -- photons Compton-backscattered ~180
        degrees in surrounding material before reaching the crystal, then
        fully absorbed there), with amplitude and width from `dome_curve`/
        `dome_sigma_curve` (interpolated vs. `E`, same median-binned-
        smoothing method as the continuum curve). Returns zeros if no
        dome fit was loaded (`dome_energy is None` -- older
        `compton_continuum.npz` files, or if fit ever fails everywhere).
        """
        energies = np.asarray(energies, dtype=np.float64)
        if self.dome_energy is None:
            return np.zeros_like(energies)
        amplitude = np.interp(E, self.dome_energy, self.dome_curve)
        dome_sigma = np.interp(E, self.dome_energy, self.dome_sigma_curve)
        center = backscatter_energy(E)
        return amplitude * np.exp(-0.5 * ((energies - center) / dome_sigma) ** 2)

    def predict_spectrum(self, gamma_energies: list[float], edges: np.ndarray) -> np.ndarray:
        """Sum of photopeak contributions for every gamma in a cascade,
        plus the Compton continuum and backscatter dome of the cascade's
        own **highest-energy** gamma only (Phase 1 -- see module
        docstring; lower gammas' continua/domes are not modeled, so the
        predicted spectrum is photopeak-only below the second-highest
        gamma)."""
        centers = 0.5 * (edges[:-1] + edges[1:])
        total = self._photopeaks_for_cascade(gamma_energies, centers)
        if len(gamma_energies) > 0:
            e_top = max(gamma_energies)
            continuum = self.compton_continuum(e_top, centers)
            total += np.where(centers <= compton_edge_energy(e_top), continuum, 0.0)
            total += self.backscatter_dome(e_top, centers)
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
        print(f"E={E:.4f} MeV: sigma_intrinsic={1000*rf.sigma_intrinsic(E):.1f} keV, "
              f"effective_sigma={1000*rf.effective_sigma(E):.1f} keV, "
              f"efficiency={rf.full_energy_efficiency(E):.5f} counts/event{continuum_note}")
