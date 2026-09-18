"""Detector-agnostic gamma-ray interaction physics.

Pure functions only -- no dependency on Ancalagon, BGO, ROOT, or any file
format used elsewhere in this repo. This module exists so a *different*
future project (padsley has mentioned HPGe spectra) can reuse this exact
physics without dragging in anything BGO/Ancalagon-specific; only
`response_function.py` and `fit_compton_continuum.py` know how to
*calibrate* these shapes against a particular detector's simulation
output.

Everything here concerns a single mono-energetic gamma of true energy
`E` (MeV) interacting once in a detector medium via Compton scattering,
depositing recoil-electron energy `T` (MeV) at the interaction point.
"""
from __future__ import annotations

import numpy as np
from scipy.special import erf

ELECTRON_MASS_MEV = 0.510999  # m_e*c^2 -- same constant/value used by the
                               # sibling G3 project's emulator.py for its
                               # own escape-peak/Compton-edge formulas.


def compton_edge_energy(E: np.ndarray) -> np.ndarray:
    """Maximum electron recoil energy (MeV, the "Compton edge") for a
    photon of true energy `E` (MeV), from backscatter (theta=180 deg):

        T_CE(E) = 2*E^2 / (m_e*c^2 + 2*E)

    Derivation: at theta=180, cos(theta)=-1, so the scattered-photon
    energy E' = E / (1 + 2*E/(m_e*c^2)) = E*m_e*c^2/(m_e*c^2+2E), and
    T_CE = E - E'.
    """
    E = np.asarray(E, dtype=np.float64)
    return 2.0 * E**2 / (ELECTRON_MASS_MEV + 2.0 * E)


def klein_nishina_continuum_shape(E: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Unnormalized Klein-Nishina Compton-continuum shape: relative
    probability density of an incident photon of true energy `E` (MeV)
    depositing recoil-electron energy `T` (MeV) at a single Compton
    scatter, for `0 <= T <= compton_edge_energy(E)`. Zero (not NaN)
    outside that range.

    Derivation (self-contained; only the T-dependence at fixed E matters
    here since callers always fit a separate overall normalization
    against real data per E -- energy-dependent prefactors independent
    of T are dropped):

    Klein-Nishina, per unit solid angle: dsigma/dOmega = (r_e^2/2) *
    (E'/E)^2 * (E'/E + E/E' - sin^2(theta)), where E' = E - T is the
    scattered-photon energy. Integrating the azimuthal angle and
    changing variables from cos(theta) to T (via
    cos(theta) = 1 - m_e*c^2*T/(E*E'), so
    |d(cos theta)/dT| = m_e*c^2/E'^2) gives, after cancelling the
    (E'/E)^2 * 1/E'^2 factors against the Jacobian's E'^2:

        dsigma/dT  proportional-to  E'/E + E/E' - sin^2(theta)

    with sin^2(theta) = 1 - (1 - m_e*c^2*T/(E*E'))^2. This rises from 2
    at T=0 (theta=0, no deflection) toward E'/E + E/E' >= 2 at the
    Compton edge (theta=180) -- the characteristic upward "shoulder"
    approaching the edge seen in real Compton continua, before whatever
    detector resolution rounds it off further.
    """
    E = np.asarray(E, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    t_ce = compton_edge_energy(E)
    valid = (T >= 0) & (T <= t_ce)

    Ep = E - T  # scattered-photon energy E'
    with np.errstate(divide="ignore", invalid="ignore"):
        x = ELECTRON_MASS_MEV * T / (E * Ep)
        sin2_theta = 2.0 * x - x**2
        shape = Ep / E + E / Ep - sin2_theta

    return np.where(valid, shape, 0.0)


def doppler_lineshape(E: np.ndarray, x: np.ndarray, beta: float, sigma_intrinsic: float) -> np.ndarray:
    """Normalized (unit-area over `x`) photopeak lineshape for a gamma of
    true energy `E` (MeV) emitted in flight by a recoil moving at
    `beta` = v/c, observed by a detector array with near-full angular
    coverage.

    **Why not a Gaussian** (found 2026-09-18 comparing both against real
    isolated, high-statistics peaks -- chi2/ndf 551 -> 218, a real,
    checked improvement, not assumed): if the recoil emits this gamma
    isotropically in its own rest frame and the array's solid-angle
    coverage is close to uniform over the full 4*pi, the Doppler-shifted
    true energy `E*(1 + beta*cos(theta))` has a *uniform* (not peaked)
    density between `E*(1-beta)` and `E*(1+beta)` -- because an isotropic
    distribution has a uniform density in `cos(theta)`, and the shift is
    linear in `cos(theta)`. This is the standard lineshape for
    recoil-in-flight gamma spectroscopy with a large-solid-angle array:
    a "box" of true-energy density, then rounded off by whatever
    additional (non-kinematic: light-collection, electronic) resolution
    the detector itself contributes, `sigma_intrinsic`. Convolving a
    uniform box with a Gaussian has the closed form used below (a
    difference of two error functions) -- a smoothly-rounded flat-top,
    not a bell curve. Real BGO angular acceptance is not a perfectly
    sharp-edged box (chi2/ndf=218, not 1, confirms some residual
    mismatch), but this is a large, physically-motivated improvement,
    not a full geometric model of the array (that would need Ancalagon's
    actual crystal positions -- out of scope here).
    """
    E = np.asarray(E, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    sigma_intrinsic = np.abs(sigma_intrinsic)
    x1, x2 = E * (1.0 - beta), E * (1.0 + beta)
    return (erf((x - x1) / (np.sqrt(2.0) * sigma_intrinsic)) - erf((x - x2) / (np.sqrt(2.0) * sigma_intrinsic))) / (2.0 * (x2 - x1))


def effective_sigma(E: np.ndarray, beta: float, sigma_intrinsic: float) -> np.ndarray:
    """Variance-equivalent width of `doppler_lineshape` -- *not* a claim
    that the lineshape is Gaussian (it isn't), just a single-number
    stand-in for "how wide is this peak" for callers that only need to
    size a fit window or a peak-exclusion zone, not the lineshape itself
    (e.g. `fit_compton_continuum.py`'s window logic). Box variance
    `(beta*E)^2/3` (a uniform distribution of half-width `beta*E`) plus
    the intrinsic Gaussian's own variance, combined in quadrature.
    """
    E = np.asarray(E, dtype=np.float64)
    return np.sqrt((beta * E) ** 2 / 3.0 + sigma_intrinsic**2)


def backscatter_energy(E: np.ndarray) -> np.ndarray:
    """Energy (MeV) of a photon Compton-backscattered (theta=180 deg) off
    material *before* reaching the sensitive crystal, then fully
    absorbed there -- the "backscatter peak/dome" seen in real gamma
    spectra. This is simply the scattered-photon energy at the Compton
    edge: `E - compton_edge_energy(E)` (`compton_edge_energy` is the
    *transferred* energy at theta=180; what's left in the photon is the
    rest). Asymptotes to `m_e*c^2/2 ~= 0.2555 MeV` for `E >> m_e*c^2/2`,
    the textbook high-energy backscatter-peak position; at the true
    energies this dataset actually probes (0.6-2.5 MeV, where the dome
    was found 2026-09-18) it instead runs ~0.18-0.23 MeV, checked by
    hand against where the dome was visually located before trusting
    this formula for it.
    """
    E = np.asarray(E, dtype=np.float64)
    return E - compton_edge_energy(E)


def escape_peak_energies(E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single- and double-escape-peak energies (MeV) for annihilation
    photons from pair production (`E > 2*m_e*c^2`). Not used until a
    later phase of the Compton-continuum work (see the plan) -- included
    here alongside the edge formula since it's the same kind of trivial,
    detector-agnostic kinematics. Callers must check `E > m_e*c^2` /
    `E > 2*m_e*c^2` themselves before using the single/double value
    respectively; this function does not mask invalid (negative) results.
    """
    E = np.asarray(E, dtype=np.float64)
    return E - ELECTRON_MASS_MEV, E - 2.0 * ELECTRON_MASS_MEV
