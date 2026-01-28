"""
Post-processing helpers:
- overlaps between trial/Wannier gauge and Bloch bands
- reconstructing real-space Wannier functions (plane-wave sum)
- charge density
"""
from __future__ import annotations

from typing import Tuple, Optional

import numpy as np

from .lattice import MoireLattice


def wannier_bloch_overlap(A: np.ndarray) -> np.ndarray:
    """
    Compute |<ψ_{m,k}|g_{n,k}>|^2 from A[ik,m,n].

    Returns
    -------
    ov : (ik, m) if you sum over n, else (ik,m,n)
    """
    A = np.asarray(A, complex)
    return np.abs(A) ** 2


def compute_real_space_wannier(
    lat: MoireLattice,
    k_list: np.ndarray,
    u_wann: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
    wann_index: int = 0,
    layer: Optional[int] = None,
) -> np.ndarray:
    """
    Reconstruct a real-space Wannier function on a grid spanned by (r1,r2).

    Parameters
    ----------
    k_list : (Nk,2) cartesian k points
    u_wann : (Nk, dim, nwann) Wannier-gauge Bloch states
    r1, r2 : 1D arrays of coordinates along real-space basis vectors (in nm)
    wann_index : which Wannier orbital to reconstruct
    layer : if 0 or 1, only include that layer; if None, include both.

    Returns
    -------
    psi : (len(r1), len(r2)) complex (summed over sublattices; you can adapt)
    """
    Nk, dim, nwann = u_wann.shape
    siteN = lat.siteN
    if wann_index < 0 or wann_index >= nwann:
        raise IndexError("wann_index out of range.")

    # Build real-space grid (cartesian)
    rr = np.zeros((len(r1), len(r2), 2), dtype=float)
    for i, x in enumerate(r1):
        for j, y in enumerate(r2):
            rr[i, j] = x * lat.a1 + y * lat.a2

    psi = np.zeros((len(r1), len(r2)), dtype=complex)

    # Plane-wave sum:
    # ψ_n(r) = (1/Nk) Σ_k e^{i k·r} Σ_G u_{k,G}^{(n)} e^{i G·r}
    # Here we sum sublattices and optionally restrict layers.
    for ik in range(Nk):
        k = k_list[ik]
        uk = u_wann[ik, :, wann_index]

        # select indices
        if layer is None:
            idxs = np.arange(4 * siteN)
        elif layer == 0:
            idxs = np.arange(0, 2 * siteN)
        elif layer == 1:
            idxs = np.arange(2 * siteN, 4 * siteN)
        else:
            raise ValueError("layer must be 0,1, or None.")

        # Sum sublattice components into one coefficient per G for the selected layer(s).
        # (A,B) entries are interleaved in our convention.
        coeff = np.zeros(siteN, dtype=complex)
        for gi in range(siteN):
            # bottom contribution
            if layer is None or layer == 0:
                coeff[gi] += uk[2*gi] + uk[2*gi+1]
            # top contribution
            if layer is None or layer == 1:
                base = 2*siteN
                coeff[gi] += uk[base + 2*gi] + uk[base + 2*gi + 1]

        # Evaluate exp(i(k+G)·r) on grid
        phase_k = np.exp(1j * (rr @ k))
        # G phases: sum over G
        phase_G = np.zeros_like(psi)
        for gi in range(siteN):
            phase_G += coeff[gi] * np.exp(1j * (rr @ lat.G[gi]))
        psi += phase_k * phase_G

    psi /= Nk
    return psi


def compute_charge_density(psi: np.ndarray) -> np.ndarray:
    """Return |psi|^2."""
    return np.abs(psi) ** 2
