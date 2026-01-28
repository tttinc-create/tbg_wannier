"""
Bistritzer–MacDonald continuum Hamiltonian for twisted bilayer graphene.

Core objects
------------
- BMModel: wraps the parameters and moiré lattice and exposes H(k).

Implementation notes
--------------------
We build the Hamiltonian in a plane-wave basis labelled by moiré reciprocal
lattice vectors G = m b1 + n b2 (with a finite truncation).

Basis ordering (single valley)
------------------------------
For each G index i:
  |layer=0 (bottom), sublattice A,B>  (2 states)
  |layer=1 (top),    sublattice A,B>  (2 states)

So dim = 4 * siteN.

If two_valleys=True, we return a block diagonal Hamiltonian with valley K and K'.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import scipy.sparse as sp

from .config import BMParameters, SolverParameters
from .lattice import MoireLattice
from .utils import sx, sy, sz, I2


def T_matrices(w0: float, w1: float, phi: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interlayer tunneling matrices T1,T2,T3 in sublattice space.

    Common BM convention:
      T1 = [[w0, w1],[w1, w0]]
      T2 = [[w0, w1 e^{-iφ}], [w1 e^{+iφ}, w0]]
      T3 = [[w0, w1 e^{+iφ}], [w1 e^{-iφ}, w0]]
    """
    T1 = np.array([[w0, w1], [w1, w0]], dtype=complex)
    e = np.exp(-1j * phi)
    T2 = np.array([[w0, w1 * e], [w1 * np.conj(e), w0]], dtype=complex)
    T3 = np.array([[w0, w1 * np.conj(e)], [w1 * e, w0]], dtype=complex)
    return T1, T2, T3


def dirac_cone(kinetic: float, kvec: np.ndarray, valley: int = +1) -> np.ndarray:
    """
    2×2 Dirac Hamiltonian for one graphene layer at one valley.

    Parameters
    ----------
    hvf : ħ v_F (meV·nm)
    kvec : (2,) momentum in lab frame (1/nm)
    theta_layer : rotation angle of the layer (+θ/2 for top, -θ/2 for bottom)
    valley : +1 for K, -1 for K' (time-reversal)

    Returns
    -------
    h : (2,2) complex
    """
    # rotate momentum into the layer frame (convention)
    # kL = rot2(-theta_layer) @ kvec
    kx, ky = float(kvec[0]), float(kvec[1])

    if valley == +1:
        return kinetic * (kx * sx + ky * sy)
    elif valley == -1:
        # Time-reversal: ky flips sign in the effective Dirac equation
        return -kinetic * (kx * sx - ky * sy)
    else:
        raise ValueError("valley must be +1 or -1")


@dataclass
class BMModel:
    params: BMParameters
    lat: MoireLattice
    solver: SolverParameters
    _hop_single_valley: Optional[sp.csr_matrix] = None

    def __post_init__(self):
        # Precompute k-independent interlayer hopping for a single valley.
        self._hop_single_valley = self._build_hop_single_valley()

    @property
    def siteN(self) -> int:
        return self.lat.siteN
    @property
    def dim_single_valley(self) -> int:
        return 4 * self.siteN

    def _build_hop_single_valley(self) -> sp.csr_matrix:
        """
        Build the k-independent interlayer coupling block for one valley.

        Uses the neighbor mapping corresponding to q1,q2,q3 shifts:
          bottom(G) couples to top(G + shift_j) via T_j.
        """
        w0, w1, phi = self.params.w0_meV, self.params.w1_meV, self.params.phi_AB
        T1, T2, T3 = T_matrices(w0, w1, phi)
        Ts = [T1, T2, T3]

        nb = self.lat.neighbor_indices()  # (siteN, 3)
        dim = self.dim_single_valley
        H = sp.lil_matrix((dim, dim), dtype=complex)

        # Indices:
        # bottom layer block: [0 .. 2*siteN)
        # top    layer block: [2*siteN .. 4*siteN)
        for i in range(self.siteN):
            for jtype in range(3):
                j = int(nb[i, jtype])
                T = Ts[jtype]
                # bottom i couples to top j
                bi = 2 * i
                tj = 2 * j + 2 * self.siteN
                H[bi:bi+2, tj:tj+2] += T
                H[tj:tj+2, bi:bi+2] += T.conj().T
        return H.tocsr()

    def _build_dirac_single_valley(self, k: np.ndarray, valley: int = +1) -> sp.csr_matrix:
        """
        Build the k-dependent intra-layer Dirac blocks for one valley.
        """
        hvf = self.params.hvf_meV_Ang * self.params.ktheta
        dim = self.dim_single_valley
        q2 = self.lat.q2
        H = sp.lil_matrix((dim, dim), dtype=complex)

        # layer angles: bottom = -θ/2, top = +θ/2
        # th_b = -theta / 2.0
        # th_t = +theta / 2.0

        for i, G in enumerate(self.lat.G):
            a = self.lat.cart_coords(self.lat.wrap(self.lat.frac_coords(k - G - q2)))
            ht = dirac_cone(hvf, a, valley=valley)
            b = self.lat.cart_coords(self.lat.wrap(self.lat.frac_coords(k + G + q2)))
            hb = dirac_cone(hvf, b, valley=valley)
            ti = 2 * i
            bi = 2 * i + 2 * self.siteN
            H[ti:ti+2, ti:ti+2] = ht
            H[bi:bi+2, bi:bi+2] = hb


        return H.tocsr()

    def H_single_valley(self, k: np.ndarray, valley: int = +1) -> sp.csr_matrix:
        """Hamiltonian for one valley (+1=K, -1=K')."""
        Hd = self._build_dirac_single_valley(k, valley=valley)
        # Interlayer hopping differs by complex conjugation in the K' valley in common conventions
        if valley == +1:
            return Hd + self._hop_single_valley
        else:
            # Conjugate the hopping matrices (keeps Hermiticity)
            return Hd + self._hop_single_valley.conjugate()

    def H(self, k: np.ndarray) -> sp.csr_matrix:
        """
        Full Hamiltonian at cartesian momentum k (2,).

        If params.two_valleys is True, returns block diagonal [K ⊕ K'].
        """
        Hk = self.H_single_valley(k, valley=+1)
        if not self.params.two_valleys:
            return Hk

        Hk2 = self.H_single_valley(k, valley=-1)
        return sp.block_diag((Hk, Hk2), format="csr")
