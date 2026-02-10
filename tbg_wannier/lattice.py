"""
Moiré reciprocal lattice, Q-lattice truncation, and k meshes.

The BM continuum model is implemented in a plane-wave basis indexed by moiré
reciprocal lattice vectors G = m b1 + n b2.

This module constructs:
- moiré reciprocal basis (b1,b2)
- the three coupling wavevectors q1,q2,q3
- the truncated integer lattice (m,n) in a finite window
- mapping from (m,n) <-> linear index
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from typing import Dict, Tuple

import numpy as np

from .config import BMParameters
from .utils import rot2, reciprocal_to_real, generate_k_frac


@dataclass(frozen=True)
class MoireLattice:
    N_L: int
    N_k: int
    b1: np.ndarray        # (2,)
    b2: np.ndarray        # (2,)
    a1: np.ndarray        # (2,)
    a2: np.ndarray        # (2,)
    q1: np.ndarray
    q2: np.ndarray
    q3: np.ndarray
    C2x: np.ndarray
    C3z: np.ndarray
    Lmn: np.ndarray       # (siteN,2) integer pairs (m,n)
    G: np.ndarray         # (siteN,2) G-vectors
    k_frac: np.ndarray
    index_of: Dict[Tuple[int, int], int]

    @property
    def siteN(self) -> int:
        return self.Lmn.shape[0]

    @staticmethod
    def build(N_L: int, N_k: int) -> "MoireLattice":
        """
        Construct moiré lattice objects for the given BMParameters.

        Conventions (standard BM)
        -------------------------
        q1,q2,q3 are the three wavevectors connecting the rotated Dirac points.
        We use a common choice:
            q1 = (0, -1)
            q2 = (√3/2,  1/2)
            q3 = (-√3/2, 1/2)

        moiré reciprocal basis:
            b1 = q2 - q1
            b2 = q3 - q1
        """

        q1 = np.array([0.0, -1.0])
        q2 = np.array([np.sqrt(3) / 2.0, 1.0 / 2.0])
        q3 = np.array([-np.sqrt(3) / 2.0, 1.0 / 2.0])

        b1 = q2 - q1
        b2 = q3 - q1

        a1, a2 = reciprocal_to_real(b1, b2)
        C2x = np.array([[1, 0], [0,-1]])
        C3z = np.array([[-.5,-np.sqrt(3)/2],[np.sqrt(3)/2,-.5]])
        # Integer lattice pairs in centered window [-N_L/2, ..., N_L/2-1]
        half = N_L // 2
        m = np.arange(N_L) - half
        n = np.arange(N_L) - half
        mm, nn = np.meshgrid(m, n, indexing="ij")
        Lmn = np.stack([mm.ravel(), nn.ravel()], axis=1).astype(int)
        k_frac = generate_k_frac(Nk = N_k)
        # Build mapping
        index_of: Dict[Tuple[int, int], int] = {}
        for idx, (mi, ni) in enumerate(Lmn):
            index_of[(int(mi), int(ni))] = idx

        G = (Lmn[:, 0:1] * b1[None, :] + Lmn[:, 1:2] * b2[None, :]).reshape(-1, 2)

        return MoireLattice(
            N_L=N_L,
            N_k=N_k,
            b1=b1,
            b2=b2,
            a1=a1,
            a2=a2,
            q1=q1,
            q2=q2,
            q3=q3,
            C2x=C2x,
            C3z=C3z,
            Lmn=Lmn,
            G=G,
            k_frac=k_frac,
            index_of=index_of,
        )
    @cached_property
    def Q_plus(self) -> np.ndarray:
        """Q_plus[i] = q2 + m*b1m + n*b2m for each integer (m,n) in `L`."""
        return self.q2[None, :] + self.Lmn[:, 0:1] * self.b1[None, :] + self.Lmn[:, 1:2] * self.b2[None, :]
    @cached_property
    def k_cart(self) -> np.ndarray:
        return np.array([self.cart_coords(k) for k in self.k_frac])

    def frac_coords(self, k_cart: np.ndarray) -> np.ndarray:
        """
        Convert k from cartesian to fractional coordinates (k = k1 b1 + k2 b2).
        """
        k1_frac = np.dot(k_cart, self.a1)/(2*np.pi)
        k2_frac = np.dot(k_cart, self.a2)/(2*np.pi)
        return np.array([k1_frac, k2_frac])

    def cart_coords(self, k_frac: np.ndarray) -> np.ndarray:
        """Convert fractional coordinates to cartesian."""
        return k_frac[0] * self.b1 + k_frac[1] * self.b2
    
    def cart_real_coords(self, r_frac: np.ndarray) -> np.ndarray:
        """Convert real-space fractional coordinates to cartesian."""
        return r_frac[0] * self.a1 + r_frac[1] * self.a2

    def wrap_mn(self, m: int, n: int) -> Tuple[int, int]:
        """
        Wrap (m,n) back into the centered finite window using modulo N_L.

        This mimics the "torus" embedding used in your notebook to keep indices
        inside a finite N_L×N_L lattice.
        """
        N = self.N_L
        half = N // 2
        m_wrapped = ((m + half) % N) - half
        n_wrapped = ((n + half) % N) - half
        return int(m_wrapped), int(n_wrapped)

    def wrap(self, k_frac: np.ndarray) -> np.ndarray:
        N = self.N_L
        half = N // 2
        out = ((k_frac + half) % N) - half
        return out
    
    def index_mn(self, m: int, n: int) -> int:
        m2, n2 = self.wrap_mn(m, n)
        return self.index_of[(m2, n2)]
    
    def embedding_matrix(self, G: Tuple[int, int]) -> np.ndarray:
        """Embedding matrix `R(G)` used in overlaps (your `mat_embedding`).

        Parameters
        ----------
        G : (g1,g2)
            Integer shift in the *fractional reciprocal basis* (the same integers that appear in nnkp).
        norb : int
            Orbitals per (m,n) per layer (2 for your notebook's single-valley BM basis).

        Returns
        -------
        R : ndarray (dim, dim)
            Acts on the orbital-space basis ordering used across the package:
            first layer block (siteN*norb) then second layer block (siteN*norb).
        """
        norb = 2
        g = np.asarray(G, dtype=int).ravel()
        g1, g2 = int(g[0]), int(g[1])

        dim = 2 * norb * self.siteN
        R = np.zeros((dim, dim), dtype=int)
        block = np.eye(norb, dtype=int)

        for i, n in enumerate(self.Lmn):
            # upper layer: shift by -G
            j = self.index_mn(*(n - np.array([g1, g2], dtype=int)))
            R[norb * i : norb * (i + 1), norb * j : norb * (j + 1)] = block

            # lower layer: shift by +G
            k = self.index_mn(*(n + np.array([g1, g2], dtype=int)))
            off_i = norb * (i + self.siteN)
            off_k = norb * (k + self.siteN)
            R[off_i : off_i + norb, off_k : off_k + norb] = block

        return R
    
    def neighbor_indices(self) -> np.ndarray:
        """
        Return neighbor table of shape (siteN, 3) giving indices for couplings between upper and bottom layers
        corresponding to (q1,q2,q3) shifts. The shift vector points from bottom layer to upper layer

        With our conventions:
          q1 shift : (m,n) -> (-m-1, -n+1) 
          q2 shift : (m,n) -> (-m-2, -n+1)
          q3 shift : (m,n) -> (-m-1, -n)
        """
        nb = np.zeros((self.siteN, 3), dtype=int)
        for i, (m, n) in enumerate(self.Lmn):
            nb[i, 0] = self.index_mn(-m - 1 , -n + 1)
            nb[i, 1] = self.index_mn(-m - 2, -n + 1)
            nb[i, 2] = self.index_mn(-m - 1, -n)
        return nb

    def high_symmetry_points(self) -> dict:
        """
        Return some standard high-symmetry points (Γ, K, M) in cartesian coords.
        """
        Gm = np.array([0.0, 0.0])
        K = np.array([np.sqrt(3)/2,1/2])
        M = np.array([np.sqrt(3)/4,3/4])
        return {"G": Gm, "K": K, "M": M}
