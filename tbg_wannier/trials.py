from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Tuple

import numpy as np
import scipy.linalg
from .config import WannierizationRecipe
from .lattice import MoireLattice
from .utils import sx, kron

from collections import defaultdict

# ----------------------------------------------------------------------
# Trial orbitals (ported from your notebook, but using MoireLattice inputs)
# ----------------------------------------------------------------------

def wannier_Ea(lat: MoireLattice, k: np.ndarray, l1: float, l2: float, alpha1: float, alpha2: float, theta: float = np.pi/4) -> np.ndarray:
    nlayer = norb = 2
    nwannier = 2
    dim = nlayer * norb * lat.siteN
    W = np.zeros((nwannier, dim), dtype=complex)
    k = np.asarray(k, float).ravel()

    for i in range(lat.siteN):
        Q = lat.Q_plus[i]  # q2 + m*b1 + n*b2
        W[0, 2*i] = alpha1 * np.exp(-0.5*l1**2*np.linalg.norm(k-Q)**2) * np.exp(1j*theta)
        W[1, 2*i] = alpha2 * (-1j*(k[0]-Q[0])-(k[1]-Q[1])) * np.exp(-0.5*l2**2*np.linalg.norm(k-Q)**2) * np.exp(-1j*theta)

        W[0, 2*(i+lat.siteN)] = alpha1 * np.exp(-0.5*l1**2*np.linalg.norm(k+Q)**2) * np.exp(-1j*theta)
        W[1, 2*(i+lat.siteN)] = -alpha2 * (-1j*(k[0]+Q[0])-(k[1]+Q[1])) * np.exp(-0.5*l2**2*np.linalg.norm(k+Q)**2) * np.exp(1j*theta)

        W[:, 2*i+1] = sx @ W[:, 2*i].conj()
        W[:, 2*(i+lat.siteN) + 1] = sx @ W[:, 2*(i+lat.siteN)].conj()

    # normalize each wannier
    for j in range(nwannier):
        W[j, :] /= np.linalg.norm(W[j, :])
    return W.T

def wannier_zhida(lat: MoireLattice, k: np.ndarray, l: float, theta: float = np.pi/4) -> np.ndarray:
    nlayer = norb = 2
    nwannier = 2
    siteN = lat.siteN
    dim = nlayer * norb * siteN
    wannier = np.zeros((nwannier, dim), dtype=complex)
    for i in range(siteN):
        Q = lat.Q_plus[i]
        wannier[0, 2*i] =  np.exp(-0.5*l**2*np.linalg.vector_norm(k-Q)**2) * np.exp(1J*theta)
        wannier[0, 2*(i+siteN)] = np.exp(-0.5*l**2*np.linalg.vector_norm(k+Q)**2) * np.exp(-1J*theta)
        wannier[1, 2*i + 1] = np.exp(-0.5*l**2*np.linalg.vector_norm(k-Q)**2) * np.exp(-1J*theta)
        wannier[1, 2*(i+siteN)+1] = np.exp(-0.5*l**2*np.linalg.vector_norm(k+Q)**2) * np.exp(1J*theta)
    for j in range(nwannier):
        wannier[j,:] = wannier[j,:]/np.linalg.vector_norm(wannier[j,:])
    return wannier.T

def wannier_A1a(lat: MoireLattice, k: np.ndarray, l: float, alpha: float, theta: float = 0.0) -> np.ndarray:
    nlayer = norb = 2
    nwannier = 1
    dim = nlayer * norb * lat.siteN
    W = np.zeros((nwannier, dim), dtype=complex)
    k = np.asarray(k, float).ravel()

    for i in range(lat.siteN):
        Q = lat.Q_plus[i]
        W[0, 2*i] = alpha * (-1j*(k[0]-Q[0])+(k[1]-Q[1])) * np.exp(-0.5*l**2*np.linalg.norm(k-Q)**2) * np.exp(1j*theta)
        W[0, 2*i+1] = W[0, 2*i].conjugate()
        W[0, 2*(i+lat.siteN)] = -alpha * (-1j*(k[0]+Q[0])+(k[1]+Q[1])) * np.exp(-0.5*l**2*np.linalg.norm(k+Q)**2) * np.exp(-1j*theta)
        W[0, 2*(i+lat.siteN)+1] = W[0, 2*(i+lat.siteN)].conjugate()

    W[0, :] /= np.linalg.norm(W[0, :])
    return W.T


def wannier_A2a(lat: MoireLattice, k: np.ndarray, l: float, alpha: float, theta: float = 0.0) -> np.ndarray:
    nlayer = norb = 2
    nwannier = 1
    dim = nlayer * norb * lat.siteN
    W = np.zeros((nwannier, dim), dtype=complex)
    k = np.asarray(k, float).ravel()

    for i in range(lat.siteN):
        Q = lat.Q_plus[i]
        W[0, 2*i] = alpha * (-1j*(k[0]-Q[0])+(k[1]-Q[1])) * np.exp(-0.5*l**2*np.linalg.norm(k-Q)**2) * np.exp(1j*theta)
        W[0, 2*i+1] = W[0, 2*i].conjugate()
        W[0, 2*(i+lat.siteN)] = alpha * (-1j*(k[0]+Q[0])+(k[1]+Q[1])) * np.exp(-0.5*l**2*np.linalg.norm(k+Q)**2) * np.exp(-1j*theta)
        W[0, 2*(i+lat.siteN)+1] = W[0, 2*(i+lat.siteN)].conjugate()

    W[0, :] /= np.linalg.norm(W[0, :])
    return W.T


def wannier_A2c(lat: MoireLattice, k: np.ndarray, l1: float, l2: float, alpha1: float, alpha2: float) -> np.ndarray:
    nlayer = norb = 2
    nwannier = 2
    dim = nlayer * norb * lat.siteN
    W = np.zeros((nwannier, dim), dtype=complex)
    k = np.asarray(k, float).ravel()

    r1 = 2*np.pi/3*np.array([-1/np.sqrt(3), 1.0])
    r2 = 2*np.pi/3*np.array([ 1/np.sqrt(3), 1.0])

    for i in range(lat.siteN):
        Q = lat.Q_plus[i]
        W[0, 2*i] = alpha1*np.exp(-0.5*l1**2*np.linalg.norm(k-Q)**2)
        W[1, 2*i] = alpha2*(-1j*(k[0]-Q[0])-(k[1]-Q[1]))*np.exp(-0.5*l2**2*np.linalg.norm(k-Q)**2)

        W[0, 2*(i+lat.siteN)] = -alpha1*np.exp(-0.5*l1**2*np.linalg.norm(k+Q)**2)
        W[1, 2*(i+lat.siteN)] = alpha2*(-1j*(k[0]+Q[0])-(k[1]+Q[1]))*np.exp(-0.5*l2**2*np.linalg.norm(k+Q)**2)

        W[:, 2*i + 1] = np.exp(-2j*np.pi/3)* sx @ W[:, 2*i].conjugate()
        W[:, 2*(i+lat.siteN) + 1 ] = np.exp(2j*np.pi/3)* sx @ W[:, 2*(i+lat.siteN)].conjugate()

        W[0, 2*i:2*i+2] *= np.exp(-1j*np.dot(r1, k-Q))
        W[1, 2*i:2*i+2] *= np.exp(-1j*np.dot(r2, k-Q))
        W[0, 2*(i+lat.siteN):2*(i+lat.siteN)+2] *= np.exp(-1j*np.dot(r1, k+Q))
        W[1, 2*(i+lat.siteN):2*(i+lat.siteN)+2] *= np.exp(-1j*np.dot(r2, k+Q))

    for j in range(nwannier):
        W[j, :] /= np.linalg.norm(W[j, :])
    return W.T


def wannier_Ec(lat: MoireLattice, k: np.ndarray, l1: float, l2: float, l3: float, l4: float,
              alpha1: float, alpha2: float, alpha3: float, alpha4: float) -> np.ndarray:
    nlayer = norb = 2
    nwannier = 4
    dim = nlayer * norb * lat.siteN
    W = np.zeros((nwannier, dim), dtype=complex)
    k = np.asarray(k, float).ravel()

    r1 = 2*np.pi/3*np.array([-1/np.sqrt(3), 1.0])
    r2 = 2*np.pi/3*np.array([ 1/np.sqrt(3), 1.0])
    mat = kron(sx, sx)

    for i in range(lat.siteN):
        Q = lat.Q_plus[i]
        W[0, 2*i] = alpha1*(-1j*(k[0]-Q[0])-(k[1]-Q[1]))*np.exp(-0.5*l1**2*np.linalg.norm(k-Q)**2)
        W[1, 2*i] = alpha2*(-1j*(k[0]-Q[0])+(k[1]-Q[1]))*np.exp(-0.5*l2**2*np.linalg.norm(k-Q)**2)
        W[2, 2*i] = alpha3*(-1j*(k[0]-Q[0])+(k[1]-Q[1]))*np.exp(-0.5*l3**2*np.linalg.norm(k-Q)**2)
        W[3, 2*i] = alpha4*np.exp(-0.5*l4**2*np.linalg.norm(k-Q)**2)

        W[0, 2*(i+lat.siteN)] = -alpha3*(-1j*(k[0]+Q[0])+(k[1]+Q[1]))*np.exp(-0.5*l3**2*np.linalg.norm(k+Q)**2)
        W[1, 2*(i+lat.siteN)] = alpha4*np.exp(-0.5*l4**2*np.linalg.norm(k+Q)**2)
        W[2, 2*(i+lat.siteN)] = -alpha1*(-1j*(k[0]+Q[0])-(k[1]+Q[1]))*np.exp(-0.5*l1**2*np.linalg.norm(k+Q)**2)
        W[3, 2*(i+lat.siteN)] = -alpha2*(-1j*(k[0]+Q[0])+(k[1]+Q[1]))*np.exp(-0.5*l2**2*np.linalg.norm(k+Q)**2)

        W[:, 2*i + 1] = np.exp(-2j*np.pi/3)* mat @ W[:, 2*i].conjugate()
        W[:, 2*(i+lat.siteN) + 1 ] = np.exp(2j*np.pi/3)* mat @ W[:, 2*(i+lat.siteN)].conjugate()

        W[0:2, 2*i:2*i+2] *= np.exp(-1j*np.dot(r1, k-Q))
        W[2:4, 2*i:2*i+2] *= np.exp(-1j*np.dot(r2, k-Q))
        W[0:2, 2*(i+lat.siteN):2*(i+lat.siteN)+2] *= np.exp(-1j*np.dot(r1, k+Q))
        W[2:4, 2*(i+lat.siteN):2*(i+lat.siteN)+2] *= np.exp(-1j*np.dot(r2, k+Q))

    for j in range(nwannier):
        W[j, :] /= np.linalg.norm(W[j, :])
    return W.T


def wannier_Af(lat: MoireLattice, k: np.ndarray, l: float, alpha: float) -> np.ndarray:
    nlayer = norb = 2
    nwannier = 3
    dim = nlayer * norb * lat.siteN
    W = np.zeros((nwannier, dim), dtype=complex)
    k = np.asarray(k, float).ravel()

    r1 = np.pi/3*np.array([ np.sqrt(3), 1.0])
    r2 = np.pi/3*np.array([-np.sqrt(3), 1.0])
    r3 = np.pi/3*np.array([0.0, 2.0])

    for i in range(lat.siteN):
        Q = lat.Q_plus[i]
        W[2, 2*i] = W[1, 2*i] = alpha * np.exp(-0.5*l**2*np.linalg.norm(k-Q)**2)
        W[0, 2*i] = np.exp(-2j*np.pi/3) * W[1, 2*i]

        W[2, 2*(i+lat.siteN)] = W[0, 2*(i+lat.siteN)] = (alpha * np.exp(-0.5*l**2*np.linalg.norm(k+Q)**2)).conj()
        W[1, 2*(i+lat.siteN)] = np.exp(2j*np.pi/3) * W[2, 2*(i+lat.siteN)]

        W[0:2, 2*i+1] = np.exp(2j*np.pi/3) * W[0:2, 2*i].conjugate()
        W[2, 2*i+1] = np.exp(-2j*np.pi/3) * W[2, 2*i].conjugate()
        W[0:2, 2*(i+lat.siteN)+1] = np.exp(-2j*np.pi/3) * W[0:2, 2*(i+lat.siteN)].conjugate()
        W[2, 2*(i+lat.siteN)+1] = np.exp(2j*np.pi/3) * W[2, 2*(i+lat.siteN)].conjugate()

        W[0, 2*i:2*i+2] *= np.exp(-1j*np.dot(r1, k-Q))
        W[1, 2*i:2*i+2] *= np.exp(-1j*np.dot(r2, k-Q))
        W[2, 2*i:2*i+2] *= np.exp(-1j*np.dot(r3, k-Q))

        W[0, 2*(i+lat.siteN):2*(i+lat.siteN)+2] *= np.exp(-1j*np.dot(r1, k+Q))
        W[1, 2*(i+lat.siteN):2*(i+lat.siteN)+2] *= np.exp(-1j*np.dot(r2, k+Q))
        W[2, 2*(i+lat.siteN):2*(i+lat.siteN)+2] *= np.exp(-1j*np.dot(r3, k+Q))

    for j in range(nwannier):
        W[j, :] /= np.linalg.norm(W[j, :])
    return W.T


def wannier_Bf(lat: MoireLattice, k: np.ndarray, l: float, alpha: float) -> np.ndarray:
    nlayer = norb = 2
    nwannier = 3
    dim = nlayer * norb * lat.siteN
    W = np.zeros((nwannier, dim), dtype=complex)
    k = np.asarray(k, float).ravel()

    r1 = np.pi/3*np.array([ np.sqrt(3), 1.0])
    r2 = np.pi/3*np.array([-np.sqrt(3), 1.0])
    r3 = np.pi/3*np.array([0.0, 2.0])

    for i in range(lat.siteN):
        Q = lat.Q_plus[i]
        W[2, 2*i] = W[1, 2*i] = alpha * np.exp(-0.5*l**2*np.linalg.norm(k-Q)**2)
        W[0, 2*i] = np.exp(-2j*np.pi/3) * W[1, 2*i]

        W[2, 2*(i+lat.siteN)] = W[0, 2*(i+lat.siteN)] = -(alpha * np.exp(-0.5*l**2*np.linalg.norm(k+Q)**2)).conj()
        W[1, 2*(i+lat.siteN)] = np.exp(2j*np.pi/3) * W[2, 2*(i+lat.siteN)]

        W[0:2, 2*i+1] = np.exp(2j*np.pi/3) * W[0:2, 2*i].conjugate()
        W[2, 2*i+1] = np.exp(-2j*np.pi/3) * W[2, 2*i].conjugate()
        W[0:2, 2*(i+lat.siteN)+1] = np.exp(-2j*np.pi/3) * W[0:2, 2*(i+lat.siteN)].conjugate()
        W[2, 2*(i+lat.siteN)+1] = np.exp(2j*np.pi/3) * W[2, 2*(i+lat.siteN)].conjugate()

        W[0, 2*i:2*i+2] *= np.exp(-1j*np.dot(r1, k-Q))
        W[1, 2*i:2*i+2] *= np.exp(-1j*np.dot(r2, k-Q))
        W[2, 2*i:2*i+2] *= np.exp(-1j*np.dot(r3, k-Q))

        W[0, 2*(i+lat.siteN):2*(i+lat.siteN)+2] *= np.exp(-1j*np.dot(r1, k+Q))
        W[1, 2*(i+lat.siteN):2*(i+lat.siteN)+2] *= np.exp(-1j*np.dot(r2, k+Q))
        W[2, 2*(i+lat.siteN):2*(i+lat.siteN)+2] *= np.exp(-1j*np.dot(r3, k+Q))

    for j in range(nwannier):
        W[j, :] /= np.linalg.norm(W[j, :])
    return W.T


# ----------------------------------------------------------------------
# Higher-level trial builders (your notebook's 'wannier_trial' + Lowdin projection)
# ----------------------------------------------------------------------


def _expand_ebr_sequence_with_l(
    ebr_sequence: tuple[str, ...],
    l0: float,
    *,
    scale: float = 1.3,
) -> list[tuple[str, float]]:
    """
    Expand an EBR sequence into (ebr, l_eff), automatically
    splitting l if an EBR appears multiple times.
    """
    counts = defaultdict(int)
    out = []
    for ebr in ebr_sequence:
        idx = counts[ebr]
        counts[ebr] += 1
        l_eff = l0 * (scale ** idx)
        out.append((ebr, l_eff))
    return out


class TrialBuilder:
    """
    Efficient builder for Wannier trial orbitals.

    - Caches Q-lattice dependent structures
    - Evaluates each EBR only once per k
    - Automatically splits l for repeated EBRs
    """

    def __init__(
        self,
        lat: MoireLattice,
        recipe: WannierizationRecipe,
        *,
        l_split_scale: float = 1.3,
    ):
        self.lat = lat
        self.recipe = recipe
        self.alpha = recipe.alpha

        # Expand EBR list with distinct l's
        self.ebr_l_list = _expand_ebr_sequence_with_l(
            recipe.ebr_sequence,
            recipe.l,
            scale=l_split_scale,
        )

        # Cache callable generators
        self._generators = {
            "A1a": lambda k, l: wannier_A1a(lat, k, l, self.alpha),
            "A2a": lambda k, l: wannier_A2a(lat, k, l, self.alpha),
            "Ea":  lambda k, l: wannier_Ea(lat, k, l, l, self.alpha, self.alpha),
            "Af":  lambda k, l: wannier_Af(lat, k, l, self.alpha),
            "Bf":  lambda k, l: wannier_Bf(lat, k, l, self.alpha),
            "Ec":  lambda k, l: wannier_Ec(lat, k, l, l, l, l,
                                            self.alpha, self.alpha,
                                            self.alpha, self.alpha),
            "zhida": lambda k, l: wannier_zhida(lat, k, l),
        }

    def build_one_k(self, k: np.ndarray) -> np.ndarray:
        """
        Build the full trial matrix at a single k.

        Returns shape: (dim, nwann_total)
        """
        blocks = []
        for ebr, l_eff in self.ebr_l_list:
            try:
                W = self._generators[ebr](k, l_eff)
            except KeyError:
                raise ValueError(f"Unsupported EBR '{ebr}' in recipe.")
            blocks.append(W)
        return np.concatenate(blocks, axis=1)

    def build_all(self, k_mesh: np.ndarray) -> np.ndarray:
        """
        Build trials for all k-points.

        Returns:
            trials: (Nk, dim, nwann_total)
        """
        return np.array([self.build_one_k(k) for k in k_mesh])

# def _build_trials_one_k(lat: MoireLattice, k: np.ndarray, recipe: WannierizationRecipe) -> np.ndarray:
#     """Build a trial set using the same concatenations as your notebook, at a single k."""
#     l = recipe.l
#     alpha = recipe.alpha
#     ebr_lst = recipe.ebr_sequence
#     wannier_trials = {
#         "A1a": wannier_A1a(lat, k, l, alpha),
#         "A2a": wannier_A2a(lat, k, l, alpha),
#         "Ea": wannier_Ea(lat, k, l, l, alpha, alpha),
#         "Ea2": wannier_Ea(lat, k, 2*l, l, alpha, alpha),
#         "Af": wannier_Af(lat, k, l, alpha),
#         "Bf": wannier_Bf(lat, k, l, alpha),
#         "Ec": wannier_Ec(lat, k, l, l, l, l, alpha, alpha, alpha, alpha)
#     }
#     return np.concatenate([wannier_trials[ebr] for ebr in ebr_lst], axis=1)

# def build_trials_kpoints(lat: MoireLattice, recipe: WannierizationRecipe) -> np.ndarray:
#     return np.array([_build_trials_one_k(lat=lat, k=k, recipe=recipe) for k in lat.k_cart])

# def lowdin_project(eigvecs: np.ndarray, trials: np.ndarray) -> np.ndarray:
#     """Löwdin orthonormalization of the projected trial subspace.

#     Parameters
#     ----------
#     eigvecs : (Nk, dim, nbands)
#     trials  : (Nk, dim, nwann)

#     Returns
#     -------
#     W : (Nk, dim, nwann) Löwdin-orthonormalized projected trial vectors.
#     """
#     Nk = eigvecs.shape[0]
#     out = []
#     for ik in range(Nk):
#         V = eigvecs[ik]
#         Wt = trials[ik]
#         A = V.conj().T @ Wt
#         S = A.conj().T @ A
#         S_mhalf = scipy.linalg.fractional_matrix_power(S, -0.5)
#         out.append(V @ A @ S_mhalf)
#     return np.asarray(out)


# def re_wannierization(w_input: np.ndarray, eigvecs: np.ndarray, kmesh: np.ndarray, sigma: float = 0.25) -> np.ndarray:
#     """Your notebook's re_wannierization (ported).

#     This is a post-processing gauge-fixing step that SVD-aligns the first 2 columns and the remaining block,
#     with a k-dependent decay factor.
#     """
#     wannier = []
#     for k_idx, k_val in enumerate(kmesh):
#         V = eigvecs[k_idx]
#         decay_factor = np.exp(-np.linalg.norm(k_val)**2/(sigma)**2)

#         weight = np.ones(6)
#         weight[2:6] = decay_factor
#         Wmat = np.diag(weight)

#         psi_now = w_input[k_idx].conj().T @ V[:, 0:6] @ Wmat
#         Vrot, _, _ = np.linalg.svd(psi_now[0:2, :].conj().T)
#         psi_now = psi_now @ Vrot[:, 0:2]

#         psi_new, _, _ = np.linalg.svd(psi_now)

#         U0, _, V0 = np.linalg.svd(psi_new[0:2, 0:2].conj().T)
#         U1, _, V1 = np.linalg.svd(psi_new[2:psi_new.shape[0], 2:psi_new.shape[0]].conj().T)
#         psi_new = psi_new @ scipy.linalg.block_diag(U0 @ V0, U1 @ V1)

#         wannier.append(w_input[k_idx] @ psi_new)
#     return np.asarray(wannier)

