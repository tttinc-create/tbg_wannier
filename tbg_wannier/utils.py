"""
Small linear-algebra and geometry utilities used across the package.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Iterable


def paulis(dtype=complex) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (I, σx, σy, σz)."""
    I = np.eye(2, dtype=dtype)
    sx = np.array([[0, 1], [1, 0]], dtype=dtype)
    sy = np.array([[0, -1j], [1j, 0]], dtype=dtype)
    sz = np.array([[1, 0], [0, -1]], dtype=dtype)
    return I, sx, sy, sz


I2, sx, sy, sz = paulis()


def kron(*ops: np.ndarray) -> np.ndarray:
    """Kronecker product of many operators."""
    out = np.array([1], dtype=complex)
    for A in ops:
        out = np.kron(out, A)
    return out


def rot2(theta: float) -> np.ndarray:
    """2D rotation matrix."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def reciprocal_to_real(b1: np.ndarray, b2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given 2D reciprocal basis (b1,b2), return real-space basis (a1,a2)
    such that a_i · b_j = 2π δ_ij.
    """
    B = np.stack([b1, b2], axis=1)  # 2x2
    A = 2 * np.pi * np.linalg.inv(B).T
    return A[:, 0], A[:, 1]


def block_diag(*blocks: np.ndarray) -> np.ndarray:
    """Dense block diagonal."""
    if len(blocks) == 0:
        return np.zeros((0, 0), dtype=complex)
    sizes = [b.shape[0] for b in blocks]
    out = np.zeros((sum(sizes), sum(sizes)), dtype=complex)
    o = 0
    for b in blocks:
        n = b.shape[0]
        out[o:o+n, o:o+n] = b
        o += n
    return out


def unitary_lowdin(W: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """
    Löwdin orthonormalization: W -> W (W†W)^(-1/2)

    Parameters
    ----------
    W : (dim, n) complex
    """
    S = W.conj().T @ W
    # eigen-decompose Hermitian overlap
    evals, evecs = np.linalg.eigh(S)
    keep = evals > rcond * evals.max()
    if not np.all(keep):
        evals = evals[keep]
        evecs = evecs[:, keep]
    Sinvhalf = (evecs * (1.0 / np.sqrt(evals))) @ evecs.conj().T
    return W @ Sinvhalf


def wrap_frac_k(k_frac: np.ndarray) -> np.ndarray:
    """Wrap fractional k coords into [0,1)."""
    return k_frac - np.floor(k_frac)

def generate_k_frac(Nk: int, gamma_centered: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a uniform Nk×Nk k mesh in the moiré BZ.

    Returns
    -------
    k_cart : (Nk*Nk, 2) cartesian k vectors
    k_frac : (Nk*Nk, 2) fractional coords in basis (b1,b2) wrapped to [0,1)
    """
    if Nk <= 0:
        raise ValueError("Nk must be positive.")

    if gamma_centered:
        grid = (np.arange(Nk) - Nk/2) / Nk
        k1, k2 = np.meshgrid(grid, grid, indexing="ij")
        k_frac = np.stack([k1.ravel(), k2.ravel()], axis=1)
    else:
        grid = np.arange(Nk) / Nk
        k1, k2 = np.meshgrid(grid, grid, indexing="ij")
        k_frac = np.stack([k1.ravel(), k2.ravel()], axis=1)

    return k_frac

def make_wanniers_from_U(U: np.ndarray, eigvecs: np.ndarray) -> np.ndarray:
    wan = eigvecs @ U
    return wan

def make_U_from_wanniers(wan: np.ndarray, eigvecs: np.ndarray) -> np.ndarray:
    U = eigvecs.conj().swapaxes(1, 2) @ wan
    return U