"""
Eigen-solvers for the BM Hamiltonian.

We keep this module separate so you can swap eigensolvers (dense for testing,
shift-invert sparse for production, etc.) without touching the model code.
"""
from __future__ import annotations

from typing import Tuple, Optional, Literal
import os
import json
import hashlib

import numpy as np
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import ArpackNoConvergence
from dataclasses import asdict
from pathlib import Path
from .bm import BMModel
from .config import BMParameters, SolverParameters
from .lattice import MoireLattice

def reorder_eigensystem(
    evals: np.ndarray,
    evecs: np.ndarray,
    order: Literal["energy", "abs"] = "energy",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reorder eigenpairs consistently.

    Parameters
    ----------
    evals : (..., nb)
    evecs : (..., dim, nb)
    order : "energy" | "abs"

    Returns
    -------
    evals_new, evecs_new
    """
    if order == "energy":
        idx = np.argsort(evals, axis=-1)
    elif order == "abs":
        idx = np.argsort(np.abs(evals), axis=-1)
    else:
        raise ValueError(f"Unknown order '{order}'")

    # broadcast-safe gather
    evals_new = np.take_along_axis(evals, idx, axis=-1)
    evecs_new = np.take_along_axis(
        evecs,
        idx[..., None, :],
        axis=-1,
    )
    return evals_new, evecs_new

def select_neutrality_bands(
    evals: np.ndarray,
    evecs: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select `n` bands closest to charge neutrality (E=0),
    then reorder them in increasing energy order.

    Parameters
    ----------
    evals : (Nk, nbands_total)
    evecs : (Nk, dim, nbands_total)
    n : int
        Number of bands to keep

    Returns
    -------
    evals_sel : (Nk, n)
        Sorted in increasing energy order
    evecs_sel : (Nk, dim, n)
        Eigenvectors reordered consistently
    """
    Nk, nb = evals.shape
    if n > nb:
        raise ValueError(f"Requested n={n} bands, but only nb={nb} available.")

    # 1) select by |E|
    idx_abs = np.argsort(np.abs(evals), axis=-1)[..., :n]

    evals_sel = np.take_along_axis(evals, idx_abs, axis=-1)
    evecs_sel = np.take_along_axis(
        evecs,
        idx_abs[..., None, :],
        axis=-1,
    )

    # 2) reorder selected bands by energy
    evals_sel, evecs_sel = reorder_eigensystem(
        evals_sel, evecs_sel, order="energy"
    )

    return evals_sel, evecs_sel

def solve_one_k(model: BMModel, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve for eigenpairs at a single k.

    Returns
    -------
    evals : (nbands,) float
    evecs : (dim, nbands) complex
    """
    H = model.H(k)
    dim = H.shape[0]
    sp_ = model.solver
    kreq = sp_.nbands + sp_.extra_eigs # a bit of padding, like your notebook
    try:
        evals, evecs = spla.eigsh(
            H, k=kreq, sigma=sp_.sigma, which=sp_.which, maxiter=sp_.maxiter,
            tol=sp_.tol, ncv=sp_.ncv
        )
    except ArpackNoConvergence as e:
        evals = e.eigenvalues
        evecs = e.eigenvectors
        if evals is None or evecs is None:
            raise

    # idx = np.argsort(np.abs(evals))
    # idx = idx[:sp_.nbands]
    # evals = evals[idx]
    # evecs = evecs[:, idx]
    # idx2 = np.argsort(evals)
    # evals = evals[idx2]
    # evecs = evecs[:, idx2]
    evecs = np.linalg.qr(evecs)[0]
    return evals, evecs

def solve_kpoints(model: BMModel) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve for eigenpairs on many k points.

    Parameters
    ----------
    k_list : (Nk,2)

    Returns
    -------
    evals : (Nk, nbands)
    evecs : (Nk, dim, nbands)
    """
    evals_all = []
    evecs_all = []
    k_list = model.lat.k_cart
    for k in k_list:
        ev, U = solve_one_k(model, k)
        evals_all.append(ev)
        evecs_all.append(U)
    evals = np.stack(evals_all, axis=0)
    evecs = np.stack(evecs_all, axis=0)
    return select_neutrality_bands(evals, evecs, n=model.solver.nbands)

def _hash_array(a: np.ndarray, nhex: int = 10) -> str:
    """Short stable hash of array bytes (for filenames)."""
    h = hashlib.sha1(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
    return h[:nhex]


def _safe_float(x: float) -> str:
    """Filename-friendly float formatting."""
    # e.g. 1.05 -> "1p05", 0.8 -> "0p8"
    s = f"{x:g}"
    return s.replace(".", "p").replace("-", "m")


def default_eigensystem_cache_path(
    model: BMModel,
    *,
    cache_dir: str | os.PathLike = "cache",
) -> Path:
    """Construct a descriptive cache filename for the eigensystem."""
    p = model.params
    l = model.lat
    nb = model.solver.nbands
    prefix = p.name
    k_hash = _hash_array(np.asarray(l.k_cart, float))
    fname = (
        f"{prefix}_eigs_"
        f"th{_safe_float(p.theta_deg)}_"
        f"wr{_safe_float(p.w_ratio)}_"
        f"w1{_safe_float(p.w1_meV)}_"
        f"NL{l.N_L}_"
        f"Nk{l.N_k}_"
        f"nb{nb}_"
        f"bv{int(bool(p.two_valleys))}_"
        f"k{ k_hash }.npz"
    )
    return Path(cache_dir) / fname


def module_name_from_model(
    model: BMModel,
) -> str:
    """
    Generate a clean module name for Wannier90 file organization.
    
    Extracts the model name and parameters into a descriptive directory name
    for organizing Wannier90 files under ./wan90/MODULE_NAME/*.
    
    Returns
    -------
    str
        Module name like: zhida_th1p05_wr0p8_w1110_NL20_Nk6
    """
    p = model.params
    l = model.lat
    prefix = p.name
    module_name = (
        f"{prefix}_"
        f"th{_safe_float(p.theta_deg)}_"
        f"wr{_safe_float(p.w_ratio)}_"
        f"w1{_safe_float(p.w1_meV)}_"
        f"NL{l.N_L}_"
        f"Nk{l.N_k}"
    )
    return module_name

def model_snapshot(model: BMModel) -> dict:
    """Return a JSON-serializable snapshot sufficient to reconstruct BMModel."""
    # BMModel has .params (BMParameters) and .lat (MoireLattice)
    # We record params + N_L. (lat is deterministically reconstructible from N_L.)
    return {
        "BMParameters": asdict(model.params),
        "N_L": int(model.lat.N_L),
        "N_k": int(model.lat.N_k),
        "BMsolver": asdict(model.solver),
    }


def model_from_snapshot(snap: dict) -> BMModel:
    """Reconstruct BMModel from a snapshot created by model_snapshot()."""
    if "BMParameters" not in snap or "N_L" not in snap or "N_k" not in snap or "BMsolver" not in snap:
        raise ValueError("Invalid model snapshot: missing components.")

    p = BMParameters(**snap["BMParameters"])
    s = SolverParameters(**snap["BMsolver"])
    lat = MoireLattice.build(N_L=int(snap["N_L"]),N_k=int(snap["N_k"]))
    return BMModel(p, lat, s)

def save_eigensystem(
    path: str | os.PathLike,
    *,
    model: BMModel,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    meta: Optional[dict] = None,
    ref_basis: Optional[np.ndarray] = None,
    compress_with_reference: bool = False,
    ortho_tol: float = 1e-8,
) -> None:
    """Save eigensystem to a compressed .npz with metadata.

    Stored keys:
      - k_mesh  (Nk,2)
      - eigvals (Nk,nb)
      - eigvecs (Nk,dim,nb)
      - meta_json (str)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = {} if meta is None else dict(meta)
    meta["model_snapshot"] = model_snapshot(model)
    meta.setdefault("k_mesh_hash", _hash_array(np.asarray(model.lat.k_cart, float)))
    meta["compressed_with_reference"] = compress_with_reference  # default, may be overridden below

    if compress_with_reference:
        if ref_basis is None:
            raise ValueError("compress_with_reference=True requires a ref_basis to be provided.")
        if ref_basis.shape != eigvecs.shape:
            raise ValueError("ref_basis must have the same shape as eigvecs (Nk,dim,nb).")

        # check orthonormality: R^H R == I
        R = ref_basis
        nb = eigvecs.shape[2]
        RHR = R.conj().swapaxes(1, 2) @ R
        I = np.broadcast_to(np.eye(nb), RHR.shape)
        if not np.allclose(RHR, I, atol=ortho_tol):
            raise ValueError("Provided ref_basis is not orthonormal within ortho_tol.")

        # overlaps: (Nk, nb, nb) = eigvecs.conj().swapaxes(1,2) @ ref_basis
        overlaps = ref_basis.conj().swapaxes(1, 2) @ eigvecs
        meta["compressed_with_reference"] = True
        try:
            meta["ref_hash"] = _hash_array(np.asarray(ref_basis, float))
        except Exception:
            pass
        meta_json = json.dumps(meta, sort_keys=True)

        np.savez_compressed(
            path,
            eigvals=np.asarray(eigvals, float),
            evecs_overlap=np.asarray(overlaps, complex),
            meta_json=np.array(meta_json),
        )
        return

    # default full-storage path
    meta_json = json.dumps(meta, sort_keys=True)
    np.savez_compressed(
        path,
        eigvals=np.asarray(eigvals, float),
        eigvecs=np.asarray(eigvecs, complex),
        meta_json=np.array(meta_json),
    )


def load_eigensystem(
    path: str | os.PathLike,
    *,
    ref_basis: Optional[np.ndarray] = None,
    output_overlaps: bool = False,
    # ortho_tol: float = 1e-8,
    # atol: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load eigensystem from a .npz.

    If the file contains a compressed overlap matrix (saved with
    `save_eigensystem(..., compress_with_reference=True)`), then a
    `ref_basis` must be provided to reconstruct the eigenvectors.

    Returns (model, eigvals, eigvecs, meta_dict).
    """
    path = Path(path)
    data = np.load(path, allow_pickle=False)

    eigvals = np.asarray(data["eigvals"], float)

    meta_json = str(data["meta_json"])
    meta = json.loads(meta_json) if meta_json else {}
    snap = meta.get("model_snapshot", None)
    if snap is None:
        raise ValueError("Cache file does not contain a model_snapshot.")
    compress_with_reference = meta.get("compressed_with_reference", False)
    model = model_from_snapshot(snap)

    # either full eigenvectors or compressed overlaps may be stored
    if "eigvecs" in data:
        eigvecs = np.asarray(data["eigvecs"], complex)
        return model, eigvals, eigvecs, meta

    if compress_with_reference:
        if ref_basis is None:
            raise ValueError(
                "Cache contains compressed eigvec overlaps; provide `ref_basis` to reconstruct eigvecs."
            )
        overlaps = np.asarray(data["evecs_overlap"], complex)

        # shapes: overlaps (Nk, nb, nb), ref_basis (Nk, dim, nb)
        Nk = eigvals.shape[0]
        if overlaps.ndim != 3:
            raise ValueError("Invalid overlap array shape in cache.")
        if ref_basis.shape != (Nk, int(model.lat.siteN * 4 if not model.params.two_valleys else model.lat.siteN * 8), overlaps.shape[2]):
            # fallback: allow matching (Nk, dim, nb) by using model.dim if available
            expected_dim = model.lat.siteN * (4 if not model.params.two_valleys else 8)
            if ref_basis.shape != (Nk, expected_dim, overlaps.shape[2]):
                raise ValueError("Provided ref_basis has incorrect shape compared to cached overlaps.")
        if output_overlaps:
            return model, eigvals, overlaps, meta
        # check orthonormality of reference basis columns per k
        # ref_basis: (Nk, dim, nb) -> R^H R should be identity (nb, nb)
        # R = ref_basis
        # nb = overlaps.shape[1]
        # # compute R^H R: (Nk, nb, nb)
        # RHR = np.einsum("kdi,kdj->kij", R.conj().swapaxes(1, 2), R)
        # I = np.broadcast_to(np.eye(nb), RHR.shape)
        # if not np.allclose(RHR, I, atol=ortho_tol):
        #     raise ValueError("Provided ref_basis is not orthonormal within ortho_tol.")

        # reconstruct eigvecs: U_k = R_k @ overlaps_k.conj().T
        eigvecs = ref_basis @ overlaps
        return model, eigvals, eigvecs, meta

    raise ValueError("Cache file missing eigenvector data (neither 'eigvecs' nor 'evecs_overlap' present).")


def get_eigensystem_cached(
    model: BMModel,
    *,
    cache_dir: str | os.PathLike = "cache",
    force_recompute: bool = False,
    ref_basis: Optional[np.ndarray] = None,
    compress_with_reference: bool = False,
    output_overlaps: bool = False,
) -> tuple[np.ndarray, np.ndarray, Path, dict]:
    """Load eigensystem from cache or compute + store it.

    Returns
    -------
    model, eigvals, eigvecs, cache_path, meta

    Notes
    -----
    - `require_match=True` enforces that cached k_mesh equals provided k_mesh.
    - Filename includes a short hash of the k_mesh to prevent accidental collisions.
    - If `compress_with_reference=True` the provided `ref_basis` will be used
      when saving the newly computed eigensystem.
    """
    k_mesh = model.lat.k_cart
    path = default_eigensystem_cache_path(
        model, cache_dir=cache_dir
    )

    if (not force_recompute) and path.exists():
        print("Cache file found, loading")
        model, ev, eV, meta = load_eigensystem(path, ref_basis=ref_basis, output_overlaps=output_overlaps)
        return model, ev, eV, path, meta

    # compute
    print("Cache file not found, computing")
    eigvals, eigvecs = solve_kpoints(model)
    print("Done with computing")
    # metadata (store params snapshot + basic shapes)
    save_eigensystem(
        path,
        model=model,
        eigvals=eigvals,
        eigvecs=eigvecs,
        ref_basis=ref_basis,
        compress_with_reference=compress_with_reference,
    )
    # return meta as well (loaded from saved file to reflect exact storage)
    model2, ev2, eV2, meta = load_eigensystem(path, ref_basis=ref_basis, output_overlaps=output_overlaps)
    return model2, ev2, eV2, path, meta
