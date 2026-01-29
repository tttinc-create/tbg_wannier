"""
Wannier90 file readers/writers.

This module is purely about file formats: it does *not* know about the BM model.
That separation makes it much easier to adapt the physics part without touching
I/O code.

Implemented formats
-------------------
- seedname.eig
- seedname.amn
- seedname.mmn
- seedname_u.mat (reader)
- seedname.dmn (symmetry-adapted Wannier90)
"""
from __future__ import annotations
from pathlib import Path
import subprocess 
import shutil
import os
from typing import List, Tuple, Optional
from .lattice import MoireLattice
from .bm import BMModel
from .config import WannierizationRecipe
from .solver import get_eigensystem_cached, module_name_from_model
from .trials import TrialBuilder
from .symmetry import SymmetryGroup,build_D_band_from_group, build_D_wann_from_group, build_D_wann_generators_from_EBRs, build_dmn_maps_trivial_irr
import numpy as np
import re


def ensure_wan90_workdir(
    module_name: str,
    wan90_root: str | Path = "wan90",
    wannier90_x_path: Optional[str | Path] = None,
) -> Path:
    """
    Ensure ./wan90/{module_name}/ directory exists and copy wannier90.x into it.
    
    Parameters
    ----------
    module_name : str
        Directory name (e.g., 'zhida_th1p05_wr0p8_w1110_NL20_Nk6').
    wan90_root : str or Path
        Root directory for Wannier90 work (default: 'wan90').
    wannier90_x_path : str, Path, or None
        Path to wannier90.x executable. If None, searches in current dir.
        
    Returns
    -------
    Path
        Full path to ./wan90/{module_name}/
    """
    wan90_root = Path(wan90_root)
    work_dir = wan90_root / module_name
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Find and copy wannier90.x if not already present
    wannier90_exe = work_dir / "wannier90.x"
    if not wannier90_exe.exists():
        # Search for wannier90.x
        if wannier90_x_path is None:
            search_paths = [
                Path.cwd() / "wannier90.x",
                Path.cwd().parent / "wannier90.x",
                shutil.which("wannier90.x"),
            ]
            for candidate in search_paths:
                if candidate and Path(candidate).exists():
                    wannier90_x_path = candidate
                    break
        
        if wannier90_x_path is None:
            raise FileNotFoundError(
                f"wannier90.x not found. Please provide wannier90_x_path or place "
                f"wannier90.x in current directory."
            )
        
        wannier90_x_path = Path(wannier90_x_path)
        if not wannier90_x_path.exists():
            raise FileNotFoundError(f"wannier90.x not found at: {wannier90_x_path}")
        
        shutil.copy2(wannier90_x_path, wannier90_exe)
        print(f"Copied wannier90.x to {wannier90_exe}")
    
    return work_dir


def generate_win_content(
    *,
    num_wann: int,
    num_bands: int,
    N_k: int,
    mp_grid: Optional[Tuple[int, int, int]] = None,
    units: str = "Ang",
    write_hr: bool = False,
    write_u_matrices: bool = True,
    num_iter: int = 5000,
    conv_tol: float = 1.0e-10,
    disentangle: bool = True,
    dis_win_max: float = 80.0,
    dis_win_min: float = -80.0,
    dis_froz_max: float = 0.01,
    dis_froz_min: float = -0.01,
    sym_adapted: bool = False,
    site_symmetry: bool = True,
    symmetrize_eps: float = 1e-6,
    # shell_list: int = 2,
    # one_dim_axis: str = "z",
    # skip_b1_tests: bool = True,
    extra_keywords: Optional[str] = None,
) -> str:
    """
    Generate Wannier90 .win file content in Python.
    
    Parameters
    ----------
    num_wann : int
        Number of Wannier functions.
    num_bands : int
        Number of Bloch bands.
    N_k : int
        k-point grid size (assumes N_k x N_k x 1).
    mp_grid : tuple(int, int, int), optional
        MP grid. If None, uses (N_k, N_k, 1).
    projections : str
        Projection specification (default: Gaussian s on origin + p orbitals).
    units : str
        Unit cell units (default: 'Ang').
    write_hr : bool
        Write real-space Hamiltonian (default: False).
    write_u_matrices : bool
        Write U matrices (default: True).
    num_iter : int
        Number of iterations (default: 10000).
    conv_tol : float
        Convergence tolerance (default: 1e-10).
    disentangle : bool
        Enable disentanglement (default: True).
    dis_win_max, dis_win_min : float
        Disentanglement window bounds (default: ±80 eV).
    dis_froz_max, dis_froz_min : float
        Frozen window bounds (default: ±0.01 eV).
    sym_adapted : bool
        Enable symmetry-adapted Wannierization (default: False).
    site_symmetry : bool
        Use site symmetry (for sym_adapted mode, default: True).
    symmetrize_eps : float
        Symmetry tolerance (default: 1e-6).
    shell_list : int
        Shell list size (for sym_adapted mode, default: 2).
    one_dim_axis : str
        One-dimensional axis (for sym_adapted mode, default: 'z').
    skip_b1_tests : bool
        Skip B1 tests (for sym_adapted mode, default: True).
    extra_keywords : str, optional
        Additional keywords to append (raw text).
    
    Returns
    -------
    str
        Complete .win file content.
    """
    if mp_grid is None:
        mp_grid = (N_k, N_k, 1)
    
    lines = [
        "! Wannier90 input file generated by tbg_wannier",
        "",
        f"num_wann          = {num_wann:6d}",
        f"num_bands         = {num_bands:6d}",
        "",
        "! Convergence & iteration",
        f"num_iter          = {num_iter}",
        "num_print_cycles  = 1000",
        f"conv_tol          = {conv_tol:.1e}",
        "conv_window       = 3",
        "",
        "! Real-space unit cell (minimal example, adjust for your system)",
        "begin unit_cell_cart",
        f"{units}",
        "3.62759873 2.0943951 0.0",
        "-3.62759873  2.0943951 0.0",
        "0.0  0.0 30.0",
        "end unit_cell_cart",
        "",
        "! Atoms (placeholder)",
        "begin atoms_frac",
        "X 0.00  0.00  0.00",
        "end atoms_frac",
        "",
        "! File writing",
        f"write_hr          = .{'true' if write_hr else 'false'}.",
        f"write_u_matrices  = .{'true' if write_u_matrices else 'false'}.",
        "",
        f"mp_grid = {mp_grid[0]} {mp_grid[1]} {mp_grid[2]}",
        "",
    ]
    
    # Add disentanglement section if enabled
    if disentangle:
        lines.extend([
            "! Disentanglement",
            f"dis_win_max       = {dis_win_max:.1f}d0",
            f"dis_win_min       = {dis_win_min:.1f}d0",
            f"dis_froz_max      = {dis_froz_max:.2f}d0",
            f"dis_froz_min      = {dis_froz_min:.2f}d0",
            "dis_num_iter      = 1000",
            "dis_mix_ratio     = 1.d0",
            "dis_conv_tol      = 1.0e-10",
            "",
            "! Projections",
            "begin projections",
            "f=0.0,0.0,0.0:px;py",
            "end projections",
            "",
        ])
    
    # Add symmetry-adapted section if enabled
    if sym_adapted:
        lines.extend([
            "! Symmetry-adapted Wannierization",
            f"site_symmetry     = .{'true' if site_symmetry else 'false'}.",
            f"symmetrize_eps    = {symmetrize_eps:.0e}",
            # f"shell_list        = {shell_list}",
            # f"one_dim_axis      = {one_dim_axis}",
            # f"skip_B1_tests     = .{'true' if skip_b1_tests else 'false'}.",
            "",
        ])
    
    if extra_keywords:
        lines.append("")
        lines.append(extra_keywords)
    
    return "\n".join(lines) + "\n"


def write_win(
    output_win: str | Path,
    *,
    num_wann: int,
    num_bands: int,
    N_k: int,
    **kwargs,
) -> None:
    """
    Write a Wannier90 .win file directly (no template needed).
    
    Parameters
    ----------
    output_win : str or Path
        Output .win file path.
    num_wann : int
        Number of Wannier functions.
    num_bands : int
        Number of Bloch bands.
    N_k : int
        k-point grid size (assumes N_k x N_k x 1).
    **kwargs
        Additional arguments passed to generate_win_content().
    
    Returns
    -------
    None
        Writes file to disk.
    """
    output_win = Path(output_win)
    content = generate_win_content(
        num_wann=num_wann,
        num_bands=num_bands,
        N_k=N_k,
        **kwargs,
    )
    output_win.parent.mkdir(parents=True, exist_ok=True)
    output_win.write_text(content)
    print(f"Wrote {output_win}  (num_wann={num_wann}, num_bands={num_bands}, mp_grid={N_k} {N_k} 1)")


def write_win_from_template(
    template_win: str | Path,
    output_win: str | Path,
    *,
    num_wann: int,
    num_bands: int,
    N_k: int
) -> None:
    """
    Write a Wannier90 .win file from a template, updating num_wann and num_bands.

    Parameters
    ----------
    template_win : str or Path
        Path to the template .win file (fixed content).
    output_win : str or Path
        Path to write the updated .win file.
    num_wann : int
        Number of Wannier functions.
    num_bands : int
        Number of Bloch bands.
    """
    template_win = Path(template_win)
    output_win = Path(output_win)

    if not template_win.exists():
        raise FileNotFoundError(f"Template win file not found: {template_win}")

    text = template_win.read_text()

    def _replace_param(text: str, key: str, value: int) -> str:
        # matches e.g. "num_wann = 8", allowing flexible whitespace
        pattern = rf"(^\s*{key}\s*=\s*)(\d+)"
        repl = rf"\g<1>{value}"
        new_text, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
        if n == 0:
            raise ValueError(f"Did not find '{key} =' in template .win file.")
        return new_text
    def _replace_mp_grid(text: str, N_k: int) -> str:
        # Matches e.g. "mp_grid = 6 6 1" (any integers, flexible whitespace)
        pattern = r"(^\s*mp_grid\s*=\s*)(\d+)\s+(\d+)\s+(\d+)"
        repl = rf"\g<1>{N_k} {N_k} 1"
        new_text, n = re.subn(pattern, repl, text, flags=re.MULTILINE)
        if n == 0:
            raise ValueError("Did not find 'mp_grid =' in template .win file.")
        return new_text
    text = _replace_param(text, "num_wann", num_wann)
    text = _replace_param(text, "num_bands", num_bands)
    text = _replace_mp_grid(text, N_k)
    output_win.write_text(text)
    print(f"Wrote {output_win}  (num_wann={num_wann}, num_bands={num_bands}, mp_grid={N_k} {N_k} 1))")
    
def read_eig(seedname: str | Path) -> np.ndarray:
    """
    Read seedname.eig written by write_eig().

    Format (one line per band,k):
        band_index  k_index  eigenvalue

    Returns
    -------
    eigvals : (Nk, Nb)
        eigvals[ik, m] = energy of band m at k-point ik
        (same shapes / ordering as passed to write_eig).
    """
    seedname = Path(seedname)
    filename = seedname.parent / f"{seedname.stem}.eig"
    rows = []
    with open(filename, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            m = int(parts[0]) - 1  # band index, 0-based
            ik = int(parts[1]) - 1 # k index, 0-based
            e = float(parts[2])
            rows.append((ik, m, e))

    if not rows:
        raise ValueError(f"No data found in {filename}")

    Nk = max(ik for ik, _, _ in rows) + 1
    Nb = max(m for _, m, _ in rows) + 1

    eigvals = np.zeros((Nk, Nb), dtype=float)
    for ik, m, e in rows:
        eigvals[ik, m] = e

    return eigvals

def write_eig(seedname: str | Path, eig: np.ndarray) -> None:
    """
    Write seedname.eig.

    Wannier90 expects energies in eV in many workflows; your notebook used meV.
    We keep the raw numbers as provided, but you can pass energies in eV if desired.

    Format:
      band_index  k_index  eigenvalue
    """
    seedname = Path(seedname)
    filename = seedname.parent / f"{seedname.stem}.eig"
    filename.parent.mkdir(parents=True, exist_ok=True)
    eig = np.asarray(eig, float)
    num_kpts, num_bands = eig.shape
    with open(filename, "w") as f:
        for ik in range(num_kpts):
            for m in range(num_bands):
                f.write(f"{m+1:5d} {ik+1:5d} {eig[ik, m]:20.12f}\n")


def write_amn(seedname: str | Path, A: np.ndarray, comment: str = "created by python") -> None:
    """
    Write seedname.amn

    A[ik, m, n] = <ψ_{m,k} | g_{n,k}>  (bands m, projections n)

    Format (Wannier90):
      comment line
      num_bands  num_kpts  num_wann
      m  ik  n  Re  Im    (one line per element)
    """
    seedname = Path(seedname)
    filename = seedname.parent / f"{seedname.stem}.amn"
    filename.parent.mkdir(parents=True, exist_ok=True)
    A = np.asarray(A, complex)
    num_kpts, num_bands, num_wann = A.shape
    with open(filename, "w") as f:
        f.write(comment.strip() + "\n")
        f.write(f"{num_bands:6d} {num_kpts:6d} {num_wann:6d}\n")
        for ik in range(num_kpts):
            for n in range(num_wann):
                for m in range(num_bands):
                    z = A[ik, m, n]
                    f.write(f"{m+1:5d} {n+1:5d} {ik+1:5d} {z.real:18.12f} {z.imag:18.12f}\n")


# def write_mmn(seedname: str, M: np.ndarray, nn_list: List[Tuple[int,int,int,int,int]], comment: str = "created by python") -> None:
#     """
#     Write seedname.mmn

#     Parameters
#     ----------
#     M : array, shape (nntot, nbands, nbands)
#         Overlap matrices for each neighbor link.
#         Convention: M[idx, m, n] corresponds to <u_{m,k}|u_{n,k+b}>
#     nn_list : list of tuples (ik, ikb, n1, n2, n3)
#         - ik: k-point index (1-based)
#         - ikb: neighbor k-point index (1-based)
#         - (n1,n2,n3): integer reciprocal-lattice vector connecting k->k+b in
#           the *real-space* lattice basis used by Wannier90 (3 integers).
#           For 2D models, you typically use n3=0.
#     """
#     M = np.asarray(M, complex)
#     nntot, nb1, nb2 = M.shape
#     assert nb1 == nb2, "M must be (nntot, nb, nb)"
#     nb = nb1
#     if len(nn_list) != nntot:
#         raise ValueError("len(nn_list) must match M.shape[0]")

#     # Infer num_kpts as max ik in nn_list
#     num_kpts = max(max(ik, ikb) for ik,ikb,_,_,_ in nn_list)

#     with open(seedname + ".mmn", "w") as f:
#         f.write(comment.strip() + "\n")
#         f.write(f"{nb:6d} {num_kpts:6d} {nntot:6d}\n")
#         for idx, (ik, ikb, n1, n2, n3) in enumerate(nn_list):
#             f.write(f"{ik:5d} {ikb:5d} {n1:5d} {n2:5d} {n3:5d}\n")
#             # nb^2 lines, n slow, m fast (Wannier90 convention)
#             for n in range(nb):
#                 for m in range(nb):
#                     z = M[idx, m, n]
#                     f.write(f"{z.real:18.12f} {z.imag:18.12f}\n")

def write_mmn(seedname: str | Path, M: np.ndarray, num_kpts: int, nn_list: np.ndarray) -> None:
    """Write seedname.mmn (Wannier90).

    M[idx, m, n] corresponds to the neighbor record nn_list[idx] = (ik, ikb, g1, g2, g3),
    with m fastest as in Wannier90.
    
    """
    seedname = Path(seedname)
    filename = seedname.parent / f"{seedname.stem}.mmn"
    filename.parent.mkdir(parents=True, exist_ok=True)
    _, num_bands, _ = M.shape
    nntot = nn_list.shape[0] // num_kpts
    with open(filename, "w") as f:
        f.write("created by python\n")
        f.write(f"{num_bands:6d} {num_kpts:6d} {nntot:6d}\n")
        for idx, (ik, ikb, n1, n2, n3) in enumerate(nn_list):
            f.write(f"{ik:5d} {ikb:5d} {n1:5d} {n2:5d} {n3:5d}\n")
            for n in range(num_bands):
                for m in range(num_bands):
                    val = M[idx, m, n]
                    f.write(f"{val.real:18.12f} {val.imag:18.12f}\n")


def _read_u_mat_int(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read Wannier90 seedname_u.mat.

    Returns
    -------
    U : (Nk, Nwann, Nwann) complex
    kpts : (Nk, 3) fractional k-point coordinates
    """
    with open(filename, "r") as f:
        _ = f.readline()  # header
        line = f.readline().split()
        if len(line) < 3:
            raise ValueError("Invalid u.mat header second line.")
        num_kpts, num_wann, num_bloch = map(int, line[:3])
        # if num_wann != num_bloch:
        #     raise ValueError("u.mat expects square matrices Nwann x Nwann.")

        kpts = np.zeros((num_kpts, 3), dtype=float)
        U = np.zeros((num_kpts, num_bloch, num_wann), dtype=complex)

        for ik in range(num_kpts):
            # Skip possible blank line(s) before k-point coords
            line = f.readline()
            while line is not None and line.strip() == "":
                line = f.readline()
            if not line:
                raise EOFError(f"Unexpected EOF while reading k-point {ik+1}")

            parts = line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"Expected 3 floats for k-point coords, got: {line}"
                )
            kpts[ik, :] = [float(x) for x in parts]

            # Now read num_wann*num_bloch lines: ((U(ik, i, j), i=1..num_bloch), j=1..num_wann)
            for j in range(num_wann):      # column index
                for i in range(num_bloch):  # row index
                    line = f.readline()
                    while line is not None and line.strip() == "":
                        line = f.readline()
                    if not line:
                        raise EOFError(
                            f"Unexpected EOF while reading U matrix at k={ik+1}"
                        )
                    vals = line.split()
                    if len(vals) < 2:
                        raise ValueError(
                            f"Expected 2 floats (Re Im), got: {line}"
                        )
                    re = float(vals[0])
                    im = float(vals[1])
                    U[ik, i, j] = re + 1j * im
    return U, kpts

def read_u_mat(seedname: str | Path, disentanglement: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read Wannier90 seedname_u.mat.
    Give the seedname as input (without .mat).
    Returns
    -------
    U : (Nk, Nwann, Nwann) complex
    kpts : (Nk, 3) fractional k-point coordinates
    """
    seedname = Path(seedname)
    filename = seedname.parent / f"{seedname.stem}.mat"
    U, kpts =  _read_u_mat_int(str(filename))
    num_kpts = kpts.shape[0]
    U_out = U.copy()
    if disentanglement:
        filename_dis = seedname.parent / f"{seedname.stem}_dis.mat"
        U_dist, kpts_dis = _read_u_mat_int(str(filename_dis))
        if not np.allclose(kpts, kpts_dis):
            raise ValueError("k mesh for U_dist and U does not match")
        U_out = np.array([U_dist[i] @ U[i] for i in range(num_kpts)])
    return U_out, kpts

def write_u_mat(
    seedname: str | Path,
    U: np.ndarray,
    kpts: np.ndarray,
    *,
    comment: str = "created by python",
) -> None:
    """
    Write Wannier90 seedname_u.mat (or seedname_u_dis.mat).

    This is the exact inverse of `read_u_mat` and is guaranteed
    to round-trip correctly.

    Parameters
    ----------
    seedname : str or Path
        Wannier90 seedname.
    U : ndarray, shape (Nk, num_bloch, num_wann)
        Gauge matrix.
    kpts : ndarray, shape (Nk, 3)
        Fractional k-point coordinates.
    comment : str
        Header comment line.
    """
    seedname = Path(seedname)
    filename = seedname.parent / f"{seedname.stem}.mat"
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    U = np.asarray(U, complex)
    kpts = np.asarray(kpts, float)

    if U.ndim != 3:
        raise ValueError("U must have shape (Nk, num_bloch, num_wann).")
    if kpts.ndim != 2 or kpts.shape[1] != 3:
        raise ValueError("kpts must have shape (Nk, 3).")

    Nk, num_bloch, num_wann = U.shape
    if kpts.shape[0] != Nk:
        raise ValueError("kpts and U must have the same number of k-points.")

    with open(filename, "w") as f:
        # header
        f.write(comment.strip() + "\n")
        f.write(f"{Nk:d} {num_wann:d} {num_bloch:d}\n")

        for ik in range(Nk):
            # k-point coordinates
            f.write(
                f"{kpts[ik,0]: .12f} "
                f"{kpts[ik,1]: .12f} "
                f"{kpts[ik,2]: .12f}\n"
            )

            # U-matrix: j outer, i inner (Wannier90 convention)
            for j in range(num_wann):
                for i in range(num_bloch):
                    z = U[ik, i, j]
                    f.write(f"{z.real: .18e} {z.imag: .18e}\n")


def parse_nnkp(filename: str) -> dict:
    """Parse a Wannier90 `.nnkp` file.

    Returns a dict containing (when present):
      - 'kpoints' : list of fractional kpoints
      - 'nn_list' : array of neighbor records (ik, ikb, g1, g2, g3) (1-based indices)
    """
    lines = open(filename, "r").read().splitlines()

    def find_block(tag: str) -> tuple[int, int]:
        begin = None
        end = None
        for i, ln in enumerate(lines):
            if ln.strip().lower() == f"begin {tag}".lower():
                begin = i + 1
            if ln.strip().lower() == f"end {tag}".lower():
                end = i
                break
        if begin is None or end is None:
            return (-1, -1)
        return (begin, end)

    out = {}

    b, e = find_block("kpoints")
    if b != -1:
        kpts = []
        for ln in lines[b:e]:
            if not ln.strip():
                continue
            parts = ln.split()
            if len(parts) >= 3:
                kpts.append([float(parts[0]), float(parts[1]), float(parts[2])])
        out["kpoints"] = np.asarray(kpts, float)

    b, e = find_block("nnkpts")
    if b != -1:
        nn = []
        for ln in lines[b:e]:
            if not ln.strip():
                continue
            parts = ln.split()
            if len(parts) >= 5:
                nn.append([int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])])
        out["nn_list"] = np.asarray(nn, int)

    return out
def build_amn_from_trials(eigvecs: np.ndarray, trials: np.ndarray) -> np.ndarray:
    """A[ik,m,n] = <psi_{m k}|g_{n k}> for each k."""
    Nk = eigvecs.shape[0]
    nbands = eigvecs.shape[-1]
    nwann = trials.shape[-1]
    A = np.zeros((Nk, nbands, nwann), dtype=complex)
    for ik in range(Nk):
        A[ik] = eigvecs[ik].conj().T @ trials[ik]
    return A

def build_mmn_from_nnkp(lat: MoireLattice, eigvecs: np.ndarray, nn_list: np.ndarray) -> np.ndarray:
    """Build the overlap matrices M for each neighbor record in nn_list.

    nn_list rows are: (ik, ikb, g1, g2, g3), with ik/ikb **1-based** (Wannier90 convention).
    """
    nn_list = np.asarray(nn_list, dtype=int)

    # Shapes:
    # eigvecs: (Nk, dim, nbands)
    # We want M[idx] = V1.conj().T @ R @ V2  -> (nbands, nbands)

    if nn_list.size == 0:
        return np.zeros((0, eigvecs.shape[2], eigvecs.shape[2]), dtype=complex)

    nrec = nn_list.shape[0]
    dim = eigvecs.shape[1]
    nbands = eigvecs.shape[2]

    ik_idx = nn_list[:, 0].astype(int) - 1
    ikb_idx = nn_list[:, 1].astype(int) - 1

    # Build stacked arrays for vectorized matmuls
    # R_stack: (nrec, dim, dim)
    R_stack = np.empty((nrec, dim, dim), dtype=complex)
    for idx, (_, _, g1, g2, g3) in enumerate(nn_list):
        R_stack[idx] = lat.embedding_matrix((g1, g2))

    # V2_stack: (nrec, dim, nbands)
    V2_stack = eigvecs[ikb_idx]
    # V1_conjT_stack: (nrec, nbands, dim)
    V1_conjT_stack = np.conjugate(eigvecs[ik_idx]).transpose(0, 2, 1)

    # tmp = R @ V2  -> (nrec, dim, nbands)
    tmp = np.matmul(R_stack, V2_stack)

    # M_stack = V1_conjT @ tmp -> (nrec, nbands, nbands)
    M_stack = np.matmul(V1_conjT_stack, tmp)

    # Verification: ensure vectorized result matches reference (previous loop)
    # Compute reference with the original loop for correctness check.
    # M_ref = []
    # for ik, ikb, g1, g2, g3 in nn_list:
    #     V1 = eigvecs[ik - 1]
    #     V2 = eigvecs[ikb - 1]
    #     R = lat.embedding_matrix((g1, g2))
    #     M_ref.append(V1.conj().T @ R @ V2)
    # M_ref = np.asarray(M_ref)

    # if not np.allclose(M_stack, M_ref, rtol=1e-8, atol=1e-12):
    #     diff = np.max(np.abs(M_stack - M_ref))
    #     raise ValueError(f"Vectorized build_mmn_from_nnkp differs from reference (max abs diff={diff})")

    return M_stack

def _write_int_block(f, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=int).ravel()
    for i in range(0, arr.size, 10):
        chunk = arr[i:i+10]
        f.write(" ".join(f"{int(x):5d}" for x in chunk) + "\n")


def write_dmn(
    seedname: str | Path,
    num_bands: int,
    num_wann: int,
    full_to_irr: np.ndarray,
    irr_kpts: np.ndarray,
    sym_kpt_map: np.ndarray,
    D_wann: np.ndarray,
    D_band: np.ndarray,
    comment: str = "created by python",
) -> None:
    """
    Write seedname.dmn for symmetry-adapted Wannier90.

    This writer follows the structure used in your notebook (complex numbers
    written as "( re, im )" per line). Wannier90 is forgiving as long as the
    block ordering is correct.

    Parameters
    ----------
    seedname : str or Path
        Wannier90 seedname.
    full_to_irr : (num_kpts,) 1-based indices mapping each full k to an irreducible k index
    irr_kpts : (nkptirr,) 1-based indices of irreducible k-points in the full list
    sym_kpt_map : (nkptirr, nsym) 1-based indices of g*k_irr in the full list
    D_wann : (nsym, nkptirr, num_wann, num_wann)
    D_band : (nsym, nkptirr, num_bands, num_bands)
    """
    seedname = Path(seedname)
    filename = seedname.parent / f"{seedname.stem}.dmn"
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    full_to_irr = np.asarray(full_to_irr, dtype=int).ravel()
    irr_kpts = np.asarray(irr_kpts, dtype=int).ravel()
    sym_kpt_map = np.asarray(sym_kpt_map, dtype=int)

    D_wann = np.asarray(D_wann, dtype=complex)
    D_band = np.asarray(D_band, dtype=complex)

    nsym, nkptirr, nw1, nw2 = D_wann.shape
    nsym2, nkptirr2, nb1, nb2 = D_band.shape
    if nsym2 != nsym or nkptirr2 != nkptirr:
        raise ValueError("D_wann and D_band must agree on (nsym, nkptirr).")
    if nw1 != num_wann or nw2 != num_wann:
        raise ValueError("D_wann last dims must be (num_wann,num_wann).")
    if nb1 != num_bands or nb2 != num_bands:
        raise ValueError("D_band last dims must be (num_bands,num_bands).")

    num_kpts = full_to_irr.size
    if irr_kpts.size != nkptirr:
        raise ValueError("irr_kpts length must equal nkptirr.")
    if sym_kpt_map.shape != (nkptirr, nsym):
        raise ValueError("sym_kpt_map shape must be (nkptirr, nsym).")

    with open(filename, "w") as f:
        f.write(comment.strip() + "\n")
        f.write(f"{num_bands:6d} {nsym:6d} {nkptirr:6d} {num_kpts:6d}\n\n")

        # full_to_irr (Wannier90 expects integer list)
        _write_int_block(f, full_to_irr)
        f.write("\n")

        # irr_kpts
        _write_int_block(f, irr_kpts)
        f.write("\n")

        # sym_kpt_map blocks, one per irr k
        for ikirr in range(nkptirr):
            _write_int_block(f, sym_kpt_map[ikirr, :])
            f.write("\n")

        # D_wann blocks: order ikirr -> isym -> j -> i (i fastest)
        for ikirr in range(nkptirr):
            for isym in range(nsym):
                M = D_wann[isym, ikirr]
                for j in range(num_wann):
                    for i in range(num_wann):
                        z = complex(M[i, j])
                        f.write(f"( {z.real:22.15e}, {z.imag:22.15e} )\n")
                f.write("\n")

        # D_band blocks
        for ikirr in range(nkptirr):
            for isym in range(nsym):
                M = D_band[isym, ikirr]
                for j in range(num_bands):
                    for i in range(num_bands):
                        z = complex(M[i, j])
                        f.write(f"( {z.real:22.15e}, {z.imag:22.15e} )\n")
                f.write("\n")

def make_wanniers_from_U(U: np.ndarray, eigvecs: np.ndarray) -> np.ndarray:
    numk = eigvecs.shape[0]
    wan = np.array([eigvecs[i] @ U[i] for i in range(numk)])
    return wan

def make_U_from_wanniers(wan: np.ndarray, eigvecs: np.ndarray) -> np.ndarray:
    numk = eigvecs.shape[0]
    U = np.array([eigvecs[i].T.conj() @ wan[i] for i in range(numk)])
    return U

def write_w90_files(
    model: BMModel,
    recipe: WannierizationRecipe,
    *,
    seedname: str | Path | None = None,
    wan90_root: str | Path = "wan90",
    eigvals: np.ndarray | None = None,
    do_write_eig: bool = True,
    eigvecs: np.ndarray | None = None,
    read_from_cache: bool = True,
    group: SymmetryGroup | None = None,
    sym_adapted: bool = False,
    disentangle: bool = False,
    no_trials: bool = False,
    force_localization: bool = False,
    wannier90_x_path: Optional[str | Path] = None,
) -> Path:
    """
    Write Wannier90 input files organized in ./wan90/{module_name}/ directory.
    
    This function orchestrates the entire Wannier90 workflow:
    1. Creates organized directory ./wan90/{module_name}/
    2. Generates and writes .win file (Python-generated, no templates)
    3. Computes or loads eigensystem
    4. Writes .eig (eigenvalues)
    5. Writes .amn (projection overlap)
    6. Runs wannier90 to generate .nnkp
    7. Writes .mmn (neighbor overlaps)
    8. Optionally writes .dmn (symmetry-adapted)
    
    Parameters
    ----------
    model : BMModel
        Model with parameters, lattice, and solver config.
    recipe : WannierizationRecipe
        Wannierization config (trial orbitals, EBR sequence).
    wan90_root : str or Path
        Root directory for Wannier90 files (default: 'wan90').
    eigvals : ndarray, optional
        Eigenvalues (Nk, nbands). If None and read_from_cache=True, loaded from cache.
    do_write_eig : bool
        Write .eig file (default: True).
    eigvecs : ndarray, optional
        Eigenvectors (Nk, dim, nbands). If None and read_from_cache=True, loaded from cache.
    read_from_cache : bool
        Load eigensystem from cache (default: True).
    group : SymmetryGroup, optional
        Symmetry group for sym_adapted mode.
    sym_adapted : bool
        Write symmetry-adapted .dmn file (default: False).
    disentangle : bool
        Enable disentanglement (default: True).
    no_trials : bool
        Skip trial orbital generation (use identity) (default: False).
    wannier90_x_path : str, Path, or None
        Path to wannier90.x executable. If None, searches automatically.
    
    Returns
    -------
    Path
        Work directory where files were written.
    """
    # 1) Setup directory structure
    module_name = module_name_from_model(model)
    work_dir = ensure_wan90_workdir(
        module_name,
        wan90_root=wan90_root,
        wannier90_x_path=wannier90_x_path,
    )
    if seedname is None:
        seedname = model.params.name    
    seed = work_dir / seedname

    lat = model.lat
    k_mesh, k_frac = lat.k_cart, lat.k_frac
    num_kpoints = k_mesh.shape[0]
    if sym_adapted and group is None:
        raise ValueError("Symmetry group must be provided if sym_adapted=True")

    # 2) Load or compute eigensystem
    if read_from_cache:
        _, eigvals, eigvecs, cache_path = get_eigensystem_cached(model, cache_dir="cache")
        print(f"Loaded eigensystem from cache: {cache_path}")
    elif eigvecs is None:
        raise ValueError("If not read from cache, eigenvectors must be provided.")
    
    num_bands = eigvecs.shape[-1]
    num_wann = recipe.num_wann
    
    # 3) Generate and write .win file (Python-generated, no template)
    win_path = work_dir / f"{seedname}.win"
    write_win(
        win_path,
        num_wann=num_wann,
        num_bands=num_bands,
        N_k=lat.N_k,
        disentangle=disentangle,
        sym_adapted=sym_adapted,
    )
    print(f"Generated .win file: {win_path}")
    
    # 4) Write .eig (eigenvalues)
    if do_write_eig:
        if eigvals is None:
            raise ValueError("Eigenvalues must be provided if do_write_eig=True.")
        write_eig(seed, eigvals)
        print(f"Wrote eigenvalues: {seed}.eig")
    
    # 5) Write .amn (projection overlap)
    if no_trials:
        print("Skipping trial functions; using identity projection.")
        A = np.eye(num_bands, num_wann)[np.newaxis, :, :].repeat(num_kpoints, axis=0)
    else:
        builder = TrialBuilder(lat, recipe)
        trials = builder.build_all(k_mesh)
        A = build_amn_from_trials(eigvecs, trials)
    write_amn(seed, A)
    print(f"Wrote projections: {seed}.amn")
    
    # 6) Run Wannier90 to generate .nnkp
    nnkp_path = work_dir / f"{seedname}.nnkp"
    if not nnkp_path.exists():
        print(f"Running wannier90.x -pp to generate .nnkp...")
        result = subprocess.run(
            ["./wannier90.x", "-pp", seedname],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError(f"wannier90.x failed with return code {result.returncode}")
        print(f"Generated .nnkp: {nnkp_path}")
    
    # 7) Write .mmn (neighbor overlaps)
    nnkp_full_path = str(nnkp_path)
    parsed = parse_nnkp(nnkp_full_path)
    nn_list = parsed.get("nn_list", None)
    if nn_list is None:
        raise RuntimeError(f"No nn_list found in {nnkp_path}")
    
    M = build_mmn_from_nnkp(lat, eigvecs, nn_list)
    write_mmn(seed, M, num_kpts=len(k_mesh), nn_list=nn_list)
    print(f"Wrote overlaps: {seed}.mmn")
    
    # 8) Optional: Write .dmn (symmetry-adapted Wannierization)
    if sym_adapted:
        print("Building symmetry-adapted Wannierization...")
        
        # D_band for full group
        D_band, elem_names = build_D_band_from_group(group, lat, eigvecs)
        
        # D_wann for full group from EBR targets
        D_wann_gens = build_D_wann_generators_from_EBRs(
            ebr_sequence=list(recipe.ebr_sequence),
            lat=lat,
            generators_needed=["C3z", "C2x"],
        )
        D_wann, elem_names2 = build_D_wann_from_group(group, lat, D_wann_gens)
        
        # DMN maps (trivial irreducible k structure)
        full_to_irr, irr_kpts, sym_kpt_map, _ = build_dmn_maps_trivial_irr(
            group, lat, elem_names=elem_names
        )
        
        # Write .dmn
        write_dmn(
            seedname=seed,
            num_bands=num_bands,
            num_wann=num_wann,
            full_to_irr=full_to_irr,
            irr_kpts=irr_kpts,
            sym_kpt_map=sym_kpt_map,
            D_wann=D_wann,
            D_band=D_band,
            comment="Generated by tbg_wannier",
        )
        print(f"Wrote symmetry matrices: {seed}.dmn")
    
    # 9) Auto-run Wannier90 full optimization if U-matrix not found
    u_mat_path = work_dir / f"{seedname}_u.mat"
    if not u_mat_path.exists() or force_localization:
        print(f"\nRunning wannier90.x for full optimization (U-matrix not found)...")
        result = subprocess.run(
            ["./wannier90.x", seedname],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise RuntimeError(f"wannier90.x failed with return code {result.returncode}")
        print(f"Generated U-matrix: {u_mat_path}")
    else:
        print(f"\nU-matrix already exists at {u_mat_path}")
    
    print(f"\nWannier90 workflow complete!")
    print(f"Work directory: {work_dir}")
    print(f"U-matrix: {u_mat_path}")
    return work_dir

