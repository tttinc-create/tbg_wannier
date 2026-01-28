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
from typing import List, Tuple, Optional
from .lattice import MoireLattice
from .bm import BMModel
from .config import WannierizationRecipe
from .solver import get_eigensystem_cached
from .trials import TrialBuilder
from .symmetry import SymmetryGroup,build_D_band_from_group, build_D_wann_from_group, build_D_wann_generators_from_EBRs, build_dmn_maps_trivial_irr
import numpy as np
import re


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
    
def read_eig(seedname: str) -> np.ndarray:
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
    filename = f"{seedname}.eig"
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

def write_eig(seedname: str, eig: np.ndarray) -> None:
    """
    Write seedname.eig.

    Wannier90 expects energies in eV in many workflows; your notebook used meV.
    We keep the raw numbers as provided, but you can pass energies in eV if desired.

    Format:
      band_index  k_index  eigenvalue
    """
    eig = np.asarray(eig, float)
    num_kpts, num_bands = eig.shape
    with open(seedname + ".eig", "w") as f:
        for ik in range(num_kpts):
            for m in range(num_bands):
                f.write(f"{m+1:5d} {ik+1:5d} {eig[ik, m]:20.12f}\n")


def write_amn(seedname: str, A: np.ndarray, comment: str = "created by python") -> None:
    """
    Write seedname.amn

    A[ik, m, n] = <ψ_{m,k} | g_{n,k}>  (bands m, projections n)

    Format (Wannier90):
      comment line
      num_bands  num_kpts  num_wann
      m  ik  n  Re  Im    (one line per element)
    """
    A = np.asarray(A, complex)
    num_kpts, num_bands, num_wann = A.shape
    with open(seedname + ".amn", "w") as f:
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

def write_mmn(seedname: str, M: np.ndarray, num_kpts: int, nn_list: np.ndarray) -> None:
    """Write seedname.mmn (Wannier90).

    M[idx, m, n] corresponds to the neighbor record nn_list[idx] = (ik, ikb, g1, g2, g3),
    with m fastest as in Wannier90.
    
    """
    _, num_bands, _ = M.shape
    nntot = nn_list.shape[0] // num_kpts
    with open(seedname + ".mmn", "w") as f:
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

def read_u_mat(seedname: str, disentanglement: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read Wannier90 seedname_u.mat.

    Returns
    -------
    U : (Nk, Nwann, Nwann) complex
    kpts : (Nk, 3) fractional k-point coordinates
    """
    filename = f"{seedname}_u.mat"
    filename = f"{seedname}_u.mat"
    U, kpts =  _read_u_mat_int(filename)
    num_kpts = kpts.shape[0]
    U_out = U.copy()
    if disentanglement:
        filename_dis = f"{seedname}_u_dis.mat"
        U_dist, kpts_dis = _read_u_mat_int(filename_dis)
        if not np.allclose(kpts, kpts_dis):
            raise ValueError("k mesh for U_dist and U does not match")
        U_out = np.array([U_dist[i] @ U[i] for i in range(num_kpts)])
    return U_out, kpts

def write_u_mat(
    seedname: str,
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
    seedname : str
        Wannier90 seedname.
    U : ndarray, shape (Nk, num_bloch, num_wann)
        Gauge matrix.
    kpts : ndarray, shape (Nk, 3)
        Fractional k-point coordinates.
    comment : str
        Header comment line.
    """
    U = np.asarray(U, complex)
    kpts = np.asarray(kpts, float)

    if U.ndim != 3:
        raise ValueError("U must have shape (Nk, num_bloch, num_wann).")
    if kpts.ndim != 2 or kpts.shape[1] != 3:
        raise ValueError("kpts must have shape (Nk, 3).")

    Nk, num_bloch, num_wann = U.shape
    if kpts.shape[0] != Nk:
        raise ValueError("kpts and U must have the same number of k-points.")
    
    filename = f"{seedname}_u.mat"

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
    M_list = []
    for ik, ikb, g1, g2, g3 in nn_list:
        V1 = eigvecs[ik-1]
        V2 = eigvecs[ikb-1]
        R = lat.embedding_matrix((g1, g2))
        M_list.append(V1.conj().T @ R @ V2)
    return np.asarray(M_list)

def _write_int_block(f, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=int).ravel()
    for i in range(0, arr.size, 10):
        chunk = arr[i:i+10]
        f.write(" ".join(f"{int(x):5d}" for x in chunk) + "\n")


def write_dmn(
    seedname: str,
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
    full_to_irr : (num_kpts,) 1-based indices mapping each full k to an irreducible k index
    irr_kpts : (nkptirr,) 1-based indices of irreducible k-points in the full list
    sym_kpt_map : (nkptirr, nsym) 1-based indices of g*k_irr in the full list
    D_wann : (nsym, nkptirr, num_wann, num_wann)
    D_band : (nsym, nkptirr, num_bands, num_bands)
    """
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

    with open(seedname + ".dmn", "w") as f:
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

def write_w90_files(seed, model: BMModel, recipe: WannierizationRecipe, * ,
                    eigvals: np.ndarray| None = None, do_write_eig: bool = True,
                    eigvecs: np.ndarray| None = None, read_from_cache: bool = True,
                    group: SymmetryGroup| None = None, sym_adapted: bool = False,
                    no_trials: bool = False) -> None:
    # 1) Model + mesh + eigensystem
    if not Path(f'{seed}.win').exists():
        raise RuntimeError("No win file found.")
    lat = model.lat
    k_mesh, k_frac = lat.k_cart, lat.k_frac

    if read_from_cache:
        _, eigvals, eigvecs, cache_path = get_eigensystem_cached(model, cache_dir="cache")
        print("Loaded/computed eigensystem at:", cache_path)
    elif eigvecs is None:
        raise ValueError("If not read from cache, eigenvectors should be provided.")        

    # 2) Trials -> Löwdin -> AMN

    if do_write_eig:
        if eigvals is None:
            raise ValueError("Eigenvalues should be provided if write .eig file.")
        write_eig(seed, eigvals)
        print(f'Wrote:  {seed}.eig')

    if no_trials:
        print("Skipping trial functions as requested.")
        A = np.eye(eigvals.shape[1], eigvals.shape[1])[np.newaxis, :, :].repeat(eigvals.shape[0], axis=0)
    else:
        builder = TrialBuilder(lat, recipe)
        trials = builder.build_all(k_mesh)
        A = build_amn_from_trials(eigvecs, trials)
    write_amn(seed, A)

    # 3) MMN (needs seed.nnkp from Wannier90)
    if not Path(f'{seed}.nnkp').exists():
        subprocess.run(["./wannier90.x", '-pp', f"{seed}"])
    parsed = parse_nnkp(seed + ".nnkp")
    nn_list = parsed.get("nn_list", None)


    M = build_mmn_from_nnkp(lat, eigvecs, nn_list)
    write_mmn(seed, M, num_kpts=len(k_mesh), nn_list=nn_list)
    print(f"Wrote:  {seed}.amn {seed}.mmn")
    if sym_adapted:
        if group is None:
            raise ValueError("Symmetry group should be provided if one intends to use symmetry adapted mode")
        # 5) D_band for full group
        D_band, elem_names = build_D_band_from_group(group, lat, eigvecs)
        # old elements we compare (same set as your old build_D_wann default)
        # D_band, num_bands, _ = build_D_band(k_mesh=k_mesh, vec=eigvecs, lat=lat, sym_labels=labels)
        # [print(np.trace(D_band[1, ik, :, :])) for ik in [0, 14, 28, 11, 16, 21, 31]]
        

        # 6) D_wann for full group from EBR targets (generators -> compose)
        D_wann_gens = build_D_wann_generators_from_EBRs(
            ebr_sequence=list(recipe.ebr_sequence),
            lat=lat,
            generators_needed=["C3z", "C2x"],
        )

        D_wann, elem_names2 = build_D_wann_from_group(group, lat, D_wann_gens)
        # assert elem_names2 == elem_names
        # D_wann, num_wann, _ = build_D_wann(ebr_sequence=recipe.ebr_sequence, k_mesh=k_mesh, lat=lat, sym_labels=labels)

        # 7) dmn maps (irr = full mesh)
        # k_maps, G_lists = build_k_maps(k_frac)
        # sym_kpt_map = np.zeros((k_mesh.shape[0], len(labels)), dtype=int)
        # for isym, sym in enumerate(labels):
        #     sym_kpt_map[:, isym] = np.array(k_maps[sym]) + 1
        # irr_kpts = full_to_irr = np.arange(1, k_mesh.shape[0] + 1, dtype=int)   # 1-based
        full_to_irr, irr_kpts, sym_kpt_map, _ = build_dmn_maps_trivial_irr(
            group, lat, elem_names=elem_names
        )

        # 8) write dmn
        num_bands = eigvecs.shape[-1]
        num_wann = D_wann.shape[-1]
        write_dmn(
            seedname=seed,
            num_bands=num_bands,
            num_wann=num_wann,
            full_to_irr=full_to_irr,
            irr_kpts=irr_kpts,
            sym_kpt_map=sym_kpt_map,
            D_wann=D_wann,
            D_band=D_band,
            comment="Generated by tbg_wannier group-aware helper",
        )
        print(f'Symmetry Adapted mode enabled, wrote {seed}.dmn')

