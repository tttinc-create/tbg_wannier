"""
Example: Complete Wannier90 workflow using the new organized structure.

This script demonstrates the new wan90/{module_name}/ directory structure with:
- Python-generated .win files (no templates)
- Organized file storage per model configuration
- Automatic wannier90.x management
- Optional symmetry-adapted Wannierization

Usage:
    python example_new_workflow.py [--symmetric]

The output will be in:
    ./wan90/zhida_th1p05_wr0p8_w1110_NL20_Nk6/
"""
import argparse
import numpy as np
from pathlib import Path

from tbg_wannier import (
    BMParameters, SolverParameters, WannierizationRecipe,
    MoireLattice, BMModel,
    write_w90_files, read_u_mat, make_wanniers_from_U,
)
from tbg_wannier.plotting import plot_real_space_wanniers
from tbg_wannier.solver import get_eigensystem_cached, select_neutrality_bands
from tbg_wannier.symmetry import (
    SymmetryOperator, SymmetryGroup, group_23, group_TR, symmetrize_u_matrix, symmetrize_with_P
)
from tbg_wannier.utils import paulis
from tbg_wannier.wannier90 import write_eig, write_u_mat, write_u_mat
from tbg_wannier.wannier90 import make_U_from_wanniers


def main():


    print("=" * 70)
    print("tbg_wannier: New Workflow Example (Organized Directory Structure)")
    print("=" * 70)

    # Setup: Define model parameters
    I2, sx, sy, sz = paulis()
    seed = "bm"
    bm = BMParameters(
        name=seed,
        theta_deg=1.05,
        w1_meV=110.0,
        w_ratio=0.8,
        two_valleys=False,
    )
    lat = MoireLattice.build(N_L=20, N_k=6)
    solver = SolverParameters(nbands=16)
    model = BMModel(bm, lat, solver=solver)
    wan90_root = "wan90"    
    _, eigvals, eigvecs, cache_path, _ = get_eigensystem_cached(
    model, cache_dir="cache")
    # Define Wannierization recipe (trial orbitals)
    recipezhida = WannierizationRecipe(
        l=0.1 * 4 * np.pi / 3,
        alpha=2.0,
        ebr_sequence=["zhida"],
    )
    print(f"EBR sequence: {recipezhida.ebr_sequence}")
    print(f"Number of Wannier functions: {recipezhida.num_wann}")
    # Optional: Setup symmetry group for sym_adapted mode

    # ========================================================================
    # KEY CHANGE: New API - everything is automatic!
    # ========================================================================
    print(f"\nGenerating Wannier90 files in: {wan90_root}/")
    work_dir = write_w90_files(
        model,
        recipezhida,
        wan90_root=wan90_root,
        read_from_cache=True,  # Load eigensystem from cache
        do_write_eig=True,
        disentangle=True,
        no_trials=False,
    )

    print(f"\n✓ Workflow complete!")
    print(f"  Work directory: {work_dir}")
    u_mat_path = work_dir / f"{seed}_u.mat"
    U_zhida, _ = read_u_mat(u_mat_path, disentanglement=True)
    U_zhida_sym = symmetrize_u_matrix(U_zhida, group=group_23, lat=lat, eigvecs=eigvecs, recipe=recipezhida)
    U_zhida_sym = symmetrize_with_P(U_zhida_sym, eigvecs=eigvecs, lat=lat)
    U_zhida_sym = symmetrize_u_matrix(U_zhida_sym, group=group_TR, lat=lat, eigvecs=eigvecs, recipe=recipezhida)
    w_zhida = make_wanniers_from_U(U_zhida_sym, eigvecs)
    eig10b, vec10b = select_neutrality_bands(eigvals, eigvecs, n=10)
    M = vec10b.swapaxes(1,2).conjugate() @ w_zhida
    U, _, _ = np.linalg.svd(M)
    psi_orthocomp = vec10b @ U[..., 2:]
    seq = ["A1a", "Bf", "Ec"]
    # seq2 = ["A1a", "A2a", "Ea", "Ec"]
    recipe8b = WannierizationRecipe(
        l=0.1, alpha=2.0, ebr_sequence=seq
    )
    seed = "bm8b"
    work_dir_8b = write_w90_files(
        model,
        recipe8b,
        seedname=seed,
        wan90_root=wan90_root,
        read_from_cache=False,  # Load eigensystem from cache
        eigvecs=psi_orthocomp,
        do_write_eig=False,
        sym_adapted=True,
        group=group_23,
        no_trials=False,
    )
    u_mat_path_8b = work_dir_8b / f"{seed}_u.mat"
    U_w90_8b, k3 = read_u_mat(u_mat_path_8b)
    U_8b_sym = symmetrize_u_matrix(U_w90_8b, group=group_TR, lat=lat, eigvecs=psi_orthocomp, recipe=recipe8b, enforce_semiunitary=True)

    w_8b = make_wanniers_from_U(U_8b_sym, psi_orthocomp)
    wanniers = np.concatenate((w_zhida, w_8b), axis=-1)
    U_res = make_U_from_wanniers(wanniers, vec10b)
    u_mat_path_10b = work_dir / "10b_merged1_u"
    write_eig(work_dir / "10b_merged1", eig10b)
    write_u_mat(u_mat_path_10b, U_res, k3)
    plot_path = work_dir / f"wannier_10b_merged1"
    plot_real_space_wanniers(lat, wanniers, beta_idx=1, layer=1, savepath=plot_path, ncols=5, cmap="inferno")




if __name__ == "__main__":
    main()
