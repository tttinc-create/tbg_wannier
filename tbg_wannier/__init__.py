"""
tbg_wannier: modular BM-model + symmetry-adapted Wannier90 interface.

The package was refactored from a monolithic Jupyter notebook into modules:
- config: parameter dataclasses
- lattice: moiré lattice + Q lattice
- bm: BM Hamiltonian
- solver: eigensolver wrappers
- grids: k meshes and high-symmetry paths
- trials: projection-based trial orbitals
- symmetry: symmetry mapping and representations (customizable)
- wannier90: readers/writers for Wannier90 files
- analysis, plotting: post-processing utilities
"""
from .config import BMParameters, SolverParameters, WannierizationRecipe
from .lattice import MoireLattice
from .bm import BMModel
from .grids import symmetry_path
from .solver import (solve_one_k, solve_kpoints, get_eigensystem_cached, load_eigensystem, 
                     module_name_from_model)
from .trials import (TrialBuilder)
from .fourier import compute_real_space_wannier
from .wannier90 import (write_win, write_win_from_template, generate_win_content, ensure_wan90_workdir,
                        read_eig, write_eig, read_u_mat, write_u_mat, parse_nnkp,
                        write_amn, write_mmn, write_dmn, build_amn_from_trials, 
                        build_mmn_from_nnkp, make_U_from_wanniers, make_wanniers_from_U, 
                        write_w90_files, run_workflow_for_angle)

__all__ = [
    "BMParameters", "SolverParameters", "WannierizationRecipe",
    "MoireLattice", "BMModel",
    "symmetry_path",
    "solve_one_k", "solve_kpoints", "get_eigensystem_cached", "load_eigensystem",
    "module_name_from_model",
    "TrialBuilder", "compute_real_space_wannier",
    "write_win", "write_win_from_template", "generate_win_content", "ensure_wan90_workdir",
    "read_eig", "write_eig", "read_u_mat", "write_u_mat", "parse_nnkp",
    "write_amn", "write_mmn", "write_dmn", 
    "build_amn_from_trials", "build_mmn_from_nnkp", "make_U_from_wanniers", "make_wanniers_from_U",
    "write_w90_files", "run_workflow_for_angle",
]

