"""
Configuration and parameter objects.

This module is intentionally "dumb": it only defines dataclasses and light
validation. No physics is computed here.

Design goals
------------
- Avoid hidden globals (the notebook relied on global state heavily).
- Make runs reproducible: parameters can be saved/loaded as JSON.
- Keep parameters grouped by responsibility (BM model vs solver vs meshes).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, Literal
import json
import math
import numpy as np

@dataclass(frozen=True)
class BMParameters:
    """
    Parameters for the Bistritzer–MacDonald continuum model for TBG.

    Notes on units
    --------------
    - Energies are in meV (matching your notebook).
    - Lengths are in nm.
    - Momenta are in 1/nm.
    - ħ v_F is in meV·nm.

    Conventions
    -----------
    The model is built around one valley (K) by default; set `two_valleys=True`
    if you want to build a block-diagonal valley-doubled Hamiltonian.
    """
    name: str = "bm"
    theta_deg: float = 1.05
    # graphene lattice constant (a ≈ 2.46 Angstrom)
    a_graphene_Angstrom: float = 2.46
    # ħ v_F ≈ 5944 meV·Angstrom (often used in continuum BM)
    hvf_meV_Ang: float = 5944

    # interlayer tunneling
    w1_meV: float = 110.0                 # "AA/AB" average, depending on convention
    w_ratio: float = 0.8                  # w0/w1 if using relaxation (w0 = w_ratio*w1)
    phi_AB: float = 2.0 * math.pi / 3.0   # phase used in T2,T3

    # momentum lattice truncation size (N_L x N_L) in reciprocal-lattice integer coords
    # degrees of freedom
    norb: int = 2         # 2 = (A,B) sublattice per layer per valley in the Dirac continuum
    two_layers: bool = True
    two_valleys: bool = False

    # numerical tolerances
    match_tol: float = 1e-8

    @property
    def theta_rad(self) -> float:
        return self.theta_deg * math.pi / 180.0

    @property
    def w0_meV(self) -> float:
        return self.w_ratio * self.w1_meV

    @property
    def ktheta(self) -> float:
        theta = self.theta_rad
        a = self.a_graphene_Angstrom
        kD = 4.0 * np.pi / (3.0 * a)
        ktheta = 2.0 * kD * np.sin(theta / 2.0)
        return ktheta
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @staticmethod
    def from_json(path: str) -> "BMParameters":
        with open(path, "r") as f:
            d = json.load(f)
        return BMParameters(**d)


@dataclass(frozen=True)
class SolverParameters:
    """
    Parameters controlling the eigen-solver.

    We default to sparse `eigsh` near zero energy (sigma shift-invert), like your
    notebook.
    """
    nbands: int = 10
    sigma: float = 1e-3
    which: str = "LM"
    maxiter: int = 10_000
    tol: float = 1e-10
    ncv: Optional[int] = None
    # Sorting: "abs" (closest to 0 by absolute value) or "energy" (ascending)
    # sort: Literal["abs", "energy"] = "abs"
    extra_eigs: int = 4


# Mapping of EBR symbols to number of Wannier orbitals
EBR_WANNIER_COUNTS = {
    "A1a": 1,
    "A2a": 1,
    "Ea": 2,
    "Ec": 4,
    "Af": 3,
    "Bf": 3,
    "zhida": 2,  # special case: Zhida's 2 Wannier trial
}


@dataclass(frozen=True)
class WannierizationRecipe:
    """
    Configuration for Wannierization trial orbitals.
    
    Parameters
    ----------
    l : float
        Gaussian trial orbital width parameter.
    alpha : float
        Gaussian trial orbital exponent.
    ebr_sequence : list[str]
        Sequence of EBR symbols (e.g., ["Ea", "A1a", "A2a", "Ec"]).
        Each symbol corresponds to an elementary band representation.
        Allowed values and their Wannier counts:
          - A1a: 1 Wannier orbital
          - A2a: 1 Wannier orbital
          - Ea: 2 Wannier orbitals
          - Ec: 4 Wannier orbitals
          - Af: 3 Wannier orbitals
          - Bf: 3 Wannier orbitals
          - zhida: 2 Wannier orbitals (special case)
    """
    l: float = 0.5
    alpha: float = 2.0
    ebr_sequence: list[str] = field(default_factory=list)

    @property
    def num_wann(self) -> int:
        """
        Total number of Wannier functions from EBR sequence.
        
        Returns
        -------
        int
            Sum of Wannier counts for all EBRs in ebr_sequence.
        
        Raises
        ------
        ValueError
            If an EBR symbol is not recognized.
        """
        total = 0
        for ebr in self.ebr_sequence:
            if ebr not in EBR_WANNIER_COUNTS:
                raise ValueError(
                    f"Unknown EBR symbol: '{ebr}'. "
                    f"Allowed values: {list(EBR_WANNIER_COUNTS.keys())}"
                )
            total += EBR_WANNIER_COUNTS[ebr]
        return total