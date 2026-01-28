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

@dataclass(frozen=True)
class WannierizationRecipe:
    l: float = 0.5
    alpha: float = 2.0
    ebr_sequence: list[str] = field(default_factory=list)