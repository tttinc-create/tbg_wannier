"""
Sweep twist angle around magic angle and analyze hopping parameter smoothness.

This script:
1. Spans 0.2° around magic angle (1.05°) at 0.02° intervals
2. For each angle, runs the Wannier90 workflow
3. Extracts real-space hopping matrices H(R)
4. Analyzes smoothness of hoppings across angles
5. Generates summary plots and statistics

Output:
    ./sweep_results/
        - hoppings_by_theta.npz  (all H(R) vs theta)
        - hopping_stats.txt      (smoothness analysis)
        - plots/                 (visualization)
"""

import numpy as np
from tbg_wannier import (
    BMParameters, SolverParameters, WannierizationRecipe,
    MoireLattice, BMModel, run_workflow_for_angle
)
def main():
    recipe_zhida = WannierizationRecipe(
        l= 0.1 * 4 * np.pi / 3,
        alpha=2.0,
        ebr_sequence=["zhida"],
    )
    recipe8b = WannierizationRecipe(
        l=0.1 * 4 * np.pi / 3,
        alpha=2.0,
        ebr_sequence=["A1a", "Bf", "Ec"],
    )
    lat = MoireLattice.build(N_L=20, N_k=6)
    solver = SolverParameters(nbands=10)
    for theta in [1.05]:
        bm = BMParameters(
        name="bm",
        theta_deg=theta,
        w1_meV=110.0,
        w_ratio=0.8,
        two_valleys=False,
    )
        model = BMModel(bm, lat, solver=solver)
        run_workflow_for_angle(
            model=model,
            recipe_zhida=recipe_zhida,
            recipe_8b=recipe8b,
            seed="bm",
            wan90_root="wan90",
            cache_dir="cache",
            verbose=True,
            plotting=True,
            wannier90_x_path="/home/tobyfeng/wannier90/tbg_wannier_package/wannier90.x",
        )
if __name__ == "__main__":
    main()
