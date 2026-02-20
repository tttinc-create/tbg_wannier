"""
Simulation of Disordered Twist-Angle Domains in TBG using Kwant.

This script stitches together domains with different twist angles by using
Fourier interpolation of the Wannier Hamiltonian. It handles the lattice
mismatch by treating site positions in continuous real space and filtering
hoppings based on the physical separation of Wannier centers.

Key Components:
1. InterpolatedHamiltonian: Manages upsampling and creates a lookup table
   for efficient retrieval of H(d) for arbitrary distance vectors d.
2. ScaledLattice: Helper to handle physical unit conversion (nm) from model params.
3. make_disordered_system: Builds the Kwant system with domain logic.
"""
import collections

import numpy as np
import kwant
import matplotlib.pyplot as plt

# --- Imports from user-provided files ---
from tbg_wannier import (
    BMParameters, SolverParameters, MoireLattice, BMModel, WannierizationRecipe,
    get_eigensystem_cached, read_u_mat, load_eigensystem, make_wanniers_from_U
)
from tbg_wannier.plotting import plot_band_structure, plot_hr_tiles_simple_triangular, plot_real_space_wanniers
from tbg_wannier.grids import symmetry_path
from tbg_wannier.solver import select_neutrality_bands
from tbg_wannier.domains import DomainDef, ShapePartitioner
from tbg_wannier.kwant import build_system, attach_square_leads
from tbg_wannier.interpolation import ThetaInterpolator
from tbg_wannier.symmetry import (symmetrize_u_matrix, group_23, group_TR, group_662)
from tbg_wannier.hoppings import filter_hopping_by_energy, solve_wannier_bands, build_hopping_from_U
# --- Helper Class for Scaled Lattice ---
# --- Kwant System Construction ---

def plot_conductance(syst, energies, filename="conductance.png"):
    # Compute conductance
    print("Computing conductance...")
    data = []
    tot_task = len(energies)
    for idx, energy in enumerate(energies):
        print(f"  Energy {idx+1}/{tot_task}: {energy:.2f} meV")
        smatrix = kwant.smatrix(syst, energy)
        data.append(smatrix.transmission(1, 0))
    plt.figure()
    plt.plot(energies, data)
    plt.xlabel("Energy [meV]")
    plt.ylabel("conductance [e^2/h]")
    plt.savefig(filename, dpi=300)

def main1():
    model, eigvals, eigvecs, _ = load_eigensystem("cache/eigensystems/bm_eigs_th1p05_wr0p8_w1110_NL20_Nk6_nb10_bv0_k4ca93f82a1.npz")
    # model, eigvals, eigvecs, _ = load_eigensystem("cache/eigensystems/bm_eigs_th1p05_wr0p8_w1110_NL20_Nk18_nb10_bv0_k4fe3c41d1a.npz")

    ktheta = model.params.ktheta
    lat = model.lat
    lattice_size = 1/ktheta * np.linalg.norm(model.lat.a1)
    print(f"real space unit cell size {lattice_size:.2f} Angstrom")
    # # Select 10 neutrality bands
    # eig10b, vec10b = select_neutrality_bands(eigvals, eigvecs, n=10)
    # u_path = "wan90/bm_th1p05_wr0p8_w1110_NL20_Nk6/bm_10b_u.mat"
    # U, _ = read_u_mat(u_path)
    # HR, R_cart = build_hopping_from_U(lat, eig10b, U)
    # trial_wann = vec10b @ U  # Use the computed trial Wannier functions
    recipes = {'2EaA1aA2aEc': WannierizationRecipe(
        l=0.1 * 4 * np.pi / 3,
        alpha=2.0,
        ebr_sequence=["Ea", "Ea", "A1a", "A2a", "Ec"],
                ),
                'EaA1aBfEc': WannierizationRecipe(
                    l=0.1 * 4 * np.pi / 3,
                    alpha=2.0,
                    ebr_sequence=["Ea", "A1a", "Bf", "Ec"],
                ),
                'EaA2aAfEc': WannierizationRecipe(
                    l=0.1 * 4 * np.pi / 3,
                    alpha=2.0,
                    ebr_sequence=["Ea", "A2a", "Af", "Ec"],
                ),
                'AfBfEc': WannierizationRecipe(
                    l=0.1 * 4 * np.pi / 3,
                    alpha=2.0,
                    ebr_sequence=["Af", "Bf", "Ec"],
                ),
                }
    brs = ['2EaA1aA2aEc', 'EaA1aBfEc', 'EaA2aAfEc', 'AfBfEc']
    for br in brs[2:3]:
        U, _ = read_u_mat(f"/home/fengw/tbg_wannier/wan90/bm_th1p05_wr0p8_w1110_NL20_Nk6/bm_{br}_10b_u.mat")
        # U, _ = read_u_mat(f"/home/fengw/tbg_wannier/wan90/bm_th1p05_wr0p8_w1110_NL20_Nk18/bm_{br}_u.mat")
        recipe = recipes[br]
        U_sym = symmetrize_u_matrix(U, group=group_662, recipe=recipe, eigvecs=eigvecs, lat=lat, n_iter=3)
        # U = symmetrize_with_C2zT(U, eigvecs, lat, recipe)
        wann = make_wanniers_from_U(U, eigvecs)
        wann_sym = make_wanniers_from_U(U_sym, eigvecs)
        plot_real_space_wanniers(
            lat, wann, beta_idx=1, layer=1,
            savepath='wann_test', ncols=5, cmap="inferno"
        )
        plot_real_space_wanniers(
            lat, wann_sym, beta_idx=1, layer=1,
            savepath='wann_sym_test', ncols=5, cmap="inferno"
        )
        thetas = [1.05]  # Example: random twist angle for disorder
    # print(f"Creating ThetaInterpolator with sample thetas: {sample_thetas}")
        plot_hr = False
        if plot_hr:
            for i, w in enumerate([wann, wann_sym]):
                out = ThetaInterpolator(w, sample_thetas=thetas, model_template=model, interp_kind='pchip').get_interpolated(1.05, 
                                                                    upscale=1, verbose=True, cutoff_Ang=np.inf)
                plot_hr_tiles_simple_triangular(out.HR, out.R_cart, savepath=f"images/18x18/hr10b_{br}_linear_{i}.png", scale="linear",
                                                 title=f"H(R) for band rep {br}")
                plot_hr_tiles_simple_triangular(out.HR, out.R_cart, savepath=f"images/18x18/hr10b_{br}_log_{i}.png", scale="log",
                                             title=f"H(R) for band rep {br}")
    return None

def main():
    theta = 1.05
    lat = MoireLattice.build(N_L=20, N_k=18)
    solver = SolverParameters(nbands=10)
    bm = BMParameters(
        name="bm",
        theta_deg=theta,
        w1_meV=110.0,
        w_ratio=0.8,
        two_valleys=False,
    )
    recipe = WannierizationRecipe(
        l=0.1 * 4 * np.pi / 3,
        alpha=2.0,
        ebr_sequence=["Ea", "Ea", "A1a", "A2a", "Ec"],
    )
    model = BMModel(bm, lat, solver=solver)
    model, eigvals, eigvecs, _, _ = get_eigensystem_cached(model, cache_dir="cache/eigensystems")
    ktheta = model.params.ktheta
    lat = model.lat
    lattice_size = 1/ktheta * np.linalg.norm(model.lat.a1)
    print(f"real space unit cell size {lattice_size:.2f} Angstrom")
    # # Select 10 neutrality bands
    # eig10b, vec10b = select_neutrality_bands(eigvals, eigvecs, n=10)
    # u_path = "wan90/bm_th1p05_wr0p8_w1110_NL20_Nk6/bm_10b_u.mat"
    # U, _ = read_u_mat(u_path)
    # HR, R_cart = build_hopping_from_U(lat, eig10b, U)
    # trial_wann = vec10b @ U  # Use the computed trial Wannier functions
    # U, _ = read_u_mat("wan90/bm_th1p05_wr0p8_w1110_NL20_Nk6/10b_2Ea_u.mat")
    # wann = make_wanniers_from_U(U, eigvecs)
    wann = np.load("/home/fengw/tbg_wannier/wan90/bm_th1p05_wr0p8_w1110_NL20_Nk6/bm_2EaA1aA2aEc_10b_wanniers.npz")['wanniers']
    # repC2zT_mat = repC2zT(lat).toarray()
    # wann_sym = (wann + repC2zT_mat @ wann.conj())
    # norm = np.linalg.norm(wann_sym, axis=1, keepdims=True)
    # wann_sym = wann_sym / norm
    # wann = np.load("wan90/bm_th1p05_wr0p8_w1110_NL20_Nk6/bm_10b_wanniers.npz")['wanniers']
    # sample_thetas = [1.05]
    # # print(f"Creating ThetaInterpolator with sample thetas: {sample_thetas}")
    # interpolator = ThetaInterpolator(wann_sym, sample_thetas=sample_thetas, interp_kind='pchip', 
    #                                  model_template=model)
    # out = interpolator.get_interpolated(1.05, upscale=1, verbose=True, cutoff_Ang=np.inf)
    # plot_hr_tiles_simple_triangular(out.HR, out.R_cart, savepath="images/hr_tr_tiles_interpolated_linear.png", scale="linear")
    # plot_hr_tiles_simple_triangular(out.HR, out.R_cart, savepath="images/hr_tr_tiles_interpolated_log.png", scale="log")
    # return None
    # k_path, dists, ticks, labels = symmetry_path(lat, nseg=20)    
    # evals = solve_wannier_bands(k_path, out.HR, out.R_cart/out.scale_factor)
    # plot_band_structure(dists, evals, ticks, labels, savepath="images/bands.png")
    width, height = 400., 600. # Angstrom dimensions for the system
    np.random.seed(42)
    thetas = [1.05]  # Example: random twist angle for disorder
    domains = ShapePartitioner.voronoi_rectangle(width, height, thetas)
    syst = build_system(domains, trial_wann=wann, cutoff_Ang=np.inf, sample_thetas=[1.05],
                         tbg_leads=True, verbose=True)
    # attach_square_leads(syst, width, height, lead_a=lattice_size, lead_t=100.0, coupling_t=100.0, cutoff=lattice_size)
    kwant.plot(syst)
    # print("System built. Finalizing...")
    syst = syst.finalized()
    energies = np.linspace(-25, 25, 10)
    # plot_conductance(syst, energies=energies, filename="images/conductance_2ea.png")
    kwant.plotter.bands(syst.leads[0])
    plot_dos = False
    if plot_dos:
        fsyst = syst.finalized()
        spectrum = kwant.kpm.SpectralDensity(fsyst)
        spectrum.add_moments(100)
        energies = np.linspace(-50, 50, 100)
        densities = spectrum(energies)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(energies, densities.real, color='purple')
        ax.set_xlabel("Energy (meV)")
        ax.set_ylabel("Density of States")
        plt.suptitle("Density of States for Pristine TBG System")
        plt.tight_layout()
        plt.savefig("images/DOS_theta1p05.png", dpi=300)
        # --- USAGE EXAMPLE ---
    # Assuming you have loaded HR and lat_vectors from your previous code:
    # plot_interpolation_check(HR, R_cart, 5, 3)


if __name__ == "__main__":
    main()
