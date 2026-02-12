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
    BMParameters, SolverParameters, MoireLattice, BMModel,
    get_eigensystem_cached, read_u_mat, load_eigensystem
)
from tbg_wannier.plotting import plot_band_structure, plot_hr_tiles_simple_triangular
from tbg_wannier.grids import symmetry_path
from tbg_wannier.solver import select_neutrality_bands
from tbg_wannier.domains import DomainDef, ShapePartitioner
from tbg_wannier.kwant import build_system, attach_square_leads
from tbg_wannier.interpolation import ThetaInterpolator
from tbg_wannier.hoppings import filter_hopping_by_energy, solve_wannier_bands, build_hopping_from_U
# --- Helper Class for Scaled Lattice ---
# --- Kwant System Construction ---

def plot_conductance(syst, energies):
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
    plt.savefig("conductance.png", dpi=300)

def main():
    model, eigvals, eigvecs, _ = load_eigensystem("cache/bm_eigs_th1p05_wr0p8_w1110_NL20_Nk6_nb10_bv0_k4ca93f82a1.npz")
    ktheta = model.params.ktheta
    lat = model.lat
    lattice_size = 1/ktheta * np.linalg.norm(model.lat.a1)
    # print(f"real space unit cell size {lattice_size:.2f} Angstrom")
    # # Select 10 neutrality bands
    # eig10b, vec10b = select_neutrality_bands(eigvals, eigvecs, n=10)
    # u_path = "wan90/bm_th1p05_wr0p8_w1110_NL20_Nk6/bm_10b_u.mat"
    # U, _ = read_u_mat(u_path)
    # HR, R_cart = build_hopping_from_U(lat, eig10b, U)
    # trial_wann = vec10b @ U  # Use the computed trial Wannier functions

    wann = np.load("wan90/bm_th1p05_wr0p8_w1110_NL20_Nk6/bm_10b_wanniers.npz")['wanniers']
    sample_thetas = [0.98, 1.05]
    # # print(f"Creating ThetaInterpolator with sample thetas: {sample_thetas}")
    # interpolator = ThetaInterpolator(wann, sample_thetas=sample_thetas)
    # out = interpolator.get_interpolated(1.05, upscale=1, verbose=True, cutoff_Ang=10)
    # plot_hr_tiles_simple_triangular(out.mask, out.R_cart, savepath="hr_tiles_interpolated.png", scale="linear")
    # k_path, dists, ticks, labels = symmetry_path(lat, nseg=20)    
    # evals = solve_wannier_bands(k_path, out.HR, out.R_cart/out.scale_factor)
    # plot_band_structure(dists, evals, ticks, labels, savepath="bands.png")
    width, height = 2000., 10000. # Angstrom dimensions for the system
    np.random.seed(42)
    thetas = [1.05]  # Example: random twist angle for disorder
    domains = ShapePartitioner.voronoi_rectangle(width, height, thetas)

    syst = build_system(domains, trial_wann=wann, cutoff_Ang=np.inf, sample_thetas=sample_thetas)
    # attach_square_leads(syst, 0.5 * width, height, lead_a=lattice_size, lead_t=100.0, coupling_t=100.0, cutoff=2*lattice_size)
    # kwant.plot(syst)
    # print("System built. Finalizing...")

    plot_dos = True
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
        plt.savefig("DOS_theta1p05.png", dpi=300)
        # --- USAGE EXAMPLE ---
    # Assuming you have loaded HR and lat_vectors from your previous code:
    # plot_interpolation_check(HR, R_cart, 5, 3)


if __name__ == "__main__":
    main()
