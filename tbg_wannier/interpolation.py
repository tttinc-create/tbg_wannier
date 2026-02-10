from typing import Dict
import numpy as np
from pyparsing import Dict
import scipy as scipy
from pathlib import Path
import numpy.typing as npt

import matplotlib.pyplot as plt
from .lattice import MoireLattice
from .config import BMParameters, SolverParameters
from .solver import select_neutrality_bands, get_eigensystem_cached
from .bm import BMModel
from .wannier90 import read_u_mat
from .trials import lowdin_project
from .hoppings import filter_hopping_by_separation, build_hopping_from_U
import tinyarray as ta

def plot_interpolation_check(HR, R_cart, HR_fine, R_cart_fine, orb_i, orb_j):
    """
    Visualizes original discrete points vs. interpolated continuous curve.
    """
    print("\nPlotting Interpolation Check...")
    num_x = HR.shape[0]
    center = num_x // 2
    num_fine_x = HR_fine.shape[0]
    upscale_factor = num_fine_x // num_x
    centre_fine = num_fine_x // 2
    dx = np.linalg.vector_norm(R_cart[1, center] - R_cart[0, center])
    dx_fine = np.linalg.vector_norm(R_cart_fine[1, centre_fine] - R_cart_fine[0, centre_fine])
    # 1. Extract Original Discrete Data (Along R_x axis, R_y=0)
    # This corresponds to the middle row of the centered HR matrix
    # We verify the onsite block is at [center, center]
    if dx != dx_fine * upscale_factor:
        print(f"Warning: Detected mismatch in grid spacing. Original dx={dx:.3e}, Fine dx={dx_fine:.3e}, Upscale factor={upscale_factor}")
    original_vals = HR[:, center, orb_i, orb_j]
    resampled_vals = HR_fine[:, centre_fine, orb_i, orb_j]
    valid_indices = np.arange(0., num_x) * dx
    fine_indices = np.arange(0., num_fine_x) * dx_fine
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Real Part
    ax1.scatter(valid_indices, original_vals.real, color='red', zorder=5, label='Original HR Points')
    ax1.plot(fine_indices, resampled_vals.real, color='blue', label='Fourier Interpolation')
    ax1.set_title(f'Real Part (Orbital {orb_i}->{orb_j})')
    ax1.set_xlabel('Distance (along a1)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Imag Part
    ax2.scatter(valid_indices, original_vals.imag, color='red', zorder=5)
    ax2.plot(fine_indices, resampled_vals.imag, color='blue')
    ax2.set_title(f'Imag Part (Orbital {orb_i}->{orb_j})')
    ax2.set_xlabel('Distance (along a1)')
    ax2.grid(True, alpha=0.3)
    # print(original_vals[1], resampled_vals[upscale_factor])
    plt.tight_layout()
    plt.show()

def resample_HR(HR, upscale_factor):
    numx, numy, _, _ = HR.shape
    tmp = scipy.signal.resample(HR, upscale_factor*numx, axis = 0)
    resampled = scipy.signal.resample(tmp, upscale_factor*numy, axis = 1)
    return resampled

class InterpolatedHamiltonian:
    """
    Manages the real-space Hamiltonian for a specific twist angle.
    Upsamples the grid and creates an efficient lookup table for hoppings.
    """
    def __init__(self, theta_deg, trial_wann, upscale_factor=16, cutoff_Ang=10.0,
                 cache_dir="cache"):
        self.theta = theta_deg
        self.upscale = upscale_factor
        self.cutoff_Ang = cutoff_Ang
        
        print(f"\n--- Initializing Model for Theta = {theta_deg:.3f}° ---")
        
        # 1. Setup TBG Model
        lat = MoireLattice.build(N_L=20, N_k=6)
        bm = BMParameters(name='bm', theta_deg=theta_deg, w1_meV=110.0, 
                          w_ratio=0.8, two_valleys=False)
        solver = SolverParameters(nbands=10)
        model = BMModel(bm, lat, solver=solver)
        
        # 2. Get Eigenvalues/Vectors (Cached)
        _, eigvals, eigvecs, _, _ = get_eigensystem_cached(model, cache_dir=cache_dir, ref_basis=trial_wann, compress_with_reference=True)
        eig10b, vec10b = select_neutrality_bands(eigvals, eigvecs, n=10)
        
        # 3. Load Projectors (U matrix)
        try:
            U = lowdin_project(vec10b, trial_wann)
        except Exception as e:
            print(f"Warning: Could not make U matrix by Lowdin projection. ({e})")
            return None

        # 6. Define Physical Lattice Scale
        # Physical lattice vectors = lat.a1 / k_theta
        ktheta = model.params.ktheta
        self.scale_factor = 1.0 / ktheta 
        self.a1 = lat.a1 * self.scale_factor
        self.a2 = lat.a2 * self.scale_factor
        
        self.basis = np.column_stack((self.a1, self.a2))
        self.inv_basis = np.linalg.inv(self.basis)
        cutoff = cutoff_Ang * ktheta  # Convert cutoff to internal units

        # 4. Construct Coarse Hoppings H(R)
        HR, R_cart = build_hopping_from_U(lat, eig10b, U)
        if upscale_factor == 1:
            print("Upscale factor is 1, skipping interpolation.")
            print(f"Filtering hoppings with separation > {cutoff} ktheta^-1...")
            self.HR_fine, self.mask = filter_hopping_by_separation(
                lat, HR, R_cart, cutoff=cutoff
            )
            self.R_fine_cart = R_cart * self.scale_factor  # Store physical coordinates for filtering
            nx, ny = self.HR_fine.shape[:2]
            self.center_idx = np.array([nx // 2, ny // 2])
        else:
            # 5. Fourier Interpolation (Upsampling)
            print(f"Upsampling Hamiltonian grid by factor {upscale_factor}...")
            self.HR_fine = resample_HR(HR, upscale_factor)
    
        
            # 7. Create Lookup Table for Separation Vectors
            # We generate the Cartesian vectors for the entire fine grid to 
            # apply the distance filter efficiently once at initialization.
            nx, ny = self.HR_fine.shape[:2]
            self.center_idx = np.array([nx // 2, ny // 2])
            
            # Generate grid indices relative to center
            i_range = np.arange(nx) - self.center_idx[0]
            j_range = np.arange(ny) - self.center_idx[1]
            
            # Meshgrid (indexing='ij' matches array layout)
            I, J = np.meshgrid(i_range, j_range, indexing='ij')
            
            # Convert indices -> fractional coords
            C1 = I / upscale_factor
            C2 = J / upscale_factor
            
            # Convert fractional -> Cartesian Coords (The "Lookup Table")
            # R_fine[i,j] is the vector R corresponding to grid point (i,j)
            self.R_fine_cart = (C1[..., np.newaxis] * self.a1 + 
                                C2[..., np.newaxis] * self.a2)
        
        # 8. Apply Separation Filter
        # Use ScaledLattice so the filter sees physical units (nm)
        
            print(f"Filtering hoppings with separation > {cutoff} ktheta^-1...")
            # Note: This modifies HR_fine to zero out distant hoppings
            self.HR_fine, self.mask = filter_hopping_by_separation(
                lat, self.HR_fine, self.R_fine_cart * ktheta, cutoff=cutoff
            )
            
        print("Model initialization complete.")

    def get_hopping(self, displacement_cart: np.ndarray):
        """
        Retrieve hopping matrix for a physical displacement vector d.
        
        Args:
            displacement_cart (np.array): Vector d = r_to - r_from in nm.
            
        Returns:
            tinyarray or None: 10x10 hopping matrix, or None if zero/masked.
        """
        # 1. Map physical displacement to fractional coordinates
        # d = c1*a1 + c2*a2
        c = self.inv_basis @ displacement_cart
        
        # 2. Map fractional coordinates to Fine Grid Indices
        # idx = center + c * upscale
        idx_float = self.center_idx + c * self.upscale
        idx = np.rint(idx_float).astype(int)
        
        # 3. Lookup
        nx, ny = self.HR_fine.shape[:2]
        if (0 <= idx[0] < nx) and (0 <= idx[1] < ny):
            # Check mask to avoid returning explicit zeros
            if self.mask[idx[0], idx[1]].any():
                # Convert to tinyarray for Kwant performance
                return ta.array(self.HR_fine[idx[0], idx[1]])
        
        return None
    
    def get_hoppings_vectorized(self, displacements_cart: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Vectorized retrieval of hopping matrices for multiple displacements.
        Optimization: Replaces Python loop with matrix operations.
        
        Args:
            displacements_cart: (N, 2) array of displacement vectors in nm.
            
        Returns:
            matrices: (N, 10, 10) complex array.
            valid_mask: (N,) bool array indicating found hoppings.
        """
        # 1. Map to fractional coords: C = D @ B_inv.T
        c = displacements_cart @ self.inv_basis.T
        
        # 2. Map to grid indices
        idx_float = self.center_idx + c * self.upscale
        idx = np.rint(idx_float).astype(int)
        
        # 3. Check bounds
        nx, ny = self.HR_fine.shape[:2]
        in_bounds = (idx[:, 0] >= 0) & (idx[:, 0] < nx) & \
                    (idx[:, 1] >= 0) & (idx[:, 1] < ny)
        
        n_hops = displacements_cart.shape[0]
        matrices = np.zeros((n_hops, 10, 10), dtype=complex)
        valid_mask = np.zeros(n_hops, dtype=bool)
        
        # 4. Filter and lookup
        # We only access array for indices that are strictly in bounds
        valid_locs = np.where(in_bounds)[0]
        
        if len(valid_locs) > 0:
            ix = idx[valid_locs, 0]
            iy = idx[valid_locs, 1]
            
            # Check the distance mask (from filter_hopping_by_separation)
            mask_vals = self.mask[ix, iy].any()
            
            # Final indices that are both in-bounds and within physical cutoff
            final_locs = valid_locs[mask_vals]
            
            if len(final_locs) > 0:
                fix = idx[final_locs, 0]
                fiy = idx[final_locs, 1]
                
                # Bulk copy data
                matrices[final_locs] = self.HR_fine[fix, fiy]
                valid_mask[final_locs] = True
                
        return matrices, valid_mask
    
    def get_lattice_hopping(self):
        center_idx = self.center_idx 
        n_x, n_y, _, _ = self.HR_fine.shape
        if self.upscale != 1:
            raise ValueError("get_lattice_hopping is only valid for non-interpolated Hamiltonian (upscale_factor=1).")
        hopping_data = []
        for i in range(n_x):
            for j in range(n_y):
                # A. Geometric Filtering (Distance & Direction)
                # ---------------------------------------------
                # Skip if distance is too large
                # if norm_R[i, j] > dist_threshold:
                #     continue
                    
                dx = i - center_idx[0]
                dy = j - center_idx[1]
                
                # Skip if it is the onsite term (0,0) - handled separately
                if dx == 0 and dy == 0:
                    continue
                
                # Direction Filter: Only keep "backwards" hoppings to avoid double counting
                # (Kwant adds the Hermitian conjugate automatically)
                if not (dx < 0 or (dx == 0 and dy < 0)):
                    continue

                # B. Energy Filtering (Matrix Magnitude)
                # --------------------------------------
                mat = self.HR_fine[i, j].copy() # Copy to avoid modifying original data
                
                # Check if the ENTIRE matrix is effectively zero
                # If the max element is all 0, this hopping contributes nothing.
                if np.all(mat == 0):
                    continue
                
                # 3. Add to list (convert to tinyarray for Kwant speed)
                hopping_data.append((dx, dy, ta.array(mat)))
        print(f"✓ Hoppings retained: {len(hopping_data)} unique vectors after filtering.")
        return hopping_data

class ModelManager:
    """Handles loading and caching of InterpolatedHamiltonians."""
    def __init__(self, trial_wann, cutoff_Ang):
        self.trial_wann = trial_wann
        self.cutoff_Ang = cutoff_Ang
        self.models: Dict[float, InterpolatedHamiltonian] = {}

    def get_model(self, theta: float, upscale: int) -> InterpolatedHamiltonian:
        key = (theta, upscale)
        if key not in self.models:
            self.models[key] = InterpolatedHamiltonian(
                theta, self.trial_wann, 
                cutoff_Ang=self.cutoff_Ang, 
                upscale_factor=upscale
            )
        return self.models[key]    
