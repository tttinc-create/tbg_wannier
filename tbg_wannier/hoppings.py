import numpy as np
import numpy.typing as npt

from .fourier import build_wannier_Hk, hk_to_hr_fft2
from .lattice import MoireLattice

lst_s = [(0,0), (0,0), (0,0), (1/2, 0), (0, 1/2), (1/2, 1/2), (1/3, 2/3), (1/3, 2/3), (2/3, 1/3), (2/3, 1/3)]

lat = MoireLattice.build(N_L=20, N_k=6)
cart_real_coords = lat.cart_coords

def filter_hopping_by_separation(
    HR: npt.NDArray[np.complex128],
    R_cart: npt.NDArray[np.float64],
    wannier_centers: list[tuple[float, float]] = lst_s,
    cutoff: float = 4.75,
    verbose: bool = False,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.bool_]]:
    """
    Filters hopping elements H_{mn}(R) where the distance between Wannier centers 
    exceeds the cutoff.

    Parameters
    ----------
    lat : MoireLattice
        Must contain lattice vectors a1, a2.
    HR : array (Nx, Ny, Norb, Norb)
        Real-space hopping matrices. 
        Convention: HR[ix, iy, m, n] is hopping FROM orbital n (at 0) TO orbital m (at R).
    R_cart : array (Nx, Ny, 2)
        Cartesian vectors corresponding to the Nx, Ny grid.
    wannier_centers : list of tuples
        Fractional coordinates of the Wannier centers within the unit cell.
    cutoff : float
        Maximum allowed distance in units of 1/k_theta.

    Returns
    -------
    HR_filtered : array
        The hopping array with distant terms set to zero.
    keep_mask : boolean array
        True where elements were kept.
    """
    
    # 1. Prepare Wannier Centers in Cartesian Coordinates
    # Convert list of fractional coordinates to (Norb, 2) array
    s_cart = np.asarray([cart_real_coords(s) for s in wannier_centers], dtype=np.float64)

    # 2. Reshape for Broadcasting
    # We want to compute: vec = R[i,j] + s_cart[m] - s_cart[n]
    # HR shape is (Nx, Ny, Norb_m, Norb_n)
    
    # R_cart: (Nx, Ny, 2) -> (Nx, Ny, 1, 1, 2)
    R_expanded = R_cart[:, :, np.newaxis, np.newaxis, :]
    
    # Target centers (m = row index): s_cart (Norb, 2) -> (1, 1, Norb, 1, 2)
    s_target = s_cart[np.newaxis, np.newaxis, :, np.newaxis, :]
    
    # Source centers (n = col index): s_cart (Norb, 2) -> (1, 1, 1, Norb, 2)
    s_source = s_cart[np.newaxis, np.newaxis, np.newaxis, :, :]

    # 3. Compute Separation Vector l
    # separation_vec shape: (Nx, Ny, Norb, Norb, 2)
    separation_vec = R_expanded + s_target - s_source
    
    # 4. Compute Distances
    # shape: (Nx, Ny, Norb, Norb)
    distances = np.linalg.norm(separation_vec, axis=-1)
    
    # 5. Create Mask
    keep_mask = distances < cutoff
    
    # 6. Apply Filter
    HR_filtered = HR.copy()
    HR_filtered[~keep_mask] = 0. + 0.j
    
    # Optional: Print stats
    total_elements = keep_mask.size
    kept_elements = np.sum(keep_mask)
    if verbose:
        print(f"Filter applied (cutoff={cutoff:.3f} units of 1/k_theta): Kept {kept_elements}/{total_elements} ({kept_elements/total_elements:.1%}) hopping elements.")
    
    return HR_filtered, keep_mask

def filter_hopping_by_energy(HR: np.ndarray, thres: float, verbose: bool = False) ->  tuple[npt.NDArray[np.complex128], npt.NDArray[np.bool_]]:
    keep_mask = (np.abs(HR) > thres)
    HR_filtered = HR.copy()
    HR_filtered[~keep_mask] = 0. + 0.j
    total_elements = keep_mask.size
    kept_elements = np.sum(keep_mask)
    if verbose:
        print(f"Filter applied (cutoff={thres:.3f} meV): Kept {kept_elements}/{total_elements} ({kept_elements/total_elements:.1%}) hopping elements.")
    
    return HR_filtered, keep_mask

def build_hopping_from_U(lat: MoireLattice, eigvals: np.ndarray, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct real-space hoppings from U-matrix."""
    k_frac = lat.k_frac
    Hk = build_wannier_Hk(eigvals, U)
    HR, R_cart = hk_to_hr_fft2(lat, Hk=Hk, k_frac=k_frac)
    return HR, R_cart

def solve_wannier_bands(k_path: np.ndarray, HR: np.ndarray, R_cart: np.ndarray):
    """
    Computes band structure along a k-path using Wannier interpolation.
    
    Parameters
    ----------
    k_path : (N_k, 2) array of k-points in Cartesian coordinates.
    HR     : (..., norbs, norbs) Real-space hopping matrices.
             Can be shape (NR, norbs, norbs) or grid (Nx, Ny, norbs, norbs).
    R_cart : (..., 2) R-vectors corresponding to HR.
    
    Returns
    -------
    bands : (N_k, norbs) array of sorted eigenvalues.
    """
    # 1. Flatten Input Data if it comes in a grid format
    # We need HR as (NR, norbs, norbs) and R_cart as (NR, 2)
    nx, ny, norbs, _ = HR.shape
    flat_HR = HR.reshape(-1, norbs, norbs)
    flat_R = R_cart.reshape(-1, 2)

        
    print(f"Computing bands for {len(k_path)} k-points using {len(flat_HR)} Wannier hoppings.")

    # 2. Compute Phase Factors: exp(i * k . R)
    # Shape: (N_k, NR)
    # k_path: (N_k, 2), flat_R: (NR, 2) -> dot product gives (N_k, NR)
    phase = np.exp(-1j * np.dot(k_path, flat_R.T))

    # 3. Sum over R to get Hk
    # H(k) = sum_R phase(k, R) * H(R)
    # We use einsum for clarity and speed:
    # 'kr' (phase), 'rab' (hoppings) -> 'kab' (Hk per k-point)
    Hk_path = np.einsum('kr,rab->kab', phase, flat_HR)
    
    # 4. Diagonalize (Hermitian solver)
    # np.linalg.eigh is optimized for Hermitian matrices
    evals = np.linalg.eigvalsh(Hk_path)
    
    return evals

