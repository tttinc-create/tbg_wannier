import os
import math
from multiprocessing import Pool
from functools import partial
import itertools
import time

# limit BLAS/FFT threads per process
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from tbg_wannier import (
    BMParameters, SolverParameters, WannierizationRecipe,
    MoireLattice, BMModel,
    get_eigensystem_cached, read_u_mat, make_wanniers_from_U
)
from tbg_wannier.utils import paulis
from tbg_wannier.solver import select_neutrality_bands
from tbg_wannier.interaction import DensityDensityDense, frac_to_cart, lst_s, filter_V_by_separation

# Wannier center fractional coordinates (10 bands)

# Global worker state (initialized once per process)
WORKER = {}
OUT_DIR = os.path.join(os.getcwd(), 'interaction_full')

def init_worker():
    """Initializer for worker processes: build lattice, load Wannier coefficients and integrator."""
    seed = 'zhida'
    I2, sx, sy, sz = paulis()
    bm = BMParameters(name=seed, theta_deg=1.05, w1_meV=110.0, w_ratio=0.8, two_valleys=False)
    lat = MoireLattice.build(N_L=20, N_k=6)
    solver = SolverParameters(nbands=16)
    model = BMModel(bm, lat, solver=solver)
    _, eigvals, eigvecs, cache_path = get_eigensystem_cached(model, cache_dir="cache")

    eig10b, vec10b = select_neutrality_bands(eigvals, eigvecs, n=10)
    U, _ = read_u_mat("wan90/zhida_th1p05_wr0p8_w1110_NL20_Nk6/10b_merged1_u.mat")
    wanniers = make_wanniers_from_U(U, vec10b)

    k_theta = bm.ktheta
    xi = 100 * k_theta
    integrator = DensityDensityDense(lat, U_xi=24, xi=xi)

    WORKER['lat'] = lat
    WORKER['wanniers'] = wanniers
    WORKER['integrator'] = integrator
    WORKER['out_dir'] = OUT_DIR


def worker_task(t_triplet):
    """Compute all V and separations for (t1,t2,t3) triple using vectorized batch.

    Returns list of (abs_V, t1, t2, t3, n1, n2, n3, n4, len_sep1, len_sep2, len_sep3,
                     sep1_x, sep1_y, sep2_x, sep2_y, sep3_x, sep3_y).
    """
    t1, t2, t3 = t_triplet
    integrator = WORKER['integrator']
    wanniers = WORKER['wanniers']
    lat = WORKER['lat']

    # Compute all 10000 matrix elements at once
    V0 = integrator.matrix_elements_batch(
        eta=1, eta_p=1,
        w_coeffs_plus=wanniers,
        t1_mn=t1, t2_mn=t2, t3_mn=t3,
        n_max=10
    )
    
    V, _ = filter_V_by_separation(lat, V0, t1, t2, t3, cutoff=4.75)
    # Convert t to cartesian
    t1c = frac_to_cart(lat, t1)
    t2c = frac_to_cart(lat, t2)
    t3c = frac_to_cart(lat, t3)

    # Precompute all s_cart for the 10 bands as (10, 2) array
    s_all = np.array([frac_to_cart(lat, lst_s[n]) for n in range(10)])  # shape (10, 2)

    # Vectorized computation of all separations
    # sep1[n1, n2] = t1c + s_all[n1] - s_all[n2], shape (10, 10, 2)
    sep1 = t1c[None, None, :] + s_all[None, :, :] - s_all[:, None, :]
    len_sep1 = np.linalg.norm(sep1, axis=2)  # shape (10, 10)

    # sep2[n3, n4] = t2c + s_all[n3] - s_all[n4], shape (10, 10, 2)
    sep2 = t2c[None, None, :] + s_all[None, :, :] - s_all[:, None, :]
    len_sep2 = np.linalg.norm(sep2, axis=2)  # shape (10, 10)

    # sep3[n2, n4] = t3c + s_all[n2] - s_all[n4], shape (10, 10, 2)
    sep3 = t3c[None, None, :] + s_all[None, :, :] - s_all[:, None, :]
    len_sep3 = np.linalg.norm(sep3, axis=2)  # shape (10, 10)

    # Find maximum element and its indices
    absV = np.abs(V)
    idx_flat = int(np.argmax(absV))
    n1_max, n2_max, n3_max, n4_max = np.unravel_index(idx_flat, V.shape)
    max_abs = float(absV.flatten()[idx_flat])

    # compute separations for the max indices only
    s1c = frac_to_cart(lat, lst_s[n1_max])
    s2c = frac_to_cart(lat, lst_s[n2_max])
    s3c = frac_to_cart(lat, lst_s[n3_max])
    s4c = frac_to_cart(lat, lst_s[n4_max])

    len_sep1 = float(np.linalg.norm(t1c + s1c - s2c))
    len_sep2 = float(np.linalg.norm(t2c + s3c - s4c))
    len_sep3 = float(np.linalg.norm(t3c + s2c - s4c))

    # save full V to compressed npz (store real if imaginary negligible)
    out_dir = WORKER.get('out_dir', 'interaction_full')
    fname = f"full_t1_{t1[0]}_{t1[1]}__t2_{t2[0]}_{t2[1]}__t3_{t3[0]}_{t3[1]}.npz"
    fpath = os.path.join(out_dir, fname)
    try:
        imag_max = float(np.max(np.abs(V.imag)))
        if imag_max < 1e-12:
            np.savez_compressed(fpath, V_real=V.real)
        else:
            np.savez_compressed(fpath, V=V)
    except Exception:
        # best-effort save
        np.savez_compressed(fpath, V=V)

    # return compact summary and filename where full data is stored
    return (max_abs, t1, t2, t3, int(n1_max), int(n2_max), int(n3_max), int(n4_max), len_sep1, len_sep2, len_sep3, fpath)


def main(smoke: bool = True):
    lat = MoireLattice.build(N_L=20, N_k=6)
    # Define the t values (expanded): include (0,0), (±1,0),(±2,0),(0,±1),(0,±2),(±1,±1)
    x_range = np.arange(-2, 3)
    y_range = np.arange(-2, 3)
    X_idx, Y_idx = np.meshgrid(x_range, y_range, indexing='ij')
    t_vectors = np.column_stack((X_idx.flatten(), Y_idx.flatten()))
    t_cart = np.array([frac_to_cart(lat, t) for t in t_vectors])
    t_norm = np.linalg.norm(t_cart, axis=1)
    t_list = t_vectors[t_norm <= 8]
    print("t_vectors to be used:\n", t_list)
    # prepare tasks
    triples = [(t1, t2, t3) for t1 in t_list for t2 in t_list for t3 in t_list]

    if smoke:
        triples = triples[:8]

    nprocs = min(12, os.cpu_count() or 1)
    print(f"Starting pool with {nprocs} processes, smoke={smoke}, tasks={len(triples)}")

    # ensure output dir exists (created by main process)
    os.makedirs(OUT_DIR, exist_ok=True)

    t0 = time.time()
    all_results = []
    with Pool(processes=nprocs, initializer=init_worker) as pool:
        for i, res_list in enumerate(pool.imap_unordered(worker_task, triples, chunksize=2), 1):
            # worker_task returns a list of tuples
            all_results.extend(res_list)
            if i % 10 == 0 or i == len(triples):
                print(f"Completed {i}/{len(triples)}")

    dt = time.time() - t0
    print(f"Done. walltime={dt:.1f}s, total elements={len(all_results)}")
if __name__ == "__main__":
    # run the full job by default
    main(smoke=False)
