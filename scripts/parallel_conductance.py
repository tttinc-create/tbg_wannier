"""
parallel_conductance.py

Compute conductance in parallel with MPI (mpi4py).

Usage (example):
  mpirun -n 10 python parallel_conductance.py --syst fsyst.sav --nsteps 400 --emin -100 --emax 100

Each MPI rank loads the finalized Kwant system file (shared filesystem),
splits the energy grid among ranks, computes transmissions, then gathers
results to rank 0 for plotting and saving.
"""
import argparse
import numpy as np
from mpi4py import MPI
import kwant
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--syst", default="fsyst.sav", help="Path to saved finalized kwant system")
    p.add_argument("--nsteps", type=int, default=400, help="Number of energy points")
    p.add_argument("--emin", type=float, default=-100.0, help="Minimum energy [meV]")
    p.add_argument("--emax", type=float, default=100.0, help="Maximum energy [meV]")
    p.add_argument("--out", default="conductance_parallel.npz", help="Output npz file")
    return p.parse_args()


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"[rank {rank}] MPI size: {size}. Loading system from {args.syst}", flush=True)

    # Each rank loads the finalized system independently (shared FS assumed)
    fsyst = None
    load_errors = []
    # 2) Try pickle / dill
    if fsyst is None:
        try:
            import pickle
            with open(args.syst, "rb") as f:
                obj = pickle.load(f)
            # If we loaded a Builder, finalize it
            if hasattr(obj, "finalized") and callable(getattr(obj, "finalized")):
                try:
                    fsyst = obj.finalized()
                except Exception:
                    fsyst = None
            else:
                # Maybe it's already a finalized system
                fsyst = obj
        except Exception as e:
            load_errors.append(("pickle", e))

    if fsyst is None:
        if rank == 0:
            print(f"Failed to load system '{args.syst}'. Attempts:\n" +
                  "  " + "\n  ".join([f"{m}: {err}" for m, err in load_errors]))
        raise RuntimeError(f"Could not load system from {args.syst}")

    # Build energy grid and split indices among ranks
    energies = np.linspace(args.emin, args.emax, args.nsteps)
    all_indices = np.arange(len(energies))
    indices_split = np.array_split(all_indices, size)
    my_indices = indices_split[rank]

    if rank == 0:
        print(f"[rank {rank}] Computing transmissions on {size} ranks:", flush=True)
    print(f"[rank {rank}] computing {len(my_indices)} energies", flush=True)

    my_trans = np.empty(len(my_indices), dtype=float)
    for i_local, idx in enumerate(my_indices):
        E = energies[idx]
        try:
            print(f"[rank {rank}] Computing transmission at E={E:.2f} meV", flush=True)
            smatrix = kwant.smatrix(fsyst, E)
            # assume two leads: lead 0 -> lead 1
            my_trans[i_local] = smatrix.transmission(1, 0)
        except Exception as e:
            print(f"[rank {rank}] error at E={E}: {e}", flush=True)
            my_trans[i_local] = np.nan

    # Gather results to root as lists (variable lengths)
    gathered_indices = comm.gather(my_indices, root=0)
    gathered_trans = comm.gather(my_trans, root=0)

    if rank == 0:
        # Reconstruct full transmission array
        trans_full = np.empty(len(energies), dtype=float)
        for idx_arr, tr_arr in zip(gathered_indices, gathered_trans):
            trans_full[idx_arr] = tr_arr

        # Save and plot
        np.savez(args.out, energies=energies, transmission=trans_full)
        print(f"[rank {rank}] Saved conductance to {args.out}", flush=True)

        plt.figure()
        plt.plot(energies, trans_full)
        plt.xlabel("Energy [meV]")
        plt.ylabel("Conductance [e^2/h]")
        plt.title(f"Conductance (parallel, {size} ranks)")
        plt.tight_layout()
        try:
            plt.show()
        except Exception:
            # Headless environments may raise; silently pass
            pass


if __name__ == "__main__":
    main()
