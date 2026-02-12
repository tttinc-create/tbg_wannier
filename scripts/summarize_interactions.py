import os
import re
import argparse
import numpy as np
from tbg_wannier import MoireLattice
from tbg_wannier.interaction import frac_to_cart, lst_s

# Wannier center fractional coordinates (same ordering used when saving)

OUT_DIR = os.path.join(os.getcwd(), 'interaction_full')
PAT = re.compile(r"full_t1_(-?\d+)_(-?\d+)__t2_(-?\d+)_(-?\d+)__t3_(-?\d+)_(-?\d+)\.npz")

def parse_t_from_name(name):
    m = PAT.search(name)
    if not m:
        raise ValueError(f"Filename not in expected format: {name}")
    t1 = (int(m.group(1)), int(m.group(2)))
    t2 = (int(m.group(3)), int(m.group(4)))
    t3 = (int(m.group(5)), int(m.group(6)))
    return t1, t2, t3


def main(sort_by: str = 'V', mode: str = 'max'):
    if not os.path.isdir(OUT_DIR):
        raise SystemExit(f"No interaction directory found at {OUT_DIR}")

    files = sorted([f for f in os.listdir(OUT_DIR) if f.endswith('.npz')])
    if not files:
        raise SystemExit("No .npz files found in interaction_full/")

    # build lattice consistent with the run that created the npz files
    lat = MoireLattice.build(N_L=20, N_k=6)

    if mode == 'max':
        results = []
        for fname in files:
            fpath = os.path.join(OUT_DIR, fname)
            t1, t2, t3 = parse_t_from_name(fname)

            with np.load(fpath, allow_pickle=True) as d:
                if 'V_real' in d:
                    V = d['V_real']
                elif 'V' in d:
                    V = d['V']
                else:
                    # pick the first array in the archive
                    arrs = [k for k in d.files]
                    V = d[arrs[0]]

            absV = np.abs(V)
            idx_flat = int(np.argmax(absV))
            n1, n2, n3, n4 = np.unravel_index(idx_flat, V.shape)
            max_abs = float(absV.flatten()[idx_flat])

            # compute separation vectors and lengths
            t1c = frac_to_cart(lat, t1)
            t2c = frac_to_cart(lat, t2)
            t3c = frac_to_cart(lat, t3)

            s1c = frac_to_cart(lat, lst_s[n1])
            s2c = frac_to_cart(lat, lst_s[n2])
            s3c = frac_to_cart(lat, lst_s[n3])
            s4c = frac_to_cart(lat, lst_s[n4])

            sep1 = t1c + s1c - s2c
            sep2 = t2c + s3c - s4c
            sep3 = t3c + 0.5 * (s1c + s2c - s3c - s4c)

            len1 = float(np.linalg.norm(sep1))
            len2 = float(np.linalg.norm(sep2))
            len3 = float(np.linalg.norm(sep3))
            if True:  # skip very long-range interactions
                results.append((max_abs, t1, t2, t3, int(n1), int(n2), int(n3), int(n4), len1, len2, len3))
                            
        # sort descending by |V| or by total separation length
        if sort_by == 'len':
            results.sort(key=lambda x: (x[8]**2 + x[9]**2 + x[10]**2), reverse=False)
        else:
            results.sort(key=lambda x: x[0], reverse=True)

        out_file = 'interaction_max_ranking_from_npz.txt'
        with open(out_file, 'w') as f:
            f.write('Rank, |V|, t1, t2, t3, n1, n2, n3, n4, len_sep1, len_sep2, len_sep3, filepath\n')
            for rank, r in enumerate(results, 1):
                (abs_v, t1, t2, t3, n1, n2, n3, n4, l1, l2, l3) = r
                f.write(f"{rank}, {abs_v:.6e}, {t1}, {t2}, {t3}, {n1}, {n2}, {n3}, {n4}, {l1:.6f}, {l2:.6f}, {l3:.6f}\n")

        # print top 20
        print('Top 20 results:')
        for r in results[:20]:
            print(f"|V|={r[0]:.6e}, t=({r[1]},{r[2]},{r[3]}), n=({r[4]},{r[5]},{r[6]},{r[7]}), len=({r[8]:.3f},{r[9]:.3f},{r[10]:.3f})")

    elif mode == 'full':
        # Vectorized processing: build a mask for all indices at once and
        # extract nonzero (within range) elements using NumPy (no nested Python loops).
        full_rows = []
        s_cart = np.array([frac_to_cart(lat, s) for s in lst_s])  # (n_s, 2)

        for fname in files:
            fpath = os.path.join(OUT_DIR, fname)
            t1, t2, t3 = parse_t_from_name(fname)

            with np.load(fpath, allow_pickle=True) as d:
                if 'V_real' in d:
                    V = d['V_real']
                elif 'V' in d:
                    V = d['V']
                else:
                    arrs = [k for k in d.files]
                    V = d[arrs[0]]

            # Create index grids for all four band indices: shape = (4, n,n,n,n)
            inds = np.indices(V.shape, dtype=int)
            n1_idx, n2_idx, n3_idx, n4_idx = inds[0], inds[1], inds[2], inds[3]

            # Lookup Wannier center cartesian coordinates for each index
            s1 = s_cart[n1_idx]  # shape (n,n,n,n,2)
            s2 = s_cart[n2_idx]
            s3 = s_cart[n3_idx]
            s4 = s_cart[n4_idx]

            # translations in cartesian
            t1c = frac_to_cart(lat, t1)
            t2c = frac_to_cart(lat, t2)
            t3c = frac_to_cart(lat, t3)

            # compute separations and lengths in a vectorized way
            sep1 = t1c + s1 - s2
            sep2 = t2c + s3 - s4
            sep3 = t3c + 0.5 * (s1 + s2 - s3 - s4)

            len1 = np.sqrt(np.sum(sep1 * sep1, axis=-1))
            len2 = np.sqrt(np.sum(sep2 * sep2, axis=-1))
            len3 = np.sqrt(np.sum(sep3 * sep3, axis=-1))

            # boolean mask: True for elements to KEEP (within cutoff)
            keep_mask = np.maximum(np.maximum(len1, len2), len3) < 4.75
            if not np.any(keep_mask):
                continue

            # apply mask and extract arrays in vectorized fashion
            Vf = V.copy()
            Vf[~keep_mask] = 0

            abs_v = np.abs(Vf)[keep_mask]
            v_real = np.real(Vf)[keep_mask]
            v_imag = np.imag(Vf)[keep_mask]

            n1s = n1_idx[keep_mask].astype(int)
            n2s = n2_idx[keep_mask].astype(int)
            n3s = n3_idx[keep_mask].astype(int)
            n4s = n4_idx[keep_mask].astype(int)

            l1s = len1[keep_mask]
            l2s = len2[keep_mask]
            l3s = len3[keep_mask]

            # t-values expanded into columns
            t1_m, t1_n = int(t1[0]), int(t1[1])
            t2_m, t2_n = int(t2[0]), int(t2[1])
            t3_m, t3_n = int(t3[0]), int(t3[1])

            k = abs_v.size
            cols = np.column_stack([
                abs_v, v_real, v_imag,
                np.full(k, t1_m, dtype=int), np.full(k, t1_n, dtype=int),
                np.full(k, t2_m, dtype=int), np.full(k, t2_n, dtype=int),
                np.full(k, t3_m, dtype=int), np.full(k, t3_n, dtype=int),
                n1s, n2s, n3s, n4s,
                l1s, l2s, l3s,
            ])

            full_rows.append(cols)

        if full_rows:
            all_rows = np.vstack(full_rows)
        else:
            all_rows = np.empty((0, 16))

        # sort descending by |V| (first column)
        if all_rows.shape[0] > 0:
            order = np.argsort(all_rows[:, 0])[::-1]
            all_rows = all_rows[order]

        out_file = 'interaction_full_ranking_from_npz.txt'
        with open(out_file, 'w') as f:
            f.write('Rank, |V|, V_real, V_imag, t1_m, t1_n, t2_m, t2_n, t3_m, t3_n, n1, n2, n3, n4, len_sep1, len_sep2, len_sep3\n')
            for rank, row in enumerate(all_rows, 1):
                (abs_v, v_real, v_imag, t1m, t1n, t2m, t2n, t3m, t3n, n1, n2, n3, n4, l1, l2, l3) = row
                f.write(f"{rank}, {abs_v:.6e}, {v_real:.6e}, {v_imag:.6e}, ({int(t1m)},{int(t1n)}), ({int(t2m)},{int(t2n)}), ({int(t3m)},{int(t3n)}), {int(n1)}, {int(n2)}, {int(n3)}, {int(n4)}, {l1:.6f}, {l2:.6f}, {l3:.6f}\n")

        # print top 20
        print('Top 20 results:')
        for row in all_rows[:20]:
            print(f"|V|={row[0]:.6e}, V_real={row[1]:.6e}, V_imag={row[2]:.6e}, t1=({int(row[3])},{int(row[4])}), t2=({int(row[5])},{int(row[6])}), t3=({int(row[7])},{int(row[8])}), n=({int(row[9])},{int(row[10])},{int(row[11])},{int(row[12])}), len=({row[13]:.3f},{row[14]:.3f},{row[15]:.3f})")

    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'max' or 'full'.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize interaction matrices from npz files.")
    parser.add_argument('--mode', type=str, default='max', choices=['max', 'full'],
                        help="'max': write only max |V| per file (default). 'full': write all V values with real/imag parts.")
    parser.add_argument('--sort-by', type=str, default='V', choices=['V', 'len'],
                        help="Sort by |V| (default) or by separation length. Only used in 'max' mode.")
    args = parser.parse_args()
    
    main(sort_by=args.sort_by, mode=args.mode)
