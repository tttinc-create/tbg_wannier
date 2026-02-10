"""
Matplotlib plotting helpers.

We keep plotting separate from analysis/computation so you can use the package
headless on clusters and only import matplotlib when needed.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from matplotlib.collections import PolyCollection
import matplotlib.colors as mcolors
from math import ceil, sqrt
from .fourier import compute_real_space_wannier

def plot_band_structure(kdist: np.ndarray, evals: np.ndarray, ticks: List[int],
                         ticklabels: List[str], savepath: str | None = None):
    """
    Plot band structure along a k path.

    Parameters
    ----------
    kdist : (N,) cumulative distance
    evals : (N, nbands)
    """
    fig, ax = plt.subplots()
    ax.plot(kdist, evals, markersize=1)
    for t in ticks:
        ax.axvline(kdist[t], color='k', linestyle='--', linewidth=0.7) 
    ax.set_xticks([kdist[t] for t in ticks])
    ax.set_xticklabels(ticklabels)
    ax.set_xlabel("k-path")
    ax.set_ylabel("Energy (meV)")
    # ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.savefig(savepath)


def plot_density(r1: np.ndarray, r2: np.ndarray, rho: np.ndarray, title: str = ""):
    fig, ax = plt.subplots()
    im = ax.imshow(rho.T, origin="lower", aspect="auto",
                   extent=[r1.min(), r1.max(), r2.min(), r2.max()])
    ax.set_xlabel("r1 (in units of a1)")
    ax.set_ylabel("r2 (in units of a2)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    return fig, ax


# ============================================================
# Plotting U(k) matrix elements on the k-mesh
# ============================================================

import numpy as np
import matplotlib.pyplot as plt


def plot_U_on_kmesh(
    U: np.ndarray,
    k_frac: np.ndarray,
    *,
    mode: str = "band",
    index: int,
    ncols: int | None = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str = "U_plot",
):
    """
    Plot |U(k)| on the k-mesh with a shared colorbar.

    Parameters
    ----------
    U : ndarray, shape (Nk, nb, nw)
        Gauge matrix from Wannier90 or symmetrization.
    k_frac : ndarray, shape (Nk, 2)
        Fractional k-coordinates on a regular mesh (0 <= k < 1).
    mode : {"band", "wann"}
        - "band": fix band index m, plot all Wannier columns n
        - "wann": fix Wannier index n, plot all band rows m
    index : int
        Index of the fixed band or Wannier.
    ncols : int, optional
        Number of subplot columns. Default chooses a square-ish layout.
    cmap : str
        Matplotlib colormap.
    figsize : tuple, optional
        Figure size.
    vmin, vmax : float, optional
        Color scale limits. If None, inferred globally.
    title_prefix : str
        Prefix used in subplot titles.

    Returns
    -------
    fig, axes
    """
    U = np.asarray(U)
    k_frac = np.asarray(k_frac)
    k_frac = k_frac[:, :2]

    Nk, nb, nw = U.shape

    # if k_frac.shape != (Nk, 2):
    #     raise ValueError("k_frac must have shape (Nk, 2)")

    # infer mesh dimensions
    kx_vals = np.unique(k_frac[:, 0])
    ky_vals = np.unique(k_frac[:, 1])
    Nkx, Nky = len(kx_vals), len(ky_vals)

    if Nkx * Nky != Nk:
        raise ValueError("k_frac does not form a regular mesh")

    # determine which slices to plot
    if mode == "band":
        if not (0 <= index < nb):
            raise IndexError("band index out of range")
        data = np.abs(U[:, index, :])**2   # (Nk, nw)
        nplots = nw
        label = lambda j: f"band {index}, wann {j}"
    elif mode == "wann":
        if not (0 <= index < nw):
            raise IndexError("wannier index out of range")
        data = np.abs(U[:, :, index])**2   # (Nk, nb)
        nplots = nb
        label = lambda j: f"band {j}, wann {index}"
    else:
        raise ValueError("mode must be 'band' or 'wann'")

    # reshape data into (nplots, Nkx, Nky)
    data_grid = np.zeros((nplots, Nkx, Nky))
    for i, (kx, ky) in enumerate(k_frac):
        ix = np.where(kx_vals == kx)[0][0]
        iy = np.where(ky_vals == ky)[0][0]
        data_grid[:, ix, iy] = data[i]

    # global color scale
    if vmin is None:
        vmin = data_grid.min()
    if vmax is None:
        vmax = data_grid.max()

    # subplot layout
    if ncols is None:
        ncols = int(np.ceil(np.sqrt(nplots)))
    nrows = int(np.ceil(nplots / ncols))

    if figsize is None:
        figsize = (3.2 * ncols, 3.0 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    im = None
    for j in range(nplots):
        ax = axes[j // ncols, j % ncols]
        im = ax.imshow(
            data_grid[j].T,
            origin="lower",
            extent=(kx_vals[0], kx_vals[-1], ky_vals[0], ky_vals[-1]),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        ax.set_title(f"({label(j)})", fontsize=10)
        ax.set_xlabel("$k_1$")
        ax.set_ylabel("$k_2$")

    # turn off unused axes
    for j in range(nplots, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    # shared colorbar
    cbar = fig.colorbar(
        im,
        ax=axes,
        shrink=0.9,
        pad=0.02,
        aspect=30,
    )
    cbar.set_label('|U|^2')
    fig.savefig(f"{title}_mode{mode}_{index}")


def _get_moire_hexagon():
    # Calculate real space basis dual to reciprocal basis b1, b2
    C3z = np.array([[-.5,-np.sqrt(3)/2],[np.sqrt(3)/2,-.5]])
    a1, a2 = 2*np.pi/3*np.array([np.sqrt(3),1]), 2*np.pi/3*np.array([-np.sqrt(3),1])
    r1 = a1/3 + 2*a2/3
    r2 = 2*a1/3 + a2/3
    r3 = C3z @ r2
    r4 = C3z @ r1
    r5 = C3z @ r3
    r6 = C3z @ r4
    return np.array([r2, r1, r3, r4, r5, r6, r2])

def plot_real_space_wanniers(
    lat,
    wanniers: np.ndarray,
    *,
    beta_idx: int,
    layer: int,
    ncols: int | None = None,
    figsize: tuple[float, float] | None = None,
    cmap: str = "inferno",
    rlim: float = 8.0,
    suptitle: str | None = "",
    savepath: str | None = None,
):
    """
    Plot real-space density profiles of Wannier functions.

    Parameters
    ----------
    lat : MoireLattice
    wanniers : ndarray, shape (Nk, dim, Nwann)
        Wannier coefficients in plane-wave basis.
    alpha_list : iterable of int
        Wannier indices to plot.
    beta_idx : int
        Sublattice index (1 or 2).
    layer : int
        Layer index: +1 (top) or -1 (bottom).
    ncols : int, optional
        Number of columns in subplot grid. If None, chosen automatically.
    figsize : tuple, optional
        Figure size. If None, chosen automatically.
    cmap : str
        Colormap for density |W(r)|^2.
    rlim : float
        Real-space plot range: [-rlim, rlim].
    savepath : str, optional
        If provided, save figure to this path.

    Returns
    -------
    fig, axes
    """
    Nk, _, Nwann = wanniers.shape
    alpha_list = range(Nwann)
    nplots = Nwann

    if ncols is None:
        ncols = int(ceil(sqrt(nplots)))
    nrows = int(ceil(nplots / ncols))

    if figsize is None:
        figsize = (4.0 * ncols, 3.6 * nrows)
    print(f"Plotting {nplots} wanniers in {ncols} columns, {nrows} rows")
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()
    N_L = lat.N_L
    hex_verts = _get_moire_hexagon()
    print("Generating real space wannier plots")
    for ax, alpha in zip(axes, alpha_list):
        w_amp, Rcart = compute_real_space_wannier(
            lat=lat,
            w_coeffs=wanniers,
            n_idx=alpha,
            alpha_idx=beta_idx,
            layer=layer,
        )

        rho = np.abs(w_amp)
        # rho[3*N_L, 3*N_L] = 1
        Rx = Rcart[..., 0]
        Ry = Rcart[..., 1]
        pcm = ax.pcolormesh(
            Rx,
            Ry,
            rho,
            shading="auto",
            cmap=cmap,
        )

        ax.plot(
            hex_verts[:, 0],
            hex_verts[:, 1],
            "c-",
            linewidth=1.5,
        )
        ax.set_title(f"$n = {alpha+1}$", fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlim(-rlim, rlim)
        ax.set_ylim(-rlim, rlim)
        ax.axis("off")
        fig.colorbar(pcm, ax=ax, shrink=0.8, pad=0.02)
    # turn off unused axes
    for ax in axes[nplots:]:
        ax.axis("off")

    layer_str = "+" if layer == 1 else "-"
    fig.suptitle(
        rf"Real-space Wannier density $|\psi_{{\alpha{layer_str},{beta_idx}}}(r)|^2$" + suptitle,
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if savepath is not None:
        fig.savefig(savepath, dpi=200)
        print(f"Saved Wannier plot to {savepath}")

    return fig, axes

def plot_hr_tiles_simple_triangular(
    HR_grid: np.ndarray,
    R_cart_grid: np.ndarray,
    *,
    component: str = "abs",   # "abs" | "real" | "imag"
    scale: str = "log",  # "linear" | "log"
    cmap: str = "inferno",
    vmin: float | None = None,
    vmax: float | None = None,
    eps: float = 1e-2,
    figsize_scale=8,
    tile_scale: float = 0.9,
    title: str | None = None,
    show_lattice_points: bool = False,
    savepath: str | None = None,
):
    """
    Plot H_{mn}(R) as square tiles arranged on a triangular lattice.

    Parameters
    ----------
    HR_grid : (Nkx, Nky, Nw, Nw) complex
        Real-space hoppings from hk_to_hr_fft2 (meta["HR_grid"]).
    R_cart_grid : (Nkx, Nky, 2)
        Cartesian real-space lattice vectors (meta["R_cart_grid"]).
        The lower-left corner of each tile is placed at R.
    m, n : int
        Wannier indices to plot.
    component : which scalar to plot.
    tile_size : float
        Side length of each square tile in real-space units.
    cmap, vmin, vmax : matplotlib options.
    eps : regularizer for logabs.
    show_lattice_points : overlay lattice points if True.
    savepath : save figure if provided.
    """
    HR_grid = np.asarray(HR_grid)
    R_cart_grid = np.asarray(R_cart_grid)

    Nx, Ny, Nw, _ = HR_grid.shape
    
    # extract scalar field
    if component == "abs":
        Z = np.abs(HR_grid)
        clabel = r"$|H_{mn}(R)|$"
    elif component == "real":
        Z = np.real(HR_grid)
        clabel = r"$\Re\,H_{mn}(R)$"
    elif component == "imag":
        Z = np.imag(HR_grid)
        clabel = r"$\Im\,H_{mn}(R)$"
    else:
        raise ValueError("Invalid component.")

    if vmin is None:
        vmin = float(np.min(Z))
    if vmax is None:
        vmax = float(np.max(Z))
    
    figsize = tuple(figsize_scale * np.array([2, 1]))
    fig = plt.figure(figsize=figsize)
    Rx_max = R_cart_grid[:, :, 0].max()
    Rx_min = R_cart_grid[:, :, 0].min()
    Ry_max = R_cart_grid[:, :, 1].max()
    Ry_min = R_cart_grid[:, :, 1].min()
    # print((Rx_max-Rx_min)/(Ry_max-Ry_min))
    tile_sizex = np.abs(R_cart_grid[1,0,0] - R_cart_grid[0,0,0])
    num_x = round((Rx_max - Rx_min) / tile_sizex) + 1
    tile_sizey = np.abs(R_cart_grid[1,0,1] - R_cart_grid[0,0,1])
    num_y = round((Ry_max - Ry_min) / tile_sizey) + 1

    x_big = 0.5*tile_scale*tile_sizex / (Rx_max - Rx_min)
    global_ax = fig.add_axes([x_big, 0, 1, 1])
    
    # Hide the frame box and ticks, but keep the labels
    global_ax.spines['top'].set_visible(False)
    global_ax.spines['right'].set_visible(False)
    global_ax.spines['bottom'].set_visible(True) # Keep bottom line if you want an axis line
    global_ax.spines['left'].set_visible(True)
    global_ax.set_xticks(np.linspace(Rx_min, Rx_max, num=num_x))
    global_ax.set_yticks(np.linspace(Ry_min, Ry_max, num=num_y))
    global_ax.tick_params(axis='both', which='both', labelsize=20)
    global_ax.set_xlabel(r"$R_x$", fontsize=30, loc='right')
    global_ax.set_ylabel(r"$R_y$", fontsize=30, loc='top')
    # draw tiles
    if scale == "linear":
        norm = mcolors.Normalize(vmin, vmax, clip=True)
    elif scale == "log":
        norm = mcolors.SymLogNorm(linthresh=eps, vmin=vmin, vmax=vmax, clip=True)
    for ix in range(Nx):
        for iy in range(Ny):
            Rx, Ry = R_cart_grid[ix, iy, :2]
            axe = fig.add_axes([
                (Rx - Rx_min) / (Rx_max - Rx_min),
                (Ry - Ry_min) / (Ry_max - Ry_min),
                2*tile_scale*tile_sizex / (Rx_max - Rx_min),
                2*tile_scale*tile_sizey / (Ry_max - Ry_min),
            ])
            axe.imshow(
                Z[ix, iy],
                origin="upper",
                aspect="equal",
                # extent=(Rx, Rx + tile_size, Ry, Ry + tile_size),
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
            )
            axe.set_xticks([])
            axe.set_yticks([])
            axe.set_aspect("equal")
            # axe.set_title(f"{Rx:<.1f}, {Ry:<.1f}", fontsize=8)

    # if show_lattice_points:
    #     pts = R_cart_grid[..., :2].reshape(-1, 2)
    #     ax.scatter(pts[:, 0], pts[:, 1], s=10, c="k")

    # cbar = fig.colorbar(
    #     plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax)),
    #     ax=ax,
    #     pad=0.02,
    # )
    # cbar.set_label(clabel)``
    # fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.9, 1.0, 0.2, 0.05])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cbar_ax,
                 orientation="horizontal", label=clabel)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(clabel, fontsize=20)
    # fig.supxlabel(r"$R_x$", y=0.05)
    # fig.supylabel(r"$R_y$", x=0.05)
    # fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")


# def plot_hr_realspace_colormesh(
#     HR_grid: np.ndarray,
#     R_cart_grid: np.ndarray,
#     *,
#     # m: int,
#     # n: int,
#     component: str = "abs",   # "abs" | "real" | "imag"
#     cmap: str = "viridis",
#     vmin: float | None = None,
#     vmax: float | None = None,
#     figsize: tuple[float, float] = (6.5, 5.5),
#     title: str | None = None,
#     show_unitcell: bool = False,
#     savepath: str | None = None,
# ):
#     """
#     Plot hopping H_{mn}(R) on the real-space lattice as a Cartesian colored mesh.

#     Parameters
#     ----------
#     HR_grid : (Nkx, Nky, Nw, Nw) complex
#         Real-space hoppings on the R grid (from hk_to_hr_fft2 meta["HR_grid"]).
#     R_cart_grid : (Nkx, Nky, d) float
#         Cartesian R vectors for each grid point (from hk_to_hr_fft2 meta["R_cart_grid"]).
#         d can be 2 or 3 (only first 2 used).
#     m, n : int
#         Wannier indices to plot H_{mn}(R).
#     component : str
#         Which scalar to plot: "abs", "real", or "imag".
#     shift_center : bool
#         If True, subtract the mean R so the plot is centered around the origin.
#         (Useful if you used fftshift and want (0,0) in the middle visually.)
#     cmap, vmin, vmax : matplotlib settings
#     show_unitcell : bool
#         If True, draw the parallelogram spanned by the two lattice steps.
#     savepath : optional str
#         If provided, saves figure.

#     Returns
#     -------
#     fig, ax
#     """
#     HR_grid = np.asarray(HR_grid)
#     R_cart_grid = np.asarray(R_cart_grid)
#     Nkx, Nky, Nw, Nw2 = HR_grid.shape
#     if Nw != Nw2:
#         raise ValueError("HR_grid must have shape (Nkx, Nky, Nw, Nw).")
#     # if not (0 <= m < Nw and 0 <= n < Nw):
#     #     raise IndexError("m or n out of range for HR_grid.")
#     if R_cart_grid.shape[0] != Nkx or R_cart_grid.shape[1] != Nky:
#         raise ValueError("R_cart_grid must match HR_grid first two dimensions.")

#     Rxy = R_cart_grid[..., :2]  # (Nkx, Nky, 2)
#     Rx = Rxy[:, :, 0]
#     Ry = Rxy[:, :, 1]
#     # Choose scalar field
#     # Hmn = HR_grid[:, :, m, n]
#     Hmn = np.max(np.abs(HR_grid), axis=(2,3))
#     if component == "abs":
#         Z = np.abs(Hmn)
#         clabel = r"$|H_{mn}(\mathbf{R})|$"
#     elif component == "real":
#         Z = np.real(Hmn)
#         clabel = r"$\Re\,H_{mn}(\mathbf{R})$"
#     elif component == "imag":
#         Z = np.imag(Hmn)
#         clabel = r"$\Im\,H_{mn}(\mathbf{R})$"
#     elif component == "logabs":
#         Z = np.log10(np.abs(Hmn) + 1e-12)
#         clabel = r"$\log_{10}|H_{mn}(\mathbf{R})|$"
#     else:
#         raise ValueError("component must be one of {'abs','real','imag'}.")

#     # Build cell polygons: each cell is a parallelogram with corners
#     # (i,j), (i+1,j), (i+1,j+1), (i,j+1)
#     polys = []
#     vals = []
#     for i in range(Nkx - 1):
#         for j in range(Nky - 1):
#             p00 = Rxy[i, j]
#             p10 = Rxy[i + 1, j]
#             p11 = Rxy[i + 1, j + 1]
#             p01 = Rxy[i, j + 1]
#             polys.append([p00, p10, p11, p01])
#             # cell value: average of the four corners
#             vals.append(0.25 * (Z[i, j] + Z[i + 1, j] + Z[i + 1, j + 1] + Z[i, j + 1]))

#     vals = np.asarray(vals)

#     if vmin is None:
#         vmin = float(np.min(vals))
#     if vmax is None:
#         vmax = float(np.max(vals))

#     fig, ax = plt.subplots(figsize=figsize)
#     pcm = ax.pcolormesh(
#             Rx,
#             Ry,
#             Z,
#             shading="auto",
#             cmap=cmap,
#         )

#     # coll = PolyCollection(polys, array=vals, cmap=cmap, edgecolors="none")
#     # coll.set_clim(vmin, vmax)
#     # ax.add_collection(coll)

#     # Autoscale to polygon extents
#     # all_pts = np.vstack([np.asarray(p) for p in polys])
#     # ax.set_xlim(all_pts[:, 0].min(), all_pts[:, 0].max())
#     # ax.set_ylim(all_pts[:, 1].min(), all_pts[:, 1].max())
#     ax.set_aspect("equal", adjustable="box")
#     ax.set_xlabel(r"$R_x$")
#     ax.set_ylabel(r"$R_y$")

#     # if title is None:
#     #     title = rf"{clabel} for (m,n)=({m},{n})"
#     # ax.set_title(title)

#     cbar = fig.colorbar(pcm, ax=ax, pad=0.02, shrink=0.9)
#     cbar.set_label(clabel)

#     # if show_unitcell and Nkx >= 2 and Nky >= 2:
#     #     # Draw cell at origin-ish: use step vectors from grid
#     #     # (these are the real-space lattice steps)
#     #     a1 = Rxy[1, 0] - Rxy[0, 0]
#     #     a2 = Rxy[0, 1] - Rxy[0, 0]
#     #     origin = np.array([0.0, 0.0])
#     #     uc = np.array([origin, origin + a1, origin + a1 + a2, origin + a2, origin])
#     #     ax.plot(uc[:, 0], uc[:, 1], "k-", lw=1.2)

#     fig.tight_layout()
#     if savepath is not None:
#         fig.savefig(savepath, dpi=200)
#     return fig, ax