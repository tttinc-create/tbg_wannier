from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Literal

import numpy as np
import numpy.typing as npt
from .fourier import compute_real_space_wannier
from .lattice import MoireLattice
# Layer = Literal[+1, -1]
# Alpha = Literal[1, 2]
# Valley = Literal[+1, -1]
# A_UC: float = 8.0 * np.pi**2 / (3.0 * np.sqrt(3.0))  # unit cell area


# def deltaK_of_layer(lat: MoireLattice, layer: Layer) -> npt.NDArray[np.float64]:
#     """ΔK_layer = q2 for + layer, and -q3 for - layer."""
#     return lat.q2 if layer == +1 else -lat.q3


# def t_cart(lat: MoireLattice, t_mn: Tuple[int, int]) -> npt.NDArray[np.float64]:
#     """t = m a1 + n a2."""
#     m, n = t_mn
#     return m * lat.a1 + n * lat.a2


# def roll_shift(lat: MoireLattice, t_mn: Tuple[int, int]) -> Tuple[int, int]:
#     """
#     On your real-space grid r = (i a1 + j a2)/N_L, translation by t=m a1+n a2
#     is exactly a roll by (m*N_L, n*N_L).
#     """
#     m, n = t_mn
#     return (m * lat.N_L, n * lat.N_L)


# def Vq_analytic(qabs: npt.NDArray[np.float64], U_xi: float, xi: float) -> npt.NDArray[np.float64]:
#     """
#     V(q) = (π U ξ^2) * tanh(|q| ξ / 2) / (|q| ξ), with q->0 handled by limit 1/2.
#     """
#     x = qabs * xi
#     out = np.empty_like(x, dtype=np.float64)
#     small = x < 1e-12
#     out[small] = 0.5
#     out[~small] = np.tanh(x[~small] / 2.0) / x[~small]
#     return (np.pi * U_xi * xi * xi) * out


# @dataclass(frozen=True)
# class SparseField:
#     """Sparse 2D field on a GxG grid stored as (flat indices, complex values)."""
#     flat: npt.NDArray[np.int64]
#     val: npt.NDArray[np.complex128]


# class DensityDensityIntegrator:
#     """
#     Numerically compute V^{(η,η')}_{n1 n2 n3 n4}(t1,t2,t3) on the FFT grid using V(q).
#     """

#     def __init__(self, lat: MoireLattice, U_xi: float, xi: float, thresh: float = 1e-12):
#         self.lat = lat
#         self.U_xi = float(U_xi)
#         self.xi = float(xi)
#         self.thresh = float(thresh)

#         self.G: int = lat.N_L * lat.N_k
#         self.dA: float = A_UC / (lat.N_L * lat.N_L)  # area per real-space grid point

#         # caches
#         self._W_sparse: Dict[Tuple[int, Layer, Alpha, int], SparseField] = {}
#         self._Vq: Optional[npt.NDArray[np.float64]] = None
#         self._qx: Optional[npt.NDArray[np.float64]] = None
#         self._qy: Optional[npt.NDArray[np.float64]] = None

#     # ---------- q-grid / V(q) cache ----------

#     def _ensure_Vq(self) -> None:
#         """
#         Build the FFT-compatible q-grid and cache V(q).

#         Your real-space supercell spans N_k*a1 and N_k*a2 (because grid has G=N_L*N_k points,
#         spacing a/N_L), hence reciprocal grid spacing is b/N_k.
#         """
#         if self._Vq is not None:
#             return

#         G = self.G
#         Nk = self.lat.N_k

#         # FFT frequencies as integers in numpy ordering
#         f = np.fft.fftfreq(G) * G  # ... 0,1,2,...,-2,-1
#         u, v = np.meshgrid(f, f, indexing="ij")

#         # q(u,v) = (u/Nk) b1 + (v/Nk) b2
#         q = (u[..., None] * self.lat.b1 + v[..., None] * self.lat.b2) / Nk
#         qx = q[..., 0].astype(np.float64)
#         qy = q[..., 1].astype(np.float64)

#         qabs = np.sqrt(qx * qx + qy * qy)

#         self._qx, self._qy = qx, qy
#         self._Vq = Vq_analytic(qabs, self.U_xi, self.xi)

#     # ---------- Wannier sparse cache ----------

#     def wannier_sparse(
#         self,
#         eta: int,
#         layer: Layer,
#         alpha: Alpha,
#         n_idx: int,
#         w_coeffs: npt.NDArray[np.complex128],
#     ) -> SparseField:
#         """
#         Return sparse W^{(η)}_{layer,alpha,n}(r) from compute_real_space_wannier,
#         thresholded by |W|>thresh and cached.
#         """
#         key = (+1, int(layer), int(alpha), int(n_idx))
#         if key not in self._W_sparse:
#             Wp, _ = compute_real_space_wannier(self.lat, w_coeffs, n_idx=n_idx,
#                                             alpha_idx=int(alpha), layer=int(layer))
#             mask = np.abs(Wp) > self.thresh
#             flat = np.flatnonzero(mask).astype(np.int64)
#             print(f"Wannier sparse cache key={key}: {flat.size} / {self.G*self.G} points retained")
#             val = Wp.ravel()[flat].astype(np.complex128)
#             order = np.argsort(flat)
#             self._W_sparse[key] = SparseField(flat=flat[order], val=val[order])
#         sf = self._W_sparse[key]
#         return sf if eta == +1 else SparseField(sf.flat, np.conj(sf.val))

#     # ---------- Sparse utilities ----------

#     def _roll_flat(self, flat: npt.NDArray[np.int64], shift_xy: Tuple[int, int]) -> npt.NDArray[np.int64]:
#         """Roll sparse indices on a GxG torus by (dx,dy)."""
#         dx, dy = shift_xy
#         G = self.G
#         x = flat // G
#         y = flat % G
#         xr = (x + dx) % G
#         yr = (y + dy) % G
#         return xr * G + yr

#     def _sparse_product(
#         self,
#         A: SparseField,
#         shiftA: Tuple[int, int],
#         B: SparseField,
#         shiftB: Tuple[int, int],
#         conjA: bool,
#     ) -> SparseField:
#         """(roll(A,shiftA))^*? * (roll(B,shiftB)) via support intersection."""
#         fa = self._roll_flat(A.flat, shiftA)
#         fb = self._roll_flat(B.flat, shiftB)

#         oa = np.argsort(fa); fa = fa[oa]; va = A.val[oa]
#         ob = np.argsort(fb); fb = fb[ob]; vb = B.val[ob]

#         common, ia, ib = np.intersect1d(fa, fb, return_indices=True)
#         if common.size == 0:
#             return SparseField(common.astype(np.int64), np.zeros(0, dtype=np.complex128))

#         a = np.conj(va[ia]) if conjA else va[ia]
#         return SparseField(common.astype(np.int64), (a * vb[ib]).astype(np.complex128))

#     def _scatter_add(self, terms: list[SparseField]) -> npt.NDArray[np.complex128]:
#         """Accumulate sparse terms into a dense (G,G) array."""
#         out = np.zeros(self.G * self.G, dtype=np.complex128)
#         for sf in terms:
#             np.add.at(out, sf.flat, sf.val)
#         return out.reshape((self.G, self.G))

#     # ---------- Main routine ----------

#     def matrix_element(
#         self,
#         eta: Valley,
#         eta_p: Valley,
#         w_coeffs: npt.NDArray[np.complex128],
#         n1: int, n2: int, n3: int, n4: int,
#         t1_mn: Tuple[int, int],
#         t2_mn: Tuple[int, int],
#         t3_mn: Tuple[int, int],
#         layers: Tuple[Layer, ...] = (+1, -1),
#         alphas: Tuple[Alpha, ...] = (1, 2),
#     ) -> np.complex128:
#         """
#         Compute V^{(η,η')}_{n1 n2 n3 n4}(t1,t2,t3) with:
#           - ΔK_layer phases in real space
#           - convolution using analytic V(q)
#           - discrete measure (dA)^2
#         """
#         self._ensure_Vq()
#         assert self._Vq is not None and self._qx is not None and self._qy is not None

#         sh_t1 = roll_shift(self.lat, t1_mn)
#         sh_t2 = roll_shift(self.lat, t2_mn)

#         t1c = t_cart(self.lat, t1_mn)
#         t2c = t_cart(self.lat, t2_mn)
#         t3c = t_cart(self.lat, t3_mn)

#         A_terms: list[SparseField] = []
#         B_terms: list[SparseField] = []

#         # Build A(r), B(r) from sparse Wannier products
#         for layer in layers:
#             dK = deltaK_of_layer(self.lat, layer)

#             phA = np.exp(1j * float(eta)   * float(dK @ t1c))
#             phB = np.exp(1j * float(eta_p) * float(dK @ t2c))

#             for alpha in alphas:
#                 W1 = self.wannier_sparse(eta,   layer, alpha, n1, w_coeffs)
#                 W2 = self.wannier_sparse(eta,   layer, alpha, n2, w_coeffs)
#                 W3 = self.wannier_sparse(eta_p, layer, alpha, n3, w_coeffs)
#                 W4 = self.wannier_sparse(eta_p, layer, alpha, n4, w_coeffs)

#                 # A(r) += e^{iηΔK·t1} * conj(W1(r-t1)) * W2(r)
#                 prodA = self._sparse_product(W1, sh_t1, W2, (0, 0), conjA=True)
#                 if prodA.flat.size:
#                     A_terms.append(SparseField(prodA.flat, (phA * prodA.val).astype(np.complex128)))

#                 # B(r') += e^{iη'ΔK·t2} * conj(W3(r'-t2)) * W4(r')
#                 prodB = self._sparse_product(W3, sh_t2, W4, (0, 0), conjA=True)
#                 if prodB.flat.size:
#                     B_terms.append(SparseField(prodB.flat, (phB * prodB.val).astype(np.complex128)))

#         if not A_terms or not B_terms:
#             return np.complex128(0.0 + 0.0j)

#         A = self._scatter_add(A_terms)
#         B = self._scatter_add(B_terms)

#         # Convolution using V(q): C(q) = V(q) e^{i q·t3} B(q)
#         phase_t3 = np.exp(1j * (self._qx * t3c[0] + self._qy * t3c[1]))
#         C = np.fft.ifft2(np.fft.fft2(B) * (self._Vq * phase_t3))

#         return np.complex128(np.sum(A * C) * (self.dA ))

Layer = Literal[+1, -1]
Alpha = Literal[1, 2]

A_UC: float = 8.0 * np.pi**2 / (3.0 * np.sqrt(3.0))  # moiré unit cell area


def deltaK_of_layer(lat: MoireLattice, layer: Layer) -> npt.NDArray[np.float64]:
    """ΔK_layer = q2 for + layer, and -q3 for - layer."""
    return lat.q2 if layer == +1 else -lat.q3


def t_cart(lat: MoireLattice, t_mn: Tuple[int, int]) -> npt.NDArray[np.float64]:
    """Convert lattice vector t = m a1 + n a2 to cartesian."""
    m, n = t_mn
    return m * lat.a1 + n * lat.a2


def roll_shift(lat: "MoireLattice", t_mn: Tuple[int, int]) -> Tuple[int, int]:
    """
    On your FFT real-space grid r = (i a1 + j a2)/N_L:
    translation by t = m a1 + n a2 is exactly a roll by (m*N_L, n*N_L).
    """
    m, n = t_mn
    return (m * lat.N_L, n * lat.N_L)


def Vq_analytic(qabs: npt.NDArray[np.float64], U_xi: float, xi: float) -> npt.NDArray[np.float64]:
    """
    Your convention:
      V(q) = ∫ d^2r V(r) e^{-iq·r}
           = (π U_xi ξ^2) * tanh(|q| ξ / 2) / (|q| ξ)

    Handle q->0 with tanh(x/2)/x -> 1/2.
    """
    x = qabs * xi / 2.0
    out = np.empty_like(x, dtype=np.float64)
    small = x < 1e-12
    out[small] = 1
    out[~small] = np.tanh(x[~small]) / x[~small]
    return (np.pi * U_xi * xi * xi) * out


@dataclass
class DenseWannierCache:
    """
    Cache dense real-space Wannier arrays W_{layer,alpha,n}^{(eta)}(r).

    If use_trs=True, only eta=+1 is computed and eta=-1 is returned as conjugate.
    """
    lat: "MoireLattice"
    use_trs: bool = True
    # key: (eta, layer, alpha, n_idx)
    W: Dict[Tuple[int, int, int, int], npt.NDArray[np.complex128]] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.W is None:
            self.W = {}

    def get(
        self,
        eta: int,
        layer: Layer,
        alpha: Alpha,
        n_idx: int,
        w_coeffs_plus: npt.NDArray[np.complex128],
        w_coeffs_minus: Optional[npt.NDArray[np.complex128]] = None,
    ) -> npt.NDArray[np.complex128]:
        """
        Return dense W^{(eta)}_{layer,alpha,n}(r).

        - If use_trs=True: requires only w_coeffs_plus and uses W(-) = conj(W(+)).
        - If use_trs=False: you must provide w_coeffs_minus as well.
        """
        if eta not in (+1, -1):
            raise ValueError("eta must be ±1")

        if self.use_trs:
            # store only eta=+1
            keyp = (+1, int(layer), int(alpha), int(n_idx))
            if keyp not in self.W:
                Wp, _ = compute_real_space_wannier(
                    self.lat, w_coeffs_plus, n_idx=n_idx, alpha_idx=int(alpha), layer=int(layer)
                )
                self.W[keyp] = Wp.astype(np.complex128, copy=False)
            return self.W[keyp] if eta == +1 else np.conj(self.W[keyp])

        # no TRS shortcut: cache each eta separately
        if w_coeffs_minus is None:
            raise ValueError("w_coeffs_minus must be provided when use_trs=False")

        key = (eta, int(layer), int(alpha), int(n_idx))
        if key in self.W:
            return self.W[key]

        coeffs = w_coeffs_plus if eta == +1 else w_coeffs_minus
        Weta, _ = compute_real_space_wannier(
            self.lat, coeffs, n_idx=n_idx, alpha_idx=int(alpha), layer=int(layer)
        )
        self.W[key] = Weta.astype(np.complex128, copy=False)
        return self.W[key]


class DensityDensityDense:
    """
    Dense evaluator of the Wannier density-density matrix element using analytic V(q).

    Convolution step (inner integral) is done as:
        C(r) = ∫ d^2r' V(r-r'+t3) B(r')
             ~ ifft2( fft2(B) * V(q) * exp(+i q·t3) )

    Final outer integral:
        ∫ d^2r A(r) C(r)  ~  ΔA * sum_r A(r) C(r)

    where ΔA = A_uc / N_L^2, and the FFT grid is G×G with G=N_L*N_k.
    """

    def __init__(self, lat: MoireLattice, U_xi: float, xi: float, use_trs: bool = True):
        self.lat = lat
        self.U_xi = float(U_xi)
        self.xi = float(xi)
        self.G = lat.N_L * lat.N_k
        self.dA = A_UC / (lat.N_L * lat.N_L)

        self.cache = DenseWannierCache(lat=lat, use_trs=use_trs)

        self._Vq: Optional[npt.NDArray[np.float64]] = None
        self._qx: Optional[npt.NDArray[np.float64]] = None
        self._qy: Optional[npt.NDArray[np.float64]] = None

    def _ensure_q_grid_and_Vq(self) -> None:
        """Build FFT-compatible physical q-grid and cache V(q) on it."""
        if self._Vq is not None:
            return

        G = self.G
        Nk = self.lat.N_k

        # FFT integer indices in numpy ordering (0.., -..)
        f = np.fft.fftfreq(G) * G  # floats but integer-valued
        u, v = np.meshgrid(f, f, indexing="ij")

        # Physical q vectors for this FFT grid:
        # q = (u/Nk) b1 + (v/Nk) b2
        q = (u[..., None] * self.lat.b1 + v[..., None] * self.lat.b2) / Nk
        qx = q[..., 0].astype(np.float64)
        qy = q[..., 1].astype(np.float64)
        qabs = np.sqrt(qx * qx + qy * qy)

        self._qx, self._qy = qx, qy
        self._Vq = Vq_analytic(qabs, self.U_xi, self.xi)

    def matrix_element(
        self,
        eta: int,
        eta_p: int,
        # Wannier coefficients for eta=+1 (and optionally eta=-1 if use_trs=False)
        w_coeffs_plus: npt.NDArray[np.complex128],
        w_coeffs_minus: Optional[npt.NDArray[np.complex128]],
        n1: int, n2: int, n3: int, n4: int,
        t1_mn: Tuple[int, int],
        t2_mn: Tuple[int, int],
        t3_mn: Tuple[int, int],
        layers: Tuple[Layer, ...] = (+1, -1),
        alphas: Tuple[Alpha, ...] = (1, 2),
    ) -> np.complex128:
        """
        Compute V^{(η,η')}_{n1 n2 n3 n4}(t1,t2,t3) with dense fields.
        """
        self._ensure_q_grid_and_Vq()
        assert self._Vq is not None and self._qx is not None and self._qy is not None

        # rolls implementing r -> r - t
        sh_t1 = roll_shift(self.lat, t1_mn)
        sh_t2 = roll_shift(self.lat, t2_mn)

        # cartesian translations for phase factors
        t1c = t_cart(self.lat, t1_mn)
        t2c = t_cart(self.lat, t2_mn)
        t3c = t_cart(self.lat, t3_mn)

        # Build A(r), B(r) densely
        A = np.zeros((self.G, self.G), dtype=np.complex128)
        B = np.zeros((self.G, self.G), dtype=np.complex128)

        for layer in layers:
            dK = deltaK_of_layer(self.lat, layer)

            # e^{i η ΔK·t1}, e^{i η' ΔK·t2}
            phA = np.exp(1j * float(eta)   * float(dK @ t1c))
            phB = np.exp(1j * float(eta_p) * float(dK @ t2c))

            for alpha in alphas:
                W1 = self.cache.get(eta,   layer, alpha, n1, w_coeffs_plus, w_coeffs_minus)
                W2 = self.cache.get(eta,   layer, alpha, n2, w_coeffs_plus, w_coeffs_minus)
                W3 = self.cache.get(eta_p, layer, alpha, n3, w_coeffs_plus, w_coeffs_minus)
                W4 = self.cache.get(eta_p, layer, alpha, n4, w_coeffs_plus, w_coeffs_minus)

                # W(r - t) implemented by roll(+shift): new[i]=old[i-shift]
                W1_shift = np.roll(W1, shift=sh_t1, axis=(0, 1))
                W3_shift = np.roll(W3, shift=sh_t2, axis=(0, 1))

                A += phA * np.conj(W1_shift) * W2
                B += phB * np.conj(W3_shift) * W4

        # Convolution using analytic V(q):
        # C(q) = V(q) * e^{+i q·t3} * B(q)
        phase_t3 = np.exp(1j * (self._qx * t3c[0] + self._qy * t3c[1]))
        C = np.fft.ifft2(np.fft.fft2(B) * (self._Vq * phase_t3))

        # Outer integral: ∫ d^2r A(r) C(r)  ~  ΔA * Σ_r A*C
        return np.complex128(np.sum(A * C) * self.dA)
