from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Literal

import numpy as np
import numpy.typing as npt
from .fourier import compute_real_space_wannier
from .lattice import MoireLattice

Layer = Literal[+1, -1]
Alpha = Literal[1, 2]

A_UC: float = 8.0 * np.pi**2 / (3.0 * np.sqrt(3.0))  # moiré unit cell area
lst_s = [(0,0), (0,0), (0,0), (1/2, 0), (0, 1/2), (1/2, 1/2), (1/3, 2/3), (1/3, 2/3), (2/3, 1/3), (2/3, 1/3)]


def deltaK_of_layer(lat: MoireLattice, layer: Layer) -> npt.NDArray[np.float64]:
    """ΔK_layer = q2 for + layer, and -q3 for - layer."""
    return lat.q2 if layer == +1 else -lat.q3


def frac_to_cart(lat: MoireLattice, t_mn: Tuple[int, int]) -> npt.NDArray[np.float64]:
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

def filter_V_by_separation(
    lat: MoireLattice,
    V: npt.NDArray[np.complex128],
    t1: Tuple[int, int],
    t2: Tuple[int, int],
    t3: Tuple[int, int],
    lst_s_local: Optional[list] = None,
    cutoff: float = 4.75,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.bool_]]:
    """
    Vectorized filter for an interaction tensor V[n1,n2,n3,n4].

    Elements for which max(|sep1|, |sep2|, |sep3|) >= cutoff are set to zero.

    Returns (V_filtered, keep_mask) where keep_mask has the same shape as V
    and is True where the element was kept.
    """
    if lst_s_local is None:
        lst_s_local = lst_s

    # precompute Wannier center positions (n_s, 2)
    s_cart = np.asarray([frac_to_cart(lat, s) for s in lst_s_local], dtype=np.float64)

    # build index grids for all four band indices
    inds = np.indices(V.shape, dtype=int)
    n1_idx, n2_idx, n3_idx, n4_idx = inds[0], inds[1], inds[2], inds[3]

    # broadcast Wannier center coordinates to full V shape
    s1 = s_cart[n1_idx]  # shape (..., 2)
    s2 = s_cart[n2_idx]
    s3 = s_cart[n3_idx]
    s4 = s_cart[n4_idx]

    # translations in cartesian
    t1c = frac_to_cart(lat, t1)
    t2c = frac_to_cart(lat, t2)
    t3c = frac_to_cart(lat, t3)

    # vectorized separations
    sep1 = t1c + s1 - s2
    sep2 = t2c + s3 - s4
    sep3 = t3c + 0.5 * (s1 + s2 - s3 - s4)

    len1 = np.sqrt(np.sum(sep1 * sep1, axis=-1))
    len2 = np.sqrt(np.sum(sep2 * sep2, axis=-1))
    len3 = np.sqrt(np.sum(sep3 * sep3, axis=-1))

    keep_mask = np.maximum(np.maximum(len1, len2), len3) < float(cutoff)

    Vf = V.copy()
    Vf[~keep_mask] = 0

    return Vf, keep_mask



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
    ) -> npt.NDArray[np.complex128]:
        """
        Return dense W^{(eta)}_{layer,alpha,n}(r).

        - If use_trs=True: requires only w_coeffs_plus and uses W(-) = conj(W(+)).
        - If use_trs=False: you must provide w_coeffs_minus as well.
        """
        if eta not in (+1, -1):
            raise ValueError("eta must be ±1")

        # store only eta=+1
        keyp = (+1, int(layer), int(alpha), int(n_idx))
        if keyp not in self.W:
            Wp, _ = compute_real_space_wannier(
                self.lat, w_coeffs_plus, n_idx=n_idx, alpha_idx=int(alpha), layer=int(layer)
            )
            self.W[keyp] = Wp.astype(np.complex128, copy=False)
        return self.W[keyp] if eta == +1 else np.conj(self.W[keyp])

    def precompute_all(
        self,
        w_coeffs_plus: npt.NDArray[np.complex128],
        n_max: int = 10,
        layers: Tuple[Layer, ...] = (+1, -1),
        alphas: Tuple[Alpha, ...] = (1, 2),
    ) -> None:
        """Precompute and cache dense real-space Wannier arrays for all (layer,alpha,n).

        After this call, `self.W_all[(layer,alpha)]` will contain an array of shape
        (n_max, G, G) with dtype complex128.
        """
        for layer in layers:
            for alpha in alphas:
                key = (int(layer), int(alpha))
                if key in getattr(self, 'W_all', {}):
                    continue
                arrs = []
                for n in range(n_max):
                    Wp, _ = compute_real_space_wannier(
                        self.lat, w_coeffs_plus, n_idx=n, alpha_idx=int(alpha), layer=int(layer)
                    )
                    arrs.append(Wp.astype(np.complex128, copy=False))
                stacked = np.stack(arrs, axis=0)
                if not hasattr(self, 'W_all'):
                    self.W_all = {}
                self.W_all[key] = stacked



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

    def __init__(self, lat: MoireLattice, U_xi: float, xi: float):
        self.lat = lat
        self.U_xi = float(U_xi)
        self.xi = float(xi)
        self.G = lat.N_L * lat.N_k
        self.dA = A_UC / (lat.N_L * lat.N_L)

        self.cache = DenseWannierCache(lat=lat)

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
        t1c = frac_to_cart(self.lat, t1_mn)
        t2c = frac_to_cart(self.lat, t2_mn)
        t3c = frac_to_cart(self.lat, t3_mn)

        # Build A(r), B(r) densely
        A = np.zeros((self.G, self.G), dtype=np.complex128)
        B = np.zeros((self.G, self.G), dtype=np.complex128)

        for layer in layers:
            dK = deltaK_of_layer(self.lat, layer)

            # e^{i η ΔK·t1}, e^{i η' ΔK·t2}
            phA = np.exp(1j * float(eta)   * float(dK @ t1c))
            phB = np.exp(1j * float(eta_p) * float(dK @ t2c))

            for alpha in alphas:
                W1 = self.cache.get(eta,   layer, alpha, n1, w_coeffs_plus)
                W2 = self.cache.get(eta,   layer, alpha, n2, w_coeffs_plus)
                W3 = self.cache.get(eta_p, layer, alpha, n3, w_coeffs_plus)
                W4 = self.cache.get(eta_p, layer, alpha, n4, w_coeffs_plus)

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

    def matrix_elements_batch(
        self,
        eta: int,
        eta_p: int,
        w_coeffs_plus: npt.NDArray[np.complex128],
        t1_mn: Tuple[int, int],
        t2_mn: Tuple[int, int],
        t3_mn: Tuple[int, int],
        n_max: int = 10,
        layers: Tuple[Layer, ...] = (+1, -1),
        alphas: Tuple[Alpha, ...] = (1, 2),
    ) -> npt.NDArray[np.complex128]:
        """Compute all V_{n1 n2 n3 n4}(t1,t2,t3) at once.

        Returns an array of shape (n_max, n_max, n_max, n_max) (complex128).
        This method vectorizes over the band indices using numpy broadcasting and FFTs.
        """
        # ensure q-grid
        self._ensure_q_grid_and_Vq()
        assert self._Vq is not None and self._qx is not None and self._qy is not None

        # Precompute all W arrays if not present
        if not hasattr(self.cache, 'W_all'):
            self.cache.precompute_all(w_coeffs_plus, n_max=n_max, layers=layers, alphas=alphas)

        # shifts and cartesian translations
        sh_t1 = roll_shift(self.lat, t1_mn)
        sh_t2 = roll_shift(self.lat, t2_mn)
        t1c = frac_to_cart(self.lat, t1_mn)
        t2c = frac_to_cart(self.lat, t2_mn)
        t3c = frac_to_cart(self.lat, t3_mn)

        G = self.G
        N = n_max

        # Accumulate A(n1,n2,r) and B(n3,n4,r) as arrays of shape (N,N,G,G)
        A = np.zeros((N, N, G, G), dtype=np.complex128)
        B = np.zeros((N, N, G, G), dtype=np.complex128)

        for layer in layers:
            dK = deltaK_of_layer(self.lat, layer)
            phA = np.exp(1j * float(eta) * float(dK @ t1c))
            phB = np.exp(1j * float(eta_p) * float(dK @ t2c))

            for alpha in alphas:
                W_all = self.cache.W_all[(int(layer), int(alpha))]  # shape (N,G,G)

                # apply spatial rolls: W_shift[n] = W[n] rolled by sh
                W1_shift = np.roll(W_all, shift=sh_t1, axis=(1, 2))
                W3_shift = np.roll(W_all, shift=sh_t2, axis=(1, 2))

                # compute outer products over band indices using broadcasting
                # conj(W1_shift)[:,None,:,:] * W_all[None,:,:,:] -> (N,N,G,G)
                A_term = phA * (np.conjugate(W1_shift)[:, None, :, :] * W_all[None, :, :, :])
                B_term = phB * (np.conjugate(W3_shift)[:, None, :, :] * W_all[None, :, :, :])

                A += A_term
                B += B_term

        # Convolution: for each (n3,n4) compute C = ifft2( fft2(B) * Vq * exp(i q·t3) )
        phase_t3 = np.exp(1j * (self._qx * t3c[0] + self._qy * t3c[1]))

        # FFT over last two axes
        B_fft = np.fft.fft2(B, axes=(2, 3))
        C_fft = B_fft * (self._Vq[None, None, :, :] * phase_t3[None, None, :, :])
        C = np.fft.ifft2(C_fft, axes=(2, 3))

        # Now compute V[n1,n2,n3,n4] = dA * sum_r A[n1,n2,r] * C[n3,n4,r]
        # reshape to (N,N,G*G)
        A_flat = A.reshape((N, N, G * G))
        C_flat = C.reshape((N, N, G * G))

        # tensordot over the spatial index to get shape (N,N,N,N)
        V = np.tensordot(A_flat, C_flat, axes=([2], [2]))
        V = V.astype(np.complex128, copy=False)
        V *= self.dA

        return V
