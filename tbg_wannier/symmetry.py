"""
Symmetry utilities for symmetry-adapted Wannier90 workflows.

This module focuses on *bookkeeping*:
- mapping k-points under symmetry operations on a discrete mesh
- building representation matrices in the plane-wave Bloch basis
- providing hooks to build Wannier-gauge representation matrices (D_wann)

Important
---------
The concise notebook you uploaded contained '...' placeholders in the symmetry
cells, so some representation matrices depend on convention choices. This
module gives *working defaults* (standard continuum-model conventions) and is
written to be easy to customize.

If you have your exact reps in a separate reference, you can replace the
`sublattice_rep_*` helpers while keeping the k-mapping / file I/O intact.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List


import numpy as np
import scipy.sparse as sp
import scipy.linalg

from .config import WannierizationRecipe
from .lattice import MoireLattice
from .utils import rot2, sx, sy, sz, I2, kron


@dataclass(frozen=True)
class SymmetryOperator:
    """Generic symmetry operator acting on the BM plane-wave (Q-lattice) basis.

    Provide:
      - `R` : real 2x2 matrix acting in momentum space, k -> Rk, Q -> RQ.
      - `D_int` : representation on internal indices (valley/sublattice).
      - `D_layer` : representation on layer indices (2x2). Use identity if no layer swap.

    Basis ordering (consistent across this package):
      layer (upper, lower) -> Q-site (lat.L order) -> internal index.

    The operator builds a k-independent plane-wave representation matrix `D_full`.
    On a finite k-mesh you also need the embedding/shift matrix `lat.embedding_matrix(G)`
    to form sewing matrices, same as your notebook.
    """

    name: str
    R: np.ndarray
    D_int: np.ndarray
    D_layer: np.ndarray
    is_antiunitary: bool = False
    tol: float = 1e-8

    def __post_init__(self) -> None:
        R = np.asarray(self.R.real, float)
        if R.shape != (2, 2):
            raise ValueError("R must be a 2x2 real matrix.")
        object.__setattr__(self, "R", R)

        D_int = np.asarray(self.D_int, complex)
        if D_int.ndim != 2 or D_int.shape[0] != D_int.shape[1]:
            raise ValueError("D_int must be a square matrix.")
        object.__setattr__(self, "D_int", D_int)

        D_layer = np.asarray(self.D_layer, complex)
        if D_layer.shape != (2, 2):
            raise ValueError("D_layer must be 2x2.")
        object.__setattr__(self, "D_layer", D_layer)

    # -------------------------
    # Integer action on Q-lattice
    # -------------------------
    # @property
    # def R_frac(self) -> np.ndarray:
    #     b1 = np.array([np.sqrt(3) / 2.0, 3.0 / 2.0])
    #     b2 = np.array([-np.sqrt(3) / 2.0, 3.0 / 2.0])
    #     B = np.column_stack([b1, b2])  # 2x2
    #     Binv = np.linalg.inv(B)
    #     RB = (self.R) @ B
    #     M_float = Binv @ RB
    #     M = np.rint(M_float).astype(int)
    #     if not np.allclose(M_float, M, atol=1e-6):
    #         raise ValueError(
    #             f"{self.name} does not map (b1,b2) to integer combos."
    #         )
    #     return M
    
    def _integer_action_on_b(self, lat: MoireLattice, sign: int) -> tuple[np.ndarray, np.ndarray]:
        """Compute integer matrices (M, t) such that:

            sign * R * (q2 + B n) = q2 + B (t + M n),

        where B = [b1 b2] (2x2), n = (m,n) integers, and M,t are integer.

        sign = +1 : Q' =  + R Q  (same-layer mapping)
        sign = -1 : Q' =  - R Q  (layer-flip mapping required by BM ±Q convention)
        """
        if sign not in (+1, -1):
            raise ValueError("sign must be +1 or -1.")

        B = np.column_stack([lat.b1, lat.b2])  # 2x2
        Binv = np.linalg.inv(B)
        RB = (sign * self.R) @ B
        M_float = Binv @ RB
        M = np.rint(M_float).astype(int)
        if not np.allclose(M_float, M, atol=1e-6):
            raise ValueError(
                f"{self.name}: Q mapping fails for sign={sign}. "
                f"(sign*R) does not map (b1,b2) to integer combos."
            )

        dq = (sign * self.R) @ lat.q2 - lat.q2
        t_float = Binv @ dq
        t = np.rint(t_float).astype(int)
        if not np.allclose(t_float, t, atol=1e-6):
            raise ValueError(
                f"{self.name}: Q mapping fails for sign={sign}. "
                f"(sign*R*q2 - q2) is not an integer combo of (b1,b2)."
            )

        return M, t

    def map_Q_indices(self, lat: MoireLattice, sign: int) -> np.ndarray:
        """mapQ[j] gives the image index of Q-site j under Q -> sign * RQ (mod N_L)."""
        M, t = self._integer_action_on_b(lat=lat, sign=sign)
        mapQ = np.empty(lat.siteN, dtype=int)
        for j in range(lat.siteN):
            n = lat.Lmn[j]  # (m,n)
            n_prime = t + M @ n
            mapQ[j] = lat.index_mn(*n_prime)
        return mapQ

    # -------------------------
    # Full plane-wave rep matrix
    # -------------------------
    def rep_matrix(self, lat: MoireLattice) -> sp.csr_matrix:
        """Full representation matrix in plane-wave basis, with correct handling of layer flips.

        For blocks that map layer_src -> layer_tgt:
          - if layer_tgt == layer_src, use sign = +1
          - if layer_tgt != layer_src, use sign = -1  (BM-required for layer flip)
        """
        n_int = self.D_int.shape[0]
        dim_layer = lat.siteN * n_int
        dim = 2 * dim_layer

        # Precompute Q-maps for the signs we actually need (based on D_layer support)
        signs_needed: set[int] = set()
        for l_src in range(2):
            for l_tgt in range(2):
                if abs(self.D_layer[l_tgt, l_src]) > 1e-14:
                    signs_needed.add(+1 if l_tgt == l_src else -1)

        mapQ_by_sign: dict[int, np.ndarray] = {s: self.map_Q_indices(lat=lat, sign=s) for s in signs_needed}

        D = sp.lil_matrix((dim, dim), dtype=complex)
        for l_src in range(2):
            for l_tgt in range(2):
                c = self.D_layer[l_tgt, l_src]
                if abs(c) < 1e-14:
                    continue

                sign = +1 if l_tgt == l_src else -1
                mapQ = mapQ_by_sign[sign]

                for j in range(lat.siteN):
                    jp = int(mapQ[j])
                    row0 = (l_tgt * lat.siteN + jp) * n_int
                    col0 = (l_src * lat.siteN + j) * n_int
                    D[row0:row0 + n_int, col0:col0 + n_int] = c * self.D_int

        return D.tocsr()

    # -------------------------
    # Mesh mapping and sewing
    # -------------------------
    def map_k_mesh(self, lat: MoireLattice) -> tuple[np.ndarray, np.ndarray]:
        """Return (k_map, G_list) with gk = k_image + G on a uniform mesh."""
        k_frac = lat.k_frac
        Nmesh = (lat.N_k, lat.N_k)
        return find_gk_image_and_G(k_frac, self.R, Nmesh, tol=self.tol)

    def sewing_matrices(
        self,
        lat: MoireLattice, 
        eigvecs: np.ndarray,
    ) -> np.ndarray:
        """Compute band sewing matrices S_g(k).

        S_g(k) = <u_{gk}| R(G)^T D_full |u_k>,
        where G is the integer reciprocal shift that brings gk back onto the chosen mesh.
        """

        k_map, G_list = self.map_k_mesh(lat)
        D_full = self.rep_matrix(lat)

        Nk = eigvecs.shape[0]
        nbands = eigvecs.shape[-1]
        S = np.zeros((Nk, nbands, nbands), dtype=complex)
        for ik in range(Nk):
            ikp = int(k_map[ik])
            G = tuple(int(x) for x in G_list[ik].ravel()[:2])
            Remb = lat.embedding_matrix(G)
            S[ik] = eigvecs[ikp].conj().T @ Remb.T @ D_full @ eigvecs[ik]  # (Nk, nb, nb)
        return S

def repC2zT(lat: MoireLattice, valley: int = +1) -> sp.csr_matrix:
    norb = 2
    dim = 2 * norb * lat.siteN
    d = sp.lil_matrix((dim, dim), dtype=complex)
    mat = sx
    for i in range(lat.siteN):
        d[norb*i:norb*(i+1), norb*i:norb*(i+1)] = mat
        d[norb*(i+lat.siteN):norb*(i+lat.siteN+1), norb*(i+lat.siteN):norb*(i+lat.siteN+1)] = mat
    return d.tocsr()


def repC3z(lat: MoireLattice, valley: int = +1) -> sp.csc_matrix:
    norb = 2
    dim = 2 * norb * lat.siteN
    d = sp.lil_matrix((dim, dim), dtype=complex)
    t_matrix = np.diag(np.exp(1j*valley*2*np.pi/3*np.array([1,-1])))

    for i in range(lat.siteN):
        for j in range(lat.siteN):
            n1, n2 = int(lat.Lmn[j,0]), int(lat.Lmn[j,1])
            rotQ = np.array([-1-n1-n2, n1+1], dtype=int)
            if np.array_equal((lat.Lmn[i] - rotQ) % lat.N_L, np.zeros(2, dtype=int)):
                d[norb*i:norb*(i+1), norb*j:norb*(j+1)] = t_matrix
                d[norb*(i+lat.siteN):norb*(i+lat.siteN+1), norb*(j+lat.siteN):norb*(j+lat.siteN+1)] = t_matrix
    return d.tocsc()


def repC2x(lat: MoireLattice, valley: int = +1) -> sp.csc_matrix:
    norb = 2
    dim = 2 * norb * lat.siteN
    d = sp.lil_matrix((dim, dim), dtype=complex)
    mat = sx

    for i in range(lat.siteN):
        n1, n2 = int(lat.Lmn[i,0]), int(lat.Lmn[i,1])
        rotQ = np.array([n2-1, n1+1], dtype=int)
        for j in range(lat.siteN):
            if np.array_equal((lat.Lmn[j] - rotQ) % lat.N_L, np.zeros(2, dtype=int)):
                d[norb*(j+lat.siteN):norb*(j+lat.siteN+1), norb*i:norb*(i+1)] = mat
                d[norb*j:norb*(j+1), norb*(i+lat.siteN):norb*(i+lat.siteN+1)] = mat
    return d.tocsc()


def repP(lat: MoireLattice, valley: int = +1) -> sp.csc_matrix:
    norb = 2
    dim = 2 * norb * lat.siteN
    d = sp.lil_matrix((dim, dim), dtype=complex)
    mat = I2
    if norb == 4:
        mat = kron(I2, sz)
    for i in range(lat.siteN):
        d[norb*i:norb*(i+1), norb*(i+lat.siteN):norb*(i+lat.siteN+1)] = -mat
        d[norb*(i+lat.siteN):norb*(i+lat.siteN+1), norb*i:norb*(i+1)] = mat
    return d.tocsc()


# ----------------------------------------------------------------------
# k-mesh mapping: g*k = k_image + G
# ----------------------------------------------------------------------

def find_gk_image_and_G(
    k_frac: np.ndarray,
    R_g: np.ndarray,
    Nmesh: Tuple[int, int],
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """    
    k_frac : array, shape (Nk, dim)
        Fractional coords of the k-mesh, with dim=2 or 3.
        For a N1 x N2 mesh in dim=2: k_frac[i] = [n1/N1, n2/N2].
    R_g : array, shape (dim, dim)
        Symmetry matrix acting on cartesian coords: x' = R_g @ x.
    Nmesh : array_like, shape (dim,)
        Number of k-points along each reciprocal direction (N1,N2,(N3)).
    tol : float
        Tolerance for rounding and consistency checks

    Assumes the k-mesh ordering is row-major with:
        for c1 in linspace(0,1,N1): for c2 in linspace(0,1,N2): append

    Then the flat index equals `n1*N2 + n2`.
    """
    k_frac = np.asarray(k_frac, float)
    b1 = np.array([np.sqrt(3) / 2.0, 3.0 / 2.0])
    b2 = np.array([-np.sqrt(3) / 2.0, 3.0 / 2.0])
    B = np.column_stack([b1, b2])  # 2x2
    Binv = np.linalg.inv(B)
    R_g = np.asarray(R_g, float)
    R_frac_float = Binv @ R_g @ B
    R_frac = np.rint(R_frac_float).astype(int)
    if not np.allclose(R_frac, R_frac_float, atol=1e-6):
        raise ValueError(
            f"{R_g} does not map (b1,b2) to integer combos."
        )
    Nmesh = np.asarray(Nmesh, int)
    Nk, dim = k_frac.shape
    assert dim == 2, "This helper is implemented for 2D meshes (as in the notebook)."

    x_prime = (R_frac @ k_frac.T).T

    x_red = x_prime - np.floor(x_prime + tol)
    inds = np.rint(x_red * Nmesh).astype(int) % Nmesh
    x_mesh = inds / Nmesh

    G_frac = x_prime - x_mesh
    G_round = np.rint(G_frac).astype(int)
    if not np.allclose(G_frac, G_round, atol=1e-6):
        max_err = np.max(np.abs(G_frac - G_round))
        raise ValueError(f"G not integer within tolerance; max deviation={max_err}")
    G_frac = G_round

    N1, N2 = int(Nmesh[0]), int(Nmesh[1])
    flat_image = inds[:, 0] * N2 + inds[:, 1]  # == list index in row-major ordering

    return flat_image.astype(int), G_frac.astype(int)
# ============================================================
# Group-aware D_band / D_wann builders (new, clean rewrite)
# ============================================================



def _round_key(z: np.ndarray, scale: float = 1e8) -> Tuple[int, ...]:
    """Hashable key for (small) float/complex arrays via rounding."""
    z = np.asarray(z)
    if np.iscomplexobj(z):
        a = np.rint(np.real(z) * scale).astype(np.int64).ravel()
        b = np.rint(np.imag(z) * scale).astype(np.int64).ravel()
        return tuple(a.tolist() + b.tolist())
    a = np.rint(z * scale).astype(np.int64).ravel()
    return tuple(a.tolist())


def _op_key(op: "SymmetryOperator") -> Tuple[int, ...]:
    """Key for identifying a group element."""
    return _round_key(op.R) + _round_key(op.D_int) + _round_key(op.D_layer)


@dataclass(frozen=True)
class GroupElement:
    """A group element represented by a SymmetryOperator and a word decomposition."""
    name: str
    op: "SymmetryOperator"
    # decomposition: elem = gen * parent (left-multiplication by generator)
    parent: Optional[str] = None
    gen_left: Optional[str] = None
    is_antiunitary: bool = False


class SymmetryGroup:
    """Finite group generated by SymmetryOperator generators via closure.

    Elements are built by left-multiplying existing elements by generators:
        new = gen * old

    This is done so that we can build representations by the clean recursion:
        D_new(k) = D_gen(old·k) @ D_old(k)
    """

    def __init__(self, elements: Dict[str, GroupElement], generators: Dict[str, "SymmetryOperator"], identity: str):
        self.elements = elements
        self.generators = generators
        self.identity = identity

    @staticmethod
    def compose_ops(name: str, a: "SymmetryOperator", b: "SymmetryOperator") -> "SymmetryOperator":
        """Return SymmetryOperator for the product a*b (apply b then a)."""
        R = a.R @ b.R
        if a.is_antiunitary:
            D_int = a.D_int @ b.D_int.conj()
            D_layer = a.D_layer @ b.D_layer.conj()
        else:   
            D_int = a.D_int @ b.D_int
            D_layer = a.D_layer @ b.D_layer
        is_antiunitary = a.is_antiunitary ^ b.is_antiunitary
        return SymmetryOperator(name=name, R=R, D_int=D_int, D_layer=D_layer, is_antiunitary=is_antiunitary, tol=min(a.tol, b.tol))

    @classmethod
    def from_generators(
        cls,
        generators: Dict[str, "SymmetryOperator"],
        *,
        identity_name: str = "E",
        max_elements: int = 256,
    ) -> "SymmetryGroup":
        """Generate full group by closure from generator set."""
        # Identity inferred from sizes of generator reps
        any_gen = next(iter(generators.values()))
        n_int = any_gen.D_int.shape[0]
        E = SymmetryOperator(
            name=identity_name,
            R=np.eye(2, dtype=float),
            D_int=np.eye(n_int, dtype=complex),
            D_layer=np.eye(2, dtype=complex),
            tol=any_gen.tol,
        )

        elements: Dict[str, GroupElement] = {identity_name: GroupElement(identity_name, E, parent=None, gen_left=None, is_antiunitary=False)}
        key_to_name: Dict[Tuple[int, ...], str] = {_op_key(E): identity_name}

        # BFS frontier of element names
        queue: List[str] = [identity_name]

        while queue:
            cur_name = queue.pop(0)
            cur_elem = elements[cur_name]

            for gname, gop in generators.items():
                new_op = cls.compose_ops(f"{gname}*{cur_name}", gop, cur_elem.op)
                k = _op_key(new_op)
                if k in key_to_name:
                    continue

                # register new element
                new_name = f"{gname}*{cur_name}"
                key_to_name[k] = new_name
                elements[new_name] = GroupElement(new_name, new_op, parent=cur_name, gen_left=gname, is_antiunitary=new_op.is_antiunitary)
                queue.append(new_name)

                if len(elements) > max_elements:
                    raise RuntimeError(
                        f"Group size exceeded max_elements={max_elements}. "
                        "Either the set is not closing as expected, or max_elements is too small."
                    )

        return cls(elements=elements, generators=generators, identity=identity_name)
    @property
    def element_names(self) -> List[str]:
        # stable order: BFS-ish based on insertion (dict preserves insertion order)
        return list(self.elements.keys())

    def k_map_for_element(self, elem_name: str, k_frac: np.ndarray, Nmesh: Tuple[int, int]) -> np.ndarray:
        """Return flat indices mapping ik -> i(gk) on the mesh for this element."""
        op = self.elements[elem_name].op
        k_map, _ = find_gk_image_and_G(k_frac, op.R, Nmesh, tol=op.tol)
        return k_map

    def k_maps_all(self, k_frac: np.ndarray, Nmesh: Tuple[int, int]) -> Dict[str, np.ndarray]:
        return {name: self.k_map_for_element(name, k_frac, Nmesh) for name in self.element_names}


def _compose_rep_on_mesh(rep_gen: np.ndarray, rep_parent: np.ndarray, k_map_parent: np.ndarray) -> np.ndarray:
    """Compose reps on a mesh for new = gen * parent:
        D_new(k) = D_gen(parent·k) @ D_parent(k)

    rep_gen:    (Nk, d, d)
    rep_parent: (Nk, d, d)
    k_map_parent: (Nk,) indices for parent·k reduced back to the mesh
    """
    Nk = rep_parent.shape[0]
    out = np.empty_like(rep_parent)
    for ik in range(Nk):
        out[ik] = rep_gen[int(k_map_parent[ik])] @ rep_parent[ik]
    return out


# -------------------------
# Clean D_band (band sewing) builder
# -------------------------

def build_D_band_from_group(
    group: SymmetryGroup,
    lat: "MoireLattice",
    eigvecs: np.ndarray,
    *,
    generator_names: Optional[List[str]] = None,
    norb_internal: Optional[int] = None,
    irr_kpts_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Build D_band (sewing matrices) for ALL group elements by composing generator sewings.

    Returns
    -------
    D_band_all : (Ne, Nk, nb, nb) or (Ne, Nirr, nb, nb)
        Ne = number of group elements in `group.element_names()` order.
        If irr_kpts_indices is provided, axis 1 will be Nirr=len(irr_kpts_indices).
    elem_names : list[str]
        Names in the same order as axis 0 of D_band_all.

    Notes
    -----
    - Computes generator sewing matrices using SymmetryOperator.sewing_matrices()
      (which uses lat.embedding_matrix(G) internally).
    - Builds all other elements by the group recursion D_{g h}(k)=D_g(hk) D_h(k).
    """
    elem_names = group.element_names
    Nk = eigvecs.shape[0]
    nb = eigvecs.shape[-1]
    k_frac = lat.k_frac
    Nmesh = (lat.N_k, lat.N_k)
    if norb_internal is None:
        # infer from operator internal rep size (matches your plane-wave basis internal block)
        any_op = next(iter(group.generators.values()))
        norb_internal = any_op.D_int.shape[0]

    if generator_names is None:
        generator_names = list(group.generators.keys())

    # sewing for identity
    reps: Dict[str, np.ndarray] = {}
    reps[group.identity] = np.tile(np.eye(nb, dtype=complex)[None, :, :], (Nk, 1, 1))

    # sewing for generators (direct)
    for gname in generator_names:
        reps[gname] = group.generators[gname].sewing_matrices(
            lat=lat, eigvecs=eigvecs
        )

    # k-maps for all elements needed for composition
    k_maps = group.k_maps_all(k_frac, Nmesh)

    # build all other elements in insertion order
    for name in elem_names:
        if name in reps:
            continue
        elem = group.elements[name]
        if elem.parent is None:
            continue
        assert elem.gen_left is not None
        gen = elem.gen_left
        parent = elem.parent

        if gen not in reps:
            # generator sewing not computed (shouldn't happen if generator_names includes all)
            reps[gen] = group.generators[gen].sewing_matrices(
                lat=lat, eigvecs=eigvecs, k_frac=k_frac, Nmesh=Nmesh, norb_internal=norb_internal
            )

        reps[name] = _compose_rep_on_mesh(reps[gen], reps[parent], k_maps[parent])

    D_band_all = np.stack([reps[name] for name in elem_names], axis=0)

    if irr_kpts_indices is not None:
        # Slice to keep only irreducible k-points
        # irr_kpts_indices should be 0-based indices into the full (Nk) mesh
        D_band_all = D_band_all[:, irr_kpts_indices, :, :]

    return D_band_all, elem_names


# -------------------------
# Clean D_wann (target Wannier rep) builder
# -------------------------

def build_D_wann_from_group(
    group: SymmetryGroup,
    lat: "MoireLattice",
    D_wann_generators: Dict[str, np.ndarray],
    *,
    generator_names: Optional[List[str]] = None,
    identity_name: Optional[str] = None,
    irr_kpts_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Build D_wann(k) for ALL group elements by composing generator target reps.

    Parameters
    ----------
    D_wann_generators : dict gen_name -> array (Nk, nw, nw)
        The target Wannier-basis representation for each *generator* on your mesh.
        These can come from your EBR construction (block-diagonal over chosen EBRs).
    irr_kpts_indices : array_like, optional
        If provided, the output array will be sliced to include only these k-points.
        (0-based indices).
    """
    elem_names = group.element_names
    k_frac = lat.k_frac
    Nk = k_frac.shape[0]
    Nmesh = (lat.N_k, lat.N_k)
    if identity_name is None:
        identity_name = group.identity

    if generator_names is None:
        generator_names = list(group.generators.keys())

    # infer nwann
    any_gen_mat = next(iter(D_wann_generators.values()))
    nw = any_gen_mat.shape[-1]

    reps: Dict[str, np.ndarray] = {}
    reps[identity_name] = np.tile(np.eye(nw, dtype=complex)[None, :, :], (Nk, 1, 1))

    # ensure generators provided
    for gname in generator_names:
        if gname not in D_wann_generators:
            raise KeyError(f"Missing D_wann generator matrix for '{gname}'")
        reps[gname] = D_wann_generators[gname]

    k_maps = group.k_maps_all(k_frac, Nmesh)

    # build all other elements
    for name in elem_names:
        if name in reps:
            continue
        elem = group.elements[name]
        if elem.parent is None:
            continue
        assert elem.gen_left is not None
        gen = elem.gen_left
        parent = elem.parent

        reps[name] = _compose_rep_on_mesh(reps[gen], reps[parent], k_maps[parent])

    D_wann_all = np.stack([reps[name] for name in elem_names], axis=0)
    
    if irr_kpts_indices is not None:
        # Slice to keep only irreducible k-points
        D_wann_all = D_wann_all[:, irr_kpts_indices, :, :]

    return D_wann_all, elem_names


# -------------------------
# Optional helper: build block-diagonal generator D_wann from your existing EBR machinery
# -------------------------
def mat_DF(choice_EBR: str, k_mesh: np.ndarray, lat: MoireLattice) -> list[dict[str, np.ndarray]]:
    """Target Wannier-basis representation matrices for each EBR choice (ported)."""
    omega_c3z = scipy.linalg.expm(2j*np.pi/3*sz)

    # Location of 2c Wyckoff locations
    r1 = 2*np.pi/3*np.array([-1/np.sqrt(3), 1.0])
    r2 = 2*np.pi/3*np.array([ 1/np.sqrt(3), 1.0])
    lst_Ec = [r1, r1, r2, r2]
    # Location of 3f Wyckoff locations
    u1 = np.pi/3*np.array([ np.sqrt(3), 1.0])
    u2 = np.pi/3*np.array([-np.sqrt(3), 1.0])
    u3 = np.pi/3*np.array([0.0, 2.0])
    lst_3f = [u1, u2, u3]

    one = np.array([[1]], dtype=complex)
    symm_dict = []

    for k in k_mesh:
        k = np.asarray(k, float).ravel()
        if choice_EBR == "A1a":
            symm_dict.append({"E": one, "C2x": one, "C3z": one, "C2zT": one})
        elif choice_EBR == "A2a":
            symm_dict.append({"E": one, "C2x": -1*one, "C3z": one, "C2zT": one})
        elif choice_EBR == "Ea":
            symm_dict.append({"E": np.eye(2, dtype=complex), "C2x": sx, "C3z": omega_c3z, "C2zT": sx})
        elif choice_EBR == "zhida":
            symm_dict.append({"E": np.eye(2, dtype=complex), "C2x": sx, "C3z": omega_c3z, "C2zT": sx})
        elif choice_EBR == "Ec":
            phase_C2x = np.array([[np.exp(1j*np.dot(lat.C2x @ k, ri - lat.C2x @ rj)) for rj in lst_Ec] for ri in lst_Ec], dtype=complex)
            phase_C3z = np.array([[np.exp(1j*np.dot(lat.C3z @ k, ri - lat.C3z @ rj)) for rj in lst_Ec] for ri in lst_Ec], dtype=complex)
            phase_C2zT = np.array([[np.exp(1j*np.dot(k, ri + rj)) for rj in lst_Ec] for ri in lst_Ec], dtype=complex)
            symm_dict.append({
                "E": np.eye(4, dtype=complex),
                "C2x": phase_C2x * kron(I2, sx),
                "C3z": phase_C3z * kron(I2, omega_c3z),
                "C2zT": phase_C2zT * kron(sx, sx),
            })
        elif choice_EBR == "Af":
            phase_C2x = np.array([[np.exp(1j*np.dot(lat.C2x @ k, ri - lat.C2x @ rj)) for rj in lst_3f] for ri in lst_3f], dtype=complex)
            phase_C3z = np.array([[np.exp(1j*np.dot(lat.C3z @ k, ri - lat.C3z @ rj)) for rj in lst_3f] for ri in lst_3f], dtype=complex)
            phase_C2zT = np.array([[np.exp(1j*np.dot(k,  ri + rj)) for rj in lst_3f] for ri in lst_3f], dtype=complex)
            symm_dict.append({
                "E": np.eye(3, dtype=complex),
                "C2x": phase_C2x * np.array([[0,1,0],[1,0,0],[0,0,1]], dtype=complex),
                "C3z": phase_C3z * np.array([[0,0,1],[1,0,0],[0,1,0]], dtype=complex),
                "C2zT": phase_C2zT * np.identity(3)
            })
        elif choice_EBR == "Bf":
            phase_C2x = np.array([[np.exp(1j*np.dot(lat.C2x @ k, ri - lat.C2x @ rj)) for rj in lst_3f] for ri in lst_3f], dtype=complex)
            phase_C3z = np.array([[np.exp(1j*np.dot(lat.C3z @ k, ri - lat.C3z @ rj)) for rj in lst_3f] for ri in lst_3f], dtype=complex)
            phase_C2zT = np.array([[np.exp(1j*np.dot(k,  ri + rj)) for rj in lst_3f] for ri in lst_3f], dtype=complex)
            symm_dict.append({
                "E": np.eye(3, dtype=complex),
                "C2x": -phase_C2x * np.array([[0,1,0],[1,0,0],[0,0,1]], dtype=complex),
                "C3z": phase_C3z * np.array([[0,0,1],[1,0,0],[0,1,0]], dtype=complex),
                "C2zT": phase_C2zT * np.identity(3)
            })
        else:
            raise ValueError(f"Unsupported EBR choice: {choice_EBR}")

    return symm_dict


def build_D_wann_generators_from_EBRs(
    ebr_sequence: List[str],
    lat: "MoireLattice",
    *,
    generators_needed: List[str],
) -> Dict[str, np.ndarray]:
    """Convenience wrapper that uses your existing `mat_DF` to produce
    generator target reps in a single dict suitable for build_D_wann_from_group().

    Returns dict: gen_name -> (Nk, nw, nw)
    """
    # precompute per-ebr dict list
    k_mesh = lat.k_cart
    ebr_data = {ebr: mat_DF(ebr, k_mesh, lat) for ebr in set(ebr_sequence)}

    Nk = k_mesh.shape[0]
    # determine nwann
    nw = sum(ebr_data[ebr][0]["E"].shape[0] for ebr in ebr_sequence)

    out: Dict[str, np.ndarray] = {}
    for g in generators_needed:
        arr = np.zeros((Nk, nw, nw), dtype=complex)
        for ik in range(Nk):
            blocks = []
            for ebr in ebr_sequence:
                blocks.append(ebr_data[ebr][ik][g])
            arr[ik] = scipy.linalg.block_diag(*blocks)
        out[g] = arr
    return out


def build_dmn_maps_trivial_irr(
    group: SymmetryGroup,
    lat: MoireLattice,
    *,
    elem_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build mapping arrays needed for Wannier90 `seedname.dmn` in the simplest case:
    **all k-points are treated as irreducible** (irr_kpts == full k-mesh).

    Returns
    -------
    full_to_irr : (Nk,) int
        1-based mapping from full k index -> irr index. Here it's just 1..Nk.
    irr_kpts : (Nk,) int
        1-based list of irreducible k-point indices. Here it's just 1..Nk.
    sym_kpt_map : (Nk, Ne) int
        1-based indices of the image of each irr k-point under each group element.
        Column order matches `elem_names`.
    elem_names : list[str]
        Group element names corresponding to the symmetry index used in sym_kpt_map.
    """
    if elem_names is None:
        elem_names = group.element_names
    k_frac = lat.k_frac
    Nk = k_frac.shape[0]
    Nmesh = (lat.N_k, lat.N_k)
    full_to_irr = np.arange(1, Nk + 1, dtype=int)
    irr_kpts = np.arange(1, Nk + 1, dtype=int)

    sym_kpt_map = np.zeros((Nk, len(elem_names)), dtype=int)
    for isym, name in enumerate(elem_names):
        k_map = group.k_map_for_element(name, k_frac, Nmesh)  # 0-based
        sym_kpt_map[:, isym] = k_map.astype(int) + 1          # 1-based for Wannier90

    return full_to_irr, irr_kpts, sym_kpt_map, elem_names


def build_dmn_maps_irreducible(
    group: SymmetryGroup,
    lat: MoireLattice,
    *,
    elem_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build mapping arrays needed for Wannier90 `seedname.dmn` using the
    true irreducible wedge of the k-mesh under the group action.
    
    Returns
    -------
    full_to_irr : (Nk,) int
        1-based mapping from full k index -> irr index.
        The value irr = full_to_irr[ik] implies that `canonical_k = irr_kpts[irr-1]`.
        Note: Wannier90 expects full_to_irr[ik] to be the 1-based index 
        within the *irreducible list*, not the full list index of the representative.
        
    irr_kpts : (Nirr,) int
        1-based list of indices of the irreducible k-points in the full mesh.
        
    sym_kpt_map : (Nk, Ne) int
        1-based indices for S*k.
        sym_kpt_map[ik, isym] gives the index of K' = Op[isym] * k[ik] in the FULL list.
        
    elem_names : list[str]
        Group element names corresponding to the symmetry index.
    """
    if elem_names is None:
        elem_names = group.element_names

    k_frac = lat.k_frac
    Nk = k_frac.shape[0]
    Nmesh = (lat.N_k, lat.N_k)
    nsym = len(elem_names)

    # Precompute maps for all symmetries: map[isym][ik] -> ik_prime (0-based)
    # all_k_maps[isym, ik] = image of k[ik] under sym[isym]
    all_k_maps = np.zeros((nsym, Nk), dtype=int)
    for isym, name in enumerate(elem_names):
        all_k_maps[isym] = group.k_map_for_element(name, k_frac, Nmesh)

    # 1. Identify orbits and pick representatives
    # visited[ik] = True if ik has been assigned to an orbit
    visited = np.zeros(Nk, dtype=bool)
    
    # irr_indices_full: 0-based indices in the full mesh of the chosen representatives
    irr_indices_full = []
    
    # mapping from full index -> unique orbit ID (0-based index into irr_indices_full)
    orbit_id = np.empty(Nk, dtype=int)
    orbit_id.fill(-1)

    for ik in range(Nk):
        if visited[ik]:
            continue
            
        # Found a new orbit representative
        curr_orbit_idx = len(irr_indices_full)
        irr_indices_full.append(ik)
        
        # Generate the star of k[ik]
        # star_indices = set(all_k_maps[:, ik])
        # To be robust, we need closure. But all_k_maps contains the full group,
        # so {Op * k} for all Op is the full orbit.
        star_indices = np.unique(all_k_maps[:, ik])
        
        for k_star in star_indices:
            if not visited[k_star]:
                visited[k_star] = True
                orbit_id[k_star] = curr_orbit_idx
            # check consistency? If already visited and assigned to *another* orbit,
            # then groups didn't close or something is wrong.
            elif orbit_id[k_star] != curr_orbit_idx:
                # This could happen if star_indices overlaps an existing orbit
                # implying the previous orbit wasn't complete?
                # With a full group, this shouldn't happen.
                pass

    # 2. Build outputs
    
    # full_to_irr: 1-based index in the *irreducible list*
    # orbit_id is 0-based index in irr_indices_full, so just add 1
    full_to_irr = orbit_id + 1
    
    # irr_kpts: 1-based indices in the *full list*
    irr_kpts = np.array(irr_indices_full, dtype=int) + 1
    
    # sym_kpt_map: map from (ik_irr, isym) -> ik_prime (1-based full index)
    # Transpose all_k_maps to (Nk, nsym), add 1, then slice to keep only irr k-points
    sym_kpt_map_full = all_k_maps.T.astype(int) + 1
    sym_kpt_map = sym_kpt_map_full[irr_indices_full, :]

    return full_to_irr, irr_kpts, sym_kpt_map, elem_names





def build_k_maps(k_frac: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build the k-index maps and G-lists for the symmetry set used in your notebook."""
    r_c3z = np.array([[-.5,-np.sqrt(3)/2],[np.sqrt(3)/2,-.5]])
    r_c2x = np.array([[1,0],[0,-1]])
    Nmesh = (round(np.sqrt(k_frac.shape[0])), round(np.sqrt(k_frac.shape[0])))
    map_C3z, G_C3z = find_gk_image_and_G(k_frac, r_c3z, Nmesh)
    map_C3z2, G_C3z2 = find_gk_image_and_G(k_frac, r_c3z @ r_c3z, Nmesh)
    map_C2x, G_C2x = find_gk_image_and_G(k_frac, r_c2x, Nmesh)
    map_C2xC3z, G_C2xC3z = find_gk_image_and_G(k_frac, r_c2x @ r_c3z, Nmesh)
    map_C2xC3z2, G_C2xC3z2 = find_gk_image_and_G(k_frac, r_c2x @ (r_c3z @ r_c3z), Nmesh)

    Nk = k_frac.shape[0]
    map_E = np.arange(Nk, dtype=int)
    G_E = np.zeros((Nk, 2), dtype=int)


    k_maps = {
        "E": map_E,
        "C3z": map_C3z,
        "C2x": map_C2x,
        "C3z2": map_C3z2,
        "C2xC3z": map_C2xC3z,
        "C3zC2x": map_C2xC3z2,
    }
    G_lists = {
        "E": G_E,
        "C3z": G_C3z,
        "C2x": G_C2x,
        "C3z2": G_C3z2,
        "C2xC3z": G_C2xC3z,
        "C3zC2x": G_C2xC3z2,
    }
    return k_maps, G_lists

def build_rep_matrices(lat: MoireLattice) -> dict[str, sp.csc_matrix]:
    DC3z = repC3z(lat)
    DC2x = repC2x(lat)
    DE        = np.eye(4*lat.siteN, dtype=complex)
    DC3z2     = DC3z @ DC3z
    DC2xC3z   = DC2x @ DC3z
    DC2xC3z2  = DC2x @ DC3z2
    D_orb = {
    "E":        DE,
    "C3z":      DC3z,
    "C2x":      DC2x,
    "C3z2":     DC3z2,
    "C2xC3z":   DC2xC3z,
    "C3zC2x":  DC2xC3z2,
    }
    return D_orb


# # ----------------------------------------------------------------------
# # Sewing matrices and target Wannier reps (your notebook's mat_sewing / mat_DF logic)
# # ----------------------------------------------------------------------

def mat_sewing(
    lat: MoireLattice,
    vec: np.ndarray,
    k_map_g: np.ndarray,
    G_lst: np.ndarray,
    D_g: np.ndarray,
) -> np.ndarray:
    """Build band sewing matrices: M(g,k) = <u_{gk}| R(G)^T D_g |u_k>."""
    Nk = len(k_map_g)
    nbands = vec.shape[-1]
    M = np.zeros((Nk, nbands, nbands), dtype=complex)
    for ik in range(Nk):
        ikp = int(k_map_g[ik])
        G = tuple(G_lst[ik])
        R = lat.embedding_matrix(G)
        M[ik] = vec[ikp].conj().T @ R.T @ D_g @ vec[ik]
    return M

def build_D_band(
    k_mesh: np.ndarray,
    vec: np.ndarray,
    lat: MoireLattice,
    sym_labels: list[str] | None = None,
) -> tuple[np.ndarray, int, list[str]]:

    if sym_labels is None:
        sym_labels = ["E", "C3z", "C2x", "C3z2", "C2xC3z", "C3zC2x"]
    nsym = len(sym_labels)
    Nk = k_mesh.shape[0]
    num_bands = vec.shape[-1]
    k_frac = np.array([lat.frac_coords(k) for k in k_mesh])
    k_maps, G_lists = build_k_maps(k_frac)

    rep_mats = build_rep_matrices(lat=lat)
    D_band = np.zeros((nsym, Nk, num_bands, num_bands), dtype=complex)
    for isym, sym in enumerate(sym_labels):
        if sym == "E":
            D_band[isym] = np.eye(num_bands)[None, :, :]
        else:
            D_band[isym] = mat_sewing(lat=lat, vec=vec, k_map_g=k_maps[sym], G_lst=G_lists[sym], D_g=rep_mats[sym])
    return D_band, num_bands, sym_labels




def build_D_wann(
    ebr_sequence: list[str],
    k_mesh: np.ndarray,
    lat: MoireLattice,
    sym_labels: list[str] | None = None,
) -> tuple[np.ndarray, int, list[str]]:
    """Build block-diagonal `D_wann(isym,ik)` following your notebook (cell 33/36)."""
    if sym_labels is None:
        sym_labels = ["E", "C3z", "C2x", "C3z2", "C2xC3z", "C3zC2x"]

    ebr_data = {ebr: mat_DF(ebr, k_mesh, lat) for ebr in set(ebr_sequence)}
    # determine num_wann from E blocks at first k
    blocks_example = [ebr_data[ebr][0]["E"] for ebr in ebr_sequence]
    num_wann = int(sum(B.shape[0] for B in blocks_example))

    nsym = len(sym_labels)
    Nk = k_mesh.shape[0]
    k_frac = np.array([lat.frac_coords(k) for k in k_mesh])
    D_wann = np.zeros((nsym, Nk, num_wann, num_wann), dtype=complex)
    k_maps, _ = build_k_maps(k_frac)

    for ik in range(Nk):
        ik_C3 = int(k_maps["C3z"][ik])
        ik_C3z2 = int(k_maps["C3z2"][ik])

        for isym, sym in enumerate(sym_labels):
            blocks = []
            for ebr in ebr_sequence:
                base = ebr_data[ebr]
                D_E = base[ik]["E"]
                D_C2 = base[ik].get("C2x", None)
                D_C3 = base[ik].get("C3z", None)
                D_C2zT = base[ik].get("C2zT", None)

                D_C3_at_C3k = base[ik_C3].get("C3z", None)
                D_C2_at_C3k = base[ik_C3].get("C2x", None)
                D_C2_at_C3z2k = base[ik_C3z2].get("C2x", None)

                if sym == "E":
                    blocks.append(D_E)
                elif sym == "C3z":
                    blocks.append(D_C3)
                elif sym == "C3z2":
                    blocks.append(D_C3_at_C3k @ D_C3)
                elif sym == "C2x":
                    blocks.append(D_C2)
                elif sym == "C2zT":
                    blocks.append(D_C2zT)
                elif sym == "C2xC3z":
                    blocks.append(D_C2_at_C3k @ D_C3)
                elif sym == "C3zC2x":
                    D_C3z2_here = D_C3_at_C3k @ D_C3
                    blocks.append(D_C2_at_C3z2k @ D_C3z2_here)
                else:
                    raise ValueError(f"Unknown sym label: {sym}")

            D_wann[isym, ik] = scipy.linalg.block_diag(*blocks)

    return D_wann, num_wann, sym_labels


group_23 = SymmetryGroup.from_generators({
    "C3z": SymmetryOperator("C3z", R=np.array([[-.5,-np.sqrt(3)/2],[np.sqrt(3)/2,-.5]], dtype=float),
                            D_int=np.diag(np.exp(1j*2*np.pi/3*np.array([1, -1]))).astype(complex), 
                            D_layer=I2),
    "C2x": SymmetryOperator("C2x", R=np.array([[1,0],[0,-1]], dtype=float)
                            , D_int=sx, D_layer=sx),
}, max_elements=16)

group_TR = SymmetryGroup.from_generators({"C2zT": SymmetryOperator("C2zT", R=I2, D_int=sx, D_layer=I2, is_antiunitary=True)}, max_elements=16)
group_662 = SymmetryGroup.from_generators({
    "C3z": SymmetryOperator("C3z", R=np.array([[-.5,-np.sqrt(3)/2],[np.sqrt(3)/2,-.5]], dtype=float),
                            D_int=np.diag(np.exp(1j*2*np.pi/3*np.array([1, -1]))).astype(complex), 
                            D_layer=I2),
    "C2x": SymmetryOperator("C2x", R=np.array([[1,0],[0,-1]], dtype=float)
                            , D_int=sx, D_layer=sx),
    "C2zT": SymmetryOperator("C2zT", R=I2, D_int=sx, D_layer=I2, is_antiunitary=True)
}, max_elements=32)



def _as_0based_sym_kpt_map(sym_kpt_map: np.ndarray) -> np.ndarray:
    m = np.asarray(sym_kpt_map, dtype=int)
    if m.ndim != 2:
        raise ValueError("sym_kpt_map must be 2D.")
    # Wannier90 .dmn is typically 1-based
    if m.min() == 1:
        m = m - 1
    return m

def symmetrize_with_P(
        U: np.ndarray,
        eigvecs: np.ndarray,
        lat: MoireLattice,
):
    U = np.asarray(U, complex)
    R_P = np.array([[-1,0],[0,-1]], dtype=float)
    Dint_P = I2
    Dlayer_P = -1j*sy
    opP = SymmetryOperator("P", R=R_P, D_int=Dint_P, D_layer=Dlayer_P)
    Nk, nb, nw = U.shape
    D_band_P = opP.sewing_matrices(lat, eigvecs)
    D_wann_P = -1j*sz
    k_maps_P, _ = opP.map_k_mesh(lat)
    U_cur = U.copy()
    U_new = np.zeros_like(U_cur)
    for ik in range(Nk):
        ikp = k_maps_P[ik]
        U_new[ik]  = scipy.linalg.polar(U_new[ik])[0]
        U_new[ik] = U_cur[ik] + D_band_P[ik].T.conj() @ U_cur[ikp] @ D_wann_P
    U_new /= 2
    return U_new

def symmetrize_u_matrix(
    U: np.ndarray,
    *,
    group: SymmetryGroup,
    recipe: WannierizationRecipe,
    eigvecs: np.ndarray,
    lat: MoireLattice,
    # D_band: np.ndarray,
    # D_wann: np.ndarray,
    # sym_kpt_map: np.ndarray,
    
    n_iter: int = 1,
    enforce_semiunitary: bool = True,
) -> np.ndarray:
    """Symmetrize U(k) by group-averaging with unitary symmetries.

    Parameters
    ----------
    U : (Nk, nb, nw)
        Gauge matrix from Wannier90 u_mat (columns = Wannier functions).
    D_band : (Ng, Nk, nb, nb)
        Band sewing/representation matrices for each symmetry element.
    D_wann : (Ng, Nk, nw, nw)
        Wannier target representation matrices for each symmetry element.
    sym_kpt_map : (Nk, Ng) or (Ng, Nk)
        Index map k -> gk on the mesh (0-based or 1-based accepted).

    Formula
    -------
        U_sym(k) = (1/Ng) * sum_g  D_band(g,k)^† @ U(gk) @ D_wann(g,k)

    Optionally we re-project each k-point matrix back to semi-unitary.
    """
    U = np.asarray(U, complex)
    D_band, _ = build_D_band_from_group(group, lat, eigvecs)
    generators = list(group.generators.keys())
    D_wann_gens = build_D_wann_generators_from_EBRs(
        ebr_sequence=recipe.ebr_sequence,
        lat=lat,
        generators_needed=generators,
    )
    elem_names = group.element_names
    D_wann, _ = build_D_wann_from_group(group, lat, D_wann_gens)
    _, _, sym_kpt_map, _ = build_dmn_maps_trivial_irr(group, lat)
    D_band = np.asarray(D_band, complex)
    D_wann = np.asarray(D_wann, complex)

    if U.ndim != 3:
        raise ValueError("U must have shape (Nk, nb, nw).")
    Nk, nb, nw = U.shape

    if D_band.shape[:2] != D_wann.shape[:2]:
        raise ValueError("D_band and D_wann must share (Ng,Nk) axes.")
    Ng, Nk2 = D_band.shape[0], D_band.shape[1]
    if Nk2 != Nk:
        raise ValueError(f"D_band Nk={Nk2} does not match U Nk={Nk}.")
    if D_band.shape[2:] != (nb, nb):
        raise ValueError(f"D_band last dims must be (nb,nb)=({nb},{nb}).")
    if D_wann.shape[2:] != (nw, nw):
        raise ValueError(f"D_wann last dims must be (nw,nw)=({nw},{nw}).")

    kmap = _as_0based_sym_kpt_map(sym_kpt_map)
    if kmap.shape != (Nk, Ng):
        raise ValueError(f"sym_kpt_map must be (Nk,Ng)=({Nk},{Ng}), got {kmap.shape}")

    U_cur = U.copy()

    # Precompute which elements are antiunitary for separate handling
    elem_names = group.element_names
    is_anti_arr = np.array([bool(getattr(group.elements[name].op, "is_antiunitary", False)) for name in elem_names])

    for _it in range(max(1, int(n_iter))):
        # Accumulator for all k-points (vectorized over k)
        acc = np.zeros_like(U_cur, dtype=complex)  # (Nk, nb, nw)

        # Loop over group elements (Ng typically small) but vectorize over Nk
        for ig, name in enumerate(elem_names):
            if not is_anti_arr[ig]:
                # Unitary symmetry: acc_k += D_band_g(k)^
                #                 @ U[gk] @ D_wann_g(k)
                D_b = D_band[ig]            # (Nk, nb, nb)
                U_g = U_cur[kmap[:, ig]]   # (Nk, nb, nw)
                D_w = D_wann[ig]           # (Nk, nw, nw)
                tmp = np.matmul(np.matmul(D_b.conj().transpose(0, 2, 1), U_g), D_w)
                acc += tmp
            else:
                # Antiunitary: S_k = eigvecs[k].H @ rep_mat @ eigvecs[gk].conj()
                op = group.elements[name].op
                rep_mat = op.rep_matrix(lat)  # (dim, dim) possibly sparse
                # ensure dense ndarray for numpy broadcasting/matmul
                if sp.issparse(rep_mat):
                    rep_mat = rep_mat.toarray()
                else:
                    rep_mat = np.asarray(rep_mat)

                E1 = eigvecs.conj().transpose(0, 2, 1)       # (Nk, nb, dim)
                E2 = eigvecs[kmap[:, ig]].conj()             # (Nk, dim, nb)
                # tmp = E1 @ rep_mat -> (Nk, nb, dim); then tmp @ E2 -> (Nk, nb, nb)
                tmp = np.matmul(E1, rep_mat)
                S = np.matmul(tmp, E2)

                U_g_conj = U_cur[kmap[:, ig]].conj()        # (Nk, nb, nw)
                tmp1 = np.matmul(S, U_g_conj)               # (Nk, nb, nw)

                # D_wann term uses transpose+conj as in original code
                D_w_T_conj = D_wann[ig].transpose(0, 2, 1).conj()  # (Nk, nw, nw)
                tmp2 = np.matmul(tmp1, D_w_T_conj)          # (Nk, nb, nw)
                acc += tmp2

        # Average over group
        acc /= float(Ng)

        if enforce_semiunitary:
            # Project each k-point back to semi-unitary via polar decomposition
            U_new = np.empty_like(U_cur)
            for ik in range(Nk):
                U_new[ik] = scipy.linalg.polar(acc[ik])[0]
            U_cur = U_new
        else:
            U_cur = acc

    return U_cur