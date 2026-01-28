"""
k-point grids and high-symmetry paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Dict, Iterable

import numpy as np

from .lattice import MoireLattice





def symmetry_path(lat: MoireLattice, labels: List[str] = ["K", "G", "M", "K"], nseg: int = 20) -> Tuple[np.ndarray, List[int], List[str]]:
    """
    Build a high-symmetry path through the moiré BZ.

    Parameters
    ----------
    labels : list like ["G","K","M","G"]
    nseg : number of points per segment

    Returns
    -------
    k_path : (N,2) cartesian
    dists : cumulative distance along the path
    ticks : indices where new label starts
    ticklabels : the labels
    """
    hs = lat.high_symmetry_points()
    pts = [hs[l] for l in labels]
    k_list = []
    ticks = [0]
    dists = [0.0]
    for i in range(len(pts)-1):
        a, b = pts[i], pts[i+1]
        for t in np.linspace(0, 1, nseg, endpoint=False):
            k_list.append((1-t)*a + t*b)
            if len(k_list)>1:
                d = np.linalg.norm(k_list[-1]-k_list[-2])
                dists.append(dists[-1] + d)
        ticks.append(len(k_list))
    k_list.append(pts[-1])
    k_path = np.array(k_list)
    dists.append(dists[-1] + np.linalg.norm(pts[-1]-k_list[-2]))

    return k_path, dists, ticks, labels


def kpath_dist(k_path: np.ndarray) -> np.ndarray:
    """Cumulative distance along a k path."""
    dk = np.diff(k_path, axis=0)
    seg = np.sqrt((dk**2).sum(axis=1))
    return np.concatenate([[0.0], np.cumsum(seg)])
