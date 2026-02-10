from dataclasses import dataclass
from typing import Callable, List, Tuple
import numpy as np

@dataclass
class DomainDef:
    """Definition of a single domain."""
    name: str
    theta: float
    shape_func: Callable[[Tuple[float, float]], bool]
    seed_point: Tuple[float, float]

class ShapePartitioner:
    """Tools to partition space and generate DomainDef objects."""
    
    @staticmethod
    def voronoi_rectangle(width: float, height: float, 
                          thetas: List[float]) -> List[DomainDef]:
        """
        Partitions a rectangle (centered at 0,0) into N=len(thetas) Voronoi cells.
        Returns list of DomainDef with random seed points inside the box.
        """
        print(thetas)
        n_domains = len(thetas)
        # Generate random seeds within the box
        margin = 0.9  # To avoid seeds being too close to the edge
        seeds = np.random.rand(n_domains, 2)
        seeds[:, 0] = (seeds[:, 0] - 0.5) * width * margin
        seeds[:, 1] = (seeds[:, 1] - 0.5) * height * margin
        
        domains = []
        for i, (seed, theta) in enumerate(zip(seeds, thetas)):
            # Define shape function using closure over 'seeds' and 'i'
            # (Check nearest neighbor)
            def make_func(target_i, all_seeds):
                def func(pos):
                    pos_arr = np.array(pos)
                    # Check if inside box
                    if abs(pos_arr[0]) > width/2 or abs(pos_arr[1]) > height/2:
                        return False
                    # Check voronoi condition
                    dists = np.linalg.norm(all_seeds - pos_arr, axis=1)
                    return np.argmin(dists) == target_i
                return func
            
            domains.append(DomainDef(
                name=f"vor_{i}",
                theta=theta,
                shape_func=make_func(i, seeds),
                seed_point=tuple(seed)
            ))
        return domains

    @staticmethod
    def concentric_rings(radii: List[float], thetas: List[float]) -> List[DomainDef]:
        """
        Creates concentric rings.
        radii: List of outer radii [r1, r2, r3]. 
        thetas: List of angles [t0 (center), t1 (ring1), t2 (ring2)...]
        Length of thetas must be len(radii) - 1
        """
        domains = []
        
        # Rings
        for i, theta in enumerate(thetas):
            r_inner = radii[i]
            # If last ring, go to infinity or reasonable bound? 
            # Let's assume user handles bounds, or we use next radius
            r_outer = radii[i+1]            
            def make_ring(ri, ro):
                return lambda p: ri**2 <= (p[0]**2 + p[1]**2) < ro**2
            
            # Point in ring: (ri + epsilon, 0)
            seed = ((r_inner + r_outer)/2, 0)
            
            domains.append(DomainDef(
                name=f"ring_{i}", theta=theta,
                shape_func=make_ring(r_inner, r_outer),
                seed_point=seed
            ))
            
        return domains