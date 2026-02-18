"""
Simulation of Disordered Twist-Angle Domains in TBG using Kwant.

This script stitches together domains with different twist angles by using
Fourier interpolation of the Wannier Hamiltonian. It handles the lattice
mismatch by treating site positions in continuous real space and filtering
hoppings based on the physical separation of Wannier centers.

Key Components:
1. InterpolatedHamiltonian: Manages upsampling and creates a lookup table
   for efficient retrieval of H(d) for arbitrary distance vectors d.
2. ScaledLattice: Helper to handle physical unit conversion (nm) from model params.
3. make_disordered_system: Builds the Kwant system with domain logic.
"""

import numpy as np
import kwant
import tinyarray as ta
from scipy.spatial import cKDTree

from .interpolation import ModelManager
from .domains import DomainDef
# --- Helper Class for Scaled Lattice ---
# --- Kwant System Construction ---

def find_valid_start(lat, shape_func, seed, search_n=10):
    """
    Robustly finds a valid lattice site inside the shape near the seed.
    This fixes 'No sites close to...' errors when seed is near boundary.
    """
    # 1. Try the closest lattice point to the geometric seed
    closest_pos = lat.n_closest(seed, search_n)
    attempts = 0
    for pos in closest_pos:
        attempts += 1
        if shape_func(pos):
            print(f"  - Found valid seed in {attempts} attempts.")
            return pos
    print(f"- Warning: Could not find valid site near seed in {attempts} attempts.")
    return seed

def attach_square_leads(syst: kwant.Builder, width: float, height: float,
                        lead_a: float = 10.0, lead_t: float = 100.0,
                        coupling_t: float = 50.0, cutoff: float = 15.0,
                        verbose: bool = False):
    """
    Attaches square lattice leads to the top and bottom of the system.
    Handles lattice mismatch by adding a 'connector' buffer region.
    
    Args:
        syst: The populated finite system builder.
        width: Width of the scattering region.
        height: Height of the scattering region.
        lead_a: Lattice constant of the square lead (Angstrom).
        lead_t: Hopping parameter for the lead (meV).
        coupling_t: Tunneling hopping between TBG and Lead (meV).
        cutoff: Distance cutoff for coupling hoppings.(Angstrom)
    """
    if verbose:
        print("\n--- Attaching Square Leads ---")
    
    # 1. Define Lead Lattice (Square, 10 orbitals to match TBG dimensionality)
    lat_lead = kwant.lattice.square(a=lead_a, norbs=10, name='Lead')
    
    # Lead Hamiltonian (Diagonal hopping for simple metal)
    lead_onsite = ta.array(np.zeros((10, 10)))
    lead_hopping = ta.array(-lead_t * np.eye(10))
    
    # 2. Add "Connector" Sites to the Finite System
    # We add a strip of square lattice sites just overlapping the domain edges
    # Thickness of buffer
    buff = 2.0 * lead_a 
    
    def shape_top(pos):
        x, y = pos
        return (-width/2 <= x <= width/2) and (height/2 - buff <= y <= height/2 )

    def shape_bot(pos):
        x, y = pos
        return (-width/2 <= x <= width/2) and (-height/2<= y <= -height/2 + buff)
    
    # Add buffer sites
    syst[lat_lead.shape(shape_top, (0, height/2 - buff/2))] = lead_onsite
    syst[lat_lead.shape(shape_bot, (0, -height/2 + buff/2))] = lead_onsite
    
    # Add internal lead hoppings within the buffer
    syst[lat_lead.neighbors()] = lead_hopping
    
    if verbose:
        print("  > Lead buffer sites added.")

    # 3. Create Couplings (Tunneling)
    # We need to connect the new 'Lead' sites to the existing 'TBG' sites
    # Use KDTree for efficient neighbor finding
    
    all_sites = list(syst.sites())
    
    # Separate by family
    sites_lead = [s for s in all_sites if s.family == lat_lead]
    sites_tbg = [s for s in all_sites if s.family != lat_lead]
    
    if not sites_lead or not sites_tbg:
        if verbose:
            print("  > Warning: Could not find sites for coupling.")
        return

    pos_lead = np.array([s.pos for s in sites_lead])
    pos_tbg = np.array([s.pos for s in sites_tbg])
    
    tree_lead = cKDTree(pos_lead)
    tree_tbg = cKDTree(pos_tbg)
    mat_coupling = coupling_t * np.eye(10)
    # Find neighbors within cutoff
    if verbose:
        print(f"  > Computing couplings (exponential decay length={cutoff} Angstrom)...")
    count = 0
    results = tree_lead.query_ball_tree(tree_tbg, r=4*cutoff)
    # Coupling Matrix (Simple scalar tunneling * Identity)
    
    for i_lead, neighbors in enumerate(results):
        site_lead = sites_lead[i_lead]
        for i_tbg in neighbors:
            site_tbg = sites_tbg[i_tbg]
            syst[site_lead, site_tbg] = ta.array(mat_coupling * np.exp(-np.linalg.norm(pos_lead[i_lead] - pos_tbg[i_tbg])/cutoff))  # Optional: exponential decay with distance
            count += 1

    # for i_lead, pos in enumerate(pos_lead):
    #     dd, ii = tree_tbg.query(pos, k=6, distance_upper_bound=4*cutoff)
    #     for dist, i_tbg in zip(dd, ii):
    #         syst[sites_lead[i_lead], sites_tbg[i_tbg]] = ta.array(mat_coupling * np.exp(-dist/cutoff))  # Optional: exponential decay with distance
    #         count += 1

            
    if verbose:
        print(f"  > Added {count} coupling hoppings.")

    # 4. Construct and Attach Infinite Leads
    # Top Lead (Translational Symmetry +y)
    sym_top = kwant.TranslationalSymmetry((0, lead_a))
    lead_top = kwant.Builder(sym_top)
    
    def lead_shape(pos):
        x, y = pos
        return -width/2 <= x <= width/2

    lead_top[lat_lead.shape(lead_shape, (0, 0))] = lead_onsite
    lead_top[lat_lead.neighbors()] = lead_hopping
    
    # Bottom Lead (Translational Symmetry -y)
    sym_bot = kwant.TranslationalSymmetry((0, -lead_a))
    lead_bot = kwant.Builder(sym_bot)
    lead_bot[lat_lead.shape(lead_shape, (0, 0))] = lead_onsite
    lead_bot[lat_lead.neighbors()] = lead_hopping
    
    # Attach
    syst.attach_lead(lead_top)
    syst.attach_lead(lead_bot)
    if verbose:
        print("  > Leads attached successfully.")


def attach_bulk_leads_matching_domain(syst: kwant.Builder, lat, onsite, hoppings,
                                     domain_sites_positions, lead_trans=(0, 1),
                                     extend: float = 1.0, verbose: bool = False):
    """
    Attach two infinite leads (top and bottom) that use the same onsite
    energy and lattice hoppings as the provided bulk domain.

    Args:
        syst: The populated finite system builder.
        lat: The Kwant lattice associated with the domain (returned by kwant.lattice.general).
        onsite: Onsite Hamiltonian matrix (numpy / tinyarray) for each site.
        hoppings: Iterable of (dx, dy, mat) lattice hoppings as returned by the model.
        domain_sites_positions: Array-like of site positions (Nx2) for computing cross-section.
        lead_trans: Translation vector for lead periodicity (tuple-like, e.g. model.a2).
        extend: Padding added to the cross-section in the x-direction (same units as positions).
        verbose: Print progress if True.
    """
    if verbose:
        print("\n--- Attaching Bulk-Matching Leads ---")
    
    x_min, x_max = float(domain_sites_positions[:, 0].min()), float(domain_sites_positions[:, 0].max())
    pad = float(extend)

    def lead_shape(p):
        x, y = p
        return (x_min - pad) <= x <= (x_max + pad)

    # Top lead: translational symmetry along lead_trans
    sym_top = kwant.TranslationalSymmetry(tuple(lead_trans))
    lead_top = kwant.Builder(sym_top)
    start_top = find_valid_start(lat, lead_shape, (0, 0))
    lead_top[lat.shape(lead_shape, start_top)] = ta.array(onsite)
    for dx, dy, mat in hoppings:
        lead_top[kwant.builder.HoppingKind((dx, dy), lat, lat)] = ta.array(mat)

    # Bottom lead: opposite translation
    # neg_trans = tuple((-np.array(lead_trans)).tolist())
    # sym_bot = kwant.TranslationalSymmetry(neg_trans)
    # lead_bot = kwant.Builder(sym_bot)
    # lead_bot[lat.shape(lead_shape, (0, 0))] = ta.array(onsite)
    # for dx, dy, mat in hoppings:
    #     lead_bot[kwant.builder.HoppingKind((dx, dy), lat, lat)] = ta.array(mat)

    syst.attach_lead(lead_top)
    syst.attach_lead(lead_top.reversed())

    if verbose:
        print("  > Bulk-matching leads attached.")


def build_system(domains: list[DomainDef], trial_wann: np.ndarray = None, cutoff_Ang: float = 400.0,
                  verbose: bool =True, sample_thetas: list[float] = None, tbg_leads: bool = False) -> kwant.Builder:
    """
    Generic builder for N-domain systems.
    """
    print(f"\n=== Building System with {len(domains)} Domains ===")
    
    manager = ModelManager(trial_wann, cutoff_Ang)
    # If a trial_wann is provided and no theta interpolator exists yet,
    # build a default ThetaInterpolator using the domain angles.
    # Interpolation grid: multiples of 0.01 degrees covering domain range.
    if sample_thetas is None and len(domains) > 0:
        thetas = np.array([d.theta for d in domains], dtype=float)
        tmin = float(np.floor(thetas.min() * 100) / 100.0)
        tmax = float(np.ceil(thetas.max() * 100) / 100.0)
        # include endpoints, step 0.01
        sample_thetas = np.round(np.arange(tmin, tmax + 1e-8, 0.01), 6)
    if verbose:
        print(f"  > Creating ThetaInterpolator with samples {sample_thetas[0]:.2f}..{sample_thetas[-1]:.2f} ({len(sample_thetas)} points)")
    manager.create_theta_interpolator(sample_thetas, cache_dir="cache/eigensystems")
    syst = kwant.Builder()
    norbs = 10  # Assuming 10 orbitals from the Wannier model
    # Store domain info for interface loop
    # structure: { domain_index: { 'sites': [], 'lattice': ... } }
    domain_data = {}
    
    # --- 1. Create Sites & Bulk Hoppings ---
    for i, dom in enumerate(domains):
        if verbose:
            print(f"Processing Domain {i}: {dom.name} (Theta={dom.theta}°)")
        
        # Get Model (upscale=1 for bulk lattice)
        model = manager.get_model(dom.theta, upscale=1)
        
        # Create Lattice
        lat = kwant.lattice.general([model.a1, model.a2], norbs=norbs, name=f"L_{dom.name}")
        # Define Shape
        # We use the seed point to help kwant if needed, but shape_func is primary
        
        # Populate sites
        # We start flood fill from seed, or just use shape
        # For disconnected domains, straightforward shape usage is safer than flood fill
        valid_seed = find_valid_start(lat, dom.shape_func, dom.seed_point)  # Just to check if seed is valid
        syst[lat.shape(dom.shape_func, valid_seed)] = ta.array(model.HR_fine[tuple(model.center_idx)])
        
        # Add Bulk Hoppings (Internal)
        for dx, dy, mat in model.get_lattice_hopping():
            syst[kwant.builder.HoppingKind((dx, dy), lat, lat)] = ta.array(mat)
            
        # Store sites for interface step
        sites = [s for s in syst.sites() if s.family == lat]
        if verbose:
            print(f"  > Sites created: {len(sites)}")
        
        domain_data[i] = {
            'sites': sites,
            'lattice': lat,
            'theta': dom.theta
        }

    # --- 2. Compute Interfaces (Vectorized) ---
    if verbose:
        print("\nComputing Interfaces...")
    
    # Iterate over unique pairs of domains
    for i in range(len(domains)):
        for j in range(i + 1, len(domains)):
            
            sites_i = domain_data[i]['sites']
            sites_j = domain_data[j]['sites']
            
            if not sites_i or not sites_j: continue
            
            # Calculate Average Theta for Interface
            theta_avg = (domain_data[i]['theta'] + domain_data[j]['theta']) / 2.0
            
            # Get Interface Model (High precision interpolation)
            model_inter = manager.get_model(theta_avg, upscale=16)
            
            # KDTree Query
            pos_i = np.array([s.pos for s in sites_i])
            pos_j = np.array([s.pos for s in sites_j])
            
            tree_i = cKDTree(pos_i)
            tree_j = cKDTree(pos_j)
            
            # Find pairs within cutoff (cutoff_Ang -> nm)
            results = tree_i.query_ball_tree(tree_j, r=cutoff_Ang)
            
            # Collect all displacements to vectorize
            # We need to map back which site pair corresponds to which displacement
            pair_indices = [] # [(idx_in_i, idx_in_j), ...]
            displacements = []
            
            count_pairs = 0
            for idx_i, neighbors in enumerate(results):
                if not neighbors: continue
                p_i = pos_i[idx_i]
                for idx_j in neighbors:
                    p_j = pos_j[idx_j]
                    d_vec = p_i - p_j
                    
                    displacements.append(d_vec)
                    pair_indices.append((idx_i, idx_j))
                    count_pairs += 1
            
            if count_pairs == 0:
                continue

            # Vectorized Lookup
            displacements = np.array(displacements)
            matrices, mask = model_inter.get_hoppings_vectorized(displacements)
            
            # Apply to System
            added_hops = 0
            for k, is_valid in enumerate(mask):
                if is_valid:
                    idx_i, idx_j = pair_indices[k]
                    site_i = sites_i[idx_i]
                    site_j = sites_j[idx_j]
                    
                    # syst[i, j] is hopping FROM j TO i
                    syst[site_i, site_j] = ta.array(matrices[k])
                    added_hops += 1
                    
            if verbose:
                print(f"  > Interface {domains[i].name} <-> {domains[j].name}: {added_hops} hoppings")
    # If only one domain, attach leads that reuse the domain's onsite and hoppings
    if len(domains) == 1 and tbg_leads == True:
        if verbose:
            print("\nSingle-domain detected — attaching leads matching domain Hamiltonian...")
        dom0 = domains[0]
        model0 = manager.get_model(dom0.theta, upscale=1)
        lat0 = domain_data[0]['lattice']
        onsite0 = ta.array(model0.HR_fine[tuple(model0.center_idx)])
        hopp0 = list(model0.get_lattice_hopping())
        positions0 = np.array([s.pos for s in domain_data[0]['sites']])
        lead_trans = tuple(model0.a2 + model0.a1)
        extend = model0.scale_factor * 2
        attach_bulk_leads_matching_domain(syst, lat0, onsite0, hopp0, positions0,
                                         lead_trans=lead_trans, extend=extend, verbose=verbose)

    return syst
