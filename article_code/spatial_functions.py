import numpy as np

from typing import Dict, Tuple

from polychrom.starting_conformations import grow_cubic, create_random_walk # type: ignore
try:
    import openmm # type: ignore
except Exception:
    import simtk.openmm as openmm # type: ignore

class bondUpdater(object):
    """
    Updates harmonic bonds between SMC complex legs during 3D simulation.
    
    This class manages the dynamic addition and removal of bonds that represent
    loop extrusion by SMC complexes. Bonds are updated based on pre-calculated
    positions from 1D simulations.
    """

    def __init__(self, LEFpositions, N):
        """
        Initialize bond updater with SMC positions.
        
        Args:
            LEFpositions: Pre-calculated positions of loop-extruding factors (SMC complexes)
                          Shape: (n_blocks, n_extruders, 2) where last dimension is [left_leg, right_leg]
            N: Total number of beads in the polymer chain
        """
        self.LEFpositions = LEFpositions
        self.curtime  = 0
        self.allBonds = []
        self.N = N

    def setParams(self, activeParamDict, inactiveParamDict):
        """
        Set parameters for active and inactive bonds.

        This method should be called before setup() to define bond parameters.
        It's separate from __init__ because you may want to create the Simulation
        object first, then set parameters.

        Args:
            activeParamDict: Dictionary of parameters for active bonds (e.g., {'length': 1.0, 'k': 100.0})
            inactiveParamDict: Dictionary of parameters for inactive bonds (typically k=0)
        """
        self.activeParamDict = activeParamDict
        self.inactiveParamDict = inactiveParamDict


    def setup(self, bondForce, nonbondForce,  blocks = 100, smcStepsPerBlock = 1, except_bonds = False, verbose = False):
        """
        Initialize bonds for the first block of simulation.
        
        Pre-calculates bonds for the specified number of blocks and sets up
        the bond force with initial bond configurations.
        
        Args:
            bondForce: OpenMM bond force object (must be newly created after simulation restart)
            nonbondForce: OpenMM nonbonded force object
            blocks: Number of blocks to precalculate (default: 100)
            smcStepsPerBlock: Number of SMC steps per block (default: 1)
            except_bonds: Whether to add bond exceptions to nonbonded force (default: False)
            verbose: Print debug information (default: False)
        
        Returns:
            tuple: (current_bonds, []) - current bonds and empty list for compatibility
        """
        

        if len(self.allBonds) != 0:
            raise ValueError("Not all bonds were used; {0} sets left".format(len(self.allBonds)))

        self.bondForce = bondForce
        self.nonbondForce = nonbondForce
        if hasattr(self.nonbondForce, "addException"):
            self.nonbond_base_exepts = nonbondForce.getNumExceptions()
        elif hasattr(self.nonbondForce, "addExclusion"):
            self.nonbond_base_exepts = nonbondForce.getNumExclusions() 

        #precalculating all bonds
        allBonds = []
        
        loaded_positions  = self.LEFpositions[self.curtime : self.curtime+blocks]
        allBonds = [[(int(loaded_positions[i, j, 0]), int(loaded_positions[i, j, 1])) 
                        for j in range(loaded_positions.shape[1])] for i in range(blocks)]
        
        allBonds = [[i for i in s if sum(i)>=0] for s in allBonds]
        self.allBonds = allBonds
        self.uniqueBonds = list(set(sum(allBonds, [])))

        #adding forces and getting bond indices
        self.bondInds = []
        self.curBonds = allBonds.pop(0)

        for bond in self.uniqueBonds:
            paramset = self.activeParamDict if (bond in self.curBonds) else self.inactiveParamDict
            ind = bondForce.addBond(bond[0], bond[1], **paramset) # changed from addBond
            self.bondInds.append(ind)
        self.bondToInd = {i:j for i,j in zip(self.uniqueBonds, self.bondInds)}
        
        self.curtime += blocks
        
        if except_bonds:
            if hasattr(self.nonbondForce, "setExclusionParticles"):
                
                tmp_bonds_in_exept = [tuple(self.nonbondForce.getExclusionParticles(i)) for i in range(self.nonbond_base_exepts)]
                bonds_for_add = []
                for bond in self.curBonds:
                    if not (bond in tmp_bonds_in_exept):
                        bonds_for_add += [bond]
                        
                if hasattr(self.nonbondForce, "addException"):
                    exc = list(set([tuple(i) for i in np.sort(np.array(bonds_for_add), axis=1)]))
                    for pair in exc:
                        self.nonbondForce.addException(pair[0], pair[1], 0, 0, 0, True)

                    num_exc = self.nonbondForce.getNumExceptions()

                # The built-in LJ nonbonded force uses "exclusions" instead of "exceptions"
                elif hasattr(self.nonbondForce, "addExclusion"):
                    # for b in bonds_for_add:
                    #     if b[0] < 0 or b[1] < 0:
                    #         print(b)
                    #     if b[0] >= 17620 or b[1] >= 17620:
                    #         print(b)
                    self.nonbondForce.createExclusionsFromBonds([(b[0], b[1]) for b in bonds_for_add], int(except_bonds))
                    num_exc = self.nonbondForce.getNumExclusions()

                if verbose:
                    print("Number of exceptions after milker.setup:", num_exc)
            else:
                print('Cannot make exeptions')
        
        return self.curBonds,[]


    def step(self, context, verbose=False, except_bonds = False):
        """
        Update bonds to the next simulation step.
        
        Automatically updates bond parameters based on pre-calculated positions.
        Bonds that are added or removed are updated in the OpenMM context.
        
        Args:
            context: OpenMM simulation context
            verbose: Print debug information (default: False)
            except_bonds: Whether to update bond exceptions in nonbonded force (default: False)
        
        Returns:
            tuple: (current_bonds, previous_bonds) - for reference only
        """
        if len(self.allBonds) == 0:
            raise ValueError("No bonds left to run; you should restart simulation and run setup  again")

        pastBonds = self.curBonds
        self.curBonds = self.allBonds.pop(0)  # getting current bonds
        bondsRemove = [i for i in pastBonds if i not in self.curBonds]
        bondsAdd = [i for i in self.curBonds if i not in pastBonds]
        bondsStay = [i for i in pastBonds if i in self.curBonds]
        if verbose:
            print("{0} bonds stay, {1} new bonds, {2} bonds removed".format(len(bondsStay),
                                                                            len(bondsAdd), len(bondsRemove)))
        bondsToChange = bondsAdd + bondsRemove
        bondsIsAdd = [True] * len(bondsAdd) + [False] * len(bondsRemove)
        for bond, isAdd in zip(bondsToChange, bondsIsAdd):
            ind = self.bondToInd[bond]
            paramset = self.activeParamDict if isAdd else self.inactiveParamDict
            self.bondForce.setBondParameters(ind, bond[0], bond[1], **paramset)  # actually updating bonds
        self.bondForce.updateParametersInContext(context)  # now run this to update things in the context
        
        if except_bonds:
            if hasattr(self.nonbondForce, "addException"):
                exepts = self.nonbondForce.getNumExceptions()
            elif hasattr(self.nonbondForce, "addExclusion"):
                exepts = self.nonbondForce.getNumExclusions()
                
            if hasattr(self.nonbondForce, "setExclusionParticles"):
                
                tmp_bonds_in_exept = [self.nonbondForce.getExclusionParticles(i) for i in range(exepts)]
                bonds_for_add = []
                for bond in self.curBonds:
                    if not (bond in tmp_bonds_in_exept):
                        bonds_for_add += [bond]
                        
                breaked = 0
                for i_b, bond in enumerate(bonds_for_add):
                    if self.nonbond_base_exepts + i_b < exepts:
                        self.nonbondForce.setExclusionParticles(self.nonbond_base_exepts + i_b, bond[0], bond[1])
                    else:
                        breaked = 1
                        break
                tmp_left_bonds = bonds_for_add[i_b + 1 - breaked:]
                
                if len(tmp_left_bonds) > 0:
                    if hasattr(self.nonbondForce, "addException"):
                        exc = list(set([tuple(i) for i in np.sort(np.array(tmp_left_bonds), axis=1)]))
                        for pair in exc:
                            self.nonbondForce.addException(pair[0], pair[1], 0, 0, 0, True)

                    # The built-in LJ nonbonded force uses "exclusions" instead of "exceptions"
                    elif hasattr(self.nonbondForce, "addExclusion"):
                        self.nonbondForce.createExclusionsFromBonds([(b[0], b[1]) for b in tmp_left_bonds], int(except_bonds))

                if verbose:
                    if hasattr(self.nonbondForce, "addException"): 
                        num_exc = self.nonbondForce.getNumExceptions()
                    elif hasattr(self.nonbondForce, "addExclusion"):
                        num_exc = self.nonbondForce.getNumExclusions()
                    print("Number of exceptions in milker.step:", num_exc)
                self.nonbondForce.updateParametersInContext(context)
            else:
                print('Cannot make exeptions')
        
        return self.curBonds, pastBonds
    
def bond_calculator(bead_sizes, beads_in_one, df_nuc = 3., df_tu = 1., base_bead_nm = 10.0, two_chains = True):
    """
    Calculate bond distances between beads based on their sizes and fractal dimension.
    
    This function computes the effective bond length between beads by considering
    the fractal dimension of the polymer and the sizes of adjacent beads.
    
    Args:
        bead_sizes: List of sizes of individual beads in base pairs
        beads_in_one: Number of beads to group together
        df_nuc: Fractal dimension for nucleosomes (default: 3.0)
        df_tu: Fractal dimension for transcription units (default: 1.0)
        base_bead_nm: Base size of a bead in nanometers (default: 10.0)
        two_chains: Whether the polymer has two chains (sister chromatids) (default: True)
    
    Returns:
        numpy.ndarray: Array of bond distances between adjacent grouped beads
    """
    # Calculate number of nucleotides in each bead
    if two_chains:
        half_bead_sizes = bead_sizes[:len(bead_sizes)//2]
        tmp_bead_sizes = [sum(half_bead_sizes[i:i+beads_in_one]) for i in range(0, len(half_bead_sizes), beads_in_one)]
        
        half_bead_sizes = bead_sizes[len(bead_sizes)//2:][::-1]
        tmp_bead_sizes += [sum(half_bead_sizes[i:i+beads_in_one]) for i in range(0, len(half_bead_sizes), beads_in_one)][::-1]
    else:
        tmp_bead_sizes = [sum(bead_sizes[i:i+beads_in_one]) for i in range(0, len(bead_sizes), beads_in_one)]
    diams = []
    # Compute effective diameter of combined beads using average fractal dimension
    for b_idx, b_size in enumerate(tmp_bead_sizes):
        # For df=1 the maximal bead size is ~225 nm, for df=3 minimal is ~45 nm;
        # the linear relation and constants are chosen to interpolate between these cases.
        dfractal = (b_size / beads_in_one) / 90 + 0.5
        diams.append(base_bead_nm * beads_in_one ** (1 / dfractal))
    diams = np.array(diams)
    # Apply these effective diameters to bonds using the arithmetic mean between neighbours
    if two_chains:
        halves = diams[:len(diams)//2], diams[len(diams)//2:]
        bonds = np.concatenate( ((halves[0][1:]+halves[0][:-1])/2, (halves[1][1:]+halves[1][:-1])/2) )
    else:
        bonds = (diams[1:]+diams[:-1])/2
    return bonds

def random_walk(N):
    """
    Generate a simple 3D random walk of N steps with unit step length.
    
    This is the standard method for creating starting conformations in the project.
    Used in fullsimulation_stepforce.py, fullsimulation_dpdforce.py, and tutorial notebooks.
    
    Args:
        N: Number of beads/steps in the random walk
    
    Returns:
        numpy.ndarray: 3D coordinates, shape (3, N) where rows are [x, y, z]
    """
    r = 1
    x = y = z = np.array([0])
    for i in range(1, N):
        theta = np.random.sample(1) * np.pi
        phi = 2 * np.random.sample(1) * np.pi
        x = np.append(x, r * np.sin(theta) * np.cos(phi) + x[-1])
        y = np.append(y, r * np.sin(theta) * np.sin(phi) + y[-1])
        z = np.append(z, r * np.cos(theta) + z[-1])
    return np.vstack([x, y, z])

def make_start_conf(N, BondDist, starting_conformation=None, starting_conf_file=None, bead_dencity=None):
    """
    Generate starting conformation for 3D polymer simulation.
    
    Supports two modes:
    - 'random_walk': Simple random walk (default for the project)
    - 'grow_cubic': Grow polymer in cubic box (fallback option)
    
    Args:
        N: Number of beads in the polymer chain
        BondDist: Bond distances between consecutive beads
        starting_conformation: Type of conformation ('random_walk' or 'grow_cubic')
        starting_conf_file: Path to file with pre-generated conformation (not implemented)
        bead_dencity: Bead density for 'grow_cubic' mode (default: 0.1)
    
    Returns:
        numpy.ndarray: Initial 3D coordinates, shape (N, 3)
    """
    if starting_conf_file is None:
        if starting_conformation == 'random_walk':
            # Most common: generate random walk starting conformation
            data = (np.mean(BondDist) * np.array(random_walk(N))).T
        elif starting_conformation == 'grow_cubic':
            # Alternative: grow polymer in cubic box
            dens = bead_dencity if bead_dencity is not None else 0.1
            box = (N / dens) ** 0.33 
            data = min(BondDist) * np.vstack(grow_cubic(N, int(box) - 2))
        else:
            # Default fallback: use grow_cubic
            print(f'No valid starting_conformation specified (got: {starting_conformation}), using grow_cubic as default')
            dens = bead_dencity if bead_dencity is not None else 0.1
            box = (N / dens) ** 0.33 
            data = min(BondDist) * grow_cubic(N, int(box) - 2)
    else:
        # Placeholder: reading initial coordinates from file can be added here
        return None
    return data

def lef_pos_calculator(pos, N_raw, gr_size):
    """
    Calculate LEF (Loop Extruding Factor) position for 3D simulation.
    
    Converts 1D lattice position to grouped bead index for 3D representation.
    Handles both single chains and sister chromatids (two chains).
    
    Args:
        pos: Position on 1D lattice
        N_raw: Total number of beads in the chain
        gr_size: Number of beads to group together
    
    Returns:
        int: Grouped bead index for 3D simulation
    """
    N = int(np.ceil(N_raw/2/gr_size)*2)
    if pos < N_raw/2:
        return pos//gr_size  
    else:
        return N - (N_raw - 1 - pos)//gr_size - 1

def dpd_repulsive(sim_object, trunc=3.0, radiusMult=1.0, name="dpd_repulsive"):
    """
    DPD (Dissipative Particle Dynamics) repulsive potential for soft sphere interactions.
    
    This potential provides smoother repulsion compared to Step potential.
    Used in fullsimulation_dpdforce.py for DPD potential simulations.
    
    Formula: U = (1 - r/sigma)^2 * E/2  (for r < sigma)
    
    Args:
        sim_object: Polychrom simulation object
        trunc: Repulsion energy coefficient (default: 3.0)
        radiusMult: Bead radius multiplier (float/int or list/tuple for per-bead radii)
        name: Force name (default: "dpd_repulsive")
    
    Returns:
        openmm.CustomNonbondedForce: DPD repulsive force object
    """
    if (isinstance(radiusMult, float)) or (isinstance(radiusMult, int)):
        repul_energy = (
        "(1 - rsc) * (1 - rsc) * REPe/2;"
        "rsc = r / REPsigma;"
        )
        force = openmm.CustomNonbondedForce(repul_energy)
        force.name = name
        radius = sim_object.conlen * radiusMult
        nbCutOffDist = radius
        force.addGlobalParameter("REPsigma", radius)
        for _ in range(sim_object.N):
            force.addParticle(())
        force.setCutoffDistance(nbCutOffDist)
    
    elif (isinstance(radiusMult, list)) or (isinstance(radiusMult, tuple)):
        repul_energy = (
        "(1 - rsc) * (1 - rsc) * REPe/2;"
        "rsc = r / REPsigma;"
        "REPsigma = REPUL1 * nm + REPUL2 * nm;"
        )

        force = openmm.CustomNonbondedForce(repul_energy)
        force.name = name
        force.addPerParticleParameter("REPUL")
        force.addGlobalParameter("nm", sim_object.conlen)
        for i in range(len(radiusMult)):
            force.addParticle([float(radiusMult[i])])
        nbCutOffDist = sim_object.conlen*max(radiusMult)*2
        force.setCutoffDistance(nbCutOffDist)
    else:
        raise ValueError("\nWrong parameters for radiusMult! It is not list or tuple!")
        
    force.addGlobalParameter("REPe", trunc * sim_object.kT)
    
    return force

def polynomial_repulsive(sim_object, trunc=3.0, radiusMult=1.0, name="polynomial_repulsive"):
    """
    Step potential (polynomial repulsive) for hard sphere interactions.
    
    This potential provides steep repulsion at bead radius, minimizing overlaps.
    Used in fullsimulation_stepforce.py for Step potential simulations.
    Modified from polychrom.forces.polynomial_repulsive for easier use.
    
    Formula: U = r^12 * (r^2 - 1) * E / emin12 + E  (for r < sigma)
    
    Args:
        sim_object: Polychrom simulation object
        trunc: Repulsion energy coefficient (default: 3.0)
        radiusMult: Bead radius multiplier (float/int or list/tuple for per-bead radii)
        name: Force name (default: "polynomial_repulsive")
    
    Returns:
        openmm.CustomNonbondedForce: Polynomial repulsive force object
    """  
    if (isinstance(radiusMult, float)) or (isinstance(radiusMult, int)):
        repul_energy = (
        "rsc12 * (rsc2 - 1.0) * REPe / emin12 + REPe;"
        "rsc12 = rsc4 * rsc4 * rsc4;"
        "rsc4 = rsc2 * rsc2;"
        "rsc2 = rsc * rsc;"
        "rsc = r / REPsigma * rmin12;"
        )
        force = openmm.CustomNonbondedForce(repul_energy)
        force.name = name
        radius = sim_object.conlen * radiusMult
        nbCutOffDist = radius
        force.addGlobalParameter("REPsigma", radius)
        for _ in range(sim_object.N):
            force.addParticle(())
        force.setCutoffDistance(nbCutOffDist)
        
    elif (isinstance(radiusMult, list)) or (isinstance(radiusMult, tuple)):
        repul_energy = (
        "(rsc12 * (rsc2 - 1.0) * REPe/ emin12 + REPe) * step(REPsigma - r);"
        "rsc12 = rsc4 * rsc4 * rsc4;"
        "rsc4 = rsc2 * rsc2;"
        "rsc2 = rsc * rsc;"
        "rsc = r / REPsigma * rmin12;"
        "REPsigma = REPUL1 * nm + REPUL2 * nm;"
        )

        force = openmm.CustomNonbondedForce(repul_energy)
        force.name = name
        force.addPerParticleParameter("REPUL")
        force.addGlobalParameter("nm", sim_object.conlen)
        for i in range(len(radiusMult)):
            force.addParticle([float(radiusMult[i])])
        nbCutOffDist = sim_object.conlen*max(radiusMult)*2
        force.setCutoffDistance(nbCutOffDist)
    else:
        raise ValueError("\nWrong parameters for radiusMult! It is not int, float, list or tuple!")
        
    force.addGlobalParameter("REPe", trunc * sim_object.kT)
    force.addGlobalParameter("emin12", 46656.0 / 823543.0)
    force.addGlobalParameter("rmin12", np.sqrt(6.0 / 7.0))

    return force


def orient_chain_along_z(data):
    """
    Orient polymer chain along Z-axis by aligning principal axis with Z.
    
    This function centers the polymer at origin and rotates it so that
    the principal axis (direction of maximum variance) aligns with the Z-axis.
    This is useful for consistent visualization and analysis.
    
    Args:
        data: Array of 3D positions, shape (N, 3)
    
    Returns:
        numpy.ndarray: Oriented positions, shape (N, 3)
    """
    # Center the data at origin
    data_centered = data - np.mean(data, axis=0)  # shape (N, 3)

    # Compute covariance matrix
    cov = np.cov(data_centered, rowvar=False)  # shape (3, 3)

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Find principal axis (direction of maximum variance)
    max_idx = np.argmax(eigvals)
    principal_axis = eigvecs[:, max_idx]  # principal axis vector (3,)

    # Target: rotate principal_axis to align with Z-axis (0,0,1)
    z_axis = np.array([0, 0, 1])

    # Rotation vector via cross product
    v = np.cross(principal_axis, z_axis)
    s = np.linalg.norm(v)
    c = np.dot(principal_axis, z_axis)

    # If vectors already aligned
    if s == 0:
        return data_centered  # chain already oriented along z

    # Rotation matrix using Rodrigues' formula
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])

    R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))

    # Rotate all points
    data_oriented = np.dot(data_centered, R.T)

    return data_oriented