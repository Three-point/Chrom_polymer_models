"""
Full simulation workflow for chromatin polymer modeling with DPD potential.

This script performs:
1. 1D loop extrusion simulation using static extruders
2. 3D molecular dynamics simulation using Polychrom with dpd_repulsive force (DPD potential)
3. Multiple cell replicates (5 cells) for statistical analysis

Parameters are based on Samejima et al. approach with modifications.
"""

# System libraries
import os
import json
import time
import shutil
import warnings
warnings.filterwarnings('ignore')

# Data processing libraries
import h5py
import numpy as np

# Polychrom libraries
import polychrom.hdf5_format as hdf5
from polychrom import forces, forcekits
from polychrom.simulation import Simulation
from polychrom.hdf5_format import HDF5Reporter

# Molecular dynamics backend (OpenMM / simtk.openmm compatibility layer)
try:
    import openmm
except Exception:
    import simtk.openmm as openmm

# Local modules
from bead import *
from smc import *
from ensemble_director import *
from file_handlers import *
from spatial_functions import (orient_chain_along_z, bond_calculator, make_start_conf, 
                                bondUpdater, lef_pos_calculator, dpd_repulsive)


# ============================================================================
# SECTION 1: Define polymer chain and 1D simulation parameters
# ============================================================================

# Polymer chain parameters (40 Mbp, 200 bp per bead)
bead_sizes = [200] * int(2e5)  # 200,000 beads × 200 bp = 40 Mbp
time_step_1d = 1/10  # seconds per 1D simulation step

# Extruder velocity parameters
naked_velocity = 0  # bp/sec (0 for static extruders, 1000 for dynamic)
velocity = naked_velocity * 6  # Effective velocity on naked DNA

# Chain parameters
bead_pars = {
    'name': 'chain',
    'objects_number': len(bead_sizes),
    'objects_sizes': bead_sizes,
    'two_chain': False,  # Single chromosome (not sister chromatids)
    'objects_general_type': 'Bead',
}

# Condensin attributes (control permeability for SMC passage on beads)
# These values determine whether other complexes can pass through this complex's subunits
# in specific directions when they encounter each other on the same bead
# "Active": Whether directional blocking is enabled (False = no blocking)
# "Condensin+": Probability of blocking passage in the positive direction
# "Condensin-": Probability of blocking passage in the negative direction
attrs_condensin1 = {
    "Active": False,
    "Condensin+": 0.0,
    "Condensin-": 0.0,
}

# Condensin arguments (physical parameters)
args_condensin1 = {
    "type": "Condensin",
    "beads_number": len(bead_sizes),
    "n_steps": 5,  # Number of beads moved per extrusion step
    "force": 0.2,  # Extrusion force
    "change_dir_prob": 0.5,  # Probability of switching active leg
    "diff_prob": 0.0,  # Diffusion probability
    "avgloopsize_bp": 1e5,  # Average loop size: 100 kbp
    "lifetime_on_ctcf": 1,
    "reborntime": 1,
    "velocity": velocity,  # bp/sec
    "pushing": False,  # Don't push other complexes
}

# Calculate step probability 
# Standard condensin step on naked DNA is 10 nm
args_condensin1['step_prob'] = args_condensin1['velocity'] / (args_condensin1['n_steps'] * 33) * time_step_1d


# ============================================================================
# SECTION 2: Run simulation for multiple cells
# ============================================================================

for cell_num in range(0, 5, 1):
    
    print(f"\n{'='*80}")
    print(f"PROCESSING CELL {cell_num}")
    print(f"{'='*80}\n")
    
    # ========================================================================
    # 2.1: Calculate lifetime and generate static extruder positions
    # ========================================================================
    
    # Calculate extruder lifetime based on velocity
    if args_condensin1["velocity"] > 0:
        args_condensin1["lifetime"] = args_condensin1['avgloopsize_bp'] / \
                    (args_condensin1["velocity"] * time_step_1d)
    else:
        args_condensin1["lifetime"] = 1e100  # Effectively infinite for static extruders
    
    # Generate static extruder positions with exponential loop size distribution
    static_condensins_positions = []
    loop_length = np.random.exponential(scale=args_condensin1["avgloopsize_bp"], size=1)
    loop_spacing = 400.  # bp - Samejima et al. reported 20 nm separator = 2 nucleosomes
    first_leg_pos = 1
    second_leg_pos = int(first_leg_pos + loop_length // 200)
    
    while True:
        if first_leg_pos > len(bead_sizes) - 2 or second_leg_pos > len(bead_sizes) - 2:
            break
        static_condensins_positions.append([first_leg_pos, second_leg_pos])
        loop_length = np.random.exponential(scale=args_condensin1["avgloopsize_bp"], size=1)
        first_leg_pos = int(second_leg_pos + loop_spacing // 200)
        second_leg_pos = int(first_leg_pos + loop_length // 200)
    
    # Condensin parameters
    condensin1_pars = {
        'name': 'condensin1',
        'objects_number': len(static_condensins_positions),
        'objects_general_type': 'Extruder',
        'args_smc': args_condensin1,
        'attrs_smc': attrs_condensin1,
        'positions': static_condensins_positions,
    }
    
    print(f"Generated {len(static_condensins_positions)} static extruders")
    
    # Combined parameters dictionary
    par_dict = {
        'chain': bead_pars,
        'condensin1': condensin1_pars,
    }
    
    # ========================================================================
    # 2.2: Setup output directories
    # ========================================================================
    
    path_to_files = './cells_cond1_gol_dpd'
    
    if not os.path.exists(path_to_files):
        os.mkdir(path_to_files)
    
    path_to_cell = path_to_files + f'/cell_{cell_num}/'
    if os.path.exists(path_to_cell):
        shutil.rmtree(path_to_cell)
    os.mkdir(path_to_cell)
    
    path_to_3D = path_to_cell + '3D_data/'
    os.mkdir(path_to_3D)
    
    path_to_gaussian = path_to_cell + '3D_gaussian/'
    os.mkdir(path_to_gaussian)
    
    # ========================================================================
    # 2.3: Run 1D simulation (loop extrusion on lattice)
    # ========================================================================
    
    print(f"\nStarting 1D simulation...")
    
    ultimate = ensemble_director(par_dict, path_to_files=path_to_cell, dinamic_model=True)
    json.dump(bead_sizes, open(f'{path_to_cell}/bead_sizes.json', 'w'))
    
    # Simulate 1800 seconds (30 minutes)
    sim_duration = 1800 / time_step_1d
    sim_duration = int(sim_duration // 100) * 100  # Round to nearest 100
    
    print(f'Total 1D simulation steps: {sim_duration}')
    
    # Run simulation in batches of 100 steps
    for batch in range(sim_duration // 100):
        ultimate.run_simulation(timestep=1/10, steps=100, merge_files_after_calculation=False)
        if (batch + 1) % 10 == 0:
            print(f'  {(batch + 1) / (sim_duration // 100) * 100:.0f}% complete')
    
    # Merge HDF5 files
    merge_h5_files(ensamble_director=ultimate, target_objects_names=None)
    
    # Save 1D simulation metadata
    save_1D_info = {
        'cond1_lifetime': args_condensin1["lifetime"],
        'timestep_per_second_1D': time_step_1d,
        'sim_duration': sim_duration,
        'beads_num': len(bead_sizes),
        'chain_length_bp': sum(bead_sizes),
    }
    json.dump(save_1D_info, open(f'{path_to_cell}/timesize_info.json', 'w'))
    
    print("✓ 1D simulation complete")
    
    # ========================================================================
    # 2.4: Setup 3D simulation parameters
    # ========================================================================
    
    print(f"\nSetting up 3D simulation...")
    
    # Load 1D simulation results
    cond1_file = h5py.File(path_to_cell + f"condensin1_steps:0-{sim_duration}.hdf5", mode='r')
    bead_sizes = json.load(open(path_to_cell + 'bead_sizes.json'))
    starting_conf_file = None
    
    # Frame selection parameters
    get_every_n_frames = 1
    beads_gr_size = 1
    
    # Calculate bond distances and radii
    bonds_dist = (np.ones_like(bond_calculator(bead_sizes, beads_gr_size, base_bead_nm=10.0, two_chains=False))).tolist()
    radiuses = (np.ones_like(bead_sizes) * 0.5).tolist()  # Bead diameter is 1 in reduced units
    
    # Physical parameters for 3D MD simulation
    length_scale = 10.0  # nm - converts reduced units to nanometers
    smcBondDist = 1.0  # SMC bond distance in reduced units
    smcBondWiggleDist = 0.1  # SMC bond fluctuation distance
    BondWiggleDist = 0.1 * min(bonds_dist)  # Bond fluctuation distance
    
    # Force parameters
    Erep = 1.5e0  # Repulsive energy scale
    p_len_coef = 3e0  # Persistence length coefficient (bending rigidity)
    
    # MD integration parameters
    collision_rate = 1e-2  # Collision rate in inverse picoseconds
    time_step = 100
    
    # Simulation control parameters
    except_bonds_from_updater = False
    steps = int(1000)  # MD steps per block
    first_molsteps_mult = 100  # Initial equilibration multiplier
    molstepsmul = 1  # Regular step multiplier
    
    # Frame range selection
    from_frame = 0
    to_frame = sim_duration
    frame_slice = slice(from_frame, to_frame, get_every_n_frames)
    
    N_raw = len(bead_sizes)
    Nframes = (frame_slice.stop - frame_slice.start) // frame_slice.step
    
    print(f'frame_slice: {frame_slice}')
    print(f'N_raw: {N_raw}, Nframes: {Nframes}')
    
    # Extract LEF (Loop Extruding Factor) positions from 1D simulation
    LEFpositions = np.vectorize(lef_pos_calculator)(cond1_file['positions'][0, frame_slice], N_raw, beads_gr_size)
    
    # Simulation restart and save intervals
    restartSimulationEveryBlocks = Nframes // 10
    saveEveryBlocks = restartSimulationEveryBlocks // 10
    savesPerSim = restartSimulationEveryBlocks // saveEveryBlocks
    simInitsTotal = Nframes // restartSimulationEveryBlocks
    
    print(f'restartSimulationEveryBlocks: {restartSimulationEveryBlocks}, saveEveryBlocks: {saveEveryBlocks}')
    print(f'simInitsTotal: {simInitsTotal}')
    
    # Assertions for code validation
    assert (Nframes % restartSimulationEveryBlocks) == 0, \
        f'Nframes = {Nframes}, restartSimulationEveryBlocks = {restartSimulationEveryBlocks}'
    assert (restartSimulationEveryBlocks % saveEveryBlocks) == 0, \
        f'restartSimulationEveryBlocks = {restartSimulationEveryBlocks}, saveEveryBlocks = {saveEveryBlocks}'
    
    # Save 3D simulation parameters
    save_3D_info = {
        'length_scale': length_scale,
        'smcBondDist': smcBondDist,
        'smcBondWiggleDist': smcBondWiggleDist,
        'BondWiggleDist': BondWiggleDist,
        'Erep': Erep,
        'p_len_coef': p_len_coef,
        'collision_rate': collision_rate,
        'time_step': time_step,
        'steps': steps,
        'first_molsteps_mult': first_molsteps_mult,
        'molstepsmul': molstepsmul,
        'restartSimulationEveryBlocks': restartSimulationEveryBlocks,
        'saveEveryBlocks': saveEveryBlocks,
        'get_every_n_frames': get_every_n_frames
    }
    json.dump(save_3D_info, open(f'{path_to_cell}/3D_params.json', 'w'))
    
    # ========================================================================
    # 2.5: Run 3D simulation with DPD potential (dpd_repulsive)
    # ========================================================================
    
    print(f"\nStarting 3D simulation with DPD potential...")
    print("=" * 100)
    
    time_hist = []
    
    # Initialize HDF5 reporter for saving trajectories
    reporter = HDF5Reporter(folder=path_to_3D, max_data_length=savesPerSim, overwrite=True, blocks_only=False)
    
    # Create initial random walk conformation
    data = make_start_conf(N_raw, bonds_dist, starting_conformation='random_walk', 
                          starting_conf_file=starting_conf_file) * length_scale
    data = orient_chain_along_z(data)
    
    # Initialize bond updater for dynamic loop constraints
    milker = bondUpdater(LEFpositions, N=N_raw)
    
    # Main 3D simulation loop
    for iteration in range(simInitsTotal):
        t1 = time.time()
        print(f'Iteration: {iteration + 1}/{simInitsTotal}')
        
        # Initialize Polychrom simulation
        a = Simulation(
            max_Ek=1e10,
            platform='CUDA',
            integrator="langevin",
            mass=1e0,
            error_tol=1e-2,
            timestep=time_step,
            GPU="0",
            collision_rate=collision_rate,  # Collision rate in inverse picoseconds
            N=len(data),
            reporters=[reporter],
            PBCbox=False,  # No periodic boundary conditions
            precision="mixed",
            save_decimals=2,
            length_scale=length_scale,
        )
        
        # Load initial conformation
        a.set_data(data, center=True)
        
        # Add polymer forces
        a.add_force(
            forcekits.polymer_chains(
                a,
                chains=[(0, None, False)],
                
                # Harmonic bonds between consecutive beads
                bond_force_func=forces.harmonic_bonds,
                bond_force_kwargs={
                    'bondLength': bonds_dist,
                    'bondWiggleDistance': BondWiggleDist,
                },
                
                # Angle force for bending rigidity (persistence length)
                angle_force_func=forces.angle_force,
                angle_force_kwargs={
                    'k': p_len_coef,
                },
                
                # Non-bonded repulsion: DPD potential (dpd_repulsive)
                # Note: Using custom dpd_repulsive from spatial_functions
                nonbonded_force_func=dpd_repulsive,
                nonbonded_force_kwargs={
                    'trunc': Erep,
                    'radiusMult': radiuses,
                },
                
                except_bonds=True,
            )
        )
        
        non_bond_force_name = "dpd_repulsive"
        
        # Setup dynamic loop constraints (SMC bonds)
        kbond = a.kbondScalingFactor / (smcBondWiggleDist ** 2)
        activeParams = {"length": smcBondDist * a.length_scale, "k": kbond}
        inactiveParams = {"length": smcBondDist * a.length_scale, "k": 0}
        milker.setParams(activeParams, inactiveParams)

        # Setup bonds: adds all bonds and sets initial bond states
        milker.setup(bondForce=a.force_dict['harmonic_bonds'],
                    nonbondForce=a.force_dict[non_bond_force_name],
                    blocks=restartSimulationEveryBlocks,
                    except_bonds=except_bonds_from_updater)
        
        # Initial equilibration
        if iteration == 0:
            a.local_energy_minimization()
            a.collisionRate = collision_rate * a.collisionRate.unit
            a.integrator.setFriction(a.collisionRate)
            a.integrator.step(steps * first_molsteps_mult)
        else:
            a._apply_forces()
            a.integrator.step(steps * first_molsteps_mult)
        
        print(f'Cutoff distance: {a.force_dict[non_bond_force_name].getCutoffDistance()}')
        
        # Run MD blocks with periodic bond updates
        for i in range(restartSimulationEveryBlocks):
            if i % saveEveryBlocks == (saveEveryBlocks - 1):
                a.do_block(steps=steps * molstepsmul)  # Save this block
            else:
                a.integrator.step(steps * molstepsmul)  # Fast step without GPU retrieval
            
            # Update loop constraints (except for last block)
            if i < restartSimulationEveryBlocks - 1:
                curBonds, pastBonds = milker.step(a.context, except_bonds=except_bonds_from_updater)
        
        # Get final conformation for next iteration
        data = a.get_data()
        del a
        
        reporter.blocks_only = True  # Write output hdf5-files only for blocks
        
        # Progress reporting
        t = time.localtime()
        t = time.strftime("%H:%M:%S", t)
        time.sleep(0.2)  # Brief pause for garbage collection
        t2 = time.time()
        time_hist.append(t2 - t1)
        remTime = np.mean(time_hist) * (simInitsTotal - iteration - 1)
        
        progress_str = (f'Completed: {round(((1 + iteration) / simInitsTotal) * 100, 1)}% | '
                       f'Remaining time: {round(remTime / 3600, 1)} hours | '
                       f'Time for iteration: {round((t2 - t1) / 60, 1)} min')
        print(progress_str)
        
        # Save progress to file
        with open(path_to_cell + "/Outputdata.txt", "a") as file:
            file.write(progress_str + f' | Current time: {t}\n')
    
    print('\n✓ Simulation complete!')
    reporter.dump_data()

print(f"\n{'='*80}")
print("ALL CELLS PROCESSED")
print(f"{'='*80}")
