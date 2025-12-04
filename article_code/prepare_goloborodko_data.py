"""
Script to prepare simulated data from Samejima et al. (2024), DOI: 10.1126/science.adq1709

This script processes 3D conformational data from mitotic chromosome models
with static condensin I extruders and prepares them for spiral correlation analysis (Fig A3).

The data comes from different condensin knockout conditions:
- SMC3-CAPH2KO: SMC3 and CAPH2 knockout
- SMC3-CAPH KO: SMC3 and CAPH knockout  
- SMC3KO: SMC3 knockout

The script:
1. Loads last frame from GSD trajectory files
2. Unfolds periodic boundary conditions
3. Saves positions as numpy arrays

Reference:
Samejima, K., et al. (2024). "Mitotic chromosomes are self-entangled and disentangle 
through a topoisomerase-II-dependent two-stage exit from mitosis." Science, 385(6714).
"""

import os
import numpy as np

try:
    import gsd.hoomd
    GSD_AVAILABLE = True
except ImportError:
    print("Warning: gsd.hoomd module not available. Install with: pip install gsd")
    GSD_AVAILABLE = False


def chromosome_unfolder(positions, cutoff_dist=1.0, pbs_box=[10., 10., 10.]):
    """
    Unfold chromosome positions across periodic boundary conditions.
    
    When polymers cross periodic boundaries, adjacent beads appear far apart
    in wrapped coordinates. This function detects such jumps and unwraps them
    by adding/subtracting box dimensions.
    
    Args:
        positions: Array of bead positions, shape (N, 3)
        cutoff_dist: Distance threshold to detect boundary crossing (in simulation units)
        pbs_box: Periodic box dimensions [Lx, Ly, Lz]
    
    Returns:
        numpy.ndarray: Unfolded positions, shape (N, 3)
    """
    tmp_positions = positions.copy()
    vecs_arr = tmp_positions[1:] - tmp_positions[:-1]
    
    for i_vec in range(len(positions) - 1):
        target_vec = vecs_arr[i_vec]
        update_flag = False
        
        for coord in range(3):
            if abs(target_vec[coord]) > cutoff_dist:
                sign = np.sign(target_vec[coord])
                tmp_positions[i_vec + 1:, coord] += -1 * sign * pbs_box[coord]
                update_flag = True
        
        if update_flag:
            update_flag = False
            vecs_arr = tmp_positions[1:] - tmp_positions[:-1]
        
        if (i_vec + 1) % 10000 == 0:
            print(f'Done with {i_vec+1} / {len(positions)-1} vecs', end='\r', flush=True)
    
    return tmp_positions


def process_reference_data(input_folder="samejima_mitotic_models/best_models/", 
                           output_folder="reference_data/",
                           max_beads=500000):
    """
    Process mitotic chromosome models from Samejima et al. (2024).
    
    Extracts last frame from GSD trajectory files and saves unfolded positions
    as numpy arrays for further analysis.
    
    Args:
        input_folder: Path to folder with simulation data
        output_folder: Path to save processed numpy arrays
        max_beads: Maximum number of beads to extract (default: 500000)
    """
    if not GSD_AVAILABLE:
        raise ImportError("gsd.hoomd module is required. Install with: pip install gsd")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Define simulation types and their unfolding parameters
    sim_configs = {
        'SMC3-CAPH2': {'cutoff': 10.0, 'box': [56.693836, 56.693836, 400.]},
        'SMC3-CAPH': {'cutoff': 1.0, 'box': [10., 10., 10.]},  # Default parameters
        'SMC3': {'cutoff': 1.0, 'box': [10., 10., 10.]}  # Default parameters
    }
    
    for sim_type, config in sim_configs.items():
        sim_folder = os.path.join(input_folder, sim_type)
        
        if not os.path.exists(sim_folder):
            print(f"Warning: Folder {sim_folder} not found, skipping {sim_type}")
            continue
        
        rep_dirs = os.listdir(sim_folder)
        
        for rep_dir in rep_dirs:
            rep_name = rep_dir.split('-')[-1]
            print(f"\nProcessing {sim_type} - {rep_name}")
            
            gsd_file = os.path.join(sim_folder, rep_dir, 'last_frame.gsd')
            
            if not os.path.exists(gsd_file):
                print(f"  Warning: File {gsd_file} not found")
                continue
            
            try:
                with gsd.hoomd.open(gsd_file, 'r') as traj:
                    frame = traj[-1]
                    positions = np.array(frame.particles.position)[:max_beads]
                    
                    # Unfold periodic boundaries if needed (especially for SMC3-CAPH2)
                    if sim_type == 'SMC3-CAPH2':
                        positions = chromosome_unfolder(
                            positions, 
                            cutoff_dist=config['cutoff'], 
                            pbs_box=config['box']
                        )
                    
                    # Save positions
                    output_name = f'{sim_type}KO_positions_{rep_name}.npy'
                    output_path = os.path.join(output_folder, output_name)
                    np.save(output_path, positions)
                    print(f"  ✓ Saved {output_name} ({positions.shape[0]} beads)")
                    
            except Exception as e:
                print(f"  ✗ Error processing {gsd_file}: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"Saved files to: {output_folder}")


if __name__ == "__main__":
    # Default paths - adjust as needed
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process mitotic chromosome models from Samejima et al. (2024)"
    )
    parser.add_argument(
        '--input', 
        default='samejima_mitotic_models/best_models/',
        help='Path to input folder with simulation data'
    )
    parser.add_argument(
        '--output', 
        default='reference_data/',
        help='Path to output folder for processed data'
    )
    parser.add_argument(
        '--max-beads', 
        type=int,
        default=500000,
        help='Maximum number of beads to extract (default: 500000)'
    )
    
    args = parser.parse_args()
    
    process_reference_data(
        input_folder=args.input,
        output_folder=args.output,
        max_beads=args.max_beads
    )

