"""
Example script for generating contact maps from simulation data.

This script demonstrates how to create .mcool files from 3D polymer simulation data,
as described in "Effects of Extruder Dynamics and Noise on Simulated Chromatin 
Contact Probability Curves".

Key steps:
1. Load 3D conformations from Polychrom HDF5 files
2. Add Gaussian noise to 3D conformations (optional smoothing)
3. Detect contacts between beads using CUDA-accelerated functions
4. Generate multi-resolution .mcool files

Parameters match those used in the article:
- Chromosome: 40 Mbp (200,000 beads × 200 bp/bead)
- Cutoff distance: 11 nm (≈1.1 bead diameters for 10 nm beads)
- Resolutions: 1000, 2000, 5000, 10000 bp per bin
- Gaussian noise (sigma): 50, 100, 200, or 400 nm

The script includes three example workflows:
1. Aggregated maps for single sigma value (active by default)
2. Aggregated maps for multiple sigma values (commented)
3. Individual cell maps for analyzing variability (commented)

Usage:
- Uncomment the desired workflow section
- Adjust experimental types and parameters as needed
- Ensure simulation data exists in ./cells_<exp_type>/cell_<num>/3D_data/
"""

import os
import json
import numpy as np
from pathlib import Path
import polychrom.hdf5_format as hdf5
from polychrom.hdf5_format import HDF5Reporter
import polychrom.polymerutils as polymerutils

from draw_and_plot import generate_contact_map_multiple_cells

# =============================================================================
# Parameters (as used in the article)
# =============================================================================

cutoff = 11  # Contact distance in nm (≈1.1 bead diameters, assuming 10 nm per bead)
interest_keys = [i for i in range(300)]  # Block keys to process
resolutions = [1000, 2000, 5000, 10000]  # Resolutions for contact maps (bp)
chrom_sizes = {'chr0': int(4e7)}  # Chromosome sizes: 40 Mbp

# =============================================================================
# Example 1: Aggregated contact maps for multiple cells
# =============================================================================
# This section creates aggregated .mcool files by combining data from 
# multiple cells (0-4) for each experimental condition.

for exp_type in [
                'cond1_gol_p_plen3',
                # 'cond1_gol_p_Erep03',
                # 'cond1_gol_p_soft_force_12>2',
                # 'cond1_gol_p_plen3_dynamic_lt_1div6',
                # 'cond1_gol_p_plen3_dynamic_lt_4div6',
                # 'cond1_gol_p_plen3_dynamic_lt_8div6',
                # 'cond1_gol_p_plen3_Erep30',
                # 'cond1_gol_p_plen30',
                # 'cond1_gol_p_plen03',
                # 'cond1_gol_p_plen0_dpdforce',
                 ]:
    
    # Process different Gaussian noise levels (sigma values in nm)
    for sigma in [50]:
        path_to_files = f'./cells_{exp_type}'

        # Step 1: Add Gaussian noise to 3D conformations for each cell
        for cell_num in range(0, 5):
            path_to_cell = path_to_files + f'/cell_{cell_num}/'
            path_to_3D = path_to_cell + '3D_data/'
            path_to_gaussian = path_to_cell + '3D_gaussian/'
            
            # Create directory for Gaussian-smoothed data
            if not os.path.exists(path_to_gaussian):
                os.mkdir(path_to_gaussian)

            # Parameters for data slicing
            n_lifetimes = 5
            savesPerSim = 10
            simInitsTotal = 10

            # Setup HDF5 reporter for writing smoothed conformations
            reporter = HDF5Reporter(folder=path_to_gaussian, max_data_length=100, overwrite=True, blocks_only=False)
            
            # Select subset of conformations (3/5 of lifetime, every 2nd save)
            slicing = (int(savesPerSim*simInitsTotal*3/n_lifetimes)-1, savesPerSim*simInitsTotal, savesPerSim*2)
            list_uris = hdf5.list_URIs(path_to_3D)[slicing[0]:slicing[1]:slicing[2]]

            # Add Gaussian noise to each conformation
            block = 0
            for uri in list_uris:
                data = polymerutils.load(uri)
                # Generate 100 noisy copies of each conformation
                for _ in range(100):
                    gauss_smooth = np.random.multivariate_normal(
                        mean=[0, 0, 0], 
                        cov=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * (sigma**2), 
                        size=data.shape[0]
                    )
                    reporter.report("data", {"pos": data + gauss_smooth, "time": 0, "block": block})
                    block += 1
            reporter.dump_data()
            
        # Step 2: Generate aggregated contact map from all cells
        output_uri = f'./coolmaps/{exp_type}_sigma{sigma}.mcool'
        paths_to_files = [f'./cells_{exp_type}/cell_{cell_num}/3D_gaussian/' for cell_num in range(0, 5)]
        bead_sizes = json.load(open(f'./cells_{exp_type}/cell_{cell_num}/' + 'bead_sizes.json'))

        # Generate contact map using CUDA acceleration
        rows_cuda, cols_cuda, data_cuda = generate_contact_map_multiple_cells(
            paths_to_files=paths_to_files,
            output_uri=output_uri,
            interest_keys=interest_keys,
            cutoff=cutoff,
            resolutions=resolutions,
            chrom_sizes=chrom_sizes,
            weight_mode='standard',
            telomere_padding=0.0,
            bead_sizes=bead_sizes,
            gpu_num=1
        )

        print(f"✓ Contact map created and saved to {output_uri}")
        

# =============================================================================
# Example 2: Aggregated contact maps with multiple sigma values
# =============================================================================
# This section demonstrates processing multiple Gaussian noise levels
# (sigma = 50, 100, 200, 400 nm) for each experimental condition.

# for exp_type in [
#                 'cond1_gol_p_plen3',
#                 'cond1_gol_p_Erep03',
#                 'cond1_gol_p_soft_force_12>2',
#                 'cond1_gol_p_plen3_dynamic_lt_1div6',
#                 'cond1_gol_p_plen3_dynamic_lt_4div6',
#                 'cond1_gol_p_plen3_dynamic_lt_8div6',
#                 'cond1_gol_p_plen3_Erep30',
#                 'cond1_gol_p_plen30',
#                 'cond1_gol_p_plen03',
#                 # 'cond1_gol_p_plen0_dpdforce',
#                  ]:
#     
#     # Process multiple Gaussian noise levels
#     for sigma in [50, 100, 200, 400]:
#         path_to_files = f'./cells_{exp_type}'
# 
#         # Step 1: Add Gaussian noise to 3D conformations for each cell
#         for cell_num in range(0, 5):
#             path_to_cell = path_to_files + f'/cell_{cell_num}/'
#             path_to_3D = path_to_cell + '3D_data/'
#             path_to_gaussian = path_to_cell + '3D_gaussian/'
#             
#             if not os.path.exists(path_to_gaussian):
#                 os.mkdir(path_to_gaussian)
# 
#             # Parameters for data slicing
#             n_lifetimes = 5
#             savesPerSim = 10
#             simInitsTotal = 10
# 
#             # Setup HDF5 reporter
#             reporter = HDF5Reporter(folder=path_to_gaussian, max_data_length=100, overwrite=True, blocks_only=False)
#             slicing = (int(savesPerSim*simInitsTotal*3/n_lifetimes)-1, savesPerSim*simInitsTotal, savesPerSim*2)
#             list_uris = hdf5.list_URIs(path_to_3D)[slicing[0]:slicing[1]:slicing[2]]
# 
#             # Add Gaussian noise
#             block = 0
#             for uri in list_uris:
#                 data = polymerutils.load(uri)
#                 for _ in range(100):
#                     gauss_smooth = np.random.multivariate_normal(
#                         mean=[0, 0, 0], 
#                         cov=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * (sigma**2), 
#                         size=data.shape[0]
#                     )
#                     reporter.report("data", {"pos": data + gauss_smooth, "time": 0, "block": block})
#                     block += 1
#             reporter.dump_data()
#             
#         # Step 2: Generate aggregated contact map
#         output_uri = f'./coolmaps/{exp_type}_sigma{sigma}.mcool'
#         paths_to_files = [f'./cells_{exp_type}/cell_{cell_num}/3D_gaussian/' for cell_num in range(0, 5)]
#         bead_sizes = json.load(open(f'./cells_{exp_type}/cell_{cell_num}/' + 'bead_sizes.json'))
# 
#         rows_cuda, cols_cuda, data_cuda = generate_contact_map_multiple_cells(
#             paths_to_files=paths_to_files,
#             output_uri=output_uri,
#             interest_keys=interest_keys,
#             cutoff=cutoff,
#             resolutions=resolutions,
#             chrom_sizes=chrom_sizes,
#             weight_mode='standard',
#             telomere_padding=0.0,
#             bead_sizes=bead_sizes,
#             gpu_num=1
#         )
# 
#         print(f"✓ Contact map created and saved to {output_uri}")


# =============================================================================
# Example 3: Individual contact maps for single cells
# =============================================================================
# This section creates separate .mcool files for each cell replica,
# useful for analyzing cell-to-cell variability.

# # Parameters with additional resolution
# cutoff = 11  # Contact distance in nm
# interest_keys = [i for i in range(300)]  # Block keys to process
# resolutions = [200, 1000, 2000, 5000, 10000]  # Resolutions including 200 bp
# chrom_sizes = {'chr0': int(4e7)}  # Chromosome sizes: 40 Mbp

# for exp_type in [
#                 # 'cond1_gol_p_plen3',
#                 # 'cond1_gol_p_Erep03',
#                 # 'cond1_gol_p_soft_force_12>2',
#                 # 'cond1_gol_p_plen3_dynamic_lt_1div6',
#                 # 'cond1_gol_p_plen3_dynamic_lt_4div6',
#                 # 'cond1_gol_p_plen3_dynamic_lt_8div6',
#                 'cond1_gol_p_plen0_dpdforce',
#                 ]:
#     
#     # Process each cell individually
#     for cell_num in range(0, 5):
#         
#         # Load bead sizes for this cell
#         bead_sizes = json.load(open(f'./cells_{exp_type}/cell_{cell_num}/' + 'bead_sizes.json'))
#         
#         # Create individual contact map for this cell
#         output_uri = f'./cells_{exp_type}/cell_{cell_num}/single_map_sigma{sigma}.mcool'
#         paths_to_files = [f'./cells_{exp_type}/cell_{cell_num}/3D_gaussian/']
#         
#         rows_cuda, cols_cuda, data_cuda = generate_contact_map_multiple_cells(
#             paths_to_files=paths_to_files,
#             output_uri=output_uri,
#             interest_keys=interest_keys,
#             cutoff=cutoff,
#             resolutions=resolutions,
#             chrom_sizes=chrom_sizes,
#             weight_mode='standard',
#             telomere_padding=0.0,
#             bead_sizes=bead_sizes,
#             gpu_num=1
#         )
#         
#         print(f"✓ Contact map created and saved to {output_uri}")

