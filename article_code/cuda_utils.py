"""
CUDA utilities for efficient contact detection in 3D polymer simulations.

This module provides GPU-accelerated functions for finding contacts between beads
in polymer conformations, which is essential for generating contact maps from
molecular dynamics trajectories.
"""

import numpy as np
from numba import cuda
import math


@cuda.jit
def compute_contacts_kernel(coordinates, counter, cutoff, contacts):
    """
    CUDA kernel for computing contacts between beads in 3D space.
    
    This kernel processes beads in parallel, computing pairwise distances
    and identifying contacts (distance < cutoff). The outer loop ensures that
    all threads participate in processing the complete set of beads.
    
    Parameters:
    -----------
    coordinates : cuda.devicearray (N, 3)
        Array of 3D coordinates for N beads
    counter : cuda.devicearray (2,)
        Atomic counters: [current_bead_index, contact_count]
    cutoff : float
        Contact distance threshold (in same units as coordinates)
    contacts : cuda.devicearray (M, 2)
        Output array for storing contact pairs [bead_i, bead_j]
    """
    # Get thread index in the grid
    idx = cuda.grid(1)
    if idx >= coordinates.shape[0]:
        return
    
    # Loop through all beads to process
    for _ in range(coordinates.shape[0]):
        
        # Atomically get and increment the current bead index
        current_idx = cuda.atomic.add(counter, 0, 1)
        
        # Check if we've processed all beads
        if current_idx >= coordinates.shape[0] - 1:
            return
        
        # Precompute squared cutoff for faster comparison
        cutoff_sqr = cutoff * cutoff
        
        # Get coordinates of current bead
        x1, y1, z1 = coordinates[current_idx]
        
        # Compare with all subsequent beads (to avoid double-counting)
        for j in range(current_idx + 1, coordinates.shape[0]):
            # Compute squared distance between beads
            x2, y2, z2 = coordinates[j]
            dx = x1 - x2
            dy = y1 - y2
            dz = z1 - z2
            distance_sqr = dx*dx + dy*dy + dz*dz
            
            # If distance < cutoff, record the contact pair
            if distance_sqr < cutoff_sqr:
                # Atomically get next available slot in contacts array
                contact_idx = cuda.atomic.add(counter, 1, 1)
                if contact_idx < contacts.shape[0]:
                    contacts[contact_idx, 0] = current_idx
                    contacts[contact_idx, 1] = j


def find_contacts(coordinates, cutoff, gpu_num=0):
    """
    Find contacts between beads using GPU acceleration.
    
    This function identifies all pairs of beads that are within a specified
    cutoff distance of each other. Uses CUDA for parallel computation.
    
    Parameters:
    -----------
    coordinates : np.ndarray (N, 3)
        Array of 3D coordinates for N beads (shape: N × 3)
    cutoff : float
        Contact distance threshold (in same units as coordinates)
    gpu_num : int, optional
        GPU device number to use (default: 0)
        
    Returns:
    --------
    contacts : np.ndarray (M, 2)
        Array of contact pairs, where M is the number of contacts found.
        Each row contains indices [bead_i, bead_j] where distance < cutoff.
    
    Notes:
    ------
    - Memory allocation for results is conservative (N × 48 pairs)
    - If maximum is exceeded, a warning is printed and some contacts may be lost
    - GPU memory is explicitly cleared after computation
    """
    
    # Select GPU device
    cuda.select_device(gpu_num)
    
    # Get GPU capabilities
    device = cuda.get_current_device()
    max_threads_per_block = device.MAX_THREADS_PER_BLOCK
    max_blocks_per_grid = device.MAX_GRID_DIM_X
    
    # Initialize counters on GPU: [current_bead_index, contact_count]
    d_counter = cuda.device_array(2, dtype=np.int32)
    d_counter[0] = 0  # Current bead being processed
    d_counter[1] = 0  # Number of contacts recorded
    
    # Transfer coordinates to GPU memory
    d_coordinates = cuda.to_device(coordinates)
    
    # Allocate result array (conservative estimate: N × 48 contacts)
    # This is much larger than typically needed to avoid overflow
    max_contacts = coordinates.shape[0] * 4 * 12
    contacts = np.full((max_contacts, 2), -1, dtype=np.int32)
    d_contacts = cuda.to_device(contacts)
    
    # Configure kernel launch parameters
    threads_per_block = max_threads_per_block
    blocks_per_grid = min(
        (coordinates.shape[0] + threads_per_block - 1) // threads_per_block,
        max_blocks_per_grid
    )
    
    # Launch CUDA kernel
    compute_contacts_kernel[blocks_per_grid, threads_per_block](
        d_coordinates, d_counter, cutoff, d_contacts
    )
    
    # Copy results back to CPU
    contacts = d_contacts.copy_to_host()
    counters = d_counter.copy_to_host()
    
    # Explicit GPU memory cleanup
    del d_counter
    del d_coordinates
    del d_contacts
    cuda.current_context().deallocations.clear()
    
    # Trim array to actual number of contacts found
    num_contacts = counters[1]
    if num_contacts >= max_contacts:
        print(
            f'WARNING: Number of contacts found ({num_contacts}) exceeds '
            f'maximum allocation ({max_contacts})! Some contacts were not recorded!'
        )
    
    return contacts[:num_contacts]
