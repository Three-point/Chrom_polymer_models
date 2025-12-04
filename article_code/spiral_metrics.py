import numpy as np


def compute_C_s(positions, d=32, min_s=1, max_s=50000, step_s=100):
    """
    Compute spiral correlation function C(s) for a polymer chain.
    
    This function calculates the correlation between normalized direction vectors
    separated by distance s along the polymer chain. The correlation is computed
    using vectors of length d (in beads).
    
    Args:
        positions: Array of 3D positions of beads, shape (N, 3)
        d: Length of direction vectors in beads (default: 32)
        min_s: Minimum separation distance to compute (default: 1)
        max_s: Maximum separation distance to compute (default: 50000)
        step_s: Step size for separation distances (default: 100)
    
    Returns:
        tuple: (C_s, S) where C_s is array of correlation values and S is array of separation distances
    """
    N = positions.shape[0]
    max_s = min(max_s, N - d - 1)

    # Compute all direction vectors of length d
    vectors = positions[d:] - positions[:-d]

    # Normalize all vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors_normalized = vectors / norms

    C_s = []
    S = []
    for s in range(min_s, max_s + 1, step_s):
        # Take pairs of normalized vectors separated by distance s
        vec_1 = vectors_normalized[:-s]
        vec_2 = vectors_normalized[s:]
        # Compute dot products for all pairs
        dots = np.einsum('ij,ij->i', vec_1, vec_2)
        C_s.append(np.mean(dots))
        S.append(s)
    return np.array(C_s), np.array(S)

def generate_cylindrical_spiral(total_points, beads_per_turn, radius=1.0, pitch=1.0):
    """
    Generate a 3D spiral along a cylindrical surface.
    
    Creates a helical structure with specified number of points and beads per turn.
    This is useful for generating idealized spiral conformations for comparison.
    
    Args:
        total_points: Total number of points (beads) in the spiral
        beads_per_turn: Number of beads per turn of the spiral
        radius: Radius of the cylinder (default: 1.0)
        pitch: Height step per turn (default: 1.0)
    
    Returns:
        numpy.ndarray: Array of 3D positions, shape (total_points, 3)
    """
    total_turns = total_points / beads_per_turn
    theta = np.linspace(0, 2 * np.pi * total_turns, total_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = pitch * theta / (2 * np.pi)
    return np.vstack((x, y, z)).T
