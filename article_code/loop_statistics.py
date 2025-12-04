import h5py
import numpy as np
import os
from pathlib import Path

def identify_nested_loops(positions):
    """
    Identify nested loops among extruders.
    
    Args:
        positions: 2D NumPy array of extruder leg positions, one row per
            extruder, ``[left_leg, right_leg]`` in bead indices.
        
    Returns:
        tuple(list, list):
            - indices of outer (non‑nested) loops
            - indices of nested loops (whose span lies inside another loop)
    """
    import numpy as np
    
    # Identify nested loops by comparing spans of all pairs
    nested_indices = set()
    outer_indices = set(range(len(positions)))
    
    for i, (left1, right1) in enumerate(positions):
        for j, (left2, right2) in enumerate(positions):
            if i != j:
                # Loop i is nested inside loop j if its span is fully enclosed by loop j.
                # Use <=, >= to also detect loops with coincident boundaries.
                if left2 <= left1 and right1 <= right2:
                    # If loops have the same span and shared boundaries, this is not treated as nesting
                    if not (left2 == left1 and right1 == right2):
                        nested_indices.add(i)
                        break
    
    # Outer loops are all loops that are not nested in any other loop
    outer_indices = outer_indices - nested_indices
    
    return list(outer_indices), list(nested_indices)

def calculate_adjacent_extruder_legs_distance(
    path_to_cell: str,
    time_frames=None,
    min_distance: int = 0,
    debug: bool = False
) -> np.ndarray:
    """
    Compute distances between neighbouring opposing extruder legs,
    excluding nested loops.

    Distances are measured in beads along the 1D lattice. Only outer loops
    (see :func:`identify_nested_loops`) are used when computing the spacing.
    
    Args:
        path_to_cell: Path to a directory with condensin HDF5 files
            (e.g. ``cell_0/condensin1_steps:*.hdf5``).
        time_frames: Frame index or list of indices to analyse.
            - ``None`` – use the last available frame
            - ``int`` – single frame
            - ``list[int]`` – multiple frames
        min_distance: Minimal spacing (in beads) to keep; smaller distances are discarded.
        debug: If ``True``, print verbose diagnostic information.
        
    Returns:
        np.ndarray: 1D array with distances (in beads) between the right leg
        of one outer loop and the left leg of the next outer loop.
    """
    
    # Locate the condensin file
    path = Path(path_to_cell)
    cond_files = list(path.glob("condensin1_steps:*.hdf5"))
    
    if not cond_files:
        raise FileNotFoundError(f"No condensin files found in {path_to_cell}")
    
    cond_file = cond_files[0]
    
    # Load condensin positions
    with h5py.File(cond_file, mode='r') as f:
        total_frames = f['positions'].shape[1]
        
        # Determine which frames to analyse
        if time_frames is None:
            time_frames = [total_frames - 1]  # Use the last frame by default
        elif isinstance(time_frames, int):
            time_frames = [time_frames]  # Convert a single index to a list
        
        # Validate requested frames
        valid_frames = [t for t in time_frames if 0 <= t < total_frames]
        if not valid_frames:
            raise ValueError(f"No valid time frames in request. Available frames: {total_frames}")
        
        # Collect distances across all requested frames
        all_distances = []
        for frame in valid_frames:
            positions = f['positions'][0, frame]
            
            # Valid extruders have both legs at positive positions
            valid_extruders = np.all(positions > 0, axis=1)
            valid_positions = positions[valid_extruders]
            
            if debug:
                print(f"Frame {frame}, total extruders: {len(valid_positions)}")
            
            if len(valid_positions) == 0:
                continue
            
            # Cast to integers for robust comparisons
            valid_positions = valid_positions.astype(int)
            
            # Ensure left_leg < right_leg for every extruder
            # This is important for computing distances between different extruders
            left_legs = np.min(valid_positions, axis=1)
            right_legs = np.max(valid_positions, axis=1)
            valid_positions = np.column_stack([left_legs, right_legs])
            
            # Exclude nested loops
            outer_indices, nested_indices = identify_nested_loops(valid_positions)
            
            if debug:
                print(f"Found nested loops: {len(nested_indices)}, outer loops: {len(outer_indices)}")
            
            if len(outer_indices) == 0:
                continue
            
            # Keep only outer loops
            outer_loops = valid_positions[outer_indices]
            
            # Sort outer loops by left leg position so that neighbours are ordered
            sorted_indices = np.argsort(outer_loops[:, 0])
            outer_loops = outer_loops[sorted_indices]
            
            if debug:
                print("Sorted outer loops (left_leg, right_leg):")
                for i, (left, right) in enumerate(outer_loops):
                    print(f"Loop {i}: ({left}, {right})")
            
            # Distances between right leg of loop i and left leg of loop i+1
            frame_distances = []
            for i in range(len(outer_loops) - 1):
                right_leg_current = outer_loops[i, 1]
                left_leg_next = outer_loops[i + 1, 0]
                
                # Distance is (left_next - right_current - 1):
                # subtract 1 bead to ignore the single-bead spacer between loops.
                distance = left_leg_next - right_leg_current - 1
                
                if debug:
                    print(f"Distance between loops {i} and {i+1}: {distance}")
                    print(f"  Right leg of loop {i}: {right_leg_current}")
                    print(f"  Left leg of loop {i+1}: {left_leg_next}")
                
                if distance >= min_distance:
                    frame_distances.append(distance)
            
            all_distances.extend(frame_distances)
        
        return np.array(all_distances)

def analyze_extruder_spacing(
    path_to_cells: str,
    cell_nums: list = None,
    time_frames: list = None,
    min_distance: int = 0,
    debug: bool = False
) -> dict:
    """
    Analyse spacing between neighbouring extruders for many cells / time points.
    
    Args:
        path_to_cells: Path to directory with cell subfolders (e.g. ``cell_0``, ``cell_1``).
        cell_nums: List of integer cell indices to analyse. ``None`` means all cells.
        time_frames: List of frame indices to analyse. ``None`` means last frame only.
        min_distance: Minimal allowed spacing (in beads).
        debug: If ``True``, print detailed progress information.
        
    Returns:
        dict: Results dictionary with per‑cell statistics and aggregated distances.
    """
    import os
    import numpy as np
    from pathlib import Path
    
    path = Path(path_to_cells)
    
    # Determine which cells to analyse
    if cell_nums is None:
        cells = list(path.glob("cell_*/"))
        cells.sort()  # Sort for reproducibility
    else:
        cells = [path / f"cell_{num}" for num in sorted(cell_nums)]
    
    all_distances = []
    results = {"cell_data": {}}
    
    for cell_path in cells:
        cell_name = cell_path.name
        
        if debug:
            print(f"\nProcessing cell: {cell_name}")
        
        try:
            distances = calculate_adjacent_extruder_legs_distance(
                str(cell_path), 
                time_frames=time_frames,
                min_distance=min_distance,
                debug=debug
            )
            
            if len(distances) > 0:
                results["cell_data"][cell_name] = {
                    "mean": float(np.mean(distances)),
                    "median": float(np.median(distances)),
                    "std": float(np.std(distances)),
                    "min": float(np.min(distances)),
                    "max": float(np.max(distances)),
                    "count": int(len(distances)),
                    "distances": distances.tolist()
                }
                all_distances.extend(distances.tolist())
                
                if debug:
                    print(f"  Distances found: {len(distances)}")
                    print(f"  Mean spacing: {np.mean(distances):.2f}")
                    print(f"  Unique distances: {np.unique(distances)}")
            else:
                if debug:
                    print(f"  No distances found for {cell_name}")
        except Exception as e:
            print(f"Error while processing {cell_name}: {e}")
    
    # Aggregate statistics across all cells
    if all_distances:
        results["overall"] = {
            "mean": float(np.mean(all_distances)),
            "median": float(np.median(all_distances)),
            "std": float(np.std(all_distances)),
            "min": float(np.min(all_distances)),
            "max": float(np.max(all_distances)),
            "count": int(len(all_distances)),
            "distances": all_distances
        }
    
    return results


def plot_extruder_spacing_histogram(
    spacing_data: dict,
    bin_size: int = 5,   # in beads
    max_distance: int = 250,  # in beads
    title: str = "Distribution of distances between neighbouring extruders",
    y_log: bool = True,  # use logarithmic scale on Y
    min_y: float = 0.5   # minimal Y value for logarithmic scale
):
    """
    Plot a histogram of distances between neighbouring extruders.
    
    Args:
        spacing_data: Dictionary produced by :func:`analyze_extruder_spacing`.
        bin_size: Histogram bin width in beads.
        max_distance: Maximum distance (in beads) to display.
        title: Plot title.
        y_log: Whether to use logarithmic scale on the y‑axis.
        min_y: Minimal y value when using log scale.
        
    Returns:
        go.Figure: Plotly figure with the spacing histogram.
    """
    import plotly.graph_objects as go
    import numpy as np
    
    fig = go.Figure()
    
    # Global histogram over all cells
    if "overall" in spacing_data:
        distances = np.array(spacing_data["overall"]["distances"])
        # Filter by maximal distance
        distances = distances[distances <= max_distance]
        
        # Use explicit integer bins to respect discreteness (distance in beads)
        bins = np.arange(0, max_distance + bin_size, bin_size)
        
        # Compute histogram manually for finer control
        hist, bin_edges = np.histogram(distances, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Max value for axis scaling
        max_y = max(hist) * 1.2
        
        fig.add_trace(go.Bar(
            x=bin_centers,
            y=hist,
            width=bin_size * 0.9,  # Small gap between bars for visual separation
            name="All cells",
            marker_color='blue',
            opacity=0.7,
            hovertemplate="Distance: %{x} beads<br>Count: %{y}<extra></extra>"
        ))
        
        # Add a vertical line at the mean spacing
        if y_log:
            fig.add_trace(go.Scatter(
                x=[spacing_data["overall"]["mean"], spacing_data["overall"]["mean"]],
                y=[min_y, max_y],
                mode="lines",
                name=f"Mean ({spacing_data['overall']['mean']:.1f})",
                line=dict(color="red", width=2, dash="dash"),
                hoverinfo="name"
            ))
        else:
            # For linear scale, use a shape spanning full y‑axis
            fig.add_shape(
                type="line",
                x0=spacing_data["overall"]["mean"],
                y0=0,
                x1=spacing_data["overall"]["mean"],
                y1=1,
                yref="paper",
                line=dict(color="red", width=2, dash="dash"),
            )
            
            # Add a text label for the mean
            fig.add_annotation(
                x=spacing_data["overall"]["mean"],
                y=max_y * 0.95,
                text=f"Mean: {spacing_data['overall']['mean']:.1f} beads",
                showarrow=False,
                font=dict(color="red")
            )
        
        # Annotation with summary statistics
        stats_text = (
            f"Mean: {spacing_data['overall']['mean']:.1f} beads<br>"
            f"Median: {spacing_data['overall']['median']:.1f} beads<br>"
            f"Std: {spacing_data['overall']['std']:.1f} beads<br>"
            f"Count: {spacing_data['overall']['count']}<br>"
            f"Unique distances: {len(np.unique(distances))}"
        )
        
        fig.add_annotation(
            x=0.95,
            y=0.95,
            xref="paper",
            yref="paper",
            text=stats_text,
            showarrow=False,
            font=dict(size=12),
            align="right",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1,
            borderpad=4
        )
    
    # Layout configuration
    fig.update_layout(
        title=title,
        xaxis_title="Distance, beads",
        yaxis_title=f"Frequency{' (log scale)' if y_log else ''}",
        template="plotly_white",
        bargap=0.1,
        margin=dict(l=80, r=80, t=60, b=60)
    )
    
    # Optional log scale on y axis
    if y_log:
        fig.update_yaxes(
            type="log", 
            range=[np.log10(min_y), np.log10(max_y)],
        )
    
    return fig

def calculate_own_extruder_legs_distance(
    path_to_cell: str,
    time_frames=None,
    min_distance: int = 0,
    debug: bool = False,
    smc_file: str = 'condensin2_steps:0-5400',
) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Compute distances between legs of each individual extruder loop.

    Three sets of distances are returned:
    - inner (nested) loops only
    - outer (non-nested) loops only
    - all loops

    Distances are expressed in beads along the 1D lattice and correspond
    to loop sizes used in the article (loop size × bead_size gives bp).
    
    Args:
        path_to_cell: Path to a directory with condensin HDF5 files.
        time_frames: Frame index or list of indices to analyse.
            ``None`` – last frame; ``int`` – single frame; list – multiple.
        min_distance: Minimal loop size (in beads) to keep.
        debug: If ``True``, print detailed diagnostics.
        
    Returns:
        Tuple of three 1D arrays:
            (inner_loops, outer_loops, all_loops), each in bead units.
    """
    import h5py
    import numpy as np
    import os
    from pathlib import Path
    
    # Locate condensin file(s) for this cell
    path = Path(path_to_cell)
    cond_files = list(path.glob(f"{smc_file}*.hdf5"))
    
    if not cond_files:
        raise FileNotFoundError(f"No SMC files matching '{smc_file}' found in {path_to_cell}")
    
    cond_file = cond_files[0]
    
    # Load condensin positions
    with h5py.File(cond_file, mode='r') as f:
        total_frames = f['positions'].shape[1]
        
        # Determine which frames to analyse
        if time_frames is None:
            time_frames = [total_frames - 1]  # Use the last frame by default
        elif isinstance(time_frames, int):
            time_frames = [time_frames]  # Convert a single index to a list
        
        # Validate requested frames
        valid_frames = [t for t in time_frames if 0 <= t < total_frames]
        if not valid_frames:
            raise ValueError(f"No valid time frames in request. Available frames: {total_frames}")
        
        # Collect distances for all requested frames
        all_outer = []
        all_inner = []
        all_all = []
        for frame in valid_frames:
            positions = f['positions'][0, frame]
            
            # Valid extruders have non-zero positions for both legs
            valid_extruders = np.all(positions > 0, axis=1)
            valid_positions = positions[valid_extruders]
            
            if debug:
                print(f"Frame {frame}, total extruders: {len(valid_positions)}")
            
            if len(valid_positions) == 0:
                continue
            
            # Convert to integer indices for robust comparisons
            valid_positions = valid_positions.astype(int)
            
            # Ensure left_leg < right_leg for every extruder
            left_legs = np.min(valid_positions, axis=1)
            right_legs = np.max(valid_positions, axis=1)
            valid_positions = np.column_stack([left_legs, right_legs])
            
            # Separate nested and outer loops
            outer_indices, nested_indices = identify_nested_loops(valid_positions)
            
            if debug:
                print(f"Found nested loops: {len(nested_indices)}, outer loops: {len(outer_indices)}")
            
            if len(outer_indices) == 0:
                continue
            
            # Select nested and outer loops
            outer_loops = valid_positions[outer_indices]
            inner_loops = valid_positions[nested_indices]
            
            if debug:
                print("Sorted outer loops (left_leg, right_leg):")
                for i, (left, right) in enumerate(outer_loops):
                    print(f"Loop {i}: ({left}, {right})")
            
            # Distances between the two legs of outer loops (loop sizes)
            frame_distances = []
            for i in range(len(outer_loops)):
                right_leg = outer_loops[i, 1]
                left_leg = outer_loops[i, 0]
                
                # Loop size is |left - right| - 1; subtract one bead spacer
                distance = abs(left_leg - right_leg) - 1
                
                if debug:
                    print(f"  Outer loop {i} legs: left={left_leg}, right={right_leg}")
                
                if distance >= min_distance:
                    frame_distances.append(distance)
            
            all_outer.extend(frame_distances)
            
            frame_distances = []
            for i in range(len(inner_loops)):
                right_leg = inner_loops[i, 1]
                left_leg = inner_loops[i, 0]
                
                # Loop size is |left - right| - 1; subtract one bead spacer
                distance = abs(left_leg - right_leg) - 1
                
                if debug:
                    print(f"  Inner loop {i} legs: left={left_leg}, right={right_leg}")
                
                if distance >= min_distance:
                    frame_distances.append(distance)
            
            all_inner.extend(frame_distances)
            
            frame_distances = []
            for i in range(len(valid_positions)):
                right_leg = valid_positions[i, 1]
                left_leg = valid_positions[i, 0]
                
                # Loop size is |left - right| - 1; subtract one bead spacer
                distance = abs(left_leg - right_leg) - 1
                
                if debug:
                    print(f"  Loop {i} legs: left={left_leg}, right={right_leg}")
                
                if distance >= min_distance:
                    frame_distances.append(distance)
            
            all_all.extend(frame_distances)
        
        return np.array(all_inner), np.array(all_outer), np.array(all_all)

def analyze_extruder_loops(
    path_to_cells: str,
    cell_nums: list = None,
    time_frames: list = None,
    min_distance: int = 0,
    debug: bool = False,
    smc_file: str = 'condensin1',
) -> dict:
    """
    Analyse distributions of loop sizes across many cells / time frames.

    This function aggregates loop sizes (in beads) produced by
    :func:`calculate_own_extruder_legs_distance` into three collections:
    inner loops, outer loops and all loops.
    
    Args:
        path_to_cells: Path to directory with cell subfolders.
        cell_nums: List of cell indices to analyse, or ``None`` for all.
        time_frames: List of frame indices, or ``None`` for last frame only.
        min_distance: Minimal loop size (in beads) to keep.
        debug: If ``True``, print verbose progress information.
        smc_file: Prefix pattern for SMC files (e.g. ``'condensin1'``).
        
    Returns:
        dict: Dictionary with keys ``'inner_agr'``, ``'outer_agr'`` and
        ``'all_agr'`` (when present), each containing a list of loop sizes.
    """
    import os
    import numpy as np
    from pathlib import Path
    
    path = Path(path_to_cells)
    
    # Determine which cells to analyse
    if cell_nums is None:
        cells = list(path.glob("cell_*/"))
        cells.sort()  # Sort for reproducibility
    else:
        cells = [path / f"cell_{num}" for num in sorted(cell_nums)]
    
    inner_loops = []
    outer_loops = []
    all_loops = []
    results = {}
    
    for cell_path in cells:
        cell_name = cell_path.name
        
        if debug:
            print(f"\nProcessing cell: {cell_name}")
        
        try:
            inner, outer, all_all = calculate_own_extruder_legs_distance(
                str(cell_path), 
                time_frames=time_frames,
                min_distance=min_distance,
                debug=debug,
                smc_file = smc_file
            )
            
            if len(inner) > 0:
                inner_loops.extend(inner.tolist())
            if len(outer) > 0:
                outer_loops.extend(outer.tolist())
            if len(all_all) > 0:
                all_loops.extend(all_all.tolist())
                
                if debug:
                    print(f"  Loops found: {len(all_all)}")
                    print(f"  Mean loop size: {np.mean(all_all):.2f}")
                    print(f"  Unique loop sizes: {np.unique(all_all)}")
            else:
                if debug:
                    print(f"  No loops found for {cell_name}")
        except Exception as e:
            print(f"Error while processing {cell_name}: {e}")
    
    # Aggregate lists of loop sizes
    if inner_loops:
        results["inner_agr"] = inner_loops
    if outer_loops:
        results["outer_agr"] = outer_loops
    if all_loops:
        results["all_agr"] = all_loops
    
    return results

def plot_extruder_loops_histograms(
    loops_data: dict,
    bin_size: int = 5,
    max_distance: int = 250,
    title: str = "Distribution of extruder loop sizes",
    min_y: float = 0.,
    loop_types: list = ['all_agr'],
    colors: list = ['blue'],
    height: int = 300,
    width: int = 1000,
):
    """
    Plot normalized loop-size density curves for several loop sets.

    Args:
        loops_data: Dictionary with aggregated loop sizes (see :func:`analyze_extruder_loops`).
        bin_size: Bin width (in beads) used for constructing densities.
        max_distance: Maximal loop size to display (in beads).
        title: Plot title.
        min_y: Minimal y-axis value (useful for log-scale overlays).
        loop_types: Keys from ``loops_data`` to plot (e.g. ``['all_agr']``).
        colors: List of colors for each loop type.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        go.Figure: Plotly figure with density curves over loop size.
    """
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    import numpy as np


    fig = go.Figure()

    # Create kernel density curves using figure_factory
    hist_data = [np.array(loops_data[loop_type]) for loop_type in loop_types]
    distplot = ff.create_distplot(hist_data,
                                    group_labels=loop_types,
                                    bin_size=bin_size,
                                    colors=colors,
                                    show_hist=False,
                                    show_rug=False)

    # y values for the first density curve
    y_values = distplot.data[0].y.copy()

    # Area under the first density curve (for normalization)
    first_area = (y_values) * bin_size
    
    
    # Normalize densities for all loop types to the same total area
    for i, loop_type in enumerate(loop_types):
        # Index of density trace for this loop type (densities come first in distplot)
        density_trace_index = i

        # Extract density values
        y_values = distplot.data[density_trace_index].y.copy()

        # Area under the density curve
        area = np.sum(y_values) * bin_size
        print(area)
        
        # Rescale to match area of the first curve
        normalized_y_values = y_values * (first_area/ area)

        # Update y values in-place
        distplot.data[density_trace_index].y = normalized_y_values
        
    # Add all density traces to the main figure
    for trace in distplot.data:
        
        fig.add_trace(trace)

    for i, loop_type in enumerate(loop_types):
        # Add vertical line at mean loop size
        mean_val = np.mean(loops_data[loop_type])
        
        max_y = np.max(distplot.data[i].y) * 1.2

        fig.add_trace(go.Scatter(
            x=[mean_val, mean_val],
            y=[0, max_y*1.2],
            mode="lines",
            line=dict(color=colors[i], width=2, dash="dash"),
            showlegend=False
        ))

        fig.add_annotation(
            x=mean_val,
            y=max_y * 0.95,
            text=f"Mean: {mean_val:.1f} beads",
            showarrow=False,
            font=dict(color=colors[i])
        )

    # Summary statistics for the main loop set (usually 'all_agr')
    if 'all_agr' in loops_data and loops_data['all_agr']:
        stats_text = (
            f"Mean: {np.mean(loops_data['all_agr']):.2f} beads<br>"
            f"Median: {np.median(loops_data['all_agr']):.1f} beads<br>"
            f"Std: {np.std(loops_data['all_agr']):.1f} beads<br>"
            f"Count: {len(loops_data['all_agr'])}<br>"
            f"Unique sizes: {len(np.unique(loops_data['all_agr']))}<br>"
        )

        fig.add_annotation(
            x=0.95,
            y=0.95,
            xref="paper",
            yref="paper",
            text=stats_text,
        showarrow=False,
        font=dict(size=12),
        align="right",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="rgba(0, 0, 0, 0.3)",
        borderwidth=1,
        borderpad=4
    )

    # Layout configuration
    fig.update_layout(
        title=title,
        xaxis_title="Loop size, beads",
        yaxis_title="Density (normalized)",
        template="plotly_white",
        margin=dict(l=80, r=80, t=60, b=60),
        height=height,
        width=width
    )

    return fig