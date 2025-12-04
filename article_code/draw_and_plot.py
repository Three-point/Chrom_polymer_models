# Standard library imports
import os
import re
import json
import logging
import h5py
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Scientific and numerical libraries
import numpy as np
import pandas as pd
from scipy import sparse as sp

# Hi-C data handling libraries
import cooler
import cooltools

# Visualization libraries
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.axes as mpl_axes
import seaborn as sns
import matplotlib.pyplot as plt

# Local modules
from spatial_functions import bond_calculator

# Optional CUDA utils import (for contact finding)
from cuda_utils import find_contacts


# polychrom modules (for 3D polymer simulations and contact maps)
import polychrom.hdf5_format as hdf5
from polychrom.hdf5_format import HDF5Reporter, list_URIs, load_URI, load_hdf5_file
from polychrom.contactmaps import monomerResolutionContactMap, monomerResolutionContactMapSubchains, binnedContactMap

def slice_str_parser(chosen_slice_str):
    chosen_slice = chosen_slice_str.split(':')
    chosen_slice = [chosen_slice[0], *chosen_slice[1].split('-')]
    chosen_slice[1] = int(chosen_slice[1].replace(',', ''))
    chosen_slice[2] = int(chosen_slice[2].replace(',', ''))
    return chosen_slice


def create_cooler_file(
    output_uri: str,
    contact_rows: np.ndarray,
    contact_cols: np.ndarray,
    contact_data: np.ndarray,
    resolutions: List[int] = [1000],
    chrom_sizes: Dict[str, float] = {'chr0': 1e3},
    weight_mode: str = 'standard',
    telomere_padding: float = 0.0
) -> None:
    """
    Create a Cooler file with contact map for different resolutions.
    
    Args:
        output_uri: Path to save .mcool file
        contact_rows: Array with row coordinates of contacts
        contact_cols: Array with column coordinates of contacts
        contact_data: Array with contact weights
        resolutions: List of resolutions (binning sizes)
        chrom_sizes: Dictionary with chromosome sizes {name: size}
        weight_mode: Weight calculation mode ('standard', 'ones', 'all')
        telomere_padding: Size of telomeric regions at chromosome edges
    """
    # Validate parameters
    if not isinstance(resolutions, list):
        raise TypeError("resolutions must be a list of integers")
    
    # Pre-calculate cumulative positions
    chrom_names = list(chrom_sizes.keys())
    chrom_lengths = np.array([chrom_sizes[name] for name in chrom_names])
    cum_lengths = np.insert(np.cumsum(chrom_lengths), 0, 0)

    for resolution in resolutions:
        if not isinstance(resolution, int) or resolution <= 0:
            raise ValueError(f"Invalid resolution: {resolution}")

        all_bins = []
        pixel_data = {'bin1_id': [], 'bin2_id': [], 'count': []}
        
        for chrom_idx, (chrom_name, chrom_len) in enumerate(chrom_sizes.items()):
            # Create bins for chromosome with telomere padding
            total_length = chrom_len + 2 * telomere_padding
            starts = np.arange(0, total_length, resolution, dtype=int)
            ends = starts + resolution
            ends[-1] = total_length  # Adjust last bin
            
            chrom_bins = pd.DataFrame({
                'chrom': chrom_name,
                'start': starts,
                'end': ends
            })
            
            # Filter contacts for current chromosome
            mask = (
                (cum_lengths[chrom_idx] <= contact_rows) & 
                (contact_rows < cum_lengths[chrom_idx+1]) &
                (cum_lengths[chrom_idx] <= contact_cols) & 
                (contact_cols < cum_lengths[chrom_idx+1])
            )
            
            # Convert coordinates to local positions
            local_rows = (
                (contact_rows[mask] - cum_lengths[chrom_idx] + telomere_padding) // resolution
            ).astype(int)
            local_cols = (
                (contact_cols[mask] - cum_lengths[chrom_idx] + telomere_padding) // resolution
            ).astype(int)
            
            # Create sparse contact matrix
            contact_matrix = sp.coo_matrix(
                (contact_data[mask], (local_rows, local_cols)),
                shape=(len(chrom_bins), len(chrom_bins)),
                dtype=float
            )
            contact_matrix.sum_duplicates()
            contact_matrix.eliminate_zeros()
            
            # Add weights
            active_bins = np.unique(np.concatenate([
                contact_matrix.row, contact_matrix.col
            ]))
            if weight_mode == 'standard':
                chrom_bins['weight'] = 0.0
                chrom_bins.loc[active_bins, 'weight'] = 1.0
            elif weight_mode == 'ones':
                chrom_bins['weight'] = 1.0
            elif weight_mode == 'all':
                chrom_bins['standard_weight'] = 0.0
                chrom_bins.loc[active_bins, 'standard_weight'] = 1.0
                chrom_bins['ones_weight'] = 1.0
            
            # Adjust global indices
            bin_offset = sum(len(b) for b in all_bins)
            pixel_data['bin1_id'].extend(contact_matrix.row + bin_offset)
            pixel_data['bin2_id'].extend(contact_matrix.col + bin_offset)
            pixel_data['count'].extend(contact_matrix.data)
            
            all_bins.append(chrom_bins)

        # Create full bins DataFrame
        full_bins = pd.concat(all_bins, ignore_index=True)
        
        # Create Cooler file
        cooler.create_cooler(
            cool_uri=f"{output_uri}::/resolutions/{resolution}",
            bins=full_bins,
            pixels=pixel_data,
            dtypes={'bin1_id': int, 'bin2_id': int, 'count': float},
            mode='w' if resolution == resolutions[0] else 'a'
        )
        

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def model_contacts_reader(
    path_to_file: Path,
    path_to_1d: Path,
    beads_gr_size: int = 1,
    bin_size: int = 5,
    cutoff: float = 50.0,
    slicing: Tuple[int, int, int] = (99, 1000, 100),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Process polymer simulation data to generate contact map.
    
    Args:
        path_to_file: Path to directory with simulation files
        path_to_1d: Path to directory with bead size metadata
        beads_gr_size: Bead group size (default: 1)
        bin_size: Bin size for aggregation (default: 5)
        cutoff: Contact distance threshold (default: 50.0)
        slicing: File selection slices (start, stop, step)
        
    Returns:
        Tuple with contact data and total genome length:
        - rows: Array of row coordinates
        - cols: Array of column coordinates
        - data: Array of contact weights
        - genome_length: Total genome length
        
    Raises:
        ValueError: For invalid input parameters
        FileNotFoundError: If files are not found
    """
    # Validate input parameters
    if bin_size <= 0:
        raise ValueError("bin_size must be a positive integer")
    if not path_to_file.exists():
        raise FileNotFoundError(f"Directory {path_to_file} not found")

    # Load and process file URIs
    start, stop, step = slicing
    try:
        file_uris = hdf5.list_URIs(str(path_to_file))[start:stop:step]
    except Exception as e:
        logger.error(f"Error getting URIs: {e}")
        raise

    # Load bead metadata
    bead_sizes_path = path_to_1d / 'bead_sizes.json'
    try:
        with bead_sizes_path.open('r') as f:
            bead_sizes = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading bead_sizes.json: {e}")
        raise

    total_beads = len(bead_sizes)
    num_bins = (total_beads + bin_size - 1) // bin_size  # Round up

    # Vectorized aggregation of bead sizes
    bin_indices = np.arange(total_beads) // bin_size
    binned_bead_sizes = np.bincount(bin_indices, weights=bead_sizes, minlength=num_bins)

    # Calculate genomic coordinates
    genomic_coords = np.cumsum(binned_bead_sizes, dtype=int)
    genome_length = genomic_coords[-1]

    # Process files
    rows, cols, data = [], [], []
    for file_idx, uri in enumerate(file_uris, 1):
        try:
            # Generate contact map
            contact_map, _ = binnedContactMap(
                filenames=[uri],
                chains=[[0, total_beads]],
                binSize=bin_size,
                cutoff=cutoff,
                n=1
            )
            
            # Convert to upper triangular matrix
            contact_map += np.eye(*contact_map.shape, dtype=contact_map.dtype)
            contact_map = sp.coo_matrix(np.triu(contact_map), dtype=float)

            # Calculate positions in genomic bins
            row_offsets = np.random.randint(1, binned_bead_sizes[contact_map.row])
            col_offsets = np.random.randint(1, binned_bead_sizes[contact_map.col])
            
            rows.extend(genomic_coords[contact_map.row] - row_offsets)
            cols.extend(genomic_coords[contact_map.col] - col_offsets)
            data.extend(contact_map.data)

            logger.info(f"Processed file {file_idx}/{len(file_uris)}")

        except Exception as e:
            logger.warning(f"Error processing file {uri}: {e}")
            continue

    return (
        np.array(rows), 
        np.array(cols), 
        np.array(data), 
        genome_length
    )

def data_loader(file_path: Path, target_resolution: int):
    """
    Load data from .mcool file.
    
    Args:
        file_path: Full path to file
        target_resolution: Target resolution
        
    Returns:
        clr: Cooler object
        bins: Bins
        genome_size: Genome size
        resolution: Actual resolution used
    """
    import cooler
    import h5py
    import logging
    
    logger = logging.getLogger(__name__)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} not found")
    
    # Load .mcool file
    with h5py.File(file_path, 'r') as f:
        resolutions = list(f['resolutions'].keys())
        resolution_str = str(target_resolution)
        
        if resolution_str not in resolutions:
            logger.warning(f"Resolution {target_resolution} not available. Available resolutions: {resolutions}")
            # Try to find closest available resolution
            if resolutions:
                closest_resolution = min(resolutions, key=lambda x: abs(int(x) - target_resolution))
                logger.info(f"Using closest available resolution: {closest_resolution}")
                resolution_str = closest_resolution
            else:
                raise ValueError(f"Resolution {target_resolution} not available and no alternatives")
    
    clr = cooler.Cooler(f'{file_path}::/resolutions/{resolution_str}')
    
    # Get bins and genome size
    bins = clr.bins()[:]
    genome_size = clr.chromsizes.sum()
    
    return clr, bins, genome_size, int(resolution_str)

def load_cvd_from_csv(csv_name, data_for_figs_dir=None):
    """
    Load CVD (Contact vs Distance) data from CSV file.
    
    Args:
        csv_name: Name of CSV file (without .csv extension) or full path
        data_for_figs_dir: Directory containing CSV files (required if csv_name is not a full path)
    
    Returns:
        pandas.DataFrame: CVD data with columns: s_bp, balanced.avg.smoothed.agg, der
    """
    import pandas as pd
    from pathlib import Path
    
    if Path(csv_name).suffix == '.csv':
        csv_path = Path(csv_name)
    else:
        if data_for_figs_dir is None:
            csv_path = Path(f"{csv_name}.csv")
        else:
            csv_path = Path(data_for_figs_dir) / f"{csv_name}.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df


def _load_data_from_params(name, params):
    """
    Helper function to load data from params dict.
    
    Supports two modes:
    - params['csv_path']: path to CSV file (will be loaded)
    - params['data']: already loaded DataFrame
    
    Args:
        name: Dataset name (for error messages)
        params: Parameters dict
    
    Returns:
        pandas.DataFrame or None if loading failed
    """
    if 'csv_path' in params:
        import pandas as pd
        from pathlib import Path
        
        # Convert to Path object (handles both str and Path inputs)
        csv_path = Path(params['csv_path'])
        
        # Check if file exists
        if not csv_path.exists():
            print(f"⚠ CSV file not found: {csv_path}, skipping {name}")
            return None
        
        # Load CSV and return DataFrame
        try:
            return pd.read_csv(str(csv_path))
        except Exception as e:
            print(f"⚠ Error loading {name} from {csv_path}: {e}")
            return None
            
    elif 'data' in params:
        return params['data']
    else:
        print(f"⚠ No 'csv_path' or 'data' for {name}, skipping")
        return None


def calculate_contact_vs_distance(view_df, target_resolution, sample_id, smooth=False, sigma=None):
    """
    Calculate contact frequency vs genomic distance using cooltools.
    
    Args:
        view_df: DataFrame with genomic regions (``chrom``, ``start``, ``end``, ``name``).
        target_resolution: Target bin size in base pairs.
        sample_id: Path-like identifier of a Cooler file. This should normally be
            a full path to a ``.cool`` or ``.mcool`` file. If no file extension
            is provided, ``.mcool`` is assumed and appended.
        smooth: Whether to apply smoothing to expected contact probabilities.
        sigma: Smoothing parameter (in log10 distance units) passed to cooltools.
        
    Returns:
        pandas.DataFrame: Contact probability vs distance table with additional
        columns:
            - ``s_bp``: genomic distance in base pairs
            - ``balanced.avg.smoothed.agg``: smoothed, balanced contact frequency
            - ``der``: derivative of log P(s) with respect to log distance
    """
    import pandas as pd
    import numpy as np
    import cooltools
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Normalize sample_id to a Path and infer extension if missing
        file_path = Path(str(sample_id))
        if file_path.suffix == "":
            # Backwards compatibility: if no extension, assume .mcool
            file_path = file_path.with_suffix(".mcool")

        # Only .cool/.mcool are supported here
        if file_path.suffix not in {".cool", ".mcool"}:
            raise ValueError(f"Unsupported Cooler file extension: {file_path.suffix} in {file_path}")

        clr, _, _, target_resolution = data_loader(file_path, target_resolution)

        # Prepare a local view_df compatible with this cooler.
        # If the provided view_df is incompatible (e.g. wrong chromosome names),
        # fall back to a view frame constructed from clr.chromsizes.
        def _build_default_viewframe(cooler_obj):
            chromsizes = cooler_obj.chromsizes.reset_index()
            chromsizes.columns = ["chrom", "length"]
            vf = chromsizes.assign(start=0, end=chromsizes["length"])
            vf = vf[["chrom", "start", "end"]]
            vf["name"] = vf["chrom"]
            return vf

        if view_df is not None:
            local_view = view_df.iloc[:, :4]
        else:
            local_view = _build_default_viewframe(clr)
        
        try:
            cvd = cooltools.expected_cis(
                clr=clr,
                view_df=local_view,
                smooth=smooth,
                smooth_sigma=sigma if sigma is not None else 0.1,
                aggregate_smoothed=True,
                nproc=10,
                intra_only=True,
            )
        except ValueError as e:
            # If the provided view_df is not compatible with the cooler,
            # rebuild a default viewframe from chromsizes and retry once.
            if "view_df is not a valid viewframe or incompatible" in str(e):
                logger.warning(
                    "Provided view_df is not compatible with cooler; "
                    "rebuilding view_df from cooler chromsizes."
                )
                local_view = _build_default_viewframe(clr)
                cvd = cooltools.expected_cis(
                    clr=clr,
                    view_df=local_view,
                    smooth=smooth,
                    smooth_sigma=sigma if sigma is not None else 0.1,
                    aggregate_smoothed=True,
                    nproc=10,
                    intra_only=True,
                )
            else:
                raise
        
        # Add column with distance in base pairs
        cvd['s_bp'] = cvd['dist'] * target_resolution
        
        # Exclude very short distances
        cvd['balanced.avg.smoothed.agg'].loc[cvd['dist'] < 2] = np.nan
        
        # Remove duplicates by distance
        cvd_merged = cvd.drop_duplicates(subset=['dist'])
        
        # Calculate derivative for slope analysis
        cvd_merged['der'] = np.gradient(
            np.log(cvd_merged['balanced.avg.smoothed.agg']),
            np.log(cvd_merged['s_bp'])
        )
        
        return cvd_merged
    
    except Exception as e:
        logger.error(f"Error calculating CVD: {e}")
        raise RuntimeError("Failed to calculate CVD") from e
    

def plot_contact_frequency(
    cvd_data: pd.DataFrame,
    x_limits: Tuple[float, float] = (1e4, 1e7),
    y_limits: Tuple[float, float] = (-2, 0),
    line_color: str = 'red',
    normalize: bool = False,
    alignment_point: int = 3,
    line_style: str = 'solid',
    title: str = '',
    show_legend: bool = True
) -> go.Figure:
    """
    Visualize contact frequency and derivative data using Plotly.
    
    Args:
        cvd_data: DataFrame with CVD data
        x_limits: X-axis limits
        y_limits: Derivative Y-axis limits
        line_color: Line color
        normalize: Normalize data
        alignment_point: Normalization point
        line_style: Line style ('solid', 'dash', 'dot')
        title: Plot title
        show_legend: Show legend
        
    Returns:
        Plotly Figure object
        
    Raises:
        ValueError: For invalid data
    """
    if cvd_data.empty or 'balanced.avg.smoothed.agg' not in cvd_data:
        raise ValueError("Invalid CVD data")

    # Create subplots
    fig = make_subplots(
        rows=2, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Contact frequency', 'Derivative')
    )

    # Prepare data
    y_values = cvd_data['balanced.avg.smoothed.agg'].copy()
    if normalize:
        y_values /= y_values.iloc[alignment_point]

    # Top plot: contact frequency
    fig.add_trace(
        go.Scatter(
            x=cvd_data['s_bp'],
            y=y_values,
            mode='lines',
            name=title if title else 'Contacts',
            line=dict(
                color=line_color,
                dash='solid' if line_style == 'solid' else 'dash'
            ),
            showlegend=show_legend
        ),
        row=1, col=1
    )

    # Bottom plot: derivative
    fig.add_trace(
        go.Scatter(
            x=cvd_data['s_bp'],
            y=cvd_data['der'],
            mode='lines',
            name=f'{title} (derivative)' if title else 'Derivative',
            line=dict(
                color=line_color,
                dash='solid' if line_style == 'solid' else 'dash'
            ),
            showlegend=show_legend
        ),
        row=2, col=1
    )

    # Configure axes and appearance
    fig.update_xaxes(
        type="log",
        range=[np.log10(x_limits[0]), np.log10(x_limits[1])],
        title_text="Genomic distance, bp",
        gridcolor='lightgray',
        row=2, col=1
    )

    fig.update_yaxes(
        type="log",
        range=[-6, np.log10(2)],
        title_text="Contact frequency",
        gridcolor='lightgray',
        row=1, col=1
    )

    fig.update_yaxes(
        range=y_limits,
        title_text="Derivative",
        gridcolor='lightgray',
        row=2, col=1
    )

    # General layout settings
    fig.update_layout(
        height=800,
        width=600,
        template='plotly_white',
        showlegend=show_legend,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        ),
        margin=dict(l=80, r=20, t=60, b=60)
    )

    return fig

def plot_multiple_contacts(
    data_dict: Dict[str, Dict[str, Any]],
    x_limits: Tuple[float, float] = (1e4, 1e7),
    y_limits: Tuple[float, float] = (-2, 0),
    normalize: bool = False,
    target_resolution: int = 200,
    alignment_s_bp: float = None
) -> go.Figure:
    """
    Create interactive plot for multiple datasets.
    
    Args:
        data_dict: Dictionary {name: {
            'data': DataFrame,
            'color': str,
            'style': str,
            'alpha': float  # Line transparency
        }}
        x_limits: X-axis limits
        y_limits: Derivative Y-axis limits
        normalize: Normalize data
        target_resolution: Target resolution
        alignment_s_bp: Parameter for alignment in base pairs
    """
    # Create subplots with shared X-axis
    fig = make_subplots(
        rows=2, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        # subplot_titles=(
        #     # f'Contact frequency (normalized at {alignment_s_bp} bp)' if normalize
        #     # else 'Contact frequency',
        #     ' ',
        #     'Derivative'
        # )
    )
    
    # Add plots for each dataset
    for name, params in data_dict.items():
        data = _load_data_from_params(name, params)
        if data is None:
            continue
        y_values = data['balanced.avg.smoothed.agg'].copy()
        alpha = params.get('alpha', 1.0)  # Get alpha value if specified, else 1.0

        if normalize and alignment_s_bp is not None:
            # Find index of closest s_bp value to alignment_s_bp
            closest_index = (data['s_bp'] - alignment_s_bp).abs().idxmin()
            norm_value = y_values.iloc[closest_index]
            if pd.isna(norm_value):
                # Find nearest non-NaN point in ascending order
                for i in range(closest_index + 1, len(y_values)):
                    if not pd.isna(y_values.iloc[i]):
                        norm_value = y_values.iloc[i]
                        break
            y_values /= norm_value

        # Top plot: contact frequency
        fig.add_trace(
            go.Scatter(
                x=data['s_bp'],
                y=y_values,
                mode='lines',
                name=name,
                line=dict(
                    color=params.get('color', 'blue'),
                    dash='solid' if params.get('style', 'solid') == 'solid' else 'dash'
                ),
                legendgroup=name,
                showlegend=True,
                opacity=alpha  # Set trace opacity
            ),
            row=1, col=1
        )

        # Bottom plot: derivative
        fig.add_trace(
            go.Scatter(
                x=data['s_bp'],
                y=data['der'],
                mode='lines',
                name=name,
                line=dict(
                    color=params.get('color', 'blue'),
                    dash='solid' if params.get('style', 'solid') == 'solid' else 'dash'
                ),
                legendgroup=name,
                showlegend=False,
                opacity=alpha  # Set transparency
            ),
            row=2, col=1
        )

    # Configure axes and appearance
    fig.update_xaxes(
        type="log",
        range=[np.log10(x_limits[0]), np.log10(x_limits[1])],
        title_text="Genomic distance, bp",
        gridcolor='lightgray',
        matches='x',  # Synchronize zoom/pan between subplots on x-axis
        exponentformat="power"
        # tickvals=[10**i for i in range(2, 10)],  # Optional explicit tick positions
        # ticktext=[r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$', r'$10^6$', r'$10^7$', r'$10^8$', r'$10^9$']  # Optional tick labels in exponential form
    )

    fig.update_yaxes(
        type="log",
        range=[-6, np.log10(2)],
        title_text="P(s)",
        gridcolor='lightgray',
        row=1, col=1,
        exponentformat="power"
    )

    fig.update_yaxes(
        range=y_limits,
        title_text="Derivative",
        gridcolor='lightgray',
        row=2, col=1
    )

    # General layout settings
    fig.update_layout(
        height=800,
        width=1000,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1
        ),
        margin=dict(l=80, r=120, t=60, b=60)
    )

    return fig


def format_metrics_results(metrics_dict, precision=3):
    """
    Format and display comparison results in readable form.
    
    Args:
        metrics_dict: Dictionary with metric tables
        precision: Number of decimal places
    """
    import pandas as pd
    import numpy as np
    
    # Dictionary with metric descriptions
    metric_descriptions = {
        'r2_reg': 'R² (linear regression) - closer to 1 is better',
        'r2_direct': 'R² (direct) - closer to 1 is better',
        'rmse': 'Root mean square error (RMSE) - lower is better',
        'mae': 'Mean absolute error (MAE) - lower is better',
        'max_error': 'Maximum error - lower is better'
    }
    
    results = {}
    
    # Format each metric and add to results
    for metric_name, metric_df in metrics_dict.items():
        if metric_name == 'figure':
            continue
            
        # Round to specified precision
        formatted_df = metric_df.round(precision)
        
        # Create empty structures for results
        best_indices = {}
        best_values = {}
        is_best = pd.DataFrame(False, index=formatted_df.index, columns=formatted_df.columns)
        
        # Process each column (reference) separately
        for col in formatted_df.columns:
            # Convert column to list of (index, value) pairs
            pairs = [(idx, val) for idx, val in zip(formatted_df.index, formatted_df[col])]
            # Filter only non-empty values
            valid_pairs = [(idx, val) for idx, val in pairs if not pd.isna(val)]
            
            if not valid_pairs:  # If all values are NaN
                best_indices[col] = "No data"
                best_values[col] = np.nan
                continue
            
            # Sort pairs
            if metric_name.startswith('r2'):
                # For R², larger values are better
                valid_pairs.sort(key=lambda x: x[1], reverse=True)
            else:
                # For errors, smaller values are better
                valid_pairs.sort(key=lambda x: x[1])
            
            # Get best value
            best_idx, best_val = valid_pairs[0]
            
            best_indices[col] = best_idx
            best_values[col] = best_val
            is_best.loc[best_idx, col] = True
        
        # Create styled table
        styled_df = formatted_df.style.apply(
            lambda col: ['font-weight: bold' if val else '' for val in is_best[col.name]], 
            axis=0
        )
        
        # Add to results
        results[metric_name] = {
            'description': metric_descriptions[metric_name],
            'dataframe': formatted_df,
            'styled_df': styled_df,
            'best_values': best_values,
            'best_indices': best_indices
        }
    
    # Display results
    print("=" * 80)
    print("P(s) CURVE COMPARISON RESULTS")
    print("=" * 80)
    
    for metric_name, result in results.items():
        print(f"\n{result['description']}")
        print("-" * 80)
        print(result['dataframe'])
        print("\nBest values:")
        
        for ref_name, best_idx in result['best_indices'].items():
            if best_idx == "No data":
                print(f"  '{ref_name}' - {best_idx}")
            else:
                best_value = result['best_values'][ref_name]
                print(f"  '{ref_name}' - '{best_idx}': {best_value}")
    
    return results

def compare_ps_curves(data_dict, reference_names, target_names=None, 
                      min_distance=10000, max_distance=4000000, 
                      alignment_s_bp=100000, normalize=True):
    """
    Compare P(s) curves in logarithmic scale using various metrics.
    Uses only common points in data, without interpolation.
    
    Args:
        data_dict: Dictionary with P(s) data for different samples
        reference_names: List of reference data names (to compare against)
        target_names: List of target data names (to compare). If None, all except reference_names
        min_distance: Minimum distance for comparison (in bp)
        max_distance: Maximum distance for comparison (in bp)
        alignment_s_bp: Normalization point in bp (if normalize=True)
        normalize: Whether to normalize data before comparison
        
    Returns:
        dict: Dictionary with tables of various metrics and plot
    """
    import numpy as np
    import pandas as pd
    from scipy import stats
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    # Ensure reference_names is a list
    if isinstance(reference_names, str):
        reference_names = [reference_names]
    
    # If target_names is not provided, use all other keys
    if target_names is None:
        target_names = [key for key in data_dict.keys() if key not in reference_names]
    elif isinstance(target_names, str):
        target_names = [target_names]
        
    # Create DataFrames to store different metrics
    metrics = {
            'r2_ps': pd.DataFrame(index=target_names, columns=reference_names),
            'r2_grad': pd.DataFrame(index=target_names, columns=reference_names),
            'rmse': pd.DataFrame(index=target_names, columns=reference_names),
            'mae': pd.DataFrame(index=target_names, columns=reference_names),
            'max_error': pd.DataFrame(index=target_names, columns=reference_names),
            'average_ratio': pd.DataFrame(index=target_names, columns=reference_names)
        }
    
    # Create a figure for visualization
    fig = make_subplots(rows=1, cols=1, subplot_titles=["P(s) curves in log scale"])
    
    # Loop over each reference dataset
    for ref_key in reference_names:
        if ref_key not in data_dict:
            print(f"Warning: {ref_key} not found in data_dict")
            continue
            
        # Use full P(s) curve for reference
        ref_df_full = _load_data_from_params(ref_key, data_dict[ref_key])
        if ref_df_full is None:
            continue
        ref_df_full = ref_df_full.copy()
        
        # Optionally normalize the full reference curve
        if normalize:
            # Find index of s_bp closest to alignment_s_bp on the full curve
            if alignment_s_bp in ref_df_full['s_bp'].values:
                # Exact match
                closest_index = ref_df_full[ref_df_full['s_bp'] == alignment_s_bp].index[0]
            else:
                # Nearest value
                closest_index = (ref_df_full['s_bp'] - alignment_s_bp).abs().idxmin()
            
            norm_value = ref_df_full.loc[closest_index, 'balanced.avg.smoothed.agg']
            
            if pd.isna(norm_value) or norm_value == 0:
                # Search for nearest non-NaN and non-zero point
                distance = 0
                while pd.isna(norm_value) or norm_value == 0:
                    distance += 1
                    idx_candidates = []
                    if closest_index - distance >= 0:
                        idx_candidates.append(closest_index - distance)
                    if closest_index + distance < len(ref_df_full):
                        idx_candidates.append(closest_index + distance)
                    
                    if not idx_candidates:
                        print(f"Failed to find suitable normalization point for {ref_key}")
                        break
                    
                    for idx in idx_candidates:
                        val = ref_df_full.loc[idx, 'balanced.avg.smoothed.agg']
                        if not pd.isna(val) and val != 0:
                            norm_value = val
                            break
            
            # Normalize full curve
            if not pd.isna(norm_value) and norm_value != 0:
                ref_df_full['balanced.avg.smoothed.agg'] /= norm_value
            else:
                print(f"Warning: normalization point for {ref_key} has value {norm_value}")
        
        # Now restrict curve to analysis range
        ref_df = ref_df_full[(ref_df_full['s_bp'] >= min_distance) & 
                              (ref_df_full['s_bp'] <= max_distance)].copy()
        
        if len(ref_df) == 0:
            print(f"Warning: no data for {ref_key} in range {min_distance}-{max_distance}")
            continue
        
        # Plot reference curve
        fig.add_trace(
            go.Scatter(
                x=ref_df['s_bp'],
                y=ref_df['balanced.avg.smoothed.agg'],
                mode='lines+markers',
                name=ref_key,
                line=dict(
                    color=data_dict[ref_key]['color'],
                    dash='solid'
                ),
                marker=dict(
                    size=8,
                    symbol='circle'
                )
            )
        )
        
        # Compare with target datasets
        for target_key in target_names:
            if target_key not in data_dict:
                print(f"Warning: {target_key} not found in data_dict")
                continue
            
            if target_key == ref_key:
                # Skip self-comparison
                continue
                
            try:
                # Get full target P(s) curve
                target_df_full = _load_data_from_params(target_key, data_dict[target_key])
                if target_df_full is None:
                    continue
                target_df_full = target_df_full.copy()
                
                # Optionally normalize target curve
                if normalize:
                    # Find index of s_bp closest to alignment_s_bp in the target curve
                    if alignment_s_bp in target_df_full['s_bp'].values:
                        # Exact match
                        closest_index = target_df_full[target_df_full['s_bp'] == alignment_s_bp].index[0]
                    else:
                        # Nearest value
                        closest_index = (target_df_full['s_bp'] - alignment_s_bp).abs().idxmin()
                    
                    norm_value = target_df_full.loc[closest_index, 'balanced.avg.smoothed.agg']
                    
                    if pd.isna(norm_value) or norm_value == 0:
                        # Search for nearest non-NaN and non-zero point
                        distance = 0
                        while pd.isna(norm_value) or norm_value == 0:
                            distance += 1
                            idx_candidates = []
                            if closest_index - distance >= 0:
                                idx_candidates.append(closest_index - distance)
                            if closest_index + distance < len(target_df_full):
                                idx_candidates.append(closest_index + distance)
                            
                            if not idx_candidates:
                                print(f"Failed to find suitable normalization point for {target_key}")
                                break
                            
                            for idx in idx_candidates:
                                val = target_df_full.loc[idx, 'balanced.avg.smoothed.agg']
                                if not pd.isna(val) and val != 0:
                                    norm_value = val
                                    break
                    
                    # Normalize full target curve
                    if not pd.isna(norm_value) and norm_value != 0:
                        target_df_full['balanced.avg.smoothed.agg'] /= norm_value
                    else:
                        print(f"Warning: normalization point for {target_key} has value {norm_value}")
                
                # Restrict to analysis distance range
                target_df = target_df_full[(target_df_full['s_bp'] >= min_distance) & 
                                           (target_df_full['s_bp'] <= max_distance)].copy()
                
                if len(target_df) == 0:
                    print(f"Warning: no data for {target_key} in range {min_distance}-{max_distance}")
                    continue
                
                # Add target curve to the plot
                fig.add_trace(
                    go.Scatter(
                        x=target_df['s_bp'],
                        y=target_df['balanced.avg.smoothed.agg'],
                        mode='lines+markers',
                        name=target_key,
                        line=dict(
                            color=data_dict[target_key]['color'],
                            dash=data_dict[target_key].get('style', 'solid')
                        ),
                        marker=dict(
                            size=6,
                            symbol='circle'
                        )
                    )
                )
                
                # Find common s_bp points between reference and target datasets
                common_s_bp = sorted(set(ref_df['s_bp']).intersection(set(target_df['s_bp'])))
                
                if len(common_s_bp) < 3:
                    print(f"Warning: not enough common points ({len(common_s_bp)}) to compare {target_key} and {ref_key}")
                    metrics['r2_reg'].loc[target_key, ref_key] = np.nan
                    metrics['r2_direct'].loc[target_key, ref_key] = np.nan
                    metrics['rmse'].loc[target_key, ref_key] = np.nan
                    metrics['mae'].loc[target_key, ref_key] = np.nan
                    metrics['max_error'].loc[target_key, ref_key] = np.nan
                    metrics['average_ratio'].loc[target_key, ref_key] = np.nan
                    continue
                
                # Extract values corresponding to common s_bp points
                ref_values = ref_df[ref_df['s_bp'].isin(common_s_bp)].sort_values('s_bp')
                target_values = target_df[target_df['s_bp'].isin(common_s_bp)].sort_values('s_bp')
                
                # Ensure that s_bp points are in the same order in both curves
                assert all(ref_values['s_bp'].values == target_values['s_bp'].values), "s_bp points do not match!"
                
                # Take logarithm of data for comparison
                ref_log_x = np.log10(ref_values['s_bp'].values)
                ref_log_y = np.log10(ref_values['balanced.avg.smoothed.agg'].values)
                target_log_y = np.log10(target_values['balanced.avg.smoothed.agg'].values)
                
                # Build mask of valid (finite and non-NaN) values
                valid_mask = ~np.isnan(ref_log_y) & ~np.isnan(target_log_y) & \
                             ~np.isinf(ref_log_y) & ~np.isinf(target_log_y)
                
                # Check that there are enough valid points for comparison
                if np.sum(valid_mask) < 3:
                    print(f"Warning: not enough valid common points ({np.sum(valid_mask)} of {len(common_s_bp)}) to compare {target_key} and {ref_key}")
                    metrics['r2_reg'].loc[target_key, ref_key] = np.nan
                    metrics['r2_direct'].loc[target_key, ref_key] = np.nan
                    metrics['rmse'].loc[target_key, ref_key] = np.nan
                    metrics['mae'].loc[target_key, ref_key] = np.nan
                    metrics['max_error'].loc[target_key, ref_key] = np.nan
                    metrics['average_ratio'].loc[target_key, ref_key] = np.nan
                    continue
                
                ref_log_x = ref_log_x[valid_mask]
                ref_log_y = ref_log_y[valid_mask]

                
                # Create 100 evenly spaced points in the range of ref_log_x
                num_points = 100
                log_min, log_max = ref_log_x.min(), ref_log_x.max()
                log_points = np.linspace(log_min, log_max, num_points)
                # For each point in log_points find the index of the nearest point from ref_log_x
                indices = np.abs(ref_log_x[:, None] - log_points).argmin(axis=0)[1:-1]

                # Compute R² between gradients of log P(s)
                ref_grad = np.gradient(ref_log_y, ref_log_x)
                target_grad = np.gradient(target_log_y, ref_log_x)
                r2_grad = r2_score(ref_grad[indices], target_grad[indices])



                # Keep only x-points near the original data
                ref_log_x = ref_log_x[indices]
                # Keep the corresponding y values
                ref_log_y = ref_log_y[indices]
                target_log_y = target_log_y[indices]



                # R² between log-transformed P(s) curves
                r2_ps = r2_score(ref_log_y, target_log_y)
                
                
                # Compute error metrics between curves
                rmse_val =np.sqrt(mean_squared_error(ref_log_y, target_log_y))
                mae_val = mean_absolute_error(ref_log_y, target_log_y)
                max_err = np.max(np.abs(ref_log_y - target_log_y))

                ratio = ref_log_y/target_log_y
                ratio_adj = np.where(ratio < 1, 1 / ratio, ratio)
                average_ratio = np.nanmean(ratio_adj)
                
                
                # Store all metrics in the results tables
                metrics['r2_ps'].loc[target_key, ref_key] = r2_ps
                metrics['r2_grad'].loc[target_key, ref_key] = r2_grad
                metrics['rmse'].loc[target_key, ref_key] = rmse_val
                metrics['mae'].loc[target_key, ref_key] = mae_val
                metrics['max_error'].loc[target_key, ref_key] = max_err
                metrics['average_ratio'].loc[target_key, ref_key] = average_ratio


                
            except Exception as e:
                print(f"Error while comparing {target_key} and {ref_key}: {e}")
                import traceback
                traceback.print_exc()
                metrics['r2_ps'].loc[target_key, ref_key] = np.nan
                metrics['r2_grad'].loc[target_key, ref_key] = np.nan
                metrics['rmse'].loc[target_key, ref_key] = np.nan
                metrics['mae'].loc[target_key, ref_key] = np.nan
                metrics['max_error'].loc[target_key, ref_key] = np.nan
                metrics['average_ratio'].loc[target_key, ref_key] = np.nan
    
    # Layout configuration
    fig.update_layout(
        title_text=(
            f"Comparison of P(s) curves (normalized at {alignment_s_bp} bp)"
            if normalize
            else "Comparison of P(s) curves"
        ),
        height=600,
        width=900,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        )
    )
    
    # Logarithmic axes for P(s) plot
    fig.update_xaxes(
        type="log", 
        title_text="Distance (bp)",
        range=[np.log10(min_distance), np.log10(max_distance)]
    )
    fig.update_yaxes(type="log", title_text="P(s)")
    
    # Add plot to results dictionary
    metrics['figure'] = fig
    
    return metrics

def make_ps(
    paths_to_3d: List[str],
    paths_to_1d: List[str],
    bin_size: int,
    exp_type: str,
    interest_keys: List[int],
    telomere_size: float = 0.0,
    cutoff: int = 11,
    chrom_slice: str = 'chr0:000,000,000-2,000,000',
    resolutions: List[int] = [200, 1000, 2000, 5000, 10000],
    output_uri: str = './coolmaps/tested_map_binsize5.mcool'
) -> None:
    """
    Generate contact maps and save them in .mcool format.

    Args:
        paths_to_3d: List of paths to folders with 3D data for each cell
        paths_to_1d: List of paths to folders with 1D data for each cell
        bin_size: Bin size
        exp_type: Experiment type
        interest_keys: List of block keys to process
        telomere_size: Size of telomeric regions at chromosome edges
        cutoff: Contact distance
        chrom_slice: String describing chromosome range
        resolutions: List of resolutions for contact map
        output_uri: Path to save .mcool file
    """
    # Parse selected chromosome slice
    chosen_slice = slice_str_parser(chrom_slice)
    start_coo = chosen_slice[1]
    chrom_size = chosen_slice[2]

    # Lists for storing data
    list_chrs = []

    for path_to_1d in paths_to_1d:
        list_chrs.append(chosen_slice[0])

    print(f'Started processing for {chosen_slice[0]} with cutoff {cutoff}')

    bead_sizes_path = Path(path_to_1d) / 'bead_sizes.json'
    try:
        with bead_sizes_path.open('r') as f:
            bead_sizes = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading bead_sizes.json: {e}")
        raise
    
    total_contacts = sp.csr_matrix((sum(bead_sizes), sum(bead_sizes)))
    prev_chr_size = start_coo
    chrom_dict = {}

    for chr_name, path_to_3d, path_to_1d in zip(list_chrs, paths_to_3d, paths_to_1d):
        print(f'Processing {path_to_3d}')
        rows, cols, data, _ = model_contacts_reader(
            Path(path_to_3d),
            Path(path_to_1d),
            bin_size=bin_size,
            beads_gr_size=1,
            cutoff=cutoff,
            slicing=(0, 300, 1)
        )

        print(max(rows), max(cols), len(bead_sizes))
        new_contacts = sp.csr_matrix((data, (rows, cols)), shape=(sum(bead_sizes), sum(bead_sizes)))
        total_contacts += new_contacts
        chrom_dict[chr_name] = chrom_size

    # Process symmetry
    rows_total, cols_total = total_contacts.nonzero()

    tmp_mask = rows_total > cols_total
    tmp_rows = cols_total[tmp_mask].copy()
    cols_total[tmp_mask] = rows_total[tmp_mask].copy()
    rows_total[tmp_mask] = tmp_rows
    total_contacts = sp.csr_matrix((total_contacts.data, (rows_total, cols_total)), shape=total_contacts.shape)

    # Create Cooler file
    if os.path.exists(output_uri):
        os.remove(output_uri)

    create_cooler_file(
        output_uri=output_uri,
        contact_rows=total_contacts.nonzero()[0],
        contact_cols=total_contacts.nonzero()[1],
        contact_data=total_contacts.data,
        resolutions=resolutions,
        chrom_sizes=chrom_dict,
        weight_mode='standard',
        telomere_padding=telomere_size
    )

    print(f'Done processing for {chosen_slice[0]} with cutoff {cutoff}')

    return total_contacts.nonzero(), total_contacts.data

def generate_contact_map_multiple_cells(
    paths_to_files: List[str],
    output_uri: str,
    interest_keys: List[int],
    cutoff: float = 11.0,
    resolutions: List[int] = [1000],
    chrom_sizes: Dict[str, float] = {'chr0': 1e3},
    bead_sizes: List[float] = None,
    weight_mode: str = 'standard',
    telomere_padding: float = 0.0,
    gpu_num = 0,
) -> None:
    """
    Generate contact maps in .mcool format based on data from multiple cells.

    Args:
        paths_to_files: List of paths to folders with h5 files for each cell
        output_uri: Path to save .mcool file
        interest_keys: List of block keys to process
        cutoff: Contact distance
        resolutions: List of resolutions for contact map
        chrom_sizes: Dictionary with chromosome sizes
        bead_sizes: List of bead sizes for correct genomic coordinate calculation
        weight_mode: Weight calculation mode ('standard', 'ones', 'all')
        telomere_padding: Size of telomeric regions at chromosome edges
        gpu_num: GPU number for contact finding
    """
    # Create dictionary to store positions from selected blocks
    positions_data = {}

    # Process each cell
    for path_to_file in paths_to_files:
        try:
            # Get cell number from path
            cell_num = int(re.search(r'cell_(\d+)', path_to_file).group(1))
        except:
            cell_num = 0

        # Get list of all h5 files in folder
        h5_files = [f for f in os.listdir(path_to_file) if f.endswith('.h5') and 'blocks' in f.lower()]

        # Process each file
        for file_name in h5_files:
            file_path = os.path.join(path_to_file, file_name)
            with h5py.File(file_path, 'r') as h5_file:
                # Check only blocks of interest
                for block_key in h5_file.keys():
                    if int(block_key) in interest_keys:
                        # Form unique key as combination of cell_num and block_key
                        unique_key = f"{cell_num}_{block_key}"
                        
                        # Extract positions from block and save to dictionary
                        if 'pos' in h5_file[block_key]:
                            positions_data[unique_key] = np.array(h5_file[block_key]['pos'])
                        else:
                            positions_data[unique_key] = np.array(h5_file[block_key])

    # Create contact matrices for each set of positions and sum them
    combined_matrix = None

    counter = 0
    for block_key, coordinates in positions_data.items():
        # Find contacts for current set of positions
        contacts = find_contacts(coordinates, cutoff, gpu_num)

        # Create symmetric sparse contact matrix
        n = coordinates.shape[0]  # number of beads
        rows = contacts[:, 0]
        cols = contacts[:, 1]

        # Create sparse matrix in COO format
        data = np.ones(len(rows), dtype=int)
        contact_matrix = sp.coo_matrix((data, (rows, cols)), shape=(n, n))

        # Convert to CSR format for more efficient operations
        contact_matrix = contact_matrix.tocsr()

        # Add to combined matrix
        if combined_matrix is None:
            combined_matrix = contact_matrix
        else:
            # Check that sizes match
            if combined_matrix.shape == contact_matrix.shape:
                combined_matrix += contact_matrix
            else:
                print(f"Skipping block {block_key} due to matrix size mismatch")
                
        logger.info(f"Processed file {counter}/{len(positions_data.items())}")
        counter += 1

    # Resulting contact matrix
    contact_matrix = combined_matrix if combined_matrix is not None else sp.csr_matrix((0, 0))

    # Create Cooler file
    contact_rows, contact_cols = contact_matrix.nonzero()  # Get indices of nonzero elements

    # Convert row and column indices to genomic coordinates
    if bead_sizes is None:
        bead_sizes = [1.0] * contact_matrix.shape[0]  # Assume bead size is 1

    print(max(contact_rows), len(bead_sizes), sum(bead_sizes))
    rows_genomic = contact_rows * np.array(bead_sizes)[contact_rows]
    cols_genomic = contact_cols * np.array(bead_sizes)[contact_cols]

    create_cooler_file(
        output_uri=output_uri,
        contact_rows=rows_genomic,  # Use genomic coordinates for rows
        contact_cols=cols_genomic,   # Use genomic coordinates for columns
        contact_data=contact_matrix.data,
        resolutions=resolutions,
        chrom_sizes=chrom_sizes,
        weight_mode=weight_mode,
        telomere_padding=telomere_padding
    )

    return rows_genomic, cols_genomic, contact_matrix.data



def analyze_metrics_vs_alignment(data_dict, reference_names, target_names=None, 
                               min_distance=10000, max_distance=4000000,
                               alignment_min_distance=None, alignment_max_distance=None,
                               max_points=20):
    """
    Analyze how comparison metrics change depending on normalization point.
    Normalization point range can differ from analysis range.
    Uses only common points in data, without interpolation.
    
    Args:
        data_dict: Dictionary with P(s) data for different samples
        reference_names: List of reference data names (to compare against)
        target_names: List of target data names (to compare). If None, all except reference_names
        min_distance: Minimum distance for comparison/metric analysis (in bp)
        max_distance: Maximum distance for comparison/metric analysis (in bp)
        alignment_min_distance: Minimum distance for normalization point sweep (in bp)
        alignment_max_distance: Maximum distance for normalization point sweep (in bp)
        max_points: Maximum number of normalization points to test
        
    Returns:
        dict: Dictionary with plots of metric dependence on normalization point
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import time
    
    # Check that reference_names is a list
    if isinstance(reference_names, str):
        reference_names = [reference_names]
    
    # If target_names not specified, use all other keys
    if target_names is None:
        target_names = [key for key in data_dict.keys() if key not in reference_names]
    elif isinstance(target_names, str):
        target_names = [target_names]
    
    # If range for normalization point sweep not specified, use full data range
    if alignment_min_distance is None:
        alignment_min_distance = 0
    if alignment_max_distance is None:
        alignment_max_distance = float('inf')
    
    # Collect all unique s_bp points from all datasets in specified range for normalization
    all_s_bp = set()
    for key in data_dict.keys():
        df = _load_data_from_params(key, data_dict[key])
        if df is None:
            continue
        # Take only points in range for normalization
        s_bp_values = df[(df['s_bp'] >= alignment_min_distance) & 
                         (df['s_bp'] <= alignment_max_distance)]['s_bp'].values
        all_s_bp.update(s_bp_values)
    
    # Sort points and convert to list
    alignment_points = sorted(list(all_s_bp))
    
    # If too many points, thin them out
    if len(alignment_points) > max_points:
        # Uniformly select points on logarithmic scale
        log_points = np.log10(alignment_points)
        log_indices = np.linspace(log_points.min(), log_points.max(), max_points)
        alignment_points = np.unique(10**log_indices)  # Convert back to normal scale

    # Metrics to track
    metric_names = ['r2_ps', 'r2_grad', 'rmse', 'mae', 'max_error']
    
    # Dictionaries to store results
    results = {
        metric: {ref: {tgt: [] for tgt in target_names} for ref in reference_names}
        for metric in metric_names
    }
    
    # Display parameter information
    print(f"Analyzing metric dependence on normalization point ({len(alignment_points)} points):")
    print(f"Metric analysis range: {min_distance} - {max_distance} bp")
    print(f"Normalization point range: {alignment_min_distance} - {alignment_max_distance} bp")
    
    # For each normalization point
    for i, alignment_s_bp in enumerate(alignment_points):
        start_time = time.time()
        
        # Display current progress
        progress = (i + 1) / len(alignment_points) * 100
        print(f"\rProgress: [{i+1}/{len(alignment_points)}] ({progress:.1f}%) - point {alignment_s_bp} bp", end="")
        
        # Run comparison with current normalization point, but use range for analysis
        metrics = compare_ps_curves(
            data_dict, 
            reference_names=reference_names,
            target_names=target_names,
            min_distance=min_distance, 
            max_distance=max_distance,
            alignment_s_bp=alignment_s_bp,
            normalize=True
        )
        
        # Save results for each metric, reference, and target set
        for metric in metric_names:
            if metric in metrics:
                for ref in reference_names:
                    for tgt in target_names:
                        if tgt in metrics[metric].index and ref in metrics[metric].columns:
                            value = metrics[metric].loc[tgt, ref]
                            results[metric][ref][tgt].append(value)
        
        elapsed = time.time() - start_time
        remaining = elapsed * (len(alignment_points) - i - 1)
        print(f" - {elapsed:.1f}s (approximately {remaining:.0f}s remaining)", end="")
    
    print("\nAnalysis complete. Building plots...")
    
    # Create plots for each metric and reference
    figures = {}
    
    # Define titles and axes
    metric_titles = {
            'r2_ps': 'R² P(s)',
            'r2_grad': 'R² Grad(P(s))',
            'rmse': 'RMSE',
            'mae': 'MAE',
            'max_error': 'Max Error'
        }
    
    # Define optimal values (higher is better or vice versa)
    metric_optimize = {
        'r2_reg': 'max',     # Higher is better
        'r2_direct': 'max',  # Higher is better
        'rmse': 'min',        # Lower is better
        'mae': 'min',        # Lower is better
        'max_error': 'min'   # Lower is better
    }
    
    # For each metric create separate plot with subplots for each reference
    for metric in metric_names:
        # Determine number of subplots (by references)
        n_refs = len(reference_names)
        
        # Create plot with subplots
        fig = make_subplots(
            rows=n_refs, 
            cols=1,
            subplot_titles=[f"Reference: {ref}" for ref in reference_names],
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        # For each reference dataset
        for i, ref in enumerate(reference_names):
            # For each target dataset
            for tgt in target_names:
                # Get metric values for current (reference, target) pair
                metric_values = results[metric][ref][tgt]
                
                # Skip if there are no finite values
                if not metric_values or all(pd.isna(val) for val in metric_values):
                    continue
                
                # Add a line to the plot
                fig.add_trace(
                    go.Scatter(
                        x=alignment_points,
                        y=metric_values,
                        mode='lines+markers',
                        name=tgt,
                        line=dict(
                            color=data_dict[tgt]['color'],
                            dash='dash' if data_dict[tgt].get('style', 'solid') == 'dash' else 'solid'
                        ),
                        marker=dict(
                            size=5,
                            color=data_dict[tgt]['color']
                        ),
                        showlegend=True,
                        legendgroup=tgt,  # Group legend elements
                        legendgrouptitle_text=tgt  # Group title
                    ),
                    row=i+1, 
                    col=1
                )
                
                # Find optimal value and mark it with star
                if metric_optimize[metric] == 'max':
                    opt_idx = np.nanargmax(metric_values)
                else:
                    opt_idx = np.nanargmin(metric_values)
                
                opt_point = alignment_points[opt_idx]
                opt_value = metric_values[opt_idx]
                
                # Add marker for optimum
                fig.add_trace(
                    go.Scatter(
                        x=[opt_point],
                        y=[opt_value],
                        mode='markers',
                        marker=dict(
                            symbol='star',
                            size=12,
                            color=data_dict[tgt]['color']
                        ),
                        name=f"{tgt} (optimum)",
                        showlegend=False
                    ),
                    row=i+1, 
                    col=1
                )
                
                # Add annotation with optimum coordinates
                fig.add_annotation(
                    x=opt_point,
                    y=opt_value,
                    text=f"Optimum: {opt_point} bp<br>({opt_value:.4f})",
                    showarrow=True,
                    arrowhead=1,
                    ax=40,
                    ay=-40,
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                    row=i+1,
                    col=1
                )
            
            # Configure subplot
            fig.update_xaxes(
                title_text="Normalization point, bp" if i == n_refs-1 else "",
                type="log",  # Set logarithmic scale
                row=i+1, 
                col=1
            )
            fig.update_yaxes(
                title_text=metric_titles[metric] if i == 0 else "",
                row=i+1, 
                col=1
            )
        
        # General plot configuration
        fig.update_layout(
            title_text=' ',
            height=100 * n_refs + 100,
            width=1000,
            legend=dict(
                orientation="v",  # Vertical orientation
                yanchor="top",
                y=0.5,
                xanchor="left",
                x=1.02,  # Legend position on right
                itemclick="toggle",  # Toggle entire group
            )
        )
        
        figures[metric] = fig
    
    # Additionally create summary plot for all metrics
    summary_fig = make_subplots(
        rows=len(metric_names), 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05
    )
    
    # For better visualization, select only first reference
    ref = reference_names[0]
    
    # For each metric
    for i, metric in enumerate(metric_names):
        # For each target set
        for tgt in target_names:
            # Get metric data for current pair (reference, target)
            metric_values = results[metric][ref][tgt]
            
            # Check for data presence
            if not metric_values or all(pd.isna(val) for val in metric_values):
                continue
            
            # Add line to plot
            summary_fig.add_trace(
                go.Scatter(
                    x=alignment_points,
                    y=metric_values,
                    mode='lines+markers',
                    name=tgt,
                    line=dict(
                        color=data_dict[tgt]['color'],
                        dash='dash' if data_dict[tgt].get('style', 'solid') == 'dash' else 'solid'
                    ),
                    marker=dict(
                        size=4,
                        color=data_dict[tgt]['color']
                    ),
                    showlegend=True,
                    legendgroup=tgt,  # Group legend elements
                    legendgrouptitle_text=tgt  # Group title
                ),
                row=i+1, 
                col=1
            )
        
        # Configure subplot
        summary_fig.update_xaxes(
            title_text="Normalization point, bp" if i == len(metric_names)-1 else "",
            type="log",  # Set logarithmic scale
            row=i+1, 
            col=1
        )
        summary_fig.update_yaxes(
            title_text=metric_titles[metric],
            row=i+1, 
            col=1
        )
    
    # General plot configuration
    summary_fig.update_layout(
        title_text=' ',
        height=100 * len(metric_names) + 100,
        width=1000,
        legend=dict(
            orientation="v",  # Vertical orientation
            yanchor="top",
            y=0.5,
            xanchor="left",
            x=1.02,  # Legend position on right
            itemclick="toggle",  # Toggle entire group
        )
    )
    
    figures['summary'] = summary_fig
    
    return {
        'figures': figures,
        'results': results,
        'alignment_points': alignment_points
    }

def plot_multiple_contacts_seaborn(
    data_dict,
    x_limits=(3e3, 2e6),
    y1_limits=None,   # limits for upper panel (P(s))
    y2_limits=None,   # limits for lower panel (slope)
    normalize=True,
    alignment_s_bp=3000,
    font_size=14,      # base font size for labels and ticks
    figsize=(8, 10)
):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=figsize, gridspec_kw={'height_ratios': [2, 1]})

    for name, params in data_dict.items():
        data = _load_data_from_params(name, params)
        if data is None:
            continue
        color = params.get('color', 'blue')
        style = params.get('style', 'solid')
        alpha = params.get('alpha', 1.0)

        y_values = data['balanced.avg.smoothed.agg'].copy()
        # Optional normalization at alignment_s_bp
        if normalize and alignment_s_bp is not None:
            closest_index = (data['s_bp'] - alignment_s_bp).abs().idxmin()
            norm_value = y_values.iloc[closest_index]
            if np.isnan(norm_value):
                for i in range(closest_index + 1, len(y_values)):
                    if not np.isnan(y_values.iloc[i]):
                        norm_value = y_values.iloc[i]
                        break
            y_values = y_values / norm_value

        linestyle = '-' if style == 'solid' else '--'

        ax1.plot(
            data['s_bp'], y_values,
            label=name,
            color=color,
            linestyle=linestyle,
            alpha=alpha
        )
        ax2.plot(
            data['s_bp'], data['der'],
            label=name,
            color=color,
            linestyle=linestyle,
            alpha=alpha
        )

        # Optional marker at a specific annotated distance
        if 'annotation_s_bp' in params and params['annotation_s_bp'] is not None:
            s_bp = params['annotation_s_bp']
            idx = (data['s_bp'] - s_bp).abs().idxmin()
            x_ann = data.iloc[idx]['s_bp']
            y_ann = data.iloc[idx]['der']
            ax2.plot(
                x_ann, y_ann, 
                marker='o', 
                color=color, 
                markersize=4,
                markeredgecolor=color,
                markeredgewidth=1.5,
                linestyle='None',
                zorder=10
            )

    # Configure axes and legend
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(x_limits)
    if y1_limits is not None:
        ax1.set_ylim(y1_limits)
    ax1.set_ylabel('Contact frequency (normalized)', fontsize=font_size)
    ax1.grid(True, which='both', linestyle=':', linewidth=0.5)

    handles, labels = ax1.get_legend_handles_labels()
    pattern = re.compile(r'.*_c\d+$')
    filtered = [(h, l) for h, l in zip(handles, labels) if not pattern.match(l)]
    if filtered:
        handles_filtered, labels_filtered = zip(*filtered)
    else:
        handles_filtered, labels_filtered = [], []
    ax1.legend(handles_filtered, labels_filtered, loc='lower left', fontsize=font_size)

    ax2.set_xscale('log')
    ax2.set_xlim(x_limits)
    if y2_limits is not None:
        ax2.set_ylim(y2_limits)
    ax2.set_xlabel('Genomic distance, bp', fontsize=font_size)
    ax2.set_ylabel('Derivative', fontsize=font_size)
    ax2.grid(True, which='both', linestyle=':', linewidth=0.5)



    # Set tick label sizes for both axes
    ax1.tick_params(axis='both', which='major', labelsize=font_size)
    ax2.tick_params(axis='both', which='major', labelsize=font_size)

    return fig

#

def analyze_metrics_vs_alignment_seaborn(data_dict, reference_names, target_names=None, 
                               min_distance=10000, max_distance=4000000,
                               alignment_min_distance=None, alignment_max_distance=None,
                               max_points=20, figsize=(12, 18), font_size=14,):
    """
    Analyse how comparison metrics depend on the normalization point.

    The range of normalization points can differ from the analysis range.
    Only common data points are used (no interpolation across distances).
    
    Args:
        data_dict: Dictionary with P(s) data for different samples.
        reference_names: List of reference dataset names.
        target_names: List of target dataset names. If None, use all except reference_names.
        min_distance: Minimal genomic distance (bp) used to compute metrics.
        max_distance: Maximal genomic distance (bp) used to compute metrics.
        alignment_min_distance: Minimal distance (bp) to consider as normalization point.
        alignment_max_distance: Maximal distance (bp) to consider as normalization point.
        max_points: Maximal number of normalization points to test.
        
    Returns:
        dict: Dictionary with figures showing metric dependence on normalization point.
    """
    import numpy as np
    import pandas as pd
    import time
    
    # Ensure reference_names is a list
    if isinstance(reference_names, str):
        reference_names = [reference_names]
    
    # If target_names is not provided, use all others
    if target_names is None:
        target_names = [key for key in data_dict.keys() if key not in reference_names]
    elif isinstance(target_names, str):
        target_names = [target_names]
    
    # If normalization range is not specified, use full span of data
    if alignment_min_distance is None:
        alignment_min_distance = 0
    if alignment_max_distance is None:
        alignment_max_distance = float('inf')
    
    # Collect all unique s_bp points across datasets within normalization range
    all_s_bp = set()
    for key in data_dict.keys():
        df = _load_data_from_params(key, data_dict[key])
        if df is None:
            continue
        # Keep only s_bp values within normalization range
        s_bp_values = df[(df['s_bp'] >= alignment_min_distance) & 
                         (df['s_bp'] <= alignment_max_distance)]['s_bp'].values
        all_s_bp.update(s_bp_values)
    
    # Sort and convert to list
    alignment_points = sorted(list(all_s_bp))
    
    # If there are too many points, downsample them
    if len(alignment_points) > max_points:
        # Select points uniformly in log10 space
        log_points = np.log10(alignment_points)
        log_indices = np.linspace(log_points.min(), log_points.max(), max_points)
        alignment_points = np.unique(10**log_indices)  # Convert back to linear scale

    # Metrics to monitor
    metric_names = ['r2_ps', 'r2_grad', 'rmse', 'mae', 'max_error', 'average_ratio']
    
    # Storage for results
    results = {
        metric: {ref: {tgt: [] for tgt in target_names} for ref in reference_names}
        for metric in metric_names
    }
    
    # Show information about analysis parameters
    print(f"Analysing metric dependence on normalization point ({len(alignment_points)} points):")
    print(f"Metric analysis range: {min_distance} - {max_distance} bp")
    print(f"Normalization point range: {alignment_min_distance} - {alignment_max_distance} bp")
    
    # Loop over normalization points
    for i, alignment_s_bp in enumerate(alignment_points):
        start_time = time.time()
        
        # Print current progress
        progress = (i + 1) / len(alignment_points) * 100
        print(f"\rProgress: [{i+1}/{len(alignment_points)}] ({progress:.1f}%) - point {alignment_s_bp} bp", end="")
        
        # Run comparison for current normalization point within analysis range
        metrics = compare_ps_curves(
            data_dict, 
            reference_names=reference_names,
            target_names=target_names,
            min_distance=min_distance, 
            max_distance=max_distance,
            alignment_s_bp=alignment_s_bp,
            normalize=True
        )
        
        # Save results for each metric, reference and target
        for metric in metric_names:
            if metric in metrics:
                for ref in reference_names:
                    for tgt in target_names:
                        if tgt in metrics[metric].index and ref in metrics[metric].columns:
                            value = metrics[metric].loc[tgt, ref]
                            results[metric][ref][tgt].append(value)
        
        elapsed = time.time() - start_time
        remaining = elapsed * (len(alignment_points) - i - 1)
        print(f" - {elapsed:.1f}s (about {remaining:.0f}s remaining)", end="")
    
    print("\nAnalysis finished. Building figures...")

    # Build figures for each metric and reference
    figures = {}
    
    # Define titles and axis labels for each metric
    metric_titles = {
        'r2_ps': 'R² P(s)',
        'r2_grad': 'R² Grad(P(s))',
        'rmse': 'RMSE',
        'mae': 'MAE',
        'max_error': 'Max Error',
        'average_ratio': 'Avr. ratio'
    }
    
    
    sns.set_style("whitegrid")
    figures = {}

    # Summary figure with all metrics stacked vertically
    summary_fig, axes = plt.subplots(len(metric_names), 1, figsize=(figsize[0], figsize[1]), 
                                   sharex=True)
    if len(metric_names) == 1:
        axes = [axes]
    
    ref = reference_names[0]
    for i, (ax, metric) in enumerate(zip(axes, metric_names)):
        for tgt in target_names:
            metric_values = results[metric][ref][tgt]
            ax.plot(alignment_points, metric_values,
                   color=data_dict[tgt]['color'],
                   alpha=data_dict[tgt].get('alpha', 1.0),
                   linestyle='--' if data_dict[tgt].get('style') == 'dash' else '-',
                #    marker='o', markersize=4,
                   label=tgt,
                   linewidth=3,)
        
        for spine in ax.spines.values():
            spine.set_edgecolor('black')

        ax.set_xmargin(0)
        ax.set_xscale('log')
        ax.set_ylabel(metric_titles[metric])
        if i == len(metric_names)-1:
            ax.set_xlabel("Normalization point, bp")
    
    pattern = re.compile(r'.*_c\d+$')

    handles, labels = ax.get_legend_handles_labels()
    filtered = [(h, l) for h, l in zip(handles, labels) if not pattern.match(l)]
    if filtered:
        handles_filtered, labels_filtered = zip(*filtered)
    else:
        handles_filtered, labels_filtered = [], []

    summary_fig.legend(handles_filtered, labels_filtered, loc='upper right', bbox_to_anchor=(1, 1), frameon=True)
    plt.tight_layout()
    plt.rcParams.update({
        'font.size': font_size,
        'axes.titlesize': font_size,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size
         })
    figures['summary'] = summary_fig

    return {
        'figures': figures,
        'results': results,
        'alignment_points': alignment_points
    }

def plot_multiple_contacts_seaborn_ratio(
    data_dict,
    compare_data,
    x_limits=(3e3, 2e6),
    y1_limits=None,
    y2_limits=None,
    normalize=True,
    alignment_s_bp=3000,
    font_size=14,
    figsize=(8, 10)
):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import re

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=figsize, gridspec_kw={'height_ratios': [2, 1]})

    # Get reference data
    compare_params = next((p for n, p in data_dict.items() if n == compare_data), None)
    if not compare_params:
        raise ValueError(f"compare_data '{compare_data}' not found")

    # Load and normalize reference data
    compare_data_df = _load_data_from_params(compare_data, compare_params)
    if compare_data_df is None:
        raise ValueError(f"Failed to load data for '{compare_data}'")
    compare_data_df = compare_data_df.copy()
    if normalize and alignment_s_bp is not None:
        closest_idx = (compare_data_df['s_bp'] - alignment_s_bp).abs().idxmin()
        norm_val = compare_data_df['balanced.avg.smoothed.agg'].iloc[closest_idx]
        if np.isnan(norm_val):
            norm_val = compare_data_df['balanced.avg.smoothed.agg'].loc[
                compare_data_df['balanced.avg.smoothed.agg'].first_valid_index()
            ]
        compare_data_df['normalized'] = compare_data_df['balanced.avg.smoothed.agg'] / norm_val
    else:
        compare_data_df['normalized'] = compare_data_df['balanced.avg.smoothed.agg']

    # Create reference DataFrame
    df_compare = pd.DataFrame({
        's_bp': compare_data_df['s_bp'],
        'ref_value': compare_data_df['normalized']
    })

    for name, params in data_dict.items():
        data_loaded = _load_data_from_params(name, params)
        if data_loaded is None:
            continue
        data = data_loaded.copy()
        color = params.get('color', 'blue')
        style = params.get('style', 'solid')
        alpha = params.get('alpha', 1.0)

        # Normalize current data
        if normalize and alignment_s_bp is not None:
            closest_idx = (data['s_bp'] - alignment_s_bp).abs().idxmin()
            norm_val = data['balanced.avg.smoothed.agg'].iloc[closest_idx]
            if np.isnan(norm_val):
                norm_val = data['balanced.avg.smoothed.agg'].loc[
                    data['balanced.avg.smoothed.agg'].first_valid_index()
                ]
            data['normalized'] = data['balanced.avg.smoothed.agg'] / norm_val
        else:
            data['normalized'] = data['balanced.avg.smoothed.agg']

        # Top plot (unchanged)
        linestyle = '-' if style == 'solid' else '--'
        ax1.plot(
            data['s_bp'], data['normalized'],
            label=name,
            color=color,
            linestyle=linestyle,
            alpha=alpha
        )

        # Safe ratio calculation via merge
        df_current = pd.DataFrame({
            's_bp': data['s_bp'],
            'current_value': data['normalized']
        })
        
        # Merge data
        merged = pd.merge(df_current, df_compare, on='s_bp', how='inner')
        if merged.empty:
            continue  # Skip if no common points

        # Calculate ratio
        ratio = merged['current_value'] / merged['ref_value']
        
        # Bottom plot
        ax2.plot(
            merged['s_bp'], ratio,
            color=color,
            linestyle=linestyle,
            alpha=alpha
        )

        # Point annotation (if required)
        if 'annotation_s_bp' in params and params['annotation_s_bp'] is not None:
            s_bp = params['annotation_s_bp']
            if s_bp in merged['s_bp'].values:
                idx = merged.index[merged['s_bp'] == s_bp][0]
                ax2.plot(
                    merged.at[idx, 's_bp'], ratio.iloc[idx],
                    marker='o',
                    color=color,
                    markersize=4,
                    markeredgecolor=color,
                    markeredgewidth=1.5,
                    zorder=10
                )

    # Plot formatting
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(x_limits)
    if y1_limits: ax1.set_ylim(y1_limits)
    ax1.set_ylabel('Contact frequency (normalized)', fontsize=font_size)
    ax1.grid(True, which='both', linestyle=':', linewidth=0.5)

    # Filter legend
    handles, labels = ax1.get_legend_handles_labels()
    pattern = re.compile(r'.*_c\d+$')
    filtered = [(h, l) for h, l in zip(handles, labels) if not pattern.match(l)]
    if filtered:
        handles, labels = zip(*filtered)
        ax1.legend(handles, labels, loc='lower left', fontsize=font_size)

    ax2.set_xscale('log')
    ax2.set_yscale('log')  # Logarithmic Y scale
    ax2.set_xlim(x_limits)
    if y2_limits: ax2.set_ylim(y2_limits)
    ax2.set_xlabel('Genomic distance, bp', fontsize=font_size)
    ax2.set_ylabel(f'Ratio to {compare_data}', fontsize=font_size)
    ax2.grid(True, which='both', linestyle=':', linewidth=0.5)

    # Configure ticks
    ax1.tick_params(axis='both', which='major', labelsize=font_size)
    ax2.tick_params(axis='both', which='major', labelsize=font_size)

    return fig

from scipy.signal import find_peaks

def _find_derivative_peak_single(df, x_col: str = 's_bp', der_col: str = 'der',
                                 x_range=(3e3, 2e6), min_distance: int = 5):
    """
    Find peak (local maximum) of derivative for a single dataset.
    
    Args:
        df: DataFrame with P(s) data
        x_col: Column name for genomic distance
        der_col: Column name for derivative
        x_range: Range of distances to search (min, max) in bp
        min_distance: Minimum distance between peaks
    
    Returns:
        float or None: s_bp value of the peak, or None if not found
    """
    try:
        x = df[x_col].values
        y = df[der_col].values
        mask = (x >= x_range[0]) & (x <= x_range[1])
        x = x[mask]
        y = y[mask]
        if len(x) == 0:
            return None
        peaks, _ = find_peaks(y, distance=min_distance)
        if len(peaks) == 0:
            return None
        peak_idx = peaks[np.argmax(y[peaks])]
        return float(x[peak_idx])
    except Exception:
        return None

def plot_combined_ps_and_metrics(data_dict, reference_names, 
                                x_limits=(3e3, 2e6), y1_limits=(1e-5, 1e0), y2_limits=(-3, 0.),
                                min_distance=3000, max_distance=2000000,
                                alignment_min_distance=3000, alignment_max_distance=2000000,
                                max_points=50, normalize=True, alignment_s_bp=3000,
                                figsize=(6.5, 7.5), dpi=300):
    """
    Create combined plot with P(s) and derivative on the left, metrics on the right.
    
    Left side: P(s) and derivative (derivative compressed to ~1/3 height)
    Right side: All metrics (R² P(s), R² Grad, RMSE, MAE, Max Error, Average Ratio)
    Size constraint: not more than 2000x2300 pixels at 300 DPI.
    
    Args:
        data_dict: Dictionary with P(s) data for different samples.
                   Each entry should have either 'csv_path' or 'data':
                   {
                       'dataset_name': {
                           'csv_path': 'path/to/file.csv',  # OR 'data': DataFrame
                           'color': 'black',
                           'style': 'solid',
                           'alpha': 1.0,
                           'show_in_legend': True
                       }
                   }
        reference_names: List of reference dataset names
        x_limits: X-axis limits (min, max) in bp
        y1_limits: Y-axis limits for P(s) plot
        y2_limits: Y-axis limits for derivative plot
        min_distance: Minimum distance for metric analysis (bp)
        max_distance: Maximum distance for metric analysis (bp)
        alignment_min_distance: Minimum distance for normalization points (bp)
        alignment_max_distance: Maximum distance for normalization points (bp)
        max_points: Maximum number of normalization points to test
        normalize: Whether to normalize data
        alignment_s_bp: Normalization point in bp
        figsize: Figure size (width, height) in inches
        dpi: Resolution in dots per inch
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import re
    import time

    # Fonts and sizes
    plt.rcParams.update({
        "font.family": "Liberation Sans",  # Arial equivalent
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    # Figure and grid (2 columns)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(nrows=1, ncols=2, wspace=0.25)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.08)

    # Left column: 2 rows with height ratio 2:1
    left_gs = gs[0].subgridspec(2, 1, hspace=0.08, height_ratios=[2, 1])
    ax1 = fig.add_subplot(left_gs[0])  # P(s)
    ax2 = fig.add_subplot(left_gs[1])  # Derivative

    # Right column: as many rows as metrics
    metric_names = ['r2_ps', 'r2_grad', 'rmse', 'mae', 'max_error', 'average_ratio']
    metric_titles = {
        'r2_ps': 'R² P(s)',
        'r2_grad': 'R² Grad(P(s))',
        'rmse': 'RMSE',
        'mae': 'MAE',
        'max_error': 'Max Error',
        'average_ratio': 'Avr. ratio',
    }
    right_gs = gs[1].subgridspec(len(metric_names), 1, hspace=0.08)
    right_axes = [fig.add_subplot(right_gs[i]) for i in range(len(metric_names))]

    # === LEFT SIDE: P(s) and derivative ===
    for name, params in data_dict.items():
        data = _load_data_from_params(name, params)
        if data is None:
            continue
            
        color = params.get('color', 'blue')
        style = params.get('style', 'solid')
        alpha = params.get('alpha', 1.0)
        show_in_legend = params.get('show_in_legend', True)  # Default: show in legend

        y_values = data['balanced.avg.smoothed.agg'].copy()
        if normalize and alignment_s_bp is not None:
            closest_index = (data['s_bp'] - alignment_s_bp).abs().idxmin()
            norm_value = y_values.iloc[closest_index]
            if np.isnan(norm_value):
                for i in range(closest_index + 1, len(y_values)):
                    if not np.isnan(y_values.iloc[i]):
                        norm_value = y_values.iloc[i]
                        break
            y_values = y_values / norm_value

        linestyle = '-' if style == 'solid' else '--'
        # Line width: base and 1.5x for reference
        base_lw_left = 2.
        linewidth = base_lw_left * 1.5 if name in reference_names else base_lw_left
        # Use label only if show_in_legend is True
        label = name if show_in_legend else None
        ax1.plot(data['s_bp'], y_values, label=label, color=color,
                 linestyle=linestyle, alpha=alpha, linewidth=linewidth)
        ax2.plot(data['s_bp'], data['der'], label=label, color=color,
                 linestyle=linestyle, alpha=alpha, linewidth=linewidth)

    # Formatting left side
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(x_limits)
    if y1_limits is not None:
        ax1.set_ylim(y1_limits)
    ax1.set_ylabel('Contact frequency (normalized)', fontsize=12)
    ax1.tick_params(axis='x', which='major', labelsize=8)
    ax1.tick_params(axis='y', which='major', labelsize=8, rotation=90)
    ax1.grid(True, which='both', linestyle=':', linewidth=0.5)

    handles, labels = ax1.get_legend_handles_labels()
    pattern = re.compile(r'.*_c\d+$')
    filtered = [(h, l) for h, l in zip(handles, labels) if not pattern.match(l)]
    if filtered:
        handles_filtered, labels_filtered = zip(*filtered)
    else:
        handles_filtered, labels_filtered = [], []
    ax1.legend(handles_filtered, labels_filtered, loc='lower left', fontsize=12)

    ax2.set_xscale('log')
    ax2.set_xlim(x_limits)
    if y2_limits is not None:
        ax2.set_ylim(y2_limits)
    ax2.set_xlabel('Genomic distance, bp', fontsize=12)
    ax2.set_ylabel('Derivative', fontsize=12)
    ax2.tick_params(axis='x', which='major', labelsize=8)
    ax2.tick_params(axis='y', which='major', labelsize=8, rotation=0)
    # Show only integer values on Y-axis for derivative
    import matplotlib.ticker as mticker
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax2.grid(True, which='both', linestyle=':', linewidth=0.5)

    # === RIGHT SIDE: all metrics ===
    if isinstance(reference_names, str):
        reference_names = [reference_names]
    target_names = [key for key in data_dict.keys() if key not in reference_names]

    # Prepare normalization points
    all_s_bp = set()
    for key in data_dict.keys():
        df = _load_data_from_params(key, data_dict[key])
        if df is None:
            continue
        s_bp_vals = df[(df['s_bp'] >= alignment_min_distance) & (df['s_bp'] <= alignment_max_distance)]['s_bp'].values
        all_s_bp.update(s_bp_vals)
    alignment_points = sorted(list(all_s_bp))
    if len(alignment_points) > max_points:
        log_points = np.log10(alignment_points)
        log_indices = np.linspace(log_points.min(), log_points.max(), max_points)
        alignment_points = np.unique(10**log_indices)

    # Collect metrics
    results = {
        metric: {ref: {tgt: [] for tgt in target_names} for ref in reference_names}
        for metric in metric_names
    }
    print(f"Analyzing metrics across {len(alignment_points)} normalization points...")
    for i, align_bp in enumerate(alignment_points):
        print(f"\r[{i+1}/{len(alignment_points)}] {align_bp} bp", end="")
        # Use compare_ps_curves from this module (defined above)
        metrics = compare_ps_curves(
            data_dict,
            reference_names=reference_names,
            target_names=target_names,
            min_distance=min_distance,
            max_distance=max_distance,
            alignment_s_bp=align_bp,
            normalize=True,
        )
        for metric in metric_names:
            if metric in metrics:
                for ref in reference_names:
                    for tgt in target_names:
                        if tgt in metrics[metric].index and ref in metrics[metric].columns:
                            results[metric][ref][tgt].append(metrics[metric].loc[tgt, ref])
    print()

    # Plot metrics (each in its own row)
    ref = reference_names[0]
    for idx, metric in enumerate(metric_names):
        ax = right_axes[idx]
        for tgt in target_names:
            values = results[metric][ref][tgt]
            if not values:
                continue
            color = data_dict[tgt]['color']
            style = data_dict[tgt].get('style', 'solid')
            alpha = data_dict[tgt].get('alpha', 1.0)
            linestyle = '-' if style == 'solid' else '--'
            # Line width for metrics: base and 1.5x for reference
            base_lw_right = 2.0
            linewidth = base_lw_right * 1.5 if tgt in reference_names else base_lw_right
            ax.plot(alignment_points, values, color=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth, label=tgt)
        ax.set_xscale('log')
        ax.set_xlim(min(alignment_points), max(alignment_points))
        ax.set_ylabel(metric_titles[metric], fontsize=12)
        ax.tick_params(axis='x', which='major', labelsize=8)
        ax.tick_params(axis='y', which='major', labelsize=8, rotation=90)
        ax.grid(True, which='both', linestyle=':', linewidth=0.5)
        if idx == len(metric_names) - 1:
            ax.set_xlabel('Normalization point, bp', fontsize=12)

    # Common legend on right top (filter single cells)
    handles_all, labels_all = right_axes[0].get_legend_handles_labels()
    filtered_all = [(h, l) for h, l in zip(handles_all, labels_all) if not pattern.match(l)]
    if filtered_all:
        h_f, l_f = zip(*filtered_all)
        fig.legend(h_f, l_f, loc='upper right', bbox_to_anchor=(0.98, 0.98), frameon=True, fontsize=12)

    return fig

def plot_combined_ps_and_loops(data_dict,
                               x_limits=(3e3, 2e6), y1_limits=(1e-3, 0), y2_limits=(-3, 0.),
                               normalize=True, alignment_s_bp=3000,
                               loops_base_dir='/home/vvkonstantinov/home/Re_model/coolmaps/draw_ps_article',
                               bead_size=200, max_distance_beads=2500,
                               figsize=(6.5, 5.5), dpi=300):
    """
    Create combined plot with P(s) and derivative on the left, loop distributions on the right.
    
    Left: P(s) and derivative
    Right: Loop size distributions (inner/outer/all) for three lifetime regimes.
    Overall figure size matches previous layouts.
    
    Args:
        data_dict: Dictionary with P(s) data
        x_limits: X-axis limits (min, max) in bp
        y1_limits: Y-axis limits for P(s) plot
        y2_limits: Y-axis limits for derivative plot
        normalize: Whether to normalize data
        alignment_s_bp: Normalization point in bp
        loops_base_dir: Base directory for loop data
        bead_size: Size of each bead in bp
        max_distance_beads: Maximum distance in beads for loop analysis
        figsize: Figure size (width, height) in inches
        dpi: Resolution in dots per inch
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import re
    from loop_statistics import analyze_extruder_loops

    plt.rcParams.update({
        "font.family": "Liberation Sans",
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    # Figure: 1x2, left subgrid 2x1, right 4x1 (for four lifetime configurations)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(nrows=1, ncols=2, wspace=0.22)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.12)

    left_gs = gs[0].subgridspec(2, 1, hspace=0.22, height_ratios=[2, 1])
    ax_ps = fig.add_subplot(left_gs[0])
    ax_der = fig.add_subplot(left_gs[1])

    # Right part: loop distributions (NOT metrics!)
    right_gs = gs[1].subgridspec(4, 1, hspace=0.22)
    ax_loops = [fig.add_subplot(right_gs[i]) for i in range(4)]

    # Left part: P(s) and derivative
    # Colors: DT40=black, static=grey, lifetimes from data_dict['color']
    for name, params in data_dict.items():
        data = _load_data_from_params(name, params)
        if data is None:
            continue
        style = params.get('style', 'solid')
        alpha = params.get('alpha', 1.0)

        # Color: fixed for DT40 and static; others from params['color']
        if name == 'DT40':
            color = 'black'
        elif 'static' in name:
            color = 'grey'
        else:
            color = params.get('color', 'red')

        y_values = data['balanced.avg.smoothed.agg'].copy()
        if normalize and alignment_s_bp is not None:
            closest_index = (data['s_bp'] - alignment_s_bp).abs().idxmin()
            norm_value = y_values.iloc[closest_index]
            if np.isnan(norm_value):
                for i in range(closest_index + 1, len(y_values)):
                    if not np.isnan(y_values.iloc[i]):
                        norm_value = y_values.iloc[i]
                        break
            y_values = y_values / norm_value

        linestyle = '-' if style == 'solid' else '--'
        base_lw = 2.0
        lw = base_lw * 1.5 if name == 'DT40' else base_lw

        label = name.replace(', free loops', '\nfree loops').replace(' free loops', '\nfree loops')
        ax_ps.plot(data['s_bp'], y_values, color=color, linestyle=linestyle, alpha=alpha, linewidth=lw, label=label)
        ax_der.plot(data['s_bp'], data['der'], color=color, linestyle=linestyle, alpha=alpha, linewidth=lw, label=label)
        
        # Add annotation marker if annotation_s_bp is provided (for mean loop size)
        annotation_s_bp = params.get('annotation_s_bp', None)
        if annotation_s_bp is not None:
            try:
                idx = (data['s_bp'] - annotation_s_bp).abs().idxmin()
                x_ann = data.iloc[idx]['s_bp']
                y_ann = data.iloc[idx]['der']
                # Use marker for mean loop size (big dots as mentioned in article)
                ax_der.plot(
                    x_ann, y_ann,
                    marker='o', linestyle='None',
                    markersize=4, markeredgewidth=1.25,
                    markeredgecolor=color, markerfacecolor=color,
                    color=color, zorder=11, alpha=0.9
                )
            except Exception:
                pass

    # Formatting left side
    ax_ps.set_xscale('log')
    ax_ps.set_yscale('log')
    ax_ps.set_xlim(x_limits)
    if y1_limits is not None:
        ax_ps.set_ylim(y1_limits)
    ax_ps.set_ylabel('Contact frequency (normalized)', fontsize=10, labelpad=4)
    ax_ps.tick_params(axis='x', which='major', labelsize=8, pad=2)
    ax_ps.tick_params(axis='y', which='major', labelsize=10, rotation=90, pad=2)
    ax_ps.grid(True, which='both', linestyle=':', linewidth=0.5)

    handles, labels = ax_ps.get_legend_handles_labels()
    pattern = re.compile(r'.*_c\d+$')
    filtered = [(h, l) for h, l in zip(handles, labels) if not pattern.match(l)]
    if filtered:
        hf, lf = zip(*filtered)
        ax_ps.legend(hf, lf, loc='upper right', fontsize=8, ncol=1, frameon=True,
                     fancybox=False, shadow=False, bbox_to_anchor=(0.98, 0.98))

    ax_der.set_xscale('log')
    ax_der.set_xlim(x_limits)
    if y2_limits is not None:
        ax_der.set_ylim(y2_limits)
    ax_der.set_xlabel('Genomic distance, bp', fontsize=10, labelpad=8)
    ax_der.set_ylabel('Derivative', fontsize=10, labelpad=4)
    ax_der.tick_params(axis='x', which='major', labelsize=10, pad=2)
    ax_der.tick_params(axis='y', which='major', labelsize=10, rotation=0, pad=2)
    import matplotlib.ticker as mticker
    ax_der.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_der.grid(True, which='both', linestyle=':', linewidth=0.5)

    # Right part: loop distributions on four rows (NOT metrics!)
    # This function creates loop distributions
    # Determine mapping of dataset names to cell paths
    # If working with free loops (unnested), use *_unnested* directories
    is_free_loops_bundle = any('free loops' in k for k in data_dict.keys())
    if is_free_loops_bundle:
        cfgs = [
            ('Condensin 1, static', 'cells_cond1_gol_p_plen3', 'Static'),
            ('Condensin 1,lt=17 sec, free loops', 'cells_cond1_gol_p_plen3_dynamic_lt_1div6_unnested', 'Lifetime 17 sec (free loops)'),
            ('Condensin 1,lt=68 sec, free loops', 'cells_cond1_gol_p_plen3_dynamic_lt_4div6_unnested', 'Lifetime 68 sec (free loops)'),
            ('Condensin 1,lt=136 sec, free loops', 'cells_cond1_gol_p_plen3_dynamic_lt_8div6_unnested', 'Lifetime 136 sec (free loops)'),
        ]
    else:
        cfgs = [
            ('Condensin 1, static', 'cells_cond1_gol_p_plen3', 'Static'),
            ('Condensin 1,lt=17 sec', 'cells_cond1_gol_p_plen3_dynamic_lt_1div6', 'Lifetime 17 sec'),
            ('Condensin 1,lt=68 sec', 'cells_cond1_gol_p_plen3_dynamic_lt_4div6', 'Lifetime 68 sec'),
            ('Condensin 1,lt=136 sec', 'cells_cond1_gol_p_plen3_dynamic_lt_8div6', 'Lifetime 136 sec'),
        ]

    loop_types = ['inner_agr', 'outer_agr', 'all_agr']
    loop_names = ['Nested loops', 'Unnested loops', 'All loops']
    colors = ['blue', 'red', 'purple']

    x_max_bp = max_distance_beads * bead_size

    # Collect mean all_agr values for each dataset (for markers on derivative)
    mean_loops_bp_by_key = {}

    # Process each loop distribution subplot
    for ax, (key_hint, rel_path, title) in zip(ax_loops, cfgs):
        cell_dir = f"{loops_base_dir}/{rel_path}"
        try:
            # Load loop statistics data
            loops = analyze_extruder_loops(cell_dir, cell_nums=None, time_frames=None, min_distance=0, debug=False)
        except Exception as e:
            # If loop data cannot be loaded, show error message on subplot
            ax.text(0.5, 0.5, f"No loops\n{title}\nError: {str(e)[:50]}", 
                   ha='center', va='center', transform=ax.transAxes, fontsize=8)
            ax.set_axis_off()
            continue

        # Scale to bp
        scaled = {k: (np.array(v) * bead_size) for k, v in loops.items() if k in loop_types}

        # Remember mean all_agr for this dataset (for marker on derivative)
        if 'all_agr' in scaled and len(scaled['all_agr']) > 0:
            mean_loops_bp_by_key[key_hint] = float(np.mean(scaled['all_agr']))

        # Plot KDE
        import seaborn as sns
        if scaled:
            if is_free_loops_bundle:
                # For free loops show only all_agr
                scaled_to_plot = {k: v for k, v in scaled.items() if k == 'all_agr' and len(v) > 0}
                palette_to_use = ['purple'] if 'all_agr' in scaled_to_plot else []
                types_to_plot = ['all_agr'] if 'all_agr' in scaled_to_plot else []
            else:
                scaled_to_plot = {k: v for k, v in scaled.items() if k in loop_types and len(v) > 0}
                palette_to_use = colors[:len(scaled_to_plot)]
                types_to_plot = [lt for lt in loop_types if lt in scaled_to_plot]

            if scaled_to_plot:
                sns.kdeplot(
                    data=scaled_to_plot,
                    palette=palette_to_use,
                    ax=ax,
                    common_norm=True,
                    legend=False
                )
                # Vertical lines for means
                for lt in types_to_plot:
                    m = float(np.mean(scaled_to_plot[lt]))
                    col = 'purple' if is_free_loops_bundle else colors[loop_types.index(lt)]
                    ax.axvline(m, color=col, linestyle='dashed', linewidth=1.5)

        # Derivative peak line (if available)
        der_peak = None
        # Search for matching key in data_dict (lt=/lifetime)
        candidates = [key_hint, key_hint.replace('lt=', 'lifetime ') if 'lt=' in key_hint else key_hint.replace('lifetime ', 'lt=')]
        for cand in candidates:
            if cand in data_dict:
                cand_data = _load_data_from_params(cand, data_dict[cand])
                if cand_data is not None:
                    der_peak = _find_derivative_peak_single(cand_data, x_range=x_limits)
                    if der_peak is not None:
                        break
        if der_peak is not None:
            ax.axvline(der_peak, color='black', linestyle='-', linewidth=2, alpha=0.8)
        # Statistics boxes (only Mean) for each type, if available
        stats_blocks = []
        if is_free_loops_bundle:
            if 'all_agr' in scaled and len(scaled['all_agr']) > 0:
                mean_val = float(np.mean(scaled['all_agr']))
                stats_blocks.append(f"All loops:\nMean {mean_val:.0f} bp")
        else:
            for lt, nm in zip(loop_types, loop_names):
                if lt in scaled and len(scaled[lt]) > 0:
                    vals = scaled[lt]
                    mean_val = float(np.mean(vals))
                    stats_blocks.append(f"{nm}:\nMean {mean_val:.0f} bp")
        if stats_blocks:
            ax.text(
                x_max_bp * 0.70,
                ax.get_ylim()[1] * 0.05,
                "\n".join(stats_blocks),
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.8, boxstyle='square,pad=0.5'),
                fontsize=8
            )

        ax.set_xlim(0, x_max_bp)
        ax.set_title(title, fontsize=10)
        ax.grid(True, linestyle=':', linewidth=0.5)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        if ax is ax_loops[-1]:
            ax.set_xlabel('Loop Size, bp', fontsize=10)
        else:
            ax.tick_params(labelbottom=False)

    # After computing mean loop sizes - add markers on derivative at these positions
    if mean_loops_bp_by_key:
        # Re-iterate through datasets and add markers
        for name, params in data_dict.items():
            # Direct match with keys in mean_loops_bp_by_key
            if name in mean_loops_bp_by_key:
                mean_bp = mean_loops_bp_by_key[name]
                df = _load_data_from_params(name, params)
                if df is None:
                    continue
                # Find nearest point and place marker
                try:
                    idx = (df['s_bp'] - mean_bp).abs().idxmin()
                    # Color as above: DT40/Static fixed, others from params['color']
                    if name == 'DT40':
                        color = 'black'
                    elif 'static' in name.lower():
                        color = 'grey'
                    else:
                        color = params.get('color', 'red')
                    # Use marker for mean loop size (big dots as mentioned in article)
                    ax_der.plot(
                        df.iloc[idx]['s_bp'], df.iloc[idx]['der'],
                        marker='o', linestyle='None',
                        markersize=4, markeredgewidth=1.25,  # Markers for mean loop sizes
                        markeredgecolor=color, markerfacecolor=color, 
                        color=color, zorder=11, alpha=0.9
                    )
                except Exception:
                    pass

    return fig

def plot_three_lifetime_comparisons(data_dict, 
                                   x_limits=(3e3, 2e6), 
                                   y1_limits=(1e-3, 0), 
                                   y2_limits=(-3, 0.),
                                   normalize=True, 
                                   alignment_s_bp=3000,
                                   figsize=(6.5, 6.5), 
                                   dpi=300):
    """
    Create 3 plots in a row, each with P(s) on top and derivative on bottom.
    
    Plot 1: DT40, static, dynamic 17s, dynamic 17s unnested
    Plot 2: DT40, static, dynamic 68s, dynamic 68s unnested  
    Plot 3: DT40, static, dynamic 136s, dynamic 136s unnested
    
    Args:
        data_dict: Dictionary with P(s) data
        x_limits: X-axis limits (min, max) in bp
        y1_limits: Y-axis limits for P(s) plot
        y2_limits: Y-axis limits for derivative plot
        normalize: Whether to normalize data
        alignment_s_bp: Normalization point in bp
        figsize: Figure size (width, height) in inches
        dpi: Resolution in dots per inch
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import re
    import matplotlib.ticker as mticker

    # Fonts and sizes
    plt.rcParams.update({
        "font.family": "Liberation Sans",
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    # Define datasets for each plot
    plot_sets = [
        {
            'title': 'Lifetime 17 sec',
            'data_keys': ['DT40', 'Condensin 1, static', 'Condensin 1, lifetime 17 sec', 'Condensin 1, lifetime 17 sec, free loops']
        },
        {
            'title': 'Lifetime 68 sec', 
            'data_keys': ['DT40', 'Condensin 1, static', 'Condensin 1, lifetime 68 sec', 'Condensin 1, lifetime 68 sec, free loops']
        },
        {
            'title': 'Lifetime 136 sec',
            'data_keys': ['DT40', 'Condensin 1, static', 'Condensin 1, lifetime 136 sec', 'Condensin 1, lifetime 136 sec, free loops']
        }
    ]

    # Create figure with 3 subplots in a row
    fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=dpi, 
                            gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.22, 'wspace': 0.20})
    # Additional outer margins
    fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.12)
    
    # Key name normalizer for robust matching
    def _normalize_key(s: str) -> str:
        s2 = s.replace('lifetime ', 'lt=')
        s2 = s2.replace(', ', ',')
        s2 = ' '.join(s2.split())  # collapse multiple spaces
        return s2.lower()

    # Map normalized_name -> real name in data_dict
    normalized_to_real = { _normalize_key(k): k for k in data_dict.keys() }

    # Process each dataset
    for col, plot_set in enumerate(plot_sets):
        ax1 = axes[0, col]  # P(s)
        ax2 = axes[1, col]  # Derivative
        
        for name in plot_set['data_keys']:
            # Robust name matching via normalization
            resolved_key = normalized_to_real.get(_normalize_key(name))
            if resolved_key is None:
                print(f"Warning: {name} not found in data_dict")
                continue

            params = data_dict[resolved_key]
            data = _load_data_from_params(resolved_key, params)
            if data is None:
                print(f"Warning: Failed to load data for {resolved_key}")
                continue
            style = params.get('style', 'solid')
            alpha = params.get('alpha', 1.0)
            annotation_s_bp = params.get('annotation_s_bp', None)

            # Fixed color scheme
            if resolved_key == 'DT40':
                color = 'black'
            elif 'static' in resolved_key:
                color = 'grey'
            elif 'unnested' in resolved_key or 'free loops' in resolved_key:
                color = 'blue'
            else:  # dynamic
                color = 'red'

            # Prepare data
            y_values = data['balanced.avg.smoothed.agg'].copy()
            if normalize and alignment_s_bp is not None:
                closest_index = (data['s_bp'] - alignment_s_bp).abs().idxmin()
                norm_value = y_values.iloc[closest_index]
                if np.isnan(norm_value):
                    for i in range(closest_index + 1, len(y_values)):
                        if not np.isnan(y_values.iloc[i]):
                            norm_value = y_values.iloc[i]
                            break
                y_values = y_values / norm_value

            linestyle = '-' if style == 'solid' else '--'
            
            # Line width: base and 1.5x for reference
            base_lw = 2.0
            linewidth = base_lw * 1.5 if resolved_key == 'DT40' else base_lw

            # Prepare compact label for legend (move "free loops" to new line)
            display_label = resolved_key.replace(', free loops', '\nfree loops').replace(' free loops', '\nfree loops')

            # Plot P(s)
            ax1.plot(data['s_bp'], y_values, 
                    label=display_label, color=color, linestyle=linestyle, 
                    alpha=alpha, linewidth=linewidth)
            
            # Plot derivative
            ax2.plot(data['s_bp'], data['der'], 
                    label=display_label, color=color, linestyle=linestyle, 
                    alpha=alpha, linewidth=linewidth)

            # Add marker at annotation_s_bp on derivative plot
            if annotation_s_bp is not None:
                idx = (data['s_bp'] - annotation_s_bp).abs().idxmin()
                x_ann = data.iloc[idx]['s_bp']
                y_ann = data.iloc[idx]['der']
                ax2.plot(x_ann, y_ann, 
                        marker='o', color=color, markersize=4,
                        markeredgecolor=color, markeredgewidth=1.5,
                        linestyle='None', zorder=10)

        # Format P(s) plot
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlim(x_limits)
        if y1_limits is not None:
            ax1.set_ylim(y1_limits)
        # For first column keep labels on left, for 2-3 columns remove
        if col == 0:
            ax1.set_ylabel('Contact frequency (normalized)', fontsize=10, labelpad=4)
        else:
            ax1.set_ylabel('')
            ax1.tick_params(labelleft=False)
        ax1.tick_params(axis='x', which='major', labelsize=8, pad=2)
        ax1.tick_params(axis='y', which='major', labelsize=8, rotation=90, pad=2)
        ax1.grid(True, which='both', linestyle=':', linewidth=0.5)
        ax1.set_title(plot_set['title'], fontsize=12)

        # Filter legend (remove single cells)
        handles, labels = ax1.get_legend_handles_labels()
        pattern = re.compile(r'.*_c\d+$')
        filtered = [(h, l) for h, l in zip(handles, labels) if not pattern.match(l)]
        if filtered:
            handles_filtered, labels_filtered = zip(*filtered)
            # Compact legend with fewer columns
            ax1.legend(handles_filtered, labels_filtered, loc='upper right', fontsize=8, 
                      ncol=1, frameon=True, fancybox=False, shadow=False,
                      bbox_to_anchor=(0.98, 0.98))

        # Format derivative plot
        ax2.set_xscale('log')
        ax2.set_xlim(x_limits)
        if y2_limits is not None:
            ax2.set_ylim(y2_limits)
        ax2.set_xlabel('Genomic distance, bp', fontsize=10, labelpad=8)
        if col == 0:
            ax2.set_ylabel('Derivative', fontsize=10, labelpad=4)
        else:
            ax2.set_ylabel('')
            ax2.tick_params(labelleft=False)
        ax2.tick_params(axis='x', which='major', labelsize=8, pad=2)
        ax2.tick_params(axis='y', which='major', labelsize=8, rotation=0, pad=2)
        # Show only integer values on Y-axis for derivative
        ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax2.grid(True, which='both', linestyle=':', linewidth=0.5)

    return fig
