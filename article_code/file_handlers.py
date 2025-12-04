"""
Helper functions for writing and post-processing 1D simulation data (HDF5).

These functions are used by ``ensemble_director`` to:
- create per-object HDF5 files with SMC leg positions,
- append positions at every simulation step,
- merge many small step-range files into a single file,
- optionally downsample trajectories in space and time.
"""

import glob
import os
import re
import traceback

import h5py
import numpy as np


def create_data_files(
    ensamble_director,
    steps: int,
    starting_step: int,
    target_objects_names=None,
):
    """
    Create empty HDF5 files to store SMC leg positions for a simulation run.

    For each selected object name with general type ``'Cohesin'`` or ``'Extruder'``
    this function creates a dataset:

    ``positions[0, step, smc_index, 2] = [left_leg, right_leg]``

    where leg positions are bead indices on the 1D lattice. Unused entries
    are later filled with (-1, -1).

    Args:
        ensamble_director: Instance of :class:`ensemble_director`.
        steps: Number of simulation steps to be performed.
        starting_step: Index of the first step in this batch (for file naming).
        target_objects_names: Name or list of names of SMC object groups.
            ``None`` means all objects in ``ensamble_director.objects_names``.
    """
    if target_objects_names is None:
        target_objects_names = ensamble_director.objects_names
    
    target_objects_names = np.atleast_1d(target_objects_names).tolist()
    
    for i in [ensamble_director.objects_names.index(name) for name in target_objects_names]:
        if ensamble_director.objects_types[i] in ["Cohesin", "Extruder"]:
            file_name = (
                f"{ensamble_director.path_to_files}"
                f"{ensamble_director.objects_names[i]}_steps:{starting_step}-{starting_step+steps}.hdf5"
            )
            with h5py.File(file_name, "a") as file:
                file.attrs["Total_beads_number"] = len(
                    ensamble_director.objects[ensamble_director.objects_types.index("Bead")]
                )
                file.create_dataset(
                    "positions",
                    shape=(
                        1,  # reserved for potential future batching
                        steps,
                        ensamble_director.params_dict[ensamble_director.objects_names[i]][
                            "objects_number"
                        ],
                        2,
                    ),
                    dtype=np.int32,
                    compression="gzip",
                )


def save_cur_positions(
    ensamble_director,
    step: int,
    steps: int,
    starting_step: int,
    target_objects_names=None,
):
    """
    Save current SMC leg positions to HDF5 files for a given simulation step.

    Args:
        ensamble_director: Instance of :class:`ensemble_director`.
        step: Index of the current step within this batch (0-based).
        steps: Total number of steps in this batch (used only for file naming).
        starting_step: Global index of the first step in this batch.
        target_objects_names: Name or list of SMC object groups to save.
            ``None`` means all objects in ``ensamble_director.objects_names``.
    """
    if target_objects_names is None:
        target_objects_names = ensamble_director.objects_names
    
    target_objects_names = np.atleast_1d(target_objects_names).tolist()
    
    for n in target_objects_names:
        if ensamble_director.objects_types[ensamble_director.objects_names.index(n)] in [
            "Cohesin",
            "Extruder",
        ]:
            links = ensamble_director.return_smc_links(n)
            file_name = (
                f"{ensamble_director.path_to_files}"
                f"{n}_steps:{starting_step}-{starting_step+steps}.hdf5"
            )
            with h5py.File(file_name, "a") as file:
                dset = file["positions"]
                # Fill existing SMCs with positions, pad unused slots with (-1, -1)
                dset[0, step] = links + [
                    (-1, -1)
                    for _ in range(
                        ensamble_director.params_dict[n]["objects_number"] - len(links)
                    )
                ]


def merge_h5_files(ensamble_director, target_objects_names=None):
    """
    Merge multiple step-range HDF5 files into a single continuous file.

    Files must follow the naming pattern:
        ``<name>_steps:<start>-<end>.hdf5``

    The merged file will cover from the earliest ``start`` to the latest ``end``
    index among all fragments found.

    Args:
        ensamble_director: Instance of :class:`ensemble_director`.
        target_objects_names: Name or list of object groups to merge.
            ``None`` means all objects in ``ensamble_director.objects_names``.
    """
    if target_objects_names is None:
        target_objects_names = ensamble_director.objects_names
    
    target_objects_names = np.atleast_1d(target_objects_names).tolist()

    for n in target_objects_names:
        named_files = glob.glob(f"{ensamble_director.path_to_files}{n}_steps:*.hdf5")
        if len(named_files) > 1:
            try:
                # Sort by starting step index encoded in the file name
                named_files.sort(key=lambda f: int(re.findall(r"\d+", f)[-3]))
                with h5py.File(named_files[0], "a") as file:
                    dataset_keys = list(file.keys())

                start_step = int(re.findall(r'\d+', named_files[0])[-3])
                end_step = int(re.findall(r'\d+', named_files[-1])[-2])
                merged_name = (
                    f"{ensamble_director.path_to_files}{n}_steps:"
                    f"{start_step}-{end_step}.hdf5"
                )
                with h5py.File(merged_name, "a") as writing_file:
                    writing_file.attrs["Total_beads_number"] = len(
                        ensamble_director.objects[
                            ensamble_director.objects_types.index("Bead")
                        ]
                    )
                    deleting_files = []
                    for data_file in named_files:
                        with h5py.File(data_file, "a") as reading_file:
                            for key in dataset_keys:
                                if key in writing_file.keys():
                                    dset = writing_file[key]
                                    new_shape = list(dset.shape)
                                    new_shape[1] = (
                                        reading_file[key].shape[1]
                                        + writing_file[key].shape[1]
                                    )
                                    dset.resize(new_shape)
                                else:
                                    new_shape = list(reading_file[key].shape)
                                    max_shape = [None] * len(new_shape)
                                    dset = writing_file.create_dataset(
                                        key,
                                        shape=new_shape,
                                        dtype=np.int32,
                                        compression="gzip",
                                        maxshape=max_shape,
                                    )
                                file_start = int(re.findall(r"\d+", data_file)[-3])
                                file_end = int(re.findall(r"\d+", data_file)[-2])
                                dset[0, file_start:file_end] = reading_file[key][0, :,]
                            deleting_files.append(data_file)
                    for data_file in deleting_files:
                        os.remove(data_file)
            except Exception:
                print("Error while merging HDF5 step files!")
                with open(
                    ensamble_director.path_to_files + "/errors_file.txt", "a"
                ) as err_file:
                    err_file.write(traceback.format_exc())


def process_h5_file(input_file, output_file, step_n: int = 10, divisor: int = 5):
    """
    Downsample an HDF5 file in time (steps) and space (bead indices).

    This helper is mainly used to coarsen long 1D trajectories before
    running 3D simulations or for faster analysis.

    Operations:
        - keep every ``step_n``-th time step
        - for the ``positions`` dataset, divide bead indices by ``divisor``

    Args:
        input_file: Path to the original HDF5 file.
        output_file: Path to the downsampled HDF5 file.
        step_n: Keep every ``step_n``-th frame along the time axis.
        divisor: Integer factor to downscale bead indices (e.g. 5 → 5-bead bins).
    """
    with h5py.File(input_file, "r") as fin, h5py.File(output_file, "w") as fout:
        # Copy and update attributes
        for attr in fin.attrs:
            if attr == "Total_beads_number":
                print("Input beads number", fin.attrs[attr])
                fout.attrs[attr] = fin.attrs[attr] // divisor
                print("Output beads number", fout.attrs[attr])
            else:
                fout.attrs[attr] = fin.attrs[attr]
        
        # Process datasets
        for key in fin.keys():
            data = fin[key][:]
            # Subsample in time
            data = data[:, ::step_n, ...]
            # Downscale only bead coordinates
            if key == "positions":
                data = data // divisor
            fout.create_dataset(
                key, data=data.astype(np.int32), dtype=np.int32, compression="gzip"
            )
