import json
import random
from typing import Any, Dict, List, Optional

import numpy as np

from bead import Bead
from smc import Border, Ctcf, Cohesin, Extruder
from file_handlers import create_data_files, merge_h5_files, save_cur_positions


def convert_data(obj):
    """
    Convert numpy data types to native Python types for JSON serialization.
    
    This function recursively processes objects to convert numpy types
    (int64, float64, etc.) to standard Python types (int, float, etc.).
    This is necessary for JSON serialization of parameters.
    
    Args:
        obj: Object to convert (can be int, float, list, dict, or numpy types)
    
    Returns:
        Converted object with native Python types
    """
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_data(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_data(v) for k, v in obj.items()}
    return obj

class ensemble_director(object):
    """
    Manages ensembles of objects in the polymer model.
    
    This class coordinates all objects (beads, SMC complexes, etc.) within a single
    chromosome or chromosomal region. The total number of objects and their parameters
    are specified during initialization through a dictionary of dictionaries for each
    object type.
    
    The class handles:
    - Creating and managing beads, SMC complexes (Extruder, Cohesin), and CTCF proteins
    - Running 1D simulations of loop extrusion
    - Saving simulation data to HDF5 files
    - Tracking object states and positions over time
    """
    
    
    def __init__(
        self,
        params_dict: Dict[str, Dict[str, Any]],
        path_to_files: Optional[str] = None,
                 objects_files=None, 
        cur_step: int = 0,
        time: float = 0.0,
        dinamic_model: bool = True,
    ):
        
        if objects_files is None:
            self.params_dict = params_dict 
            self.objects = []
            self.objects_names = []
            self.data_arrays = []
            self.objects_types = []
            self.borders = []
            self.path_to_files = path_to_files
            self.objects_files = objects_files
            self.cur_step = cur_step
            self.time = time
            # Default time step for 1D extrusion dynamics (in seconds)
            self.timestep = 1 / 5
            self.dinamic_model = dinamic_model
            self.setup()
        else:
            # Reading parameters from file (not implemented)
            pass
        
        
    def setup(self):
        """
        Create all simulation objects from ``params_dict``.

        For each entry in ``params_dict`` this method instantiates:
        - a bead chain (``objects_general_type == 'Bead'``),
        - SMC complexes (``'Cohesin'`` or ``'Extruder'``),
        - CTCF barriers (``'CTCF'``).

        It also writes per-object JSON files with the original parameters
        into ``path_to_files`` for reproducibility.
        """
        for obj in self.params_dict.values():
            
            assert obj['objects_general_type'] in ['Bead', 'Cohesin', 'Extruder', 'CTCF'], \
                    f"Object type '{obj['objects_general_type']}' is not supported!"
            
            if obj['objects_general_type'] == 'Bead':
                self.creating_chain(obj)
            elif obj['objects_general_type'] == 'Cohesin' or obj['objects_general_type'] == 'Extruder':
                self.creating_smcs(obj)
            elif obj['objects_general_type'] == 'CTCF':
                self.creating_ctcf(obj)
            else:
                raise ValueError(f"Unsupported object type: {obj['objects_general_type']}")
            
        for name in self.params_dict.keys():
            # Save object-specific parameters to separate JSON files
            json.dump(
                convert_data(self.params_dict[name]),
                open(f"{self.path_to_files}/{name}.json", "w"),
                indent=4,
            )
        
        print('Setup complete!')


    def creating_chain(self, chain_params: Dict[str, Any]) -> None:
        """
        Initialize beads for the polymer chain.
        
        Creates a list of Bead objects representing the chromatin fiber segments.
        Each bead can have associated border states and chain assignments.
        
        Args:
            chain_params: Dictionary containing chain parameters:
                - 'objects_number': Number of beads
                - 'objects_sizes': List of bead sizes in base pairs
                - 'two_chain': Whether this represents two chains (sister chromatids)
                - 'borders': Optional list of border positions
        """
        beads: List[Bead] = []
        bead_sizes = chain_params.get('objects_sizes', [])
        # two_chain = chain_params.get('two_chain', False)
        
        # Default attributes for chain borders and obstacles
        border_attrs = {"Active": False}
        for obj in self.params_dict.values():
            # Determine where the 'type' field is stored for this object
            obj_type = None
            if 'args' in obj and 'type' in obj['args']:
                obj_type = obj['args']['type']
            elif 'args_smc' in obj and 'type' in obj['args_smc']:
                obj_type = obj['args_smc']['type']
            if obj_type is not None:
                border_attrs[obj_type] = 0.0
        
        borders = chain_params.get('borders', [])
        chain_num = 0
        for i in range(chain_params['objects_number']):
            
            if i in borders:
                border_state = False
                chain_num = borders.index(i)//2
            else:
                border_state = True
            
            # Create bead with given genomic size and chain/border state
            bead = Bead(
                indx=i,
                border_state=border_state,
                chain=chain_num,
                size=bead_sizes[i]
            )
            beads.append(bead)
            if border_state == False:
                self.borders.append(Border(position=bead,
                                      attrs=border_attrs))
        
        # Register beads in the internal object lists
        self.objects.append(beads)
        self.objects_names.append('chain')
        self.objects_types.append('Bead')
        
        
        
    def change_bead_sizes(self, new_bead_sizes):
        pass
        
        
    def creating_smcs(self, obj: Dict[str, Any]) -> None:
        """
        Initialize SMC complexes (Extruder or Cohesin).
        
        Creates a list of SMC complex objects that will perform loop extrusion
        or sister chromatid cohesion.
        
        Args:
            obj: Dictionary containing SMC parameters:
                - 'objects_number': Number of SMC complexes
                - 'args_smc': Arguments for SMC behavior (velocity, lifetime, etc.)
                - 'attrs_smc': Attributes for SMC complexes
                - 'positions': List of initial positions [left_leg, right_leg] for each complex
        """
        smcs = []
        if obj['objects_general_type'] == 'Cohesin':
            kind_of = Cohesin
        elif obj['objects_general_type'] == 'Extruder':
            kind_of = Extruder
            
        for smc in range(obj['objects_number']):
            smcs.append(kind_of(smc_indx=smc,
                                    args=obj['args_smc'], 
                                    attrs=obj['attrs_smc'],
                                    positions = obj['positions'][smc]))
            
        self.objects.append(smcs)
        self.objects_names.append(obj['name'])
        self.objects_types.append(obj['objects_general_type'])
        print(self.get_object_info(obj['name']))
        
    def creating_ctcf(self, obj: Dict[str, Any]) -> None:
        """
        Initialize CTCF proteins.
        
        CTCF proteins act as barriers to loop extrusion, stopping SMC complexes
        at specific genomic positions.
        
        Args:
            obj: Dictionary containing CTCF parameters:
                - 'objects_number': Number of CTCF proteins
                - 'args': Arguments for CTCF behavior
                - 'attrs': List of attributes for each CTCF
                - 'positions': List of positions for each CTCF
        """
        ctcfs = []
        for ctcf in range(obj['objects_number']):
            ctcfs.append(Ctcf(indx=ctcf,
                              args=obj['args'], 
                              attrs=obj['attrs'][ctcf],
                              positions=obj['positions'][ctcf]
                              ))
            
        self.objects.append(ctcfs)
        self.objects_names.append(obj['name'])
        self.objects_types.append(obj['objects_general_type'])
        print(self.get_object_info(obj['name']))
        
    
    def capture_smcs_ctcfs(self, name: str, amount: Optional[int] = None) -> None:
        """
        Capture SMC complexes or CTCF proteins onto the polymer chain.
        
        This function binds free SMC complexes or CTCF proteins to available
        positions on the polymer chain. For static models, positions are pre-assigned.
        For dynamic models, positions are found randomly.
        
        Args:
            name: Name of the object type to capture (e.g., 'condensin1')
            amount: Number of objects to capture. If None, captures all available.
        """
        def extract_random_pair(pairs_list):
            if not pairs_list:
                return None
            idx = random.randrange(len(pairs_list))
            return pairs_list.pop(idx)
        
        
        free_smcs = self.get_objects_array(name, capture_state=False)
        if amount is None or amount > len(free_smcs):
            amount = len(free_smcs)
        assert amount >= 0, f'Cannot capture negative amount of SMC complexes: {amount}'
        
        bead_pairs = []
        if amount > 0:
            if len(bead_pairs) > 0:
                if free_smcs[0].positions == None:
                    for s in range(amount):
                        free_smcs[s].capture(extract_random_pair(bead_pairs))
                else:
                    for s in range(amount):
                        bead_pairs = [[self.objects[self.objects_types.index('Bead')][free_smcs[s].positions[0]], 
                                      self.objects[self.objects_types.index('Bead')][free_smcs[s].positions[1]]]]
                        free_smcs[s].capture(extract_random_pair(bead_pairs))
            else:
                for s in range(amount):
                    free_smcs[s].old_capture(self.objects[self.objects_types.index('Bead')])
            
            
    def get_free_beads(self, filter_mode: str):
        """
        Get beads that are not occupied by SMC complexes or other objects.
        
        Args:
            filter_mode: Filtering mode:
                - 'legs_only': Return beads not occupied by any SMC legs, CTCF, or borders
                - object_name: Return beads not inside loops of the specified SMC complex
                - other: Return all beads
        
        Returns:
            list: List of free Bead objects
        """
        beads = self.objects[self.objects_types.index('Bead')]
        n_beads = len(beads)
        mask = np.ones(n_beads, dtype=bool)

        if filter_mode == 'legs_only':
            # Collect all positions occupied by SMC legs, CTCF, or borders
            occupied_indices = set()

            # SMC complexes: Extruder and Cohesin
            for obj_type in ['Cohesin', 'Extruder']:
                if obj_type in self.objects_types:
                    name = self.objects_names[self.objects_types.index(obj_type)]
                    for left, right in self.return_smc_links(name):
                        occupied_indices.add(left)
                        occupied_indices.add(right)

            # CTCF proteins
            for obj_type in ['CTCF']:
                if obj_type in self.objects_types:
                    name = self.objects_names[self.objects_types.index(obj_type)]
                    for ctcf in self.get_objects_array(name, capture_state=True):
                        occupied_indices.add(ctcf.get_position())
                        
            # Chain borders
            for obj_type in ['Chain_border', 'Border']:
                if obj_type in self.objects_types:
                    name = self.objects_names[self.objects_types.index(obj_type)]
                    for border in self.get_objects_array(name, capture_state=True):
                        occupied_indices.add(border.get_position())

            # Mark occupied beads as False
            for idx in occupied_indices:
                if 0 <= idx < n_beads:
                    mask[idx] = False

            # Return only free beads
            return list(np.array(beads)[mask])

        elif filter_mode in self.objects_names:
            # Return beads not inside loops of the specified SMC complex
            beads = np.array(beads)
            mask = np.ones(n_beads, dtype=bool)
            smc_links = self.return_smc_links(filter_mode)
            for left, right in smc_links:
                l, r = min(left, right), max(left, right)
                mask[l:r+1] = False
            return list(beads[mask])

        else:
            # Default: return all beads
            return beads
        
    def get_free_bead_pairs(self, filter_mode: str = "legs_only"):
        """
        Get pairs of adjacent beads suitable for extruder binding.
        
        Args:
            filter_mode: Filtering mode:
                - 'legs_only': Beads not occupied by any legs
                - SMC complex name (e.g., 'condensin1'): Beads not inside loops of that complex
        
        Returns:
            numpy.ndarray: Array of bead pairs, shape (num_pairs, 2)
        """
        beads = self.objects[self.objects_types.index('Bead')]
        n_beads = len(beads)
        bead_array = np.array(beads, dtype=object)

        mask = np.ones(n_beads, dtype=bool)

        if filter_mode == 'legs_only':
            # Collect all occupied positions (legs of all objects)
            occupied_indices = set()

            # SMC complexes: Extruder and Cohesin
            for obj_type in ['Cohesin', 'Extruder']:
                if obj_type in self.objects_types:
                    name = self.objects_names[self.objects_types.index(obj_type)]
                    for left, right in self.return_smc_links(name):
                        occupied_indices.add(left)
                        occupied_indices.add(right)
            # CTCF proteins
            for obj_type in ['CTCF']:
                if obj_type in self.objects_types:
                    name = self.objects_names[self.objects_types.index(obj_type)]
                    for ctcf in self.get_objects_array(name, capture_state=True):
                        occupied_indices.add(ctcf.get_position())
            # Chain borders
            for obj_type in ['Chain_border', 'Border']:
                if obj_type in self.objects_types:
                    name = self.objects_names[self.objects_types.index(obj_type)]
                    for border in self.get_objects_array(name, capture_state=True):
                        occupied_indices.add(border.get_position())

            # Mark occupied beads as False
            occupied = np.array(list(occupied_indices), dtype=int)
            valid_mask = (occupied >= 0) & (occupied < n_beads)
            mask[occupied[valid_mask]] = False

        elif filter_mode in self.objects_names:
            # Exclude beads inside loops of the specified SMC complex
            mask = np.ones(n_beads, dtype=bool)
            smc_links = self.return_smc_links(filter_mode)
            for left, right in smc_links:
                mask[left:right+1] = False

        # Find pairs of adjacent free beads
        mask_pairs = mask[:-1] & mask[1:]
        indices = np.where(mask_pairs)[0]
        # Create array of bead pairs
        pairs = np.stack([bead_array[indices], bead_array[indices+1]], axis=1)
        return pairs

        
            
    def release_smcs_by_numpy(self, name: str) -> None:
        """
        Optimized SMC release using vectorized numpy operations.
        
        This function efficiently releases SMC complexes based on their lifetime
        and rebinds them after a reborn time. Uses numpy for vectorized probability
        calculations instead of iterating through each complex individually.
        
        Args:
            name: Name of the SMC complex type to process
        """
        # 1 / lifetime is the per-step release probability (Poisson process)
        prob = 1 / self.params_dict[name]["args_smc"]["lifetime"]
        smcs_on = self.get_objects_array(name, capture_state=True)
        mask = np.random.random(size=len(smcs_on)) < prob
        
        # Check for CTCF-stalled complexes (different lifetime)
        smcs_falling = [smcs_on[i] for i in np.arange(len(mask))[mask]]
        prob = 1 / self.params_dict[name]["args_smc"]["lifetime_on_ctcf"]
        mask = np.random.random(size=len(smcs_falling)) < prob
        for i in np.arange(len(mask))[mask]:
            smcs_falling[i].release()
        
        # Rebind released complexes after reborn time
        # Rebinding probability for free complexes (reborn time)
        prob = 1 / self.params_dict[name]["args_smc"]["reborntime"]
        mask = sum(np.random.random(size=len(self.get_objects_array(name, capture_state=False))) < prob)
        if mask > 0:
            self.capture_smcs_ctcfs(name, mask)
                
            
    def get_object_info(self, names=None):
        """
        Get information about objects: how many are bound and how many are free.
        
        Args:
            names: Name(s) of object types to query. If None, returns info for all objects.
                  Can be a string, list of strings, or None.
        
        Returns:
            dict: Dictionary with object information:
                - 'object_general_type': Type of object (Bead, Extruder, etc.)
                - 'onbead_number': Number of objects currently bound to the chain
                - 'total_number': Total number of objects (bound + free)
        """
        if names == None:
            names = self.objects_names
        elif type(names) is list:
            names = names
        elif type(names) is str:
            names = [names]
        else:
            raise ValueError(f'Invalid format for names: {names}. Expected string, list, or None.')
            
        objects_info = {}
        for name in names:
            objects_info[name] = {
                    'object_general_type': self.objects_types[self.objects_names.index(name)],
                    'onbead_number': len(self.get_objects_array(name, capture_state=True)),
                    'total_number': len(self.get_objects_array(name, capture_state=False)),
            }
            
        return objects_info
    
    
    def get_objects_array(self, name: str, capture_state=True):
        """
        Get array of objects of specified type.
        
        Args:
            name: Name of the object type
            capture_state: Filter by capture state:
                - True: Return only objects bound to the chain
                - False: Return only objects not bound to the chain
                - other: Return all objects regardless of state
        
        Returns:
            list: List of objects matching the criteria
        """
        if type(capture_state) is bool or capture_state == 'True' or capture_state == 'False':
            # Return objects of specified type that are bound or unbound
            try:
                return [x for x in self.objects[self.objects_names.index(name)] if x.onbead == bool(capture_state)]
            except:
                # Object type doesn't have onbead attribute
                return [x for x in self.objects[self.objects_names.index(name)]]
        else:
            # Return all objects of specified type regardless of state
            return [x for x in self.objects[self.objects_names.index(name)]]
        
        
    def make_one_step(self, name: str) -> None:
        """
        Perform one simulation step for objects of specified type.
        
        For SMC complexes, this includes:
        1. Releasing complexes based on lifetime
        2. Moving active complexes along the chain
        3. Rebinding released complexes
        
        Args:
            name: Name of the object type to update
        """
        if self.objects_types[self.objects_names.index(name)] in ["Extruder", "Cohesin"]:
            self.release_smcs_by_numpy(name)
        for elem in self.get_objects_array(name, capture_state=True):
            elem.go_one(self.objects[self.objects_types.index('Bead')], self.timestep)
            
            
    def return_smc_links(self, name: Optional[str] = None):
        """
        Get positions of SMC complex legs (loop boundaries).
        
        Returns an array of (left_leg, right_leg) positions for all bound
        SMC complexes of the specified type. This is used to track loop
        positions during simulation.
        
        Args:
            name: Name of the SMC complex type. If None, returns empty list.
        
        Returns:
            list: List of tuples (left_leg_pos, right_leg_pos) for each bound complex
        """
        if name != None:
            returning_array = []
            for elem in self.get_objects_array(name, capture_state=True):
                returning_array.append(elem.get_position())
            return returning_array
        return []
        
        
    def catch_smcs(self) -> None:
        """
        Capture all free SMC complexes onto the chain.
        
        This is a convenience function that captures all available
        SMC complexes (both Extruder and Cohesin types) to random
        free positions on the chain.
        """
        for i in range(len(self.objects_types)):
            if self.objects_types[i] in ["Cohesin","Extruder"]:
                free_smcs = self.get_objects_array(self.objects_names[i], False)
                for s in range(len(free_smcs)):
                    free_smcs[s].capture(self.get_free_beads(self.objects_names))
            
                        
                
    def run_simulation(
        self,
        timestep: float = 1 / 5,
        steps: Optional[int] = None,
        time_total: Optional[float] = None,
        merge_files_after_calculation: bool = True,
    ) -> None:
        """
        Run 1D simulation of loop extrusion.
        
        This function performs the 1D lattice simulation where SMC complexes
        bind, extrude loops, and unbind from the polymer chain. Positions
        are saved to HDF5 files for later use in 3D simulations.
        
        Args:
            timestep: Time step in seconds (default: 0.2)
            steps: Number of simulation steps. If None, calculated from time_total
            time_total: Total simulation time in seconds. If None, uses steps
            merge_files_after_calculation: Whether to merge output files after simulation
        """
        self.timestep = timestep
        starting_step = self.cur_step
        if time_total != None:
            steps = int(time_total//timestep)
        elif steps != None:
            time_total = int(steps*timestep)
        else:
            raise ValueError("Must specify either 'steps' or 'time_total'")
        
        # Create data files for saving
        create_data_files(
                       ensamble_director = self,
                       steps = steps,
                       starting_step = starting_step,
                       target_objects_names = None,
                         )
        
        # Capture initial SMC complexes and CTCF proteins
        for i in range(len(self.objects_types)):
            if self.objects_types[i] in ["Cohesin","Extruder"]:
                self.capture_smcs_ctcfs(name = self.objects_names[i])
            if self.objects_types[i] in ["CTCF"]:
                self.capture_smcs_ctcfs(name = self.objects_names[i])
                    
        # Run simulation steps
        for step in range(steps):
            for i in range(len(self.objects_types)):
                if self.objects_types[i] in ["Cohesin","Extruder"]:
                    self.make_one_step(self.objects_names[i])
                    print(' '*200, end = '\r')
                    print(f'Current step {self.cur_step+1}', end = '\r')
                    
                    save_cur_positions(
                       ensamble_director = self,
                       step = step, 
                       steps = steps, 
                       starting_step = starting_step, 
                       target_objects_names = self.objects_names[i],
                      )
                    
            self.cur_step+=1
        
        if merge_files_after_calculation:
            merge_h5_files(
                           ensamble_director = self,
                           target_objects_names = self.objects_names,
                          )
        