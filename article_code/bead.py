class Bead:
    """
    Class representing a bead in a polymer chain.
    
    A bead is a fundamental unit in the polymer model, representing a segment of chromatin.
    Each bead can have multiple "residents" (objects like SMC complexes) associated with it.
    
    Attributes:
        residents (list): List of objects currently located on this bead
        indx (int): Index of the bead in the chain
        border_state (bool): Whether this bead is at a chain boundary
        chain (int): Chain number this bead belongs to
        size (int): Size of the bead in base pairs
    """
    
    def __init__(self, indx: int, border_state: bool, chain: int, size: int = 225):
        """
        Initialize a bead.
        
        Args:
            indx: Index of the bead in the chain
            border_state: Flag indicating if this is a boundary bead
            chain: Chain number this bead belongs to
            size: Size of the bead in base pairs (default: 225)
        """
        self.residents = []
        self.indx = int(indx)
        self.border_state = bool(border_state)
        self.chain = int(chain)
        self.size = int(size)
        
    def reg_resident(self, resident) -> None:
        """
        Register a new resident object on this bead.
        
        Args:
            resident: Object to register on this bead
        """
        if resident not in self.residents:
            self.residents.append(resident)
    
    def del_resident(self, resident) -> None:
        """
        Remove a resident object from this bead.
        
        Args:
            resident: Object to remove from this bead
        """
        try:
            self.residents.remove(resident)
        except ValueError:
            pass
    
    def get_residents(self) -> list:
        """
        Get a list of all residents on this bead.
        
        Returns:
            list: Copy of the residents list
        """
        return [resident for resident in self.residents]
    
    def get_forces(self) -> list:
        """
        Get a list of forces from all residents.
        
        Returns:
            list: List of force values from residents
        """
        return [resident.args.get('force', 0) for resident in self.residents]

    def get_border_state(self) -> bool:
        """
        Get the border state of this bead.
        
        Returns:
            bool: True if bead is at boundary, False otherwise
        """
        return self.border_state
    
    def get_attrs(self) -> list:
        """
        Get all attributes of this bead.
        
        Returns:
            list: [residents, indx, border_state, chain]
        """
        return [self.residents, self.indx, self.border_state, self.chain]
    
    def __str__(self) -> str:
        """
        String representation of the bead.
        
        Returns:
            str: Information about the bead in dictionary format
        """
        return str({
            'Residents': [(r.type, r.indx) for r in self.residents],
            'Index': self.indx, 
            'Border_state': self.border_state,
            'Chain': self.chain,
            'Size': self.size
        })
    
    def __repr__(self) -> str:
        """
        Representation for debugging.
        
        Returns:
            str: Brief information about the bead
        """
        return f"Bead(indx={self.indx}, chain={self.chain}, residents={len(self.residents)})"