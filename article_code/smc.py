import numpy as np
import random


class Leg(object):
    """
    Single SMC leg located on a bead of the 1D lattice.

    Each leg stores:
    - ``pos``: current bead object (or ``None`` if detached)
    - ``direction``: +1 or -1, preferred extrusion direction
    - ``attrs``: dictionary of state flags (e.g. ``{\"Active\": True}``)
    """
    
    def __init__(self, pos, direction, attrs={"Active": False}):
        self.pos = pos
        if pos is not None:
            self.pos.reg_resident(self)
        self.attrs = dict(attrs)
        self.parent = None
        self.direction = direction
    
    def setup(self, parent):
        """Attach a reference to the parent SMC-like object."""
        self.parent = parent
        
    def change_pos(self, new_pos):
        """
        Move the leg to a new bead and update bead residency lists.
        """
        if self.pos is not None:
            self.pos.del_resident(self)
        self.pos = new_pos
        if new_pos is not None:
            self.pos.reg_resident(self)
        
    def __getattr__(self, name):
        """Forward attribute access to the parent (for shared parameters)."""
        return getattr(self.parent, name)
    
    def reverse_active_state(self):
        """
        Toggle the leg ``Active`` flag used to decide which leg is currently extruding.
        """
        self.attrs['Active'] = not self.attrs['Active']
        
    def push(self, beads, n_steps=1, force=None, direction=1):
        """
        Attempt to push other objects along the lattice in a given direction.

        The leg first tries to push all residents in front of it up to ``n_steps``
        away, spending available ``force`` on overcoming their individual
        ``resident.force`` values. If there is enough remaining force the leg
        finally moves itself by the maximal possible number of beads.

        This is a generic pushing routine used when ``SMC.pushing`` is enabled.

        Args:
            beads: list of bead objects forming the 1D lattice.
            n_steps: maximum number of beads to test ahead.
            force: available pushing force (defaults to ``self.force``).
            direction: +1 or -1, direction along the lattice.
        """
        if force is None:
            force = self.force
        possible_step = 0
        continue_flag = True
        for step in range(1, n_steps+1):
            
            if beads[self.pos.indx+direction*step].border_state<=0:
                continue_flag = False
                break
            else:
            
            # Start checking residents from the farthest bead along the direction of motion
                for resident in beads[self.pos.indx+direction*step].residents[::-direction]:
                    # If the current object is allowed to share a bead with the next one,
                    # push without checking force thresholds

                    if resident.type not in self.neighbour_list:
                        if force > resident.force:
                            force -= resident.force
                            # Recursively push the resident further along the same direction
                            answer = resident.push(
                                beads,
                                n_steps=n_steps-step+1,
                                force=force,
                                direction=direction,
                            )
                            if not answer:
                                continue_flag = False
                                possible_step = answer
                                break
                        else:
                            continue_flag = False
                            break
            
            
            if continue_flag:
                possible_step+=1
            else:
                break
        if possible_step>0:
            self.change_pos(beads[self.pos.indx+direction*possible_step])
        
        return int(possible_step)

    
class SMC(object):
    """
    Base class for SMC-like complexes (e.g. condensins, cohesins) on a 1D lattice.

    An SMC has two legs that can bind to beads and move to extrude loops.
    The physical behaviour is controlled by ``args``:

    - ``type``: string label used to index leg preferences in bead/CTCF attributes
    - ``force``: effective pushing force used in ``Leg.push`` (in arbitrary units)
    - ``lifetime``: average lifetime in units of simulation steps (see article)
    - ``diff_prob``: probability of a diffusive step per time step
    - ``step_prob``: probability of an active extrusion step per time step
    - ``change_dir_prob``: probability to switch the active leg before a step
    - ``n_steps``: maximal lattice distance per attempted step
    """
    def __init__(self, smc_indx, args, attrs, positions):
        
        self.type = args['type']
        self.force = args['force']
        self.indx = smc_indx
        # Flag showing whether the complex is currently attached to beads
        self.onbead = False
        self.left = Leg(None, direction=-1, attrs = attrs)
        self.right = Leg(None, direction=1, attrs = attrs)
        self.args = args
        self.neighbour_list = []
        # Optional preferred positions (for static capture)
        self.positions = positions
        # If True, use force-based pushing of other residents; otherwise, hop only to free beads
        self.pushing = args['pushing']
        
        self.left.setup(self)
        self.right.setup(self)
        self.neighbour_list = []
    
    def get_active_leg(self):
        """Return the currently active leg (used for directional extrusion)."""
        if self.left.attrs['Active']:
            return self.left
        else:
            return self.right
        
    def change_active_leg(self):
        """Swap which leg is considered active."""
        self.left.attrs['Active'] = not self.left.attrs['Active']
        self.right.attrs['Active'] = not self.right.attrs['Active']
    
    
    def __getitem__(self, item):
        
        if item == -1:
            return self.left
        elif item == 1:
            return self.right
        else:
            raise ValueError()     

    def unloadProb(self):
        """
        Probability to release the complex from the chain in a single time step.

        This is simply ``1 / lifetime`` in simulation time units, where
        ``lifetime`` is specified in ``args`` (see article for mapping to seconds).
        """
        return 1 / self.args["lifetime"]    
    
    def __str__(self):
        
        if self.onbead:
            
            return str({'Type': self.type,
                        'Index': self.indx, 
                        'Onbead':self.onbead,
                        'LLeg': self.left.pos.indx,
                        'RLeg': self.right.pos.indx,
                        'Args': str(self.args)
                       })
        
        else:
            
            return str({'Type': self.type,
                        'Index': self.indx, 
                        'Onbead':self.onbead,
                        'LLeg': 'None',
                        'RLeg': 'None',
                        'Args': str(self.args)
                       })
        
    def release(self):
        """
        Detach both legs from the chain and reset the active leg state.
        """
        self.onbead = False
        self.left.change_pos(None)
        self.right.change_pos(None)
        self.get_active_leg().reverse_active_state()
        
        
    
    def get_passprob_for_step(self, beads, leg, direction, n_steps):
        """
        Compute passing probabilities for a leg attempting to move ``n_steps`` beads.

        For each candidate bead ahead, the probability to pass is computed as the
        product of leg-specific acceptance probabilities stored in bead residents'
        attributes. This encodes local barriers such as CTCF or other SMCs.

        Args:
            beads: list of bead objects.
            leg: -1 for left, +1 for right leg.
            direction: +1 or -1 along the lattice.
            n_steps: number of candidate beads to inspect.

        Returns:
            probs, positions: lists of probabilities and corresponding bead objects,
            ordered from farthest to closest bead.
        """
        b_num = self[leg].pos.indx
        probs = []
        positions = []
        
        for st in range(1, n_steps+1):
            next_pos = b_num + direction*st
            
            # Check lattice boundaries
            if next_pos >= len(beads) or next_pos < 0:
                return probs[::-1], positions[::-1]
            
            current_bead = beads[next_pos]
            
            if current_bead.border_state > 0:
                # Use residents list directly
                residents = current_bead.residents
                
                if not residents:
                    probs.append(1.0)
                else:
                    place = 1.0
                    for res in residents:
                        try:
                            place *= res.attrs[self.type+'+' if direction == 1 else self.type+'-']
                        except:
                            place *= res.attrs[self.type]
                    probs.append(place)
                    
                positions.append(current_bead)
            else:
                return probs[::-1], positions[::-1]

        return probs[::-1], positions[::-1]
    
    def own_step(self, beads, leg, direction, n_steps):
        """
        Perform an intrinsic step of a leg in the given direction.

        This routine does not push other objects; instead it samples whether
        the leg can overstep local barriers using the probabilities returned
        by :meth:`get_passprob_for_step`.

        Args:
            beads: list of bead objects.
            leg: -1 for left, +1 for right leg.
            direction: +1 or -1 along the lattice.
            n_steps: maximal number of beads the leg may advance.

        Returns:
            int: remaining number of steps that were not used (for chaining moves).
        """
        probs, positions = self.get_passprob_for_step(beads, leg, direction, n_steps)
        # We start from index 1 so that for each candidate bead we multiply
        # probabilities *beyond* it when computing the cumulative pass probability.
        step = 1
        for pr, bead in zip(probs, positions):
            is_free = len(bead.residents)==0 and bead.border_state>0
            prob = np.prod(np.array(probs[step:]))
            will_pass = np.random.random()<=prob
            step+=1
            if will_pass and is_free:
                self[leg].change_pos(beads[bead.indx])
                break
        
        return n_steps-step+1
    
    def get_position(self):
        return self.left.pos.indx, self.right.pos.indx
    
class Extruder(SMC):
    """
    Loop-extruding SMC complex with two legs moving along one or two chromatin chains.
    
    In the article this class is used to model condensin-like loop extruders.
    Key physical parameters are passed via ``args`` (see :class:`SMC`), where
    ``diff_prob`` and ``step_prob`` control the relative contribution of
    diffusive hopping vs. directed extrusion per time step.
    """
    def old_capture(self, beads):
        """
        Capture the extruder onto two adjacent free beads selected randomly.
        
        This routine is used for dynamic models without predefined binding sites.
        """
        if self.positions != None:
            self.left.change_pos(beads[self.positions[0]])
            self.right.change_pos(beads[self.positions[1]])
            setattr(self[random.choice([-1,1])], 'Active', True)
            self.onbead = True
        else:
            for _ in range(100):  # Try up to 100 times to find a free neighbouring pair of beads
                a = np.random.randint(self.args["beads_number"] - 1)
                if len(beads[a].residents[:])==0 and len(beads[a+1].residents[:])==0 \
                    and beads[a].chain==beads[a+1].chain \
                    and beads[a].border_state>0 and beads[a+1].border_state>0:
                    setattr(self[random.choice([-1,1])], 'Active', True)
                    self.left.change_pos(beads[a])
                    self.right.change_pos(beads[a+1])
                    self.onbead = True
                    break
    
    
    def static_capture(self, beads):
        """
        Capture the extruder to a predefined bead pair given in ``self.positions``.
        
        This is used for static models where all complexes start at the same sites.
        """
        self.left.change_pos(beads[self.positions[0]])
        self.right.change_pos(beads[self.positions[1]])
        setattr(self[random.choice([-1,1])], 'Active', True)
        self.onbead = True
    
    def capture(self, bead_pair):
        """
        Generic capture method that binds legs to an explicit bead pair.
        
        Args:
            bead_pair: tuple/list of two bead objects (left, right).
        """
        self.left.change_pos(bead_pair[0])
        self.right.change_pos(bead_pair[1])
        setattr(self[random.choice([-1,1])], 'Active', True)
        self.onbead = True
        

    def go_one(self, beads, *args, **kwargs):
        """
        Perform one time step update for the extruder.

        The step consists of:
        1. With probability ``diff_prob``: attempt a diffusive move of a random leg.
        2. With probability ``step_prob``: attempt a directed extrusion step of
           the active leg (which may be switched with ``change_dir_prob``).
        If ``self.pushing`` is True, extrusion uses :meth:`Leg.push` and can
        move other objects; otherwise it calls :meth:`own_step`.
        """
        if np.random.random()<self.args['diff_prob']:
            leg = random.choice([-1,1])
            self.own_step(beads, leg=leg, direction=random.choice([-1,1]), n_steps=1)

        if np.random.random()<self.args['step_prob']:
            if np.random.random()<self.args['change_dir_prob']:
                self.change_active_leg()
            leg = self.get_active_leg().direction
            if self.pushing:
                self[leg].push(beads, direction=leg, force = self.args['force'], n_steps=self.args['n_steps'])
            else:
                self.own_step(beads, leg=leg, direction=leg, n_steps=self.args['n_steps'])
            
    
class Cohesin(SMC):
    """
    Cohesin-like SMC complex connecting two distant beads, typically
    on different sister chromatids.
    
    The capture routine samples symmetric positions from the two halves
    of the lattice, reflecting the geometry used in the article.
    """
    def capture(self, beads):
        """
        Capture cohesin by connecting symmetric positions on two chains.
        
        A random index ``a`` is chosen in the first half of the lattice and the
        second leg is placed at the mirrored position ``-a-1`` on the other chain.
        """
        while True:
            a = np.random.randint(self.args["beads_number"]//2) 
            if len(beads[a].residents[:])==0 and len(beads[-a-1].residents[:])==0 \
                and beads[a].chain!=beads[-a-1].chain \
                and beads[a].border_state>0:
                
                self.left.change_pos(beads[a])
                self.right.change_pos(beads[-a-1])
                self.onbead = True
                break
                
    def extra_capture(self, beads):
        """Placeholder for alternative cohesin capture modes (not implemented)."""
                
    def go_one(self, beads, *args, **kwargs):
        
        if np.random.random()<self.args['diff_prob']:
            leg = random.choice([-1,1])
            self.own_step(beads, leg=leg, direction=random.choice([-1,1]), n_steps=1)
            
class Ctcf(object):
    """
    Simple CTCF-like barrier bound to a single bead of the chain.
    
    CTCF does not move in this implementation; it only occupies its bead
    and modifies pass probabilities of SMC legs through its attributes.
    """
    def __init__(self, indx, args, attrs, positions):
        
        self.type = args['type']
        self.force = args['force']
        self.indx = indx
        self.onbead = False
        self.leg = Leg(None, direction=-1, attrs = attrs)
        self.args = args
        self.positions = positions
        self.leg.setup(self)

    def old_capture(self, beads):
        self.leg.change_pos(beads[self.positions[0]])
        self.onbead = True
        
    def go_one(self, beads, *args, **kwargs):
        pass

    def get_position(self):
        return self.leg.pos.indx
    
class Border(object):
    """
    Chain border object representing a hard boundary between chromatin segments.
    
    This object has a very large ``force`` and is always ``onbead``, effectively
    preventing SMC legs from crossing into the next chain segment.
    """
    def __init__(self, position, attrs):
        
        self.type = 'Chain_border'
        self.force = 1000000.
        self.onbead = True
        self.leg = Leg(position, direction=-1, attrs = attrs)
        self.leg.setup(self)
        
    def go_one(self, beads, *args, **kwargs):
        pass
        
    def get_position(self):
        return self.leg.pos.indx