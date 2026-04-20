
"""
Replay Memory utilities for Deep Q-Learning.

This module defines:
- the Transition data structure used to store experience tuples
- a ReplayMemory buffer for storing and sampling transitions

The replay memory enables off-policy learning by breaking temporal
correlations between consecutive samples and improving training stability.
"""
import random
from collections import namedtuple, deque
import numpy as np


def createTransition(cfg):
    """
    Define the transition structure used in replay memory.

    Each transition stores:
    - pHidden: previous hidden state (policy network)
    - state: current state
    - action: action taken
    - tHidden: target hidden state
    - next_state: next state after action
    - reward: reward received
    - video: episode identifier
    - frame: frame index within the episode
    - percentage_completion_video: progress within the episode

    Returns
    -------
    namedtuple
        Transition structure used across training.
    """
    transition = namedtuple('Transition',
                                ('pHidden', 'state', 'action', 'tHidden', 'next_state', 'reward', 'video', 'frame', 'percentage_completion_video'))
    return transition


class ReplayMemory(object):

    """
    Replay Memory buffer for storing experience transitions.

    This is a FIFO (First-In-First-Out) buffer with fixed capacity.
    When the buffer is full, the oldest transitions are discarded.

    It supports:
    - adding new transitions
    - random sampling for training batches
    - tracking recently added transitions (optional)
    """

    def __init__(self, cfg, capacity):
        """
        Initialize the replay memory.

        Parameters
        ----------
        cfg : object
            Configuration object.
        capacity : int
            Maximum number of stored transitions.
        """
        self.cfg = cfg
        self.transition = createTransition(cfg)
        # Main memory buffer
        self.memory = deque([], maxlen=capacity)
        # Auxiliary array to track which transitions were recently added
        self.selected_transitions = np.zeros(capacity)
        
        return
    
    def push(self, *args):
        """
        Store a new transition in memory.

        Parameters
        ----------
        *args : tuple
            Elements of the Transition tuple.
        """
        self.memory.append(self.transition(*args))
        # Mark this transition as "selected"
        self.selected_transitions[len(self.memory)-1] = 1
        return
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions.

        Uses random sampling with replacement (random.choices).

        Parameters
        ----------
        batch_size : int
            Number of samples.

        Returns
        -------
        list
            List of sampled transitions.
        """
        return random.choices(self.memory, k=batch_size) #random.sample()
    def get_index_selected_transitions(self):
        """
        Retrieve indices of recently added transitions.

        After retrieval, the selection flags are reset.

        Returns
        -------
        np.ndarray
            Indices of selected transitions.
        """
        idx = np.where(self.selected_transitions == 1)[0]
        self.selected_transitions[idx] = 0
        return idx
    def get_elements(self, x):
        """
        Print the first x transitions (for debugging).

        Displays:
        - action meaning
        - reward

        NOTE:
        This function is primarily for debugging/inspection.
        """
        ROBOT_ACTIONS_MEANINGS = ['bring butter', 'bring jam', 'bring milk', 'bring nutella', 'bring tomato sauce','idle']

        for transition in list(self.memory)[:x]:

            print('{} {} '.format(ROBOT_ACTIONS_MEANINGS[transition.action],transition.reward[0]))

    def load_memory(self, *args):
        """
        Load a transition into memory (used when restoring from disk).
        """
        self.memory.append(self.transition(*args))

    def last_element(self):
        """
        Get the most recent transition.

        Returns
        -------
        Transition
        """
        return list(self.memory)[-1]

    def __len__(self):
        """
        Return current memory size.
        """
        return len(self.memory)
    
    



