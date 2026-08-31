"""
Demonstration buffer for storing source domain transitions and transferring them to target domain.
Conforms to Section VI of the research paper (Niknia et al., 2024).
"""

from typing import List, Tuple, Optional
import pickle
import os
import numpy as np

from memory.prioritized_buffer import PrioritizedReplayBuffer, Transition


class DemonstrationBuffer:
    """Manages expert demonstration transitions collected from the source domain."""

    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.transitions: List[Transition] = []

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        tau: float,
        n_step_state: Optional[np.ndarray] = None,
        n_step_return: Optional[float] = None,
        n_step_tau: Optional[float] = None,
    ):
        """Adds a transition marked as expert demonstration."""
        if len(self.transitions) >= self.capacity:
            self.transitions.pop(0)

        t = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            tau=tau,
            n_step_state=n_step_state,
            n_step_return=n_step_return,
            n_step_tau=n_step_tau,
            expert_action=action,
            is_expert=True,
        )
        self.transitions.append(t)

    def populate_target_buffer(self, target_buffer: PrioritizedReplayBuffer):
        """
        Loads all demonstration transitions into target domain PER buffer (Section VI).
        Remaining capacity in target buffer is left vacant for newly collected target transitions.
        """
        for t in self.transitions:
            target_buffer._store_transition(t, is_expert=True)

    def save(self, filepath: str):
        """Saves demonstrations to file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.transitions, f)

    def load(self, filepath: str):
        """Loads demonstrations from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Demonstration file not found: {filepath}")
        with open(filepath, 'rb') as f:
            self.transitions = pickle.load(f)

    def __len__(self) -> int:
        return len(self.transitions)

