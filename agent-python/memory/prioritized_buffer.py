"""
Prioritized Experience Replay (PER) Buffer with continuous SMDP discount and reward-adjusted priority (Eq. 6).
Conforms to Section VI-1 and Table II of the research paper (Niknia et al., 2024).
"""

from typing import Tuple, List, Optional
from collections import deque
import numpy as np
import torch

from memory.sum_tree import SumTree


class Transition:
    """Represents an SMDP transition with n-step trajectory metadata."""
    __slots__ = (
        'state', 'action', 'reward', 'next_state', 'tau',
        'n_step_state', 'n_step_return', 'n_step_tau',
        'expert_action', 'is_expert'
    )

    def __init__(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        tau: float,
        n_step_state: Optional[np.ndarray] = None,
        n_step_return: Optional[float] = None,
        n_step_tau: Optional[float] = None,
        expert_action: int = 0,
        is_expert: bool = False,
    ):
        self.state = np.array(state, dtype=np.float32)
        self.action = int(action)
        self.reward = float(reward)
        self.next_state = np.array(next_state, dtype=np.float32)
        self.tau = float(tau)
        self.n_step_state = np.array(n_step_state if n_step_state is not None else next_state, dtype=np.float32)
        self.n_step_return = float(n_step_return if n_step_return is not None else reward)
        self.n_step_tau = float(n_step_tau if n_step_tau is not None else tau)
        self.expert_action = int(expert_action)
        self.is_expert = bool(is_expert)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer using proportional priority (Eq. 6).
    Supports n-step SMDP returns and demonstration data preservation/overwriting.
    """

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.4,
        beta_start: float = 0.6,
        beta_end: float = 1.0,
        beta_steps: int = 5000,
        priority_epsilon: float = 1e-5,
        n_step: int = 3,
        gamma: float = 0.99,
        use_proposed_priority: bool = True,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.priority_epsilon = priority_epsilon
        self.n_step = n_step
        self.gamma = gamma
        self.use_proposed_priority = use_proposed_priority

        self.tree = SumTree(capacity)
        self.data: List[Optional[Transition]] = [None] * capacity
        self.n_step_buffer: deque = deque(maxlen=n_step)

        self.step_count = 0
        self.num_demonstrations = 0

    @property
    def current_beta(self) -> float:
        """Linearly anneals beta from beta_start (0.6) to beta_end (1.0)."""
        fraction = min(1.0, self.step_count / max(1, self.beta_steps))
        return self.beta_start + fraction * (self.beta_end - self.beta_start)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        tau: float,
        is_expert: bool = False,
        expert_action: int = 0,
    ):
        """
        Pushes a 1-step transition to n_step_buffer and stores computed n-step transition to PER.
        """
        self.step_count += 1
        self.n_step_buffer.append((state, action, reward, next_state, tau, is_expert, expert_action))

        if len(self.n_step_buffer) < self.n_step:
            return

        # Compute n-step return and cumulative tau
        transition = self._create_n_step_transition()
        self._store_transition(transition, is_expert=is_expert)

    def flush_n_step(self):
        """Flushes remaining transitions in n-step buffer when episode resets."""
        while len(self.n_step_buffer) > 0:
            transition = self._create_n_step_transition()
            self._store_transition(transition, is_expert=self.n_step_buffer[0][5])
            self.n_step_buffer.popleft()

    def _create_n_step_transition(self) -> Transition:
        state, action, _, _, _, is_expert, expert_action = self.n_step_buffer[0]
        n_step_state = self.n_step_buffer[-1][3]

        n_step_return = 0.0
        n_step_tau = 0.0

        for s, a, r, ns, tau, _, _ in self.n_step_buffer:
            n_step_tau += tau
            discount = (self.gamma ** n_step_tau)
            n_step_return += discount * r

        first_r = self.n_step_buffer[0][2]
        first_ns = self.n_step_buffer[0][3]
        first_tau = self.n_step_buffer[0][4]

        return Transition(
            state=state,
            action=action,
            reward=first_r,
            next_state=first_ns,
            tau=first_tau,
            n_step_state=n_step_state,
            n_step_return=n_step_return,
            n_step_tau=n_step_tau,
            expert_action=expert_action if is_expert else action,
            is_expert=is_expert,
        )

    def _store_transition(self, transition: Transition, is_expert: bool):
        max_p = self.tree.max_priority
        if max_p <= 0.0:
            max_p = 1.0

        idx = self.tree.data_pointer
        # If overwriting an existing expert demonstration, decrement counter
        if self.data[idx] is not None and self.data[idx].is_expert and not is_expert:
            if self.num_demonstrations > 0:
                self.num_demonstrations -= 1

        self.data[idx] = transition
        self.tree.add(max_p)

        if is_expert:
            self.num_demonstrations += 1

    def sample(self, batch_size: int = 64) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        np.ndarray, torch.Tensor
    ]:
        """
        Samples a mini-batch with priority proportional sampling P(i) = p_i^alpha / sum p^alpha.
        Returns batch tensors and normalized importance sampling weights omega_i.
        """
        current_size = self.tree.size
        if current_size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        batch_size = min(batch_size, current_size)
        tree_indices = np.empty(batch_size, dtype=np.int32)
        transitions: List[Transition] = []
        priorities = np.empty(batch_size, dtype=np.float64)

        segment = self.tree.total_priority / batch_size
        beta = self.current_beta

        for i in range(batch_size):
            a_val = segment * i
            b_val = segment * (i + 1)
            v = np.random.uniform(a_val, b_val)
            tree_idx, priority, data_idx = self.tree.get_leaf(v)

            # Ensure valid data index
            data_idx = data_idx % self.capacity
            while self.data[data_idx] is None:
                v = np.random.uniform(0, self.tree.total_priority)
                tree_idx, priority, data_idx = self.tree.get_leaf(v)
                data_idx = data_idx % self.capacity

            tree_indices[i] = tree_idx
            priorities[i] = max(priority, self.priority_epsilon)
            transitions.append(self.data[data_idx])

        # Compute sampling probabilities P(i) and importance sampling weights omega_i (Section VI-1)
        total_p = self.tree.total_priority
        if total_p <= 0.0:
            total_p = 1.0

        probs = priorities / total_p
        weights = (1.0 / (current_size * probs)) ** beta
        # Normalize weights by max weight
        weights = weights / np.max(weights)

        # Batch tensors
        states = torch.tensor(np.array([t.state for t in transitions]), dtype=torch.float32)
        actions = torch.tensor([t.action for t in transitions], dtype=torch.int64)
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
        next_states = torch.tensor(np.array([t.next_state for t in transitions]), dtype=torch.float32)
        taus = torch.tensor([t.tau for t in transitions], dtype=torch.float32)
        n_step_states = torch.tensor(np.array([t.n_step_state for t in transitions]), dtype=torch.float32)
        n_step_returns = torch.tensor([t.n_step_return for t in transitions], dtype=torch.float32)
        n_step_taus = torch.tensor([t.n_step_tau for t in transitions], dtype=torch.float32)
        expert_actions = torch.tensor([t.expert_action for t in transitions], dtype=torch.int64)
        is_experts = torch.tensor([1.0 if t.is_expert else 0.0 for t in transitions], dtype=torch.float32)
        is_weights = torch.tensor(weights, dtype=torch.float32)

        return (
            states, actions, rewards, next_states, taus,
            n_step_states, n_step_returns, n_step_taus,
            expert_actions, is_experts,
            tree_indices, is_weights
        )

    def update_priorities(
        self,
        tree_indices: np.ndarray,
        td_errors: np.ndarray,
        rewards: np.ndarray,
        avg_reward: float,
    ):
        """
        Updates transition priorities:
        Proposed Eq. (6): p_i = (avgR(t) - r_i) + |TDE_i| + varsigma
        Standard Eq. (5): p_i = |TDE_i| + varsigma
        """
        for i, tree_idx in enumerate(tree_indices):
            tde = abs(float(td_errors[i]))
            if self.use_proposed_priority:
                r = float(rewards[i])
                p = (avg_reward - r) + tde + self.priority_epsilon
            else:
                p = tde + self.priority_epsilon

            if p <= self.priority_epsilon:
                p = self.priority_epsilon

            # Priority in tree is p^alpha
            priority_val = p ** self.alpha
            self.tree.update(int(tree_idx), priority_val)

    def __len__(self) -> int:
        return self.tree.size

