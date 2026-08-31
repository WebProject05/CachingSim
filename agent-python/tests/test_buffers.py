"""
Unit tests for SumTree and PrioritizedReplayBuffer.
"""

import numpy as np
import pytest
from memory.sum_tree import SumTree
from memory.prioritized_buffer import PrioritizedReplayBuffer
from memory.demo_buffer import DemonstrationBuffer


def test_sum_tree_operations():
    capacity = 4
    tree = SumTree(capacity)

    tree.add(10.0)
    tree.add(20.0)
    tree.add(30.0)
    tree.add(40.0)

    assert tree.total_priority == 100.0

    # Retrieve leaf for value 25.0 (should be in segment [10, 30], item 1)
    tree_idx, priority, data_idx = tree.get_leaf(25.0)
    assert data_idx == 1
    assert priority == 20.0

    # Update priority of item 0 from 10 to 50
    tree.update(tree_index=tree.capacity - 1 + 0, priority=50.0)
    assert tree.total_priority == 140.0


def test_prioritized_replay_buffer():
    buffer = PrioritizedReplayBuffer(
        capacity=100,
        alpha=0.4,
        beta_start=0.6,
        beta_end=1.0,
        n_step=3,
        gamma=0.99,
        use_proposed_priority=True,
    )

    state = np.ones(251, dtype=np.float32)
    next_state = np.ones(251, dtype=np.float32) * 2

    # Add 10 transitions
    for i in range(10):
        buffer.add(
            state=state * i,
            action=i % 2,
            reward=float(i * 10),
            next_state=next_state * i,
            tau=1.5,
        )

    # 10 1-step additions with n_step=3 creates 8 n-step transitions
    assert len(buffer) == 8

    # Sample batch
    (
        states, actions, rewards, next_states, taus,
        n_step_states, n_step_returns, n_step_taus,
        expert_actions, is_experts,
        tree_indices, is_weights
    ) = buffer.sample(batch_size=4)

    assert states.shape == (4, 251)
    assert actions.shape == (4,)
    assert is_weights.shape == (4,)
    assert is_weights.max() <= 1.0 + 1e-5


def test_demonstration_buffer():
    demo_buf = DemonstrationBuffer(capacity=10)
    target_buf = PrioritizedReplayBuffer(capacity=20)

    state = np.zeros(251, dtype=np.float32)
    for i in range(5):
        demo_buf.add(state=state, action=1, reward=10.0, next_state=state, tau=1.0)

    assert len(demo_buf) == 5

    # Populate target buffer
    demo_buf.populate_target_buffer(target_buf)
    assert target_buf.num_demonstrations == 5
    assert len(target_buf) == 5

