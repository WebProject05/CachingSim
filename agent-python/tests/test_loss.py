"""
Unit tests for multi-task DQfDLoss function.
"""

import torch
import pytest
from models.dqn_mlp import DQNNetwork
from algorithms.dqfd_loss import DQfDLoss


def test_dqfd_loss_computation():
    state_dim = 251
    hidden_dim = 16
    action_dim = 2
    batch_size = 8

    online_net = DQNNetwork(state_dim, hidden_dim, action_dim)
    target_net = DQNNetwork(state_dim, hidden_dim, action_dim)
    loss_fn = DQfDLoss(gamma=0.99, lambda_1=1.0, lambda_2=1.0, lambda_3=1e-5, expert_margin=0.8)

    states = torch.randn(batch_size, state_dim)
    actions = torch.randint(0, action_dim, (batch_size,))
    rewards = torch.randn(batch_size)
    next_states = torch.randn(batch_size, state_dim)
    taus = torch.rand(batch_size) + 0.1
    n_step_states = torch.randn(batch_size, state_dim)
    n_step_returns = torch.randn(batch_size)
    n_step_taus = torch.rand(batch_size) * 3 + 0.3
    expert_actions = torch.randint(0, action_dim, (batch_size,))
    is_experts = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    is_weights = torch.ones(batch_size)

    loss, td_errors, metrics = loss_fn(
        online_net, target_net,
        states, actions, rewards, next_states, taus,
        n_step_states, n_step_returns, n_step_taus,
        expert_actions, is_experts, is_weights
    )

    assert not torch.isnan(loss)
    assert loss.item() > 0.0
    assert td_errors.shape == (batch_size,)
    assert "loss_dq" in metrics
    assert "loss_n_step" in metrics
    assert "loss_expert" in metrics
    assert "loss_l2" in metrics

