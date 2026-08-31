"""
Unit tests for Q-network architecture and initialization.
"""

import torch
import pytest
from models.dqn_mlp import DQNNetwork


def test_dqn_network_architecture():
    state_dim = 251
    hidden_dim = 16
    action_dim = 2

    net = DQNNetwork(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim)

    # Check layer shapes
    assert net.fc1.in_features == 251
    assert net.fc1.out_features == 16
    assert net.fc2.in_features == 16
    assert net.fc2.out_features == 2

    # Check weight initialization in [-0.1, 0.1]
    for param in [net.fc1.weight, net.fc2.weight]:
        assert param.min().item() >= -0.1
        assert param.max().item() <= 0.1

    # Check bias initialization to 0.1
    for param in [net.fc1.bias, net.fc2.bias]:
        assert torch.allclose(param, torch.tensor(0.1))


def test_dqn_network_forward():
    net = DQNNetwork(state_dim=251, hidden_dim=16, action_dim=2)

    # Single state
    state_single = torch.randn(251)
    q_single = net(state_single)
    assert q_single.shape == (1, 2)

    # Batch of states
    state_batch = torch.randn(64, 251)
    q_batch = net(state_batch)
    assert q_batch.shape == (64, 2)


def test_copy_weights():
    net1 = DQNNetwork(state_dim=251, hidden_dim=16, action_dim=2)
    net2 = DQNNetwork(state_dim=251, hidden_dim=16, action_dim=2)

    # Modify net1
    with torch.no_grad():
        net1.fc1.weight.fill_(0.05)

    net2.copy_weights_from(net1)
    assert torch.allclose(net2.fc1.weight, torch.tensor(0.05))

