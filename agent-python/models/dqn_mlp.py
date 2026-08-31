"""
PyTorch Multi-Layer Perceptron (MLP) Q-Network architecture for Double Deep Q-Learning.
Conforms strictly to Section VII-B of the research paper (Niknia et al., 2024):
- 2-layer MLP with 16 nodes in the hidden layer.
- ReLU activation function.
- Uniform weight initialization in [-0.1, 0.1] and bias initialization to 0.1.
"""

import torch
import torch.nn as nn
from typing import Optional


class DQNNetwork(nn.Module):
    """Q-Network approximating action-values Q(s, a; theta)."""

    def __init__(self, state_dim: int = 251, hidden_dim: int = 16, action_dim: int = 2):
        super(DQNNetwork, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, action_dim)

        self.reinitialize_weights()

    def reinitialize_weights(self):
        """
        Initializes weights uniformly in [-0.1, 0.1] and biases to 0.1
        as specified in Section VII-B of the paper.
        """
        for layer in [self.fc1, self.fc2]:
            nn.init.uniform_(layer.weight, -0.1, 0.1)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing Q-values for all actions given a state batch.
        Input shape: (batch_size, state_dim) or (state_dim,)
        Output shape: (batch_size, action_dim)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = self.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

    def copy_weights_from(self, source_net: 'DQNNetwork'):
        """Hard update copying weights theta -> theta' for target network synchronization."""
        self.load_state_dict(source_net.state_dict())

