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
    """
    Q-Network approximating action-values Q(s, a; theta).
    Supports standard MLP (Niknia et al., 2024) and enhanced Dueling DQN architecture (Wang et al., 2016).
    """

    def __init__(
        self,
        state_dim: int = 251,
        hidden_dim: int = 16,
        action_dim: int = 2,
        dueling: bool = False,
    ):
        super(DQNNetwork, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.dueling = dueling

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, action_dim)

        # Dueling branch components (Value stream and Advantage stream)
        val_hidden = max(8, hidden_dim // 2)
        adv_hidden = max(8, hidden_dim // 2)
        self.fc_val = nn.Linear(hidden_dim, val_hidden)
        self.val_head = nn.Linear(val_hidden, 1)

        self.fc_adv = nn.Linear(hidden_dim, adv_hidden)
        self.adv_head = nn.Linear(adv_hidden, action_dim)

        self.reinitialize_weights()

    def reinitialize_weights(self):
        """
        Initializes weights uniformly in [-0.1, 0.1] and biases to 0.1
        as specified in Section VII-B of the paper.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing Q-values for all actions given a state batch.
        Input shape: (batch_size, state_dim) or (state_dim,)
        Output shape: (batch_size, action_dim)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        h = self.relu(self.fc1(state))

        if not self.dueling:
            return self.fc2(h)

        val = self.val_head(self.relu(self.fc_val(h)))  # (batch_size, 1)
        adv = self.adv_head(self.relu(self.fc_adv(h)))  # (batch_size, action_dim)
        # Q(s, a) = V(s) + (A(s, a) - 1/|A| sum A(s, a'))
        q_values = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_values

    def copy_weights_from(self, source_net: 'DQNNetwork'):
        """Hard update copying weights theta -> theta' for target network synchronization."""
        self.load_state_dict(source_net.state_dict())

