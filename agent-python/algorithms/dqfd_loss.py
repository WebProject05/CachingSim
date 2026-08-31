"""
Multi-Task Loss Function for Deep Q-Learning from Demonstrations (DQfD) and Transfer Learning.
Conforms strictly to Section VI-2 of the research paper (Niknia et al., 2024):
J = J_DQ + lambda_1 * J_n + lambda_2 * J_E + lambda_3 * J_L2
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dqn_mlp import DQNNetwork


class DQfDLoss(nn.Module):
    """
    Computes the combined transfer learning loss:
    1-step Double Q-Learning TD loss + n-step TD loss + Large Margin Supervised Loss + L2 Regularization.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        lambda_1: float = 1.0,
        lambda_2: float = 1.0,
        lambda_3: float = 1e-5,
        expert_margin: float = 0.8,
    ):
        super(DQfDLoss, self).__init__()
        self.gamma = gamma
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.expert_margin = expert_margin

    def forward(
        self,
        online_net: DQNNetwork,
        target_net: DQNNetwork,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        taus: torch.Tensor,
        n_step_states: torch.Tensor,
        n_step_returns: torch.Tensor,
        n_step_taus: torch.Tensor,
        expert_actions: torch.Tensor,
        is_experts: torch.Tensor,
        is_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Calculates combined loss J and TD errors for priority updating.
        """
        # --- 1. Current Q-values Q(s, a; theta) ---
        q_values = online_net(states)  # (batch_size, action_dim)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # (batch_size,)

        # --- 2. 1-Step Double Q-Learning Target (Section V-B) ---
        with torch.no_grad():
            # Online network selects best action b in next state: argmax_b Q(s', b; theta)
            next_q_online = online_net(next_states)
            best_actions = next_q_online.argmax(dim=1, keepdim=True)
            # Target network evaluates Q-value of selected action: Q(s', b; theta')
            next_q_target = target_net(next_states)
            max_next_q = next_q_target.gather(1, best_actions).squeeze(1)
            # SMDP continuous discount: gamma^tau
            discount_1 = torch.pow(self.gamma, taus)
            target_q_1 = rewards + discount_1 * max_next_q

        # 1-step TD error & loss J_DQ
        td_errors_1 = target_q_1 - current_q
        j_dq = (is_weights * (td_errors_1 ** 2)).mean()

        # --- 3. n-Step Double Q-Learning Loss J_n (Eq. 7) ---
        with torch.no_grad():
            # Action selection with online net at n-step next state
            n_next_q_online = online_net(n_step_states)
            best_n_actions = n_next_q_online.argmax(dim=1, keepdim=True)
            # Action evaluation with target net
            n_next_q_target = target_net(n_step_states)
            max_n_next_q = n_next_q_target.gather(1, best_n_actions).squeeze(1)
            # Cumulative continuous discount: gamma^(tau^(n))
            discount_n = torch.pow(self.gamma, n_step_taus)
            target_q_n = n_step_returns + discount_n * max_n_next_q

        td_errors_n = target_q_n - current_q
        j_n = (is_weights * (td_errors_n ** 2)).mean()

        # --- 4. Supervised Large Margin Classification Loss J_E (Section VI-2) ---
        # J_E = max_a [ Q(s, a) + L(a_E, a) - Q(s, a_E) ]
        # L(a_E, a) is 0 when a == a_E, and expert_margin otherwise.
        batch_size = states.size(0)
        action_dim = online_net.action_dim

        # Construct margin matrix: 0 on expert action, margin on non-expert actions
        margin_matrix = torch.full((batch_size, action_dim), self.expert_margin, device=states.device)
        margin_matrix.scatter_(1, expert_actions.unsqueeze(1), 0.0)

        # Augmented Q-values: Q(s, a) + L(a_E, a)
        augmented_q = q_values + margin_matrix
        max_augmented_q, _ = augmented_q.max(dim=1)
        expert_q = q_values.gather(1, expert_actions.unsqueeze(1)).squeeze(1)

        # Margin loss for each transition
        margin_loss = max_augmented_q - expert_q  # Non-negative
        # Apply only to expert transitions in the batch
        expert_count = is_experts.sum()
        if expert_count > 0:
            j_e = (is_experts * margin_loss).sum() / expert_count
        else:
            j_e = torch.tensor(0.0, device=states.device)

        # --- 5. L2 Regularization Loss J_L2 ---
        j_l2 = torch.tensor(0.0, device=states.device)
        for param in online_net.parameters():
            j_l2 = j_l2 + torch.sum(param ** 2)
        j_l2 = 0.5 * j_l2

        # --- 6. Total Loss J ---
        total_loss = j_dq + self.lambda_1 * j_n + self.lambda_2 * j_e + self.lambda_3 * j_l2

        loss_metrics = {
            'loss_total': total_loss.item(),
            'loss_dq': j_dq.item(),
            'loss_n_step': j_n.item(),
            'loss_expert': j_e.item(),
            'loss_l2': j_l2.item(),
        }

        # Return total loss, 1-step TD errors for priority update, and loss breakdown
        return total_loss, td_errors_1.detach(), loss_metrics

