"""
Configuration dataclasses for SMDP Edge Caching RL and Transfer Learning Agents.
Matches Table I and Table II from research paper (Niknia et al., 2024).
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class AgentConfig:
    # --- System & Environment Parameters (Table II) ---
    total_files: int = 50                 # F: Number of file types
    cache_capacity: float = 10000.0       # M: Cache capacity in MiB
    sliding_window_n: int = 100           # N: Popularity history sliding window length
    lambda_source: float = 0.2            # lambda_S: Request rate in source domain
    lambda_target: float = 0.3            # lambda_T: Request rate in target domain
    zipf_eta: float = 1.0                 # eta: Zipf distribution skewness

    # --- Neural Network Parameters (Section VII-B) ---
    hidden_dim: int = 16                  # 16 nodes in hidden layer
    weight_init_min: float = -0.1         # Uniform [-0.1, 0.1]
    weight_init_max: float = 0.1
    bias_init: float = 0.1                # Bias initialized to 0.1
    action_dim: int = 2                   # Actions: {0 (skip), 1 (cache)}

    # --- DDQL Reinforcement Learning Hyperparameters (Table II & Section V) ---
    gamma: float = 0.99                   # gamma: Discount factor
    learning_rate: float = 0.0005         # psi: Learning rate
    batch_size: int = 64                  # Mini-batch size
    target_sync_steps: int = 100          # zeta: Target network update frequency
    replay_capacity: int = 5000           # |Lambda|: Source replay buffer capacity
    target_buffer_cap: int = 10000        # |Lambda'|: Target domain replay buffer capacity
    epsilon_start: float = 1.0            # Initial exploration rate
    epsilon_min: float = 0.02             # Minimum exploration rate
    epsilon_decay: float = 0.998          # Epsilon decay rate per step

    # --- Transfer Learning & Loss Hyperparameters (Section VI & Table II) ---
    alpha: float = 0.4                    # alpha: Prioritization exponent for PER
    beta_start: float = 0.6               # beta: Initial importance sampling weight
    beta_end: float = 1.0                 # beta: Final importance sampling weight
    beta_steps: int = 5000                # Steps to linearly anneal beta from 0.6 to 1.0
    priority_epsilon: float = 1e-5        # varsigma: Small constant to avoid zero priority
    n_step_horizon: int = 3               # n: Horizon for n-step return J_n
    lambda_1: float = 1.0                 # lambda_1: Weight for n-step TD loss
    lambda_2: float = 1.0                 # lambda_2: Weight for supervised margin loss J_E
    lambda_3: float = 1e-5                # lambda_3: Weight for L2 regularization J_L2
    expert_margin: float = 0.8            # L(a_E, a): Large margin penalty for non-expert action

    # --- Training & Experiment Parameters ---
    source_train_steps: int = 5000        # Steps to train on source domain
    target_train_steps: int = 8000        # Steps to train on target domain
    eval_requests: int = 1000             # Requests for performance evaluation (Section VII-D)
    seed: int = 42                        # Random seed

    # --- gRPC Server Connection ---
    server_host: str = "localhost"
    server_port: int = 50051

    @property
    def state_dim(self) -> int:
        """
        Calculates input state dimension for Q-network:
        Mem(1) + D(F) + Y(F) + Z(F) + B(F) + RequestedFile(one-hot F) = 1 + 5*F
        For F=50, state_dim = 1 + 5*50 = 251.
        """
        return 1 + 5 * self.total_files


def get_default_config() -> AgentConfig:
    """Returns default configuration instance."""
    return AgentConfig()

