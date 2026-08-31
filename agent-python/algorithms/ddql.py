"""
Double Deep Q-Learning (DDQL) and Transfer Learning Agent for SMDP Edge Caching.
Conforms strictly to Sections V and VI of the research paper (Niknia et al., 2024).
"""

from typing import Tuple, Dict, Any, Optional
import os
import random
import numpy as np
import torch
import torch.optim as optim

from config import AgentConfig, get_default_config
from models.dqn_mlp import DQNNetwork
from memory.prioritized_buffer import PrioritizedReplayBuffer
from memory.demo_buffer import DemonstrationBuffer
from algorithms.dqfd_loss import DQfDLoss
from client.grpc_env_client import GrpcEnvClient


class DDQLAgent:
    """
    DDQL Agent supporting SMDP continuous discounting, prioritized experience replay,
    and multi-task transfer learning from source to target domain.
    """

    def __init__(self, cfg: Optional[AgentConfig] = None):
        self.cfg = cfg or get_default_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Online and Target Q-Networks (Section VII-B)
        self.online_net = DQNNetwork(
            state_dim=self.cfg.state_dim,
            hidden_dim=self.cfg.hidden_dim,
            action_dim=self.cfg.action_dim,
        ).to(self.device)

        self.target_net = DQNNetwork(
            state_dim=self.cfg.state_dim,
            hidden_dim=self.cfg.hidden_dim,
            action_dim=self.cfg.action_dim,
        ).to(self.device)
        self.target_net.copy_weights_from(self.online_net)

        # Optimizer (learning rate psi = 0.0005)
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.cfg.learning_rate)

        # Multi-task Loss Function (Section VI-2)
        self.loss_fn = DQfDLoss(
            gamma=self.cfg.gamma,
            lambda_1=self.cfg.lambda_1,
            lambda_2=self.cfg.lambda_2,
            lambda_3=self.cfg.lambda_3,
            expert_margin=self.cfg.expert_margin,
        ).to(self.device)

        # Experience Replay Buffers
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.cfg.replay_capacity,
            alpha=self.cfg.alpha,
            beta_start=self.cfg.beta_start,
            beta_end=self.cfg.beta_end,
            beta_steps=self.cfg.beta_steps,
            priority_epsilon=self.cfg.priority_epsilon,
            n_step=self.cfg.n_step_horizon,
            gamma=self.cfg.gamma,
            use_proposed_priority=True,
        )
        self.demo_buffer = DemonstrationBuffer(capacity=self.cfg.replay_capacity)

        # Exploration & Tracking State
        self.epsilon = self.cfg.epsilon_start
        self.total_steps = 0
        self.train_steps = 0
        self.running_reward_sum = 0.0
        self.running_reward_count = 0

    @property
    def avg_reward(self) -> float:
        """Computes running average reward avgR(t) = sum r_i / t (Section VI-1)."""
        if self.running_reward_count == 0:
            return 0.0
        return self.running_reward_sum / self.running_reward_count

    def update_running_reward(self, reward: float):
        """Records a new reward into the running average tracker."""
        self.running_reward_count += 1
        self.running_reward_sum += reward

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Selects an action using epsilon-greedy policy:
        pi*(s) = argmax_a Q(s, a; theta)
        """
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.cfg.action_dim - 1)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
            action = int(q_values.argmax(dim=1).item())
        return action

    def step_learn(self) -> Optional[Dict[str, float]]:
        """Samples a mini-batch from PER and executes one gradient descent update step."""
        if len(self.replay_buffer) < self.cfg.batch_size:
            return None

        # Sample prioritized batch
        (
            states, actions, rewards, next_states, taus,
            n_step_states, n_step_returns, n_step_taus,
            expert_actions, is_experts,
            tree_indices, is_weights
        ) = self.replay_buffer.sample(self.cfg.batch_size)

        # Transfer to target device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        taus = taus.to(self.device)
        n_step_states = n_step_states.to(self.device)
        n_step_returns = n_step_returns.to(self.device)
        n_step_taus = n_step_taus.to(self.device)
        expert_actions = expert_actions.to(self.device)
        is_experts = is_experts.to(self.device)
        is_weights = is_weights.to(self.device)

        # Compute multi-task loss
        loss, td_errors, loss_metrics = self.loss_fn(
            self.online_net,
            self.target_net,
            states, actions, rewards, next_states, taus,
            n_step_states, n_step_returns, n_step_taus,
            expert_actions, is_experts, is_weights,
        )

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update PER priorities using Eq. (6) or Eq. (5)
        td_errors_np = td_errors.cpu().numpy()
        rewards_np = rewards.cpu().numpy()
        self.replay_buffer.update_priorities(
            tree_indices,
            td_errors_np,
            rewards_np,
            self.avg_reward,
        )

        self.train_steps += 1
        # Synchronize target network every zeta steps: theta' <- theta
        if self.train_steps % self.cfg.target_sync_steps == 0:
            self.target_net.copy_weights_from(self.online_net)

        return loss_metrics

    def decay_epsilon(self):
        """Decays epsilon exploration rate."""
        if self.epsilon > self.cfg.epsilon_min:
            self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

    def train_source_domain(
        self,
        env: GrpcEnvClient,
        total_steps: int = 5000,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Trains the DDQL agent in the source domain (lambda_S = 0.2)
        and populates the demonstration buffer.
        """
        state, _ = env.reset(seed=self.cfg.seed, lambda_rate=self.cfg.lambda_source, eta=self.cfg.zipf_eta)
        self.running_reward_sum = 0.0
        self.running_reward_count = 0
        self.epsilon = self.cfg.epsilon_start

        hits = 0
        total_utility = 0.0
        avg_reward_history = []

        for step in range(1, total_steps + 1):
            action = self.select_action(state)
            next_state, reward, tau, is_hit, utility = env.step(action)

            if is_hit:
                hits += 1
                total_utility += utility

            self.update_running_reward(reward)

            # Store transition in PER buffer and demonstration buffer
            self.replay_buffer.add(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                tau=tau,
                is_expert=False,
            )
            self.demo_buffer.add(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                tau=tau,
            )

            state = next_state
            self.decay_epsilon()
            self.step_learn()

            if step % 100 == 0:
                avg_reward_history.append((step, self.avg_reward))
                if verbose and step % 1000 == 0:
                    print(f"[Source Domain] Step [{step:5d}/{total_steps:5d}] | Hit Rate: {hits/step*100:5.2f}% | "
                          f"Avg Reward: {self.avg_reward:6.2f} | Epsilon: {self.epsilon:.3f}")

        self.replay_buffer.flush_n_step()
        return {
            'hit_rate': hits / total_steps * 100.0,
            'total_utility': total_utility,
            'avg_reward': self.avg_reward,
            'reward_history': avg_reward_history,
        }

    def prepare_transfer_learning(self, mode: str = "proposed"):
        """
        Configures agent for target domain according to specified TL methodology (Section VI & VII-C-2):
        - 'proposed': Reinitializes weights, creates target buffer |Lambda'|=10000 with demonstrations,
                      uses proposed priority Eq. (6), allows overwriting demonstrations.
        - 'dqfd': Standard DQfD (does not reinitialize weights, uses standard TDE priority Eq. 5).
        - 'lfs': Learning From Scratch (reinitializes weights, empty buffer, no demonstrations).
        - 'dpr': Direct Policy Reuse (keeps source policy without training in target domain).
        """
        mode = mode.lower()
        if mode in ["proposed", "lfs"]:
            # Reinitialize network weights as specified in Section VI
            self.online_net.reinitialize_weights()
            self.target_net.copy_weights_from(self.online_net)
            self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.cfg.learning_rate)

        use_proposed_priority = (mode == "proposed")

        # Initialize Target Domain Replay Buffer |Lambda'| = 10000
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.cfg.target_buffer_cap,
            alpha=self.cfg.alpha,
            beta_start=self.cfg.beta_start,
            beta_end=self.cfg.beta_end,
            beta_steps=self.cfg.beta_steps,
            priority_epsilon=self.cfg.priority_epsilon,
            n_step=self.cfg.n_step_horizon,
            gamma=self.cfg.gamma,
            use_proposed_priority=use_proposed_priority,
        )

        # Pre-fill target replay buffer with demonstrations if applicable
        if mode in ["proposed", "dqfd"]:
            self.demo_buffer.populate_target_buffer(self.replay_buffer)

        self.epsilon = self.cfg.epsilon_start if mode != "dpr" else 0.0
        self.running_reward_sum = 0.0
        self.running_reward_count = 0

    def train_target_domain(
        self,
        env: GrpcEnvClient,
        total_steps: int = 8000,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Trains the agent in target domain (lambda_T = 0.3)."""
        state, _ = env.reset(seed=self.cfg.seed + 100, lambda_rate=self.cfg.lambda_target, eta=self.cfg.zipf_eta)
        self.running_reward_sum = 0.0
        self.running_reward_count = 0

        hits = 0
        total_utility = 0.0
        avg_reward_history = []

        for step in range(1, total_steps + 1):
            action = self.select_action(state)
            next_state, reward, tau, is_hit, utility = env.step(action)

            if is_hit:
                hits += 1
                total_utility += utility

            self.update_running_reward(reward)

            # Store in target replay buffer (gradually overwrites demonstrations once full)
            self.replay_buffer.add(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                tau=tau,
                is_expert=False,
            )

            state = next_state
            self.decay_epsilon()
            self.step_learn()

            if step % 100 == 0:
                avg_reward_history.append((step, self.avg_reward))
                if verbose and step % 1000 == 0:
                    print(f"[Target Domain] Step [{step:5d}/{total_steps:5d}] | Hit Rate: {hits/step*100:5.2f}% | "
                          f"Avg Reward: {self.avg_reward:6.2f} | Epsilon: {self.epsilon:.3f}")

        return {
            'hit_rate': hits / total_steps * 100.0,
            'total_utility': total_utility,
            'avg_reward': self.avg_reward,
            'reward_history': avg_reward_history,
        }

    def evaluate(
        self,
        env: GrpcEnvClient,
        eval_requests: int = 1000,
        lambda_rate: float = 0.2,
        eta: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Evaluates policy performance on 1000 user requests without exploration (Section VII-D).
        Returns hit count, hit rate, total utility, and average reward.
        """
        state, _ = env.reset(seed=self.cfg.seed + 999, lambda_rate=lambda_rate, eta=eta)
        hits = 0
        total_utility = 0.0
        total_reward = 0.0

        for step in range(1, eval_requests + 1):
            # Greedy action selection (evaluate = True)
            action = self.select_action(state, evaluate=True)
            next_state, reward, tau, is_hit, utility = env.step(action)

            if is_hit:
                hits += 1
                total_utility += utility

            total_reward += reward
            state = next_state

        return {
            'requests': eval_requests,
            'hits': hits,
            'hit_rate': (hits / eval_requests) * 100.0,
            'total_utility': total_utility,
            'total_reward': total_reward,
            'avg_reward': total_reward / eval_requests,
        }

    def save_checkpoint(self, filepath: str):
        """Saves model weights and optimizer state."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save({
            'online_state': self.online_net.state_dict(),
            'target_state': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
        }, filepath)

    def load_checkpoint(self, filepath: str):
        """Loads model weights."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        self.online_net.load_state_dict(checkpoint['online_state'])
        self.target_net.load_state_dict(checkpoint['target_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint.get('epsilon', self.cfg.epsilon_min)
        self.total_steps = checkpoint.get('total_steps', 0)

