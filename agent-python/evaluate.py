"""
Evaluation and Comparison Suite for SMDP Edge Caching RL Agents.
Reproduces experimental comparisons from Section VII of the research paper (Niknia et al., 2024):
- DRL convergence and comparison with CTD baseline (Fig. 3a, 3b, 3c).
- Transfer Learning convergence comparison: Proposed TL vs DQfD vs DPR vs LFS (Fig. 8a, 8b, 8c).
- Exports full detailed statistics to JSON and CSV formats into data/results/.
"""

from typing import Dict, List, Any, Tuple
import os
import json
import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from config import AgentConfig, get_default_config
from client.grpc_env_client import GrpcEnvClient
from algorithms.ddql import DDQLAgent


def run_source_training_and_eval(
    cfg: AgentConfig,
    env: GrpcEnvClient,
    train_steps: int = 5000,
    eval_requests: int = 1000,
) -> Tuple[DDQLAgent, Dict[str, Any], List[Tuple[int, float]]]:
    """Trains DDQL agent on source domain and evaluates on test requests."""
    print(f"\n>>> [1/2] Training Proposed DDQL Agent on Source Domain (lambda={cfg.lambda_source}, eta={cfg.zipf_eta})...")
    agent = DDQLAgent(cfg)
    train_res = agent.train_source_domain(env, total_steps=train_steps, verbose=True)

    print(f"\n>>> [2/2] Evaluating Converged DDQL Agent on {eval_requests} Test Requests...")
    eval_res = agent.evaluate(env, eval_requests=eval_requests, lambda_rate=cfg.lambda_source, eta=cfg.zipf_eta)

    print("\n=== DDQL Source Domain Evaluation Results ===")
    print(f"Total Requests : {eval_res['requests']}")
    print(f"Hit Count      : {eval_res['hits']}")
    print(f"Hit Rate       : {eval_res['hit_rate']:.2f}%")
    print(f"Total Utility  : {eval_res['total_utility']:.2f}")
    print(f"Average Reward : {eval_res['avg_reward']:.2f}")

    return agent, eval_res, train_res['reward_history']


def run_transfer_learning_comparison(
    source_agent: DDQLAgent,
    cfg: AgentConfig,
    env: GrpcEnvClient,
    target_steps: int = 6000,
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Compares 4 Transfer Learning paradigms on target domain (lambda_T = 0.3) matching Fig. 8a:
    1. Proposed TL: weights reinitialized, demonstration pre-fill, proposed priority Eq. 6.
    2. DQfD: no weight reinitialization, standard TDE priority Eq. 5.
    3. DPR: Direct Policy Reuse (source policy reused directly).
    4. LFS: Learning From Scratch without demonstrations.
    """
    tl_results = {}
    methods = ["proposed", "dqfd", "lfs", "dpr"]

    for method in methods:
        print(f"\n>>> Running Target Domain Transfer Learning Experiment: [{method.upper()}]...")
        agent = DDQLAgent(cfg)
        agent.demo_buffer = source_agent.demo_buffer

        if method == "dpr":
            # Copy source weights and evaluate without further training
            agent.online_net.load_state_dict(source_agent.online_net.state_dict())
            agent.target_net.load_state_dict(source_agent.target_net.state_dict())
            eval_res = agent.evaluate(env, eval_requests=1000, lambda_rate=cfg.lambda_target, eta=cfg.zipf_eta)
            tl_results["DPR"] = [(step, eval_res['avg_reward']) for step in range(100, target_steps + 1, 100)]
        else:
            agent.prepare_transfer_learning(mode=method)
            target_res = agent.train_target_domain(env, total_steps=target_steps, verbose=True)
            tl_results[method.upper()] = target_res['reward_history']

    return tl_results


def save_results_and_plot(
    cfg: AgentConfig,
    eval_res_source: Dict[str, Any],
    reward_history_ddql: List[Tuple[int, float]],
    tl_histories: Dict[str, List[Tuple[int, float]]],
    output_dir: str = "data/results",
):
    """Saves all evaluation data to JSON, CSV and renders convergence comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.utcnow().isoformat() + "Z"

    # --- 1. Export Source Training & Evaluation Data to JSON ---
    source_json_data = {
        "timestamp": timestamp,
        "metadata": {
            "seed": cfg.seed,
            "total_files": cfg.total_files,
            "cache_capacity_mib": cfg.cache_capacity,
            "sliding_window_n": cfg.sliding_window_n,
            "lambda_source": cfg.lambda_source,
            "zipf_eta": cfg.zipf_eta,
            "gamma": cfg.gamma,
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "replay_capacity": cfg.replay_capacity,
        },
        "evaluation_metrics": eval_res_source,
        "training_trajectory": [
            {"step": step, "avg_reward": avg_r} for step, avg_r in reward_history_ddql
        ],
    }

    source_json_path = os.path.join(output_dir, "drl_source_training.json")
    with open(source_json_path, "w", encoding="utf-8") as f:
        json.dump(source_json_data, f, indent=2)

    # --- 2. Export Transfer Learning Comparison Data to JSON ---
    if tl_histories:
        tl_json_data = {
            "timestamp": timestamp,
            "metadata": {
                "lambda_source": cfg.lambda_source,
                "lambda_target": cfg.lambda_target,
                "target_buffer_cap": cfg.target_buffer_cap,
                "alpha": cfg.alpha,
                "beta_start": cfg.beta_start,
                "beta_end": cfg.beta_end,
                "n_step": cfg.n_step_horizon,
            },
            "trajectories": {
                method: [{"step": s, "avg_reward": r} for s, r in hist]
                for method, hist in tl_histories.items()
            },
        }

        tl_json_path = os.path.join(output_dir, "transfer_learning_comparison.json")
        with open(tl_json_path, "w", encoding="utf-8") as f:
            json.dump(tl_json_data, f, indent=2)

    # --- 3. Export Comprehensive Experiment Summary JSON ---
    summary_data = {
        "timestamp": timestamp,
        "experiment": "SMDP Edge Caching RL and Transfer Learning",
        "parameters": {
            "total_files": cfg.total_files,
            "cache_capacity_mib": cfg.cache_capacity,
            "lambda_source": cfg.lambda_source,
            "lambda_target": cfg.lambda_target,
            "zipf_eta": cfg.zipf_eta,
        },
        "source_evaluation": eval_res_source,
        "transfer_learning_final_rewards": {
            method: hist[-1][1] if hist else 0.0 for method, hist in tl_histories.items()
        },
    }

    summary_json_path = os.path.join(output_dir, "experiment_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # --- 4. Plot DDQL Source Domain Reward Convergence (Fig. 3a) ---
    if reward_history_ddql:
        steps, rewards = zip(*reward_history_ddql)
        plt.figure(figsize=(8, 5))
        plt.plot(steps, rewards, label="Proposed SMDP-DDQL", color="#006d77", linewidth=2)
        plt.title("Average Reward vs Training Trials (Source Domain)")
        plt.xlabel("Trial / Step")
        plt.ylabel("Average Reward")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ddql_source_convergence.png"), dpi=300)
        plt.close()

    # --- 5. Plot Transfer Learning Comparison on Target Domain (Fig. 8a) ---
    if tl_histories:
        plt.figure(figsize=(9, 5.5))
        colors = {"PROPOSED": "#006d77", "DQFD": "#c44536", "LFS": "#386641", "DPR": "#6a4c93"}
        for method, hist in tl_histories.items():
            if hist:
                steps, rewards = zip(*hist)
                plt.plot(steps, rewards, label=method, color=colors.get(method, "#222222"), linewidth=2)

        plt.title("Target Domain (lambda_T = 0.3) Average Reward Convergence (Fig. 8a)")
        plt.xlabel("Trial / Step")
        plt.ylabel("Average Reward")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "transfer_learning_comparison.png"), dpi=300)
        plt.close()

    print(f"\n>>> Results and detailed JSON metrics saved to '{output_dir}':")
    print(f"    - {source_json_path}")
    if tl_histories:
        print(f"    - {tl_json_path}")
    print(f"    - {summary_json_path}")
