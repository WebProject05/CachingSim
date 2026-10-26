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
from ui_utils import render_banner, render_section_header, render_card, render_table


def run_source_training_and_eval(
    cfg: AgentConfig,
    env: GrpcEnvClient,
    train_steps: int = 5000,
    eval_requests: int = 1000,
    eval_lambda: float = 5.0,
) -> Tuple[DDQLAgent, Dict[str, Any], List[Tuple[int, float]]]:
    """Trains DDQL agent on source domain and evaluates on test requests across request rates."""
    render_section_header(f"Phase 1: Source Domain Training (lambda_S={cfg.lambda_source}, eta={cfg.zipf_eta}, Steps={train_steps:,})")
    agent = DDQLAgent(cfg)
    train_res = agent.train_source_domain(env, total_steps=train_steps, verbose=True)

    render_section_header(f"Phase 2: Multi-Rate Evaluation Across Request Arrival Rates (Table III & Fig. 6b)")
    eval_res_source = agent.evaluate(env, eval_requests=eval_requests, lambda_rate=cfg.lambda_source, eta=cfg.zipf_eta)
    eval_res_benchmark = agent.evaluate(env, eval_requests=eval_requests, lambda_rate=eval_lambda, eta=cfg.zipf_eta)

    # Multi-rate evaluation for Table III and Figure 6b verification
    multi_lambda_results = {}
    table_headers = ["Lambda", "Mean Tau", "Requests", "Hits", "Hit Rate", "Byte Hit Rate", "Total Utility", "Avg Reward"]
    table_rows = []

    for lam in [0.20, 0.50, 1.00, 1.66, 2.50, 5.00, 6.50]:
        res_lam = agent.evaluate(env, eval_requests=eval_requests, lambda_rate=lam, eta=cfg.zipf_eta, verbose=True)
        multi_lambda_results[str(lam)] = res_lam
        mean_tau = 1.0 / lam
        table_rows.append([
            f"{lam:5.2f} req/s",
            f"{mean_tau:5.2f}s",
            eval_requests,
            f"{res_lam['hits']:,}",
            f"{res_lam['hit_rate']:5.2f}%",
            f"{res_lam.get('byte_hit_rate', 0.0):5.2f}%",
            f"{res_lam['total_utility']:6.2f}",
            f"{res_lam['avg_reward']:6.2f}",
        ])

    print("\n")
    render_table(
        table_headers,
        table_rows,
        alignments=['L', 'R', 'R', 'R', 'R', 'R', 'R', 'R'],
    )

    render_card(
        "Key Evaluation Benchmarks",
        {
            f"Source Domain (lambda={cfg.lambda_source:.2f}, tau={1.0/cfg.lambda_source:.1f}s)": f"Hit Rate: {eval_res_source['hit_rate']:5.2f}% | Utility: {eval_res_source['total_utility']:6.2f}",
            f"Table III Benchmark (lambda={eval_lambda:.2f}, tau={1.0/eval_lambda:.2f}s)": f"Hit Rate: {eval_res_benchmark['hit_rate']:5.2f}% | Utility: {eval_res_benchmark['total_utility']:6.2f}",
            f"Table III Byte Hit Rate": f"{eval_res_benchmark.get('byte_hit_rate', 0.0):5.2f}%",
            f"Table III Average Reward": f"{eval_res_benchmark.get('avg_reward', 0.0):6.2f}",
        },
    )

    combined_eval_res = {
        **eval_res_source,
        'benchmark_lambda': eval_lambda,
        'benchmark_hits': eval_res_benchmark['hits'],
        'benchmark_hit_rate': eval_res_benchmark['hit_rate'],
        'benchmark_byte_hit_rate': eval_res_benchmark.get('byte_hit_rate', 0.0),
        'benchmark_total_utility': eval_res_benchmark['total_utility'],
        'benchmark_avg_reward': eval_res_benchmark.get('avg_reward', 0.0),
        'multi_lambda_eval': multi_lambda_results,
    }

    return agent, combined_eval_res, train_res['reward_history']


def run_transfer_learning_comparison(
    source_agent: DDQLAgent,
    cfg: AgentConfig,
    env: GrpcEnvClient,
    target_steps: int = 6000,
) -> Dict[str, List[Tuple[int, float]]]:
    """
    Compares Transfer Learning paradigms on target domain (lambda_T = 0.3) matching Fig. 8a, 8b, 8c:
    1. PROPOSED: Reinitialized weights, PER (Eq. 6), supervised demonstrations, overwrite enabled.
    2. DQFD: Standard DQfD (retains weights, PER Eq. 5, no overwrite).
    3. LFS: Learning From Scratch (no demonstrations).
    4. DPR: Direct Policy Reuse (source policy deployed zero-shot).
    5. DQFD_OVERWRITE: Ablation - DQfD with overwrite enabled (Fig. 8b).
    6. DQFD_NEW_PRIO: Ablation - DQfD with Eq. 6 priority (Fig. 8c).
    """
    tl_results = {}
    methods = [
        ("proposed", "Proposed TL (Reinit + Eq.6 + Demos + Overwrite)"),
        ("dqfd", "Standard DQfD (Keep Weights + Eq.5 + No Overwrite)"),
        ("lfs", "Learning From Scratch (Reinit, No Demos)"),
        ("dpr", "Direct Policy Reuse (Zero-shot Source Policy)"),
        ("dqfd_overwrite", "DQfD with Overwriting Demos (Fig. 8b)"),
        ("dqfd_new_prio", "DQfD with Proposed Priority Eq.6 (Fig. 8c)"),
    ]

    method_metrics = []

    for method_key, method_desc in methods:
        render_section_header(f"Target Domain Experiment: [{method_key.upper()}] - {method_desc}")
        agent = DDQLAgent(cfg)
        agent.demo_buffer = source_agent.demo_buffer

        if method_key == "dpr":
            agent.online_net.load_state_dict(source_agent.online_net.state_dict())
            agent.target_net.load_state_dict(source_agent.target_net.state_dict())
            target_res = agent.run_dpr_target_domain(env, total_steps=target_steps, verbose=True)
            tl_results["DPR"] = target_res['reward_history']
        else:
            agent.prepare_transfer_learning(mode=method_key)
            target_res = agent.train_target_domain(
                env, total_steps=target_steps, method_name=f"Target: {method_key.upper()}", verbose=True
            )
            tl_results[method_key.upper()] = target_res['reward_history']

        hist = target_res['reward_history']
        init_r = hist[0][1] if hist else 0.0
        final_r = hist[-1][1] if hist else 0.0
        method_metrics.append([
            method_key.upper(),
            f"{target_res['hit_rate']:5.2f}%",
            f"{target_res['total_utility']:6.2f}",
            f"{init_r:6.2f}",
            f"{final_r:6.2f}",
            f"{final_r - init_r:+6.2f}",
        ])

    render_section_header("Target Domain Transfer Learning Final Comparative Summary (lambda_T = 0.30)")
    render_table(
        headers=["Method", "Hit Rate", "Utility", "Initial AvgR", "Final AvgR", "Delta AvgR"],
        rows=method_metrics,
        alignments=['L', 'R', 'R', 'R', 'R', 'R'],
    )

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
        for method in ["PROPOSED", "DQFD", "LFS", "DPR"]:
            hist = tl_histories.get(method)
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

        # Fig. 8b: The impact of overwriting demonstration data
        if "DQFD" in tl_histories and "DQFD_OVERWRITE" in tl_histories:
            plt.figure(figsize=(8, 5))
            s1, r1 = zip(*tl_histories["DQFD"])
            s2, r2 = zip(*tl_histories["DQFD_OVERWRITE"])
            plt.plot(s1, r1, label="DQfD", color="#c44536", linewidth=2)
            plt.plot(s2, r2, label="DQfD with overwriting demonstration data", color="#006d77", linewidth=2)
            plt.title("Fig. 8b: The impact of overwriting demonstration data")
            plt.xlabel("Trial / Step")
            plt.ylabel("Average Reward")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "fig8b_overwriting_impact.png"), dpi=300)
            plt.close()

        # Fig. 8c: The impact of assigning priorities with Eq. (6)
        if "DQFD" in tl_histories and "DQFD_NEW_PRIO" in tl_histories:
            plt.figure(figsize=(8, 5))
            s1, r1 = zip(*tl_histories["DQFD"])
            s3, r3 = zip(*tl_histories["DQFD_NEW_PRIO"])
            plt.plot(s1, r1, label="DQfD", color="#c44536", linewidth=2)
            plt.plot(s3, r3, label="DQfD with new priorities (Eq. 6)", color="#006d77", linewidth=2)
            plt.title("Fig. 8c: The impact of assigning priorities with Eq. (6)")
            plt.xlabel("Trial / Step")
            plt.ylabel("Average Reward")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "fig8c_priorities_impact.png"), dpi=300)
            plt.close()

    print(f"\n>>> Results and detailed JSON metrics saved to '{output_dir}':")
    print(f"    - {source_json_path}")
    if tl_histories:
        print(f"    - {tl_json_path}")
    print(f"    - {summary_json_path}")
