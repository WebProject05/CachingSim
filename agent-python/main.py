"""
Main CLI entry point for SMDP Edge Caching Deep Reinforcement Learning and Transfer Learning Agents.
"""

import argparse
import sys
import os

from config import AgentConfig, get_default_config
from client.grpc_env_client import GrpcEnvClient
from algorithms.ddql import DDQLAgent
from evaluate import (
    run_source_training_and_eval,
    run_transfer_learning_comparison,
    save_results_and_plot,
)


def parse_args():
    parser = argparse.ArgumentParser(description="SMDP Edge Caching RL & Transfer Learning Agent")
    parser.add_argument(
        "--mode",
        type=str,
        default="full_experiment",
        choices=["train_source", "transfer_target", "full_experiment", "eval"],
        help="Execution mode (default: full_experiment)",
    )
    parser.add_argument("--source-steps", type=int, default=5000, help="Source domain training steps")
    parser.add_argument("--target-steps", type=int, default=6000, help="Target domain training steps")
    parser.add_argument("--eval-requests", type=int, default=1000, help="Evaluation test requests")
    parser.add_argument("--host", type=str, default="localhost", help="gRPC server host")
    parser.add_argument("--port", type=int, default=50051, help="gRPC server port")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_ckpt = os.path.join(project_root, "agent-python", "checkpoints")
    default_out = os.path.join(project_root, "data", "results")

    parser.add_argument("--checkpoint-dir", type=str, default=default_ckpt, help="Directory to store model checkpoints")
    parser.add_argument("--output-dir", type=str, default=default_out, help="Directory to store evaluation plots and CSVs")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = get_default_config()
    cfg.server_host = args.host
    cfg.server_port = args.port
    cfg.seed = args.seed
    cfg.source_train_steps = args.source_steps
    cfg.target_train_steps = args.target_steps
    cfg.eval_requests = args.eval_requests

    print("=" * 70)
    print(" SMDP Edge Caching Reinforcement Learning & Transfer Learning System")
    print(f" Mode: {args.mode} | Host: {args.host}:{args.port} | Seed: {args.seed}")
    print(f" Files (F): {cfg.total_files} | Cache (M): {cfg.cache_capacity:.0f} MiB")
    print(f" Source Lambda: {cfg.lambda_source} | Target Lambda: {cfg.lambda_target}")
    print("=" * 70)

    # Initialize gRPC environment connection to Go simulator
    try:
        env = GrpcEnvClient(host=cfg.server_host, port=cfg.server_port, total_files=cfg.total_files)
        # Test connection with a probe reset
        env.reset(seed=cfg.seed, lambda_rate=cfg.lambda_source, eta=cfg.zipf_eta)
        print(" Connected to Go SMDP Caching Server over gRPC.")
    except Exception as e:
        print(f" Error connecting to gRPC server at {cfg.server_host}:{cfg.server_port}.")
        print("Please ensure the Go server is running via 'go run ./cmd/server' or 'bin/server.exe'.")
        print(f"Details: {e}")
        sys.exit(1)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        if args.mode == "train_source":
            agent, eval_res, hist = run_source_training_and_eval(cfg, env, cfg.source_train_steps, cfg.eval_requests)
            ckpt_path = os.path.join(args.checkpoint_dir, "source_agent.pt")
            agent.save_checkpoint(ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")
            save_results_and_plot(cfg, eval_res, hist, {}, output_dir=args.output_dir)

        elif args.mode == "full_experiment":
            # 1. Train on Source Domain
            source_agent, eval_res, source_hist = run_source_training_and_eval(
                cfg, env, cfg.source_train_steps, cfg.eval_requests
            )
            ckpt_path = os.path.join(args.checkpoint_dir, "source_agent.pt")
            source_agent.save_checkpoint(ckpt_path)

            # 2. Run Transfer Learning Comparison on Target Domain
            tl_histories = run_transfer_learning_comparison(
                source_agent, cfg, env, target_steps=cfg.target_train_steps
            )

            # 3. Generate Plots and Save JSON/CSV Data
            save_results_and_plot(cfg, eval_res, source_hist, tl_histories, output_dir=args.output_dir)

        elif args.mode == "eval":
            agent = DDQLAgent(cfg)
            ckpt_path = os.path.join(args.checkpoint_dir, "source_agent.pt")
            if os.path.exists(ckpt_path):
                agent.load_checkpoint(ckpt_path)
                print(f"Loaded checkpoint from {ckpt_path}")
            eval_res = agent.evaluate(env, eval_requests=cfg.eval_requests, lambda_rate=cfg.lambda_source, eta=cfg.zipf_eta)
            print("\n=== Evaluation Results ===")
            for k, v in eval_res.items():
                print(f"{k:15s}: {v}")

    finally:
        env.close()
        print("\n=== Experiment Execution Complete ===")


if __name__ == "__main__":
    main()
