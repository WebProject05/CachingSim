"""
Main CLI entry point for SMDP Edge Caching Deep Reinforcement Learning and Transfer Learning Agents.
"""

import argparse
import sys
import os
import time
import subprocess
import atexit
from typing import Optional

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
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Disable automatic launching of the Go gRPC server if not already running",
    )
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_ckpt = os.path.join(project_root, "agent-python", "checkpoints")
    default_out = os.path.join(project_root, "data", "results")

    parser.add_argument("--checkpoint-dir", type=str, default=default_ckpt, help="Directory to store model checkpoints")
    parser.add_argument("--output-dir", type=str, default=default_out, help="Directory to store evaluation plots and CSVs")
    return parser.parse_args()


def try_connect_env(host: str, port: int, total_files: int, seed: int, lambda_source: float, zipf_eta: float) -> Optional[GrpcEnvClient]:
    """Attempts to connect to the Go gRPC environment and perform a probe reset."""
    try:
        env = GrpcEnvClient(host=host, port=port, total_files=total_files)
        env.reset(seed=seed, lambda_rate=lambda_source, eta=zipf_eta)
        return env
    except Exception:
        return None


def ensure_server_running(
    cfg: AgentConfig,
    auto_start: bool = True,
) -> GrpcEnvClient:
    """
    Connects to the Go gRPC server. If unavailable and auto_start is enabled,
    automatically launches bin/server.exe or go run ./cmd/server.
    """
    # 1. First probe check: Is server already running?
    env = try_connect_env(cfg.server_host, cfg.server_port, cfg.total_files, cfg.seed, cfg.lambda_source, cfg.zipf_eta)
    if env is not None:
        print(f" Connected to active Go SMDP Caching Server at {cfg.server_host}:{cfg.server_port}.")
        return env

    if not auto_start:
        _print_connection_error_guide(cfg)
        sys.exit(1)

    # 2. Auto-start attempt
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    bin_path = os.path.join(project_root, "bin", "server.exe")
    if not os.path.exists(bin_path):
        bin_path = os.path.join(project_root, "bin", "server")

    server_process = None
    if os.path.exists(bin_path):
        print(f"[*] Go server not detected at {cfg.server_host}:{cfg.server_port}.")
        print(f"[*] Auto-launching '{bin_path}' in the background...")
        server_process = subprocess.Popen(
            [bin_path, "-port", str(cfg.server_port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=project_root,
        )
    else:
        # Fallback to 'go run ./cmd/server' if binary does not exist
        sim_dir = os.path.join(project_root, "simulator-go")
        if os.path.exists(sim_dir):
            print(f"[*] Binary not found. Auto-launching Go server via 'go run ./cmd/server'...")
            server_process = subprocess.Popen(
                ["go", "run", "./cmd/server", "-port", str(cfg.server_port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=sim_dir,
            )

    if server_process is None:
        _print_connection_error_guide(cfg)
        sys.exit(1)

    # Register process termination on script exit
    def cleanup_server():
        if server_process and server_process.poll() is None:
            print("\n[*] Stopping auto-spawned Go gRPC server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                server_process.kill()

    atexit.register(cleanup_server)

    # Retry connection with polling
    print("[*] Waiting for gRPC server to initialize...")
    for attempt in range(1, 10):
        time.sleep(1.0)
        env = try_connect_env(cfg.server_host, cfg.server_port, cfg.total_files, cfg.seed, cfg.lambda_source, cfg.zipf_eta)
        if env is not None:
            print(f" Successfully connected to auto-spawned Go SMDP Caching Server at {cfg.server_host}:{cfg.server_port}.")
            return env

    # If still unable to connect
    cleanup_server()
    _print_connection_error_guide(cfg)
    sys.exit(1)


def _print_connection_error_guide(cfg: AgentConfig):
    print("\n" + "=" * 70)
    print(f"❌ ERROR: Cannot connect to gRPC server at {cfg.server_host}:{cfg.server_port}")
    print("=" * 70)
    print("The Python RL agent requires the Go SMDP simulation engine to be running.")
    print("\nChoose one of the following methods to resolve this:\n")
    print("METHOD 1 (Recommended - 1-click execution):")
    print("    Run the automated batch script from the project root:")
    print("    > scripts\\run_full_experiment.bat\n")
    print("METHOD 2 (Separate Terminals):")
    print("    Terminal 1 (Start Go Server):")
    print("    > cd simulator-go")
    print(f"    > go run .\\cmd\\server -port {cfg.server_port}")
    print("      (or: .\\bin\\server.exe -port 50051)\n")
    print("    Terminal 2 (Run Python Agent):")
    print("    > cd agent-python")
    print("    > python main.py --mode full_experiment\n")
    print("METHOD 3 (Auto-start):")
    print("    Ensure bin/server.exe exists by running:")
    print("    > scripts\\build_go_windows.bat")
    print("    Then re-run: python main.py (auto-start will boot it automatically)")
    print("=" * 70 + "\n")


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

    # Connect to or auto-spawn Go SMDP server
    env = ensure_server_running(cfg, auto_start=not args.no_auto_start)

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
