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
from ui_utils import render_banner, render_card, render_section_header


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
    parser.add_argument(
        "--eval-lambda",
        type=float,
        default=5.0,
        help="Request arrival rate lambda for evaluation (default: 5.0 matching Paper Table III / Fig. 6b; use 0.2 for source training rate)",
    )
    parser.add_argument(
        "--capacity",
        type=float,
        default=10000.0,
        help="Cache capacity in MiB (default: 10000.0, e.g. 5000, 15000, 20000)",
    )
    parser.add_argument(
        "--files",
        type=int,
        default=50,
        help="Total number of file types F (default: 50)",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="Zipf popularity skewness parameter eta (default: 1.0)",
    )
    parser.add_argument(
        "--lambda-source",
        type=float,
        default=0.2,
        help="Poisson request arrival rate in source domain lambda_S (default: 0.2)",
    )
    parser.add_argument(
        "--lambda-target",
        type=float,
        default=0.3,
        help="Poisson request arrival rate in target domain lambda_T (default: 0.3)",
    )
    return parser.parse_args()


def try_connect_env(
    host: str,
    port: int,
    total_files: int,
    seed: int,
    lambda_source: float,
    zipf_eta: float,
    use_candidate_features: bool = True,
) -> Optional[GrpcEnvClient]:
    """Attempts to connect to the Go gRPC environment and perform a probe reset."""
    try:
        env = GrpcEnvClient(
            host=host,
            port=port,
            total_files=total_files,
            use_candidate_features=use_candidate_features,
        )
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
    env = try_connect_env(
        cfg.server_host,
        cfg.server_port,
        cfg.total_files,
        cfg.seed,
        cfg.lambda_source,
        cfg.zipf_eta,
        cfg.use_candidate_features,
    )
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
            [
                bin_path,
                "-port", str(cfg.server_port),
                "-capacity", str(cfg.cache_capacity),
                "-files", str(cfg.total_files),
                "-lambda", str(cfg.lambda_source),
                "-eta", str(cfg.zipf_eta),
                "-seed", str(cfg.seed),
            ],
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
                [
                    "go", "run", "./cmd/server",
                    "-port", str(cfg.server_port),
                    "-capacity", str(cfg.cache_capacity),
                    "-files", str(cfg.total_files),
                    "-lambda", str(cfg.lambda_source),
                    "-eta", str(cfg.zipf_eta),
                    "-seed", str(cfg.seed),
                ],
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
        env = try_connect_env(
            cfg.server_host,
            cfg.server_port,
            cfg.total_files,
            cfg.seed,
            cfg.lambda_source,
            cfg.zipf_eta,
            cfg.use_candidate_features,
        )
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
    cfg.cache_capacity = args.capacity
    cfg.total_files = args.files
    cfg.zipf_eta = args.eta
    cfg.lambda_source = args.lambda_source
    cfg.lambda_target = args.lambda_target

    render_banner(
        "SMDP Edge Caching Reinforcement Learning & Transfer Learning System",
        details={
            "Execution Mode": args.mode,
            "gRPC Server Target": f"{args.host}:{args.port}",
            "Random Seed": args.seed,
            "File Types (F)": cfg.total_files,
            "Cache Capacity (M)": f"{cfg.cache_capacity:,.0f} MiB",
            "Zipf Exponent (Eta)": f"{cfg.zipf_eta:.2f}",
            "Source Lambda (lambda_S)": f"{cfg.lambda_source:.2f} req/s",
            "Target Lambda (lambda_T)": f"{cfg.lambda_target:.2f} req/s",
            "Evaluation Lambda (lambda)": f"{args.eval_lambda:.2f} req/s",
            "Source Steps": f"{cfg.source_train_steps:,}",
            "Target Steps": f"{cfg.target_train_steps:,}",
            "Evaluation Requests": f"{cfg.eval_requests:,}",
        },
    )

    # Connect to or auto-spawn Go SMDP server
    env = ensure_server_running(cfg, auto_start=not args.no_auto_start)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        if args.mode == "train_source":
            agent, eval_res, hist = run_source_training_and_eval(
                cfg, env, cfg.source_train_steps, cfg.eval_requests, eval_lambda=args.eval_lambda
            )
            ckpt_path = os.path.join(args.checkpoint_dir, "source_agent.pt")
            agent.save_checkpoint(ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")
            save_results_and_plot(cfg, eval_res, hist, {}, output_dir=args.output_dir)

        elif args.mode == "transfer_target":
            source_agent = DDQLAgent(cfg)
            ckpt_path = os.path.join(args.checkpoint_dir, "source_agent.pt")
            if os.path.exists(ckpt_path):
                source_agent.load_checkpoint(ckpt_path)
                print(f"Loaded source checkpoint from {ckpt_path}")
            else:
                print("[*] No checkpoint found. Pre-training source agent...")
                source_agent, _, _ = run_source_training_and_eval(
                    cfg, env, cfg.source_train_steps, cfg.eval_requests, eval_lambda=args.eval_lambda
                )
                source_agent.save_checkpoint(ckpt_path)

            # Populate demonstrations if empty
            if len(source_agent.demo_buffer) < 1000:
                print("[*] Populating demonstration buffer with expert decisions...")
                source_agent.collect_expert_demonstrations(env, count=2500)

            tl_histories = run_transfer_learning_comparison(
                source_agent, cfg, env, target_steps=cfg.target_train_steps
            )
            save_results_and_plot(cfg, {}, [], tl_histories, output_dir=args.output_dir)

        elif args.mode == "full_experiment":
            # 1. Train on Source Domain
            source_agent, eval_res, source_hist = run_source_training_and_eval(
                cfg, env, cfg.source_train_steps, cfg.eval_requests, eval_lambda=args.eval_lambda
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

            render_section_header(f"Evaluating Model at Table III Benchmark (lambda = {args.eval_lambda:.2f})")
            eval_res = agent.evaluate(env, eval_requests=cfg.eval_requests, lambda_rate=args.eval_lambda, eta=cfg.zipf_eta, verbose=True)
            render_card("Benchmark Evaluation Results", {
                "Requests": eval_res["requests"],
                "Hits": eval_res["hits"],
                "Hit Rate": f"{eval_res['hit_rate']:.2f}%",
                "Byte Hit Rate": f"{eval_res.get('byte_hit_rate', 0.0):.2f}%",
                "Total Utility": f"{eval_res['total_utility']:.2f}",
                "Average Reward": f"{eval_res['avg_reward']:.2f}",
            })

            if args.eval_lambda != cfg.lambda_source:
                render_section_header(f"Evaluating Model at Source Domain Rate (lambda = {cfg.lambda_source:.2f})")
                eval_res_src = agent.evaluate(env, eval_requests=cfg.eval_requests, lambda_rate=cfg.lambda_source, eta=cfg.zipf_eta, verbose=True)
                render_card("Source Rate Evaluation Results", {
                    "Requests": eval_res_src["requests"],
                    "Hits": eval_res_src["hits"],
                    "Hit Rate": f"{eval_res_src['hit_rate']:.2f}%",
                    "Byte Hit Rate": f"{eval_res_src.get('byte_hit_rate', 0.0):.2f}%",
                    "Total Utility": f"{eval_res_src['total_utility']:.2f}",
                    "Average Reward": f"{eval_res_src['avg_reward']:.2f}",
                })

    finally:
        env.close()
        render_banner(
            "Experiment Execution Complete",
            details={
                "Status": "SUCCESS",
                "Results Directory": os.path.abspath(args.output_dir),
                "Checkpoints": os.path.abspath(args.checkpoint_dir),
            },
        )


if __name__ == "__main__":
    main()
