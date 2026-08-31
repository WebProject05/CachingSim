# Python DRL & Transfer Learning Agent

This package implements the Double Deep Q-Learning (DDQL) and Transfer Learning algorithms for the SMDP Edge Caching Framework.

## Architecture

- **`models/dqn_mlp.py`**: PyTorch 2-layer MLP (16 hidden units, ReLU, uniform weight initialization in `[-0.1, 0.1]`, biases `0.1` per $\S$ VII-B).
- **`memory/sum_tree.py`**: $O(\log N)$ SumTree data structure.
- **`memory/prioritized_buffer.py`**: Prioritized Experience Replay buffer supporting the paper's proposed reward-adjusted priority equation $p_i = (\text{avg}R(t) - r_i) + |TDE_i| + \varsigma$ (Eq. 6), continuous SMDP discount $\gamma^\tau$, and $n$-step returns ($n=3$).
- **`memory/demo_buffer.py`**: Stores expert demonstrations collected in the source domain ($\lambda_S = 0.2$) and populates the target domain buffer ($|\Lambda'| = 10000$).
- **`algorithms/dqfd_loss.py`**: Multi-task transfer learning loss $J = J_{DQ} + \lambda_1 J_n + \lambda_2 J_E + \lambda_3 J_{L2}$ ($\S$ VI-2).
- **`algorithms/ddql.py`**: High-level DDQL agent managing action selection, training routines, network synchronization ($\zeta = 100$), and transfer learning modes (`proposed`, `dqfd`, `dpr`, `lfs`).
- **`evaluate.py`**: Evaluation routines, comparison runners, and JSON/PNG export functions.
- **`client/grpc_env_client.py`**: gRPC client connecting to the Go SMDP simulation server.

## Installation

```bash
pip install -r requirements.txt
```

## Running Experiments

```bash
# Full Experiment (Source Training + Evaluation + Target Transfer Learning)
python main.py --mode full_experiment --source-steps 5000 --target-steps 6000 --eval-requests 1000

# Source Domain Training Only
python main.py --mode train_source --source-steps 5000 --eval-requests 1000

# Policy Evaluation Only
python main.py --mode eval --eval-requests 1000
```

## Running Tests

```bash
python -m pytest -v tests
```

