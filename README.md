# SMDP Edge Caching Framework

An end-to-end implementation of the research paper:
> **"Edge Caching Based on Deep Reinforcement Learning and Transfer Learning"**  
> *Farnaz Niknia, Ping Wang, Zixu Wang, Aakash Agarwal, and Adib S. Rezaei (IEEE / arXiv:2402.14576v2, 2024)*

---

## 📖 System Overview & Architecture

The framework is structured into a high-performance **Go simulation engine** and an **agentic Python Deep Reinforcement Learning (DRL) & Transfer Learning (TL) system**, communicating seamlessly over **gRPC Protocol Buffers**.

```
smdp-edge-caching-framework/
├── proto/                         # Protocol Buffer definitions
│   └── cache_env.proto            # gRPC Contract (Reset, Step, BatchStep)
│
├── simulator-go/                  # High-performance Go SMDP Simulation Engine
│   ├── cmd/server/main.go         # gRPC server on :50051 & standalone simulator
│   ├── cmd/baseline/main.go       # Baseline runner (FIFO, LRU, LFU, SIEVE, CTD, MDP)
│   ├── cmd/baseline/graphs.go     # Parameter sweeps & SVG vector graph generator
│   ├── pkg/config/config.go       # Table I & II system configuration parameters
│   ├── pkg/core/                  # CacheEngine, FileMetadata, LowestUtilityEvictor, Utility
│   ├── pkg/smdp/                  # Poisson inter-arrival intervals, Zipf sampling, Rewards
│   ├── pkg/baselines/             # Eviction baselines & Table III MDP runner
│   └── tests/                     # Go test suite
│
├── agent-python/                  # DRL & Transfer Learning Agent System
│   ├── pb/                        # Compiled Python gRPC stubs
│   ├── client/grpc_env_client.py  # gRPC client for Go simulation environment
│   ├── config.py                  # Agent hyperparameters & state encoding
│   ├── models/dqn_mlp.py          # 2-layer 16-node PyTorch Q-Network
│   ├── memory/sum_tree.py         # SumTree for O(log N) prioritized replay sampling
│   ├── memory/prioritized_buffer.py # PER buffer with Eq. (6) and n-step returns
│   ├── memory/demo_buffer.py      # Demonstration buffer for transfer learning
│   ├── algorithms/dqfd_loss.py    # Multi-task loss: J = J_DQ + lambda_1*J_n + lambda_2*J_E + lambda_3*J_L2
│   ├── algorithms/ddql.py         # DDQLAgent (Source & Target domain workflows)
│   ├── evaluate.py                # Comparative evaluation & convergence plotting
│   ├── main.py                    # Main Python CLI entry point
│   ├── checkpoints/               # Saved model checkpoints (.pt)
│   └── tests/                     # Pytest suite for models, buffers, and loss functions
│
├── data/
│   ├── results/                   # Detailed experiment results (JSON, CSV, PNG plots)
│   └── offline_traces/            # Storage for workload traces
│
└── scripts/
    ├── build_go_windows.bat       # Compiles Go tests and builds bin/server.exe
    ├── compile_proto.bat          # Compiles proto for Go and Python
    └── run_full_experiment.bat    # Automated full experiment pipeline
```

---

## 🔬 Mathematical Methodology & Research Mapping

### 1. File Characteristics & Profiles ($\S$ III-A, Table II)
- **Total File Types**: $F = 50$, **Cache Capacity**: $M = 10000$ MiB.
- **Popularity**: Zipf distributed $p_f = \frac{1}{\sigma f^\eta}$ with $\sigma = \sum_{f=1}^F \frac{1}{f^\eta}$.
- **Lifetime**: $w_l^f \in [10, 30]$, **Importance**: $i_f \in [0.1, 0.9]$, **Size**: $z_f \in [100, 1000]$ MiB.
- **Generation Timestamp**: $w_g^f$ reset when file is requested/fetched.

### 2. Normalized Freshness & Non-Linear Utility ($\S$ III-A, Eq. 1)
- **Freshness**:
  $$h^f(t) = \frac{t - w_g^f}{w_l^f}, \quad 0 \le h^f(t) \le 1$$
- **Utility**:
  $$y_f(t) = \left( -\text{Curve} \cdot e^{h^f(t)} + UT_{\max} + \text{Curve} \right) \times i_f$$
  where $\text{Curve} = \frac{UT_{\max} - UT_{\min}}{e - 1}$, $UT_{\max} = 1.5$, $UT_{\min} = 0.1$.
  - At $h=0$ (brand new): $y_f(t) = UT_{\max} \times i_f$.
  - At $h=1$ (expired): $y_f(t) = UT_{\min} \times i_f$.

### 3. Lowest-Utility Eviction Policy ($\S$ IV-A-2)
When the cache lacks capacity for a new file $f_r$, it iteratively finds and removes the cached item with the lowest current utility $\arg\min_{f \in \text{Cached}} y_f(t)$ until sufficient free memory is available.

### 4. Semi-Markov Decision Process (SMDP) Formulation ($\S$ IV-A)
- **State Space**: $s(t) = \{\text{Mem}(t), d(t), y(t), z(t), b(t), f_r\}$.
  - $\text{Mem}(t) = \frac{M - \sum b_f(t) z_f}{M} \in [0, 1]$ (unoccupied memory proportion).
  - $d(t) \in \mathbb{R}^F$: request counts over sliding window $N=100$.
  - $b(t) \in \{0, 1\}^F$: cache indicator vector.
- **Continuous Transition Interval**: Poisson inter-arrival time $\tau = -\frac{\ln(U)}{\lambda}$.
- **Instant Reward** (Eq. 2 & 3):
  $$r(t) = W(t) - \text{Mem}(t) \times 100$$
  $$W(t) = \sum_{f=1}^F b_f(t) \cdot d_f(t) \cdot y_f(t) \quad (\text{Worth of cached files})$$

### 5. Double Deep Q-Learning (DDQL) with Continuous Discounting ($\S$ V-B)
- **Continuous SMDP Bellman Update**:
  $$Q(s, a) = r + \gamma^\tau Q\left(s', \arg\max_{b \in \mathcal{A}} Q(s', b; \theta); \theta'\right)$$
  where $\theta$ is the online network and $\theta'$ is the target network updated every $\zeta = 100$ steps.

### 6. Transfer Learning & Prioritized Replay with Reward Adjustment ($\S$ VI)
- **Proposed Sampling Priority (Eq. 6)**:
  $$p_i = (\text{avg}R(t) - r_i) + |TDE_i| + \varsigma$$
  where $\text{avg}R(t) = \frac{\sum_{k=1}^t r_k}{t}$, $TDE_i = Q(s, a)_i - \hat{Q}(s, a)_i$, $\varsigma = 10^{-5}$.
- **Importance Sampling**: $\omega_i = \left(\frac{1}{|\Lambda'|} \frac{1}{P(i)}\right)^\beta$ with $\beta \in [0.6, 1.0]$.
- **Multi-Task Loss Function (Eq. 7 & $\S$ VI-2)**:
  $$J = J_{DQ} + \lambda_1 J_n + \lambda_2 J_E + \lambda_3 J_{L2}$$
  - $J_{DQ}$: 1-step SMDP Double Q-Learning loss with discount $\gamma^\tau$.
  - $J_n$: $n$-step SMDP TD loss with cumulative discount $\gamma^{\tau^{(n)}}$.
  - $J_E$: Supervised Large Margin Classification Loss $\max_a [Q(s, a) + L(a_E, a) - Q(s, a_E)]$.
  - $J_{L2}$: L2 parameter regularization.

---

## 🚀 Step-by-Step Workflow & Commands

### Prerequisites
- **Go**: 1.22+
- **Python**: 3.10+ with `torch`, `grpcio`, `protobuf`, `numpy`, `matplotlib`, `pytest`

---

### Workflow 1: Running the Automated Full Experiment Pipeline
Run the all-in-one automation batch script:
```bat
scripts\run_full_experiment.bat
```
This automatically:
1. Builds the Go simulation server into `bin/server.exe`.
2. Starts the gRPC server in the background on port `50051`.
3. Trains the DDQL agent in the source domain ($\lambda_S = 0.2$).
4. Evaluates the learned policy on 1000 test requests.
5. Runs the transfer learning comparison on the target domain ($\lambda_T = 0.3$) across PROPOSED, DQFD, DPR, and LFS.
6. Saves model checkpoints to `agent-python/checkpoints/` and detailed JSON metrics & plots to `data/results/`.
7. Stops the background server cleanly.

---

### Workflow 2: Running Go Simulator Baselines & Parameter Sweeps
Generate traditional cache baseline comparisons (FIFO, LRU, LFU, SIEVE, CTD, SMDP) and 13 SVG parameter sweep charts:

```powershell
cd simulator-go
go run .\cmd\baseline -requests 5000 -files 50 -capacity 10000 -g -mdp-table
```

#### CLI Flags for `cmd/baseline`:
- `-requests <int>`: Number of requests in experiment trace (default: 50000).
- `-files <int>`: Number of generated file types $F$ (default: 50).
- `-capacity <float>`: Cache capacity $M$ in MiB (default: 10000.0).
- `-eta <float>`: Zipf skewness parameter $\eta$ (default: 1.0).
- `-seed <int>`: Random seed (default: 42).
- `-g`: Generate all 13 SVG vector charts, CSV summaries, and JSON outputs.
- `-mdp-table`: Evaluate and print **Table III** comparing discrete-time MDP vs continuous SMDP.
- `-concurrent`: Run cache simulations concurrently across goroutines.

---

### Workflow 3: Running the Go gRPC Server Manually
Start the gRPC simulation environment:
```powershell
cd simulator-go
go run .\cmd\server -port 50051 -files 50 -capacity 10000 -lambda 0.2 -eta 1.0
```
Or run the pre-built binary:
```powershell
.\bin\server.exe -port 50051
```

#### Standalone Simulation Mode (without gRPC):
```powershell
go run .\cmd\server -standalone -trials 1000 -files 50 -capacity 10000
```

---

### Workflow 4: Running the Python RL & Transfer Learning Agent
Ensure the Go server is running, then execute the Python agent:

```powershell
cd agent-python

# 1. Full Experiment (Source Training + Evaluation + Target Transfer Learning)
python main.py --mode full_experiment --source-steps 5000 --target-steps 6000 --eval-requests 1000

# 2. Source Domain Training Only
python main.py --mode train_source --source-steps 5000 --eval-requests 1000

# 3. Policy Evaluation Only
python main.py --mode eval --eval-requests 1000
```

#### CLI Flags for `agent-python/main.py`:
- `--mode <str>`: `full_experiment`, `train_source`, `transfer_target`, or `eval`.
- `--source-steps <int>`: Number of training steps in source domain ($\lambda_S = 0.2$, default: 5000).
- `--target-steps <int>`: Number of training steps in target domain ($\lambda_T = 0.3$, default: 6000).
- `--eval-requests <int>`: Number of test requests for policy evaluation (default: 1000).
- `--host <str>`: gRPC server hostname (default: `localhost`).
- `--port <int>`: gRPC server port (default: `50051`).
- `--seed <int>`: Random seed (default: `42`).
- `--checkpoint-dir <path>`: Directory for model weights (default: `checkpoints/`).
- `--output-dir <path>`: Directory for JSON results and plots (default: `data/results/`).

---

### Workflow 5: Running Automated Tests
Run unit and integration tests across both languages:

```powershell
# Go Test Suite (6 packages + integration tests)
cd simulator-go
go test -v ./...

# Python Test Suite (models, buffers, loss)
cd ..\agent-python
python -m pytest -v tests
```

---

## 📊 Recorded JSON Data Schemas in `data/results/`

All simulation runs and agent experiments automatically record structured JSON data into `data/results/`:

| Output JSON File | Description | Contents |
| :--- | :--- | :--- |
| [`baseline_results.json`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/data/results/baseline_results.json) | Complete baseline benchmark run | Request count, hits, misses, hit rate %, byte hit rate %, evictions, utilization %, total utility, latency (ns/request), operations/sec, and cumulative miss rate trajectories for FIFO, LRU, LFU, SIEVE, CTD, and SMDP. |
| [`parameter_sweeps.json`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/data/results/parameter_sweeps.json) | Multi-parameter sweep experiments | Numerical series for Cache Size (Figs. 6c/7c), Request Rate $\lambda$ (Figs. 6b/7b), Zipf $\eta$ (Figs. 6a/7a), Lifetime (Figs. 4a/5a), Size (Figs. 4b/5b), and Importance (Figs. 4c/5c). |
| [`mdp_vs_smdp_comparison.json`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/data/results/mdp_vs_smdp_comparison.json) | Table III reproduction | Discrete MDP vs Continuous SMDP hit counts and hit rates across time quotas $\Delta t \in \{0.2, 0.4, 0.6, 0.8, 1.0\}$ and request rates $\lambda \in \{5.0, 1.66, 1.0, 0.2\}$. |
| [`drl_source_training.json`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/data/results/drl_source_training.json) | DDQL Source Domain Training | Full training trajectory (steps, moving average reward), hyperparameters, and 1000-request evaluation metrics (hit rate %, total utility, avg reward). |
| [`transfer_learning_comparison.json`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/data/results/transfer_learning_comparison.json) | Target Domain Transfer Learning | Step-by-step convergence trajectories for PROPOSED TL, DQFD, DPR, and LFS matching Fig. 8a. |
| [`experiment_summary.json`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/data/results/experiment_summary.json) | Executive Summary | Unified high-level comparison metrics across all algorithms and domains. |

---

## 📈 Visual Assets in `graphs/` & `data/results/`

1. **SVG Vector Charts (`simulator-go/graphs/`)**:
   - `cumulative_miss_rate.svg` (Miss-rate convergence over trials)
   - `miss_rate_vs_cache_size.svg` (Fig. 6c) & `total_utility_vs_cache_size.svg` (Fig. 7c)
   - `miss_rate_vs_request_rate.svg` (Fig. 6b) & `total_utility_vs_request_rate.svg` (Fig. 7b)
   - `miss_rate_vs_zipf_eta.svg` (Fig. 6a) & `total_utility_vs_zipf_eta.svg` (Fig. 7a)
   - `miss_rate_vs_file_lifetime.svg` (Fig. 4a) & `total_utility_vs_file_lifetime.svg` (Fig. 5a)
   - `miss_rate_vs_file_size.svg` (Fig. 4b) & `total_utility_vs_file_size.svg` (Fig. 5b)
   - `miss_rate_vs_file_importance.svg` (Fig. 4c) & `total_utility_vs_file_importance.svg` (Fig. 5c)
2. **Matplotlib High-Resolution Convergence Plots (`data/results/`)**:
   - `ddql_source_convergence.png` (Source domain reward learning curve, Fig. 3a)
   - `transfer_learning_comparison.png` (Target domain convergence comparison: Proposed vs DQfD vs DPR vs LFS, Fig. 8a)

