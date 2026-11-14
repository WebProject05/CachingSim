# Edge Caching Simulation Framework

A high-performance edge caching simulation platform that combines a fast simulation engine written in Go with an intelligent decision-making agent written in Python using Deep Reinforcement Learning.

---

## What is this Simulator?

In mobile networks and content delivery systems, edge caching places small storage units (caches) inside cell towers and local network routers close to users. When users request videos, web pages, or app data, the cache serves the content locally instead of making users wait for a distant central cloud data center.

This simulator models that entire environment:
1. **The Go Simulation Engine (`simulator-go`)**: A lightning-fast engine that manages cache memory, tracks file expiration timers, generates incoming user traffic, and benchmarks standard caching policies.
2. **The Python AI Agent (`agent-python`)**: An intelligent neural network agent that learns over time which files are worth storing in the cache and which files should be skipped to prevent wasting memory.
3. **gRPC Communication**: The Go engine and the Python agent communicate in real-time over fast network messages, exchanging state updates and caching decisions.

---

## How the Simulator Works (The Process in Simple English)

The simulator runs an ongoing loop that mimics real-world internet traffic:

```
                  HOW THE SIMULATION PROCESS WORKS
                  
  1. User Request Arrives
     A user asks for a file at an edge router.
               |
               v
  2. Cache Check (Hit or Miss?)
     Does the cache have the file?
     - YES & FRESH  --> Cache Hit! Served immediately in milliseconds.
     - YES & EXPIRED--> Cache Miss! Stale file is cleaned up.
     - NO           --> Cache Miss! File must be fetched from cloud.
               |
               v (On a Cache Miss)
  3. AI Admission Decision
     The Python agent looks at the requested file (its size, importance,
     and recent request frequency) and the available cache space.
     The agent decides:
     - Action 0: Skip (Do not cache; stream directly from cloud).
     - Action 1: Cache (Store it locally for future users).
               |
               v (If Caching and Memory is Full)
  4. Smart Replacement (Eviction)
     If the cache does not have enough room, it removes the files
     that provide the lowest value per megabyte to fit the new file.
               |
               v
  5. Learning from Experience
     The agent receives feedback (rewards) based on hits delivered
     and memory used, continuously improving its decision policy.
```

### Key Working Features:
* **Time-Aware Content**: Every file has a validity lifetime (expiration timer) and freshness score. Files that are too old are automatically purged so stale content is never served.
* **Size & Value Awareness**: Files have realistic sizes (e.g., 100 MiB to 1,000 MiB) and importance ratings. The cache avoids kicking out five small high-value items just to fit one massive low-value file.
* **Realistic Traffic Patterns**: Incoming requests follow realistic popularity skews (a few hot files receive most requests, while many files receive rare requests) and arrive randomly over time.

---

## Repository Structure

```
smdp-edge-caching-framework/
├── proto/                     # Network communication contracts between Go and Python
│   └── cache_env.proto        # Definitions for Reset, Step, and State messages
│
├── simulator-go/              # High-speed simulation engine in Go
│   ├── cmd/server/            # gRPC server that runs the caching environment
│   ├── cmd/baseline/          # Runner for comparing standard industry caching policies
│   ├── pkg/core/              # Cache memory state, file metadata, and eviction logic
│   ├── pkg/smdp/              # Request arrival models and traffic generators
│   └── pkg/baselines/         # Implementations of FIFO, LRU, LFU, Sieve, and CTD
│
├── agent-python/              # Intelligent caching agent in Python
│   ├── client/                # gRPC client connecting to the Go simulation engine
│   ├── models/                # Neural network architectures for decision making
│   ├── algorithms/            # Reinforcement learning and transfer learning workflows
│   ├── memory/                # Prioritized experience replay buffers
│   └── checkpoints/           # Saved trained model weights
│
├── bin/                       # Compiled executable binaries for fast execution
├── data/results/              # Experiment outputs, metrics, and visual comparison charts
└── scripts/                   # Automated one-click execution scripts
```

---

## Installation & Setup

### Prerequisites
Make sure the following tools are installed on your machine:
* **Go**: Version 1.22 or newer ([Download Go](https://go.dev/dl/))
* **Python**: Version 3.10 or newer ([Download Python](https://www.python.org/downloads/))
* **Git**: To clone the repository

### Step 1: Clone the Repository
Open a terminal (or PowerShell on Windows) and clone the project:
```bash
git clone https://github.com/your-username/smdp-edge-caching-framework.git
cd smdp-edge-caching-framework
```

### Step 2: Set Up Python Dependencies
Create and activate a virtual environment (optional but recommended), then install the required Python packages:
```bash
# On Windows PowerShell:
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r agent-python/requirements.txt
```

### Step 3: Build the Go Binaries
Compile the Go simulation server and baseline tools:
```bash
# On Windows PowerShell:
cd simulator-go
go build -o ../bin/server.exe ./cmd/server
go build -o ../bin/baseline.exe ./cmd/baseline
cd ..
```
*(On Linux/macOS, replace `.exe` with standard binary names).*

---

## How to Use the Simulator (The 4 Core Commands)

To keep workflows simple and focused, all simulation, benchmark, and evaluation workflows are consolidated into **4 core commands** (strictly under 5 commands):

### [Command 1] One-Click Full Simulation Pipeline (Entire Simulation)
Runs the entire system end-to-end: builds Go binaries $\to$ runs Go baselines and sweeps $\to$ boots background Go gRPC server $\to$ trains Python PyTorch Dueling-DDQN neural agent $\to$ evaluates policy $\to$ runs target domain transfer learning $\to$ renders publication comparison figures (PNG/SVG) $\to$ compiles comprehensive performance reports (MD, TXT, JSON):

```powershell
# Windows PowerShell (Recommended):
.\scripts\run_full_experiment.ps1

# Or Windows Command Prompt:
scripts\run_full_experiment.bat
```

---

### [Command 2] Instant 3-Way Benchmark (~1-2 Seconds)
Instantly compares **CTD Baseline** vs **Original Paper (Niknia '24 DDQL)** vs **AdaLife-Pro (Ours)** directly in the Go engine, printing the 3-Way performance table and Table III (Discrete MDP vs Continuous SMDP):

```powershell
.\bin\baseline.exe -requests 1000 -lambda 5.0 -mdp-table
```

---

### [Command 3] Severe Contention Stress Benchmark
Tests the algorithms under extreme memory contention (100 competing files, restricted 5,000 MiB cache capacity, and flatter popularity $\eta = 0.85$):

```powershell
.\bin\baseline.exe -requests 10000 -files 100 -capacity 5000 -lambda 5.0 -eta 0.85
```

---

### [Command 4] Visual Graph & Plot Generation
Runs parameter sweeps across cache size, request rate $\lambda$, Zipf $\eta$, file lifetimes, and file sizes, exporting vector SVG charts and raw CSV trajectories:

```powershell
.\bin\baseline.exe -requests 5000 -lambda 5.0 -g -graphs simulator-go/graphs
```

---

## Complete Command Flags Reference

All available configuration flags across both the pipeline orchestrator and the Go engine are documented below:

### 1. Full Pipeline Flags (`run_full_experiment.ps1` / `.bat`)

| Flag | Type | Default | Description |
|---|:---:|:---:|---|
| `-EvalRequests` | `int` | `1000` | Number of test requests evaluated during policy verification. |
| `-SourceSteps` | `int` | `5000` | Number of training steps for the DDQL agent in the source domain ($\lambda_S = 5.0$). |
| `-TargetSteps` | `int` | `6000` | Number of transfer learning steps in the target domain ($\lambda_T = 0.3$). |
| `-Capacity` | `float` | `10000` | Base cache storage capacity in MiB ($M$). |
| `-EvalLambda` | `float` | `5.0` | Continuous request arrival rate ($\lambda$) in requests/second. |
| `-ZipfEta` | `float` | `1.0` | Zipf popularity skewness exponent ($\eta$). |
| `-Files` | `int` | `50` | Total number of distinct file types in the system ($F$). |

*Example with custom pipeline flags:*
```powershell
.\scripts\run_full_experiment.ps1 -EvalRequests 2000 -SourceSteps 3000 -TargetSteps 2000 -Capacity 10000 -EvalLambda 5.0 -ZipfEta 1.0 -Files 50
```

---

### 2. Go Baseline Simulator Flags (`bin\baseline.exe`)

| Flag | Type | Default | Description |
|---|:---:|:---:|---|
| `-requests` | `int` | `50000` | Total number of synthetic requests in the simulation trace. |
| `-files` | `int` | `50` | Number of unique file types ($F$) generated in the system. |
| `-capacity` | `float` | `10000` | Cache storage capacity in MiB ($M$). |
| `-lambda` | `float` | `0.2` | Continuous Poisson request arrival rate ($\lambda$) in req/s. |
| `-eta` | `float` | `1.0` | Zipf popularity skewness exponent ($\eta$). |
| `-seed` | `int64` | `42` | Random seed for file generation and request sequences. |
| `-mdp-table` | `bool` | `false` | Evaluates and prints Table III (Discrete MDP vs Continuous SMDP). |
| `-g` | `bool` | `false` | Enables automatic SVG vector chart and parameter sweep generation. |
| `-graphs` | `string` | `"graphs"` | Target directory where generated SVG charts and CSVs are written. |
| `-graph-interval` | `int` | `100` | Step interval between recorded data points for convergence curves. |
| `-window` | `int` | `0` | Sliding window for warm-up and steady-state miss rates (0 = 10% of trace). |
| `-concurrent` | `bool` | `false` | Runs caching policies concurrently across multiple CPU threads. |
| `-log` | `string` | `"logs/baseline_runs.log"` | Append-only execution log file path. |

#### Algorithmic Feature Toggles (AdaLife-Pro Layers):
* `-adalife=true/false`: Dynamic lifetime lease extensions on cache hits (Layer 1).
* `-elastic=true/false`: Load-adaptive elastic cache capacity headroom up to $+35\%$ (Layer 2).
* `-probation=true/false`: Two-tier probationary shield protecting the core cache (Layer 3).
* `-fetch-cost=true/false`: Origin retrieval cost weighting in utility & admission (Layer 4).
* `-ghost=true/false`: Counterfactual ghost cache metadata tracking (Layer 5).
* `-part1-only`: Disables all Part 2 layers to isolate the Part 1 baseline.

---

### Running Automated Test Suites
To verify all engine components, neural networks, and replay buffers:

```powershell
# 1. Test Go Engine:
cd simulator-go ; go test -count=1 ./... ; cd ..

# 2. Test Python Neural Modules:
python -m pytest -v agent-python/tests
```

---

## Where Results and Graphs are Stored

* **Numerical Results & JSON Summaries**: Saved in `data/results/` (contains exact hit counts, hit rate percentages, byte hit rates, and total utilities).
* **Model Checkpoints**: Saved in `agent-python/checkpoints/` (contains trained PyTorch `.pt` files).
* **Visual Charts**: Saved in `simulator-go/graphs/` and `data/results/` (shows hit rate trajectories, eviction counts, and comparison bar charts).
