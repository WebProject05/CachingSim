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

## How to Use the Simulator

You can use the simulator in several ways, depending on what you want to test.

### Option 1: One-Click Full Automated Run (Easiest)
Run the automated script from the root directory. This compiles the Go binaries, starts the background server, trains the AI agent, runs test evaluations, and saves all results:

```powershell
# Windows PowerShell:
.\scripts\run_full_experiment.ps1

# Or Windows Command Prompt:
scripts\run_full_experiment.bat
```

---

### Option 2: Running Baseline Comparisons (Instant Benchmark)
You can compare the proposed algorithm against standard caching algorithms (FIFO, LRU, LFU, Sieve, and CTD) directly inside the Go engine. This takes just a few seconds:

```powershell
# Run a 5,000-request benchmark with 50 files and 10,000 MiB cache capacity
.\bin\baseline.exe -requests 5000 -files 50 -capacity 10000 -lambda 5.0 -eta 1.0
```

#### Useful Settings for `baseline.exe`:
* `-requests <int>`: Total number of requests in the test run (e.g., `5000`, `10000`, `50000`).
* `-files <int>`: Number of unique file types in the system (e.g., `50`, `100`, `250`).
* `-capacity <float>`: Cache storage capacity in MiB (e.g., `5000`, `10000`, `20000`).
* `-lambda <float>`: Request arrival rate (higher means busier traffic, e.g., `0.2` for slow, `5.0` for active, `20.0` for heavy rush hours).
* `-eta <float>`: Popularity skew parameter (e.g., `1.0` means a few files are very hot; `0.7` means requests are more spread out).
* `-g`: Automatically generate comparison charts in the `simulator-go/graphs/` folder.

---

### Option 3: Running the Python AI Agent Separately

#### Step A: Start the Go Simulation Server
Open Terminal 1 and start the simulation server:
```powershell
.\bin\server.exe -port 50051 -files 50 -capacity 10000 -lambda 5.0 -eta 1.0
```
The server will start listening for caching decisions.

#### Step B: Run the Python Agent
Open Terminal 2 and run the AI agent:

* **Evaluate an existing trained model on 1,000 requests:**
  ```powershell
  python agent-python/main.py --mode eval --eval-lambda 5.0 --eval-requests 1000
  ```

* **Train the agent from scratch:**
  ```powershell
  python agent-python/main.py --mode train_source --source-steps 2000 --eval-requests 1000 --eval-lambda 5.0
  ```

* **Run a full experiment (Source training + Target domain transfer learning):**
  ```powershell
  python agent-python/main.py --mode full_experiment --source-steps 3000 --target-steps 3000 --eval-requests 1000
  ```

---

### Option 4: Running Automated Tests
To verify that everything is working properly on your system, run the test suites:

```powershell
# 1. Test the Go Simulation Engine
cd simulator-go
go test ./...
cd ..

# 2. Test the Python Neural Networks and Buffers
python -m pytest agent-python/tests
```

---

## Where Results and Graphs are Stored

* **Numerical Results & JSON Summaries**: Saved in `data/results/` (contains exact hit counts, hit rate percentages, byte hit rates, and total utilities).
* **Model Checkpoints**: Saved in `agent-python/checkpoints/` (contains trained PyTorch `.pt` files).
* **Visual Charts**: Saved in `simulator-go/graphs/` and `data/results/` (shows hit rate trajectories, eviction counts, and comparison bar charts).
