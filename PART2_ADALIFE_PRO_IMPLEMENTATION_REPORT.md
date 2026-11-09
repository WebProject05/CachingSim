# Part 2 Implementation & Verification Report: AdaLife-Pro Architecture

**Project:** SMDP Edge Caching Deep Reinforcement Learning & Transfer Learning Framework  
**Scope:** Part 1 Foundation Fixes + Part 2 (`AdaLife-Pro`) Single-Node Edge Caching Innovations  
**Date:** November 9, 2026  

---

## 1. Executive Summary

This report documents the end-to-end engineering implementation, bug resolutions, CLI toggles, unit tests, and empirical benchmarks for **Part 2 (`AdaLife-Pro`)**, implemented cleanly across both the Go discrete-event / SMDP simulation engine (`simulator-go`) and the Python Dueling-DDQN / Transfer Learning agent (`agent-python`).

### Scope Boundary
* **Implemented (Part 1 + Part 2):**
  * **Part 1 Fixes:** Size-aware Utility-Density Eviction (`UtilityDensityEvictor`), Regularized Unoccupied Memory Penalty ($\omega_{\text{mem}} = 10.0$), 10-dim Candidate-Centric Feature Vector (`state_dim = 261`), Dueling Double-DQN (`hidden_dim = 64`), and Selective Yield-Oracle Expert Demonstrations.
  * **Part 2 (`AdaLife-Pro`) Innovations:**
    1. **Layer 1 — Dynamic Lifetime (`AdaLife` Lease Extensions on Hit)**
    2. **Layer 2 — Elastic Capacity Headroom (Load-Adaptive $M_{\text{eff}}(t)$)**
    3. **Layer 3 — Two-Tier Probationary Shield (Probation / Protected Split + Idle Overdraft)**
    4. **Layer 4 — Fetch-Cost & Latency Asymmetry ($C_{\text{fetch}}(f)$ in Cost-Utility Density & Worth)**
    5. **Layer 5 — Counterfactual Ghost Cache (Regret Feedback & Fast-Track Promotion)**
  * **Backward-Compatibility Toggles:** Full CLI switches (`-part1-only` / `--part1-only` and per-layer flags) so Part 1 vs. Part 2 ablations run deterministically side by side.
* **Excluded (Deferred to Part 3):** Scale-Invariant 24-dim $O(1)$ State Space, Decaying Count-Min Sketch popularity estimator, million-request external trace parsers, and multi-edge cooperative peering.

### Key Measured Impact (Verified Engine Runs)
* **Strict Contention (`F=100, M=5,000 MiB, lambda=5.0 req/s, eta=0.85, 10k reqs`):**
  * Go Native Engine: Hit rate rose from **21.65% (`CTD`)** and **32.08% (`Part 1 Only`)** to **40.20% (`AdaLife-Pro`)** (and **41.25%** without `FetchCost` size bias), cutting final window miss rate from **78.10%** (`CTD`) / **67.30%** (`Part 1`) down to **57.40%** and doubling total utility (`3,114.75` vs `1,552.64` `CTD`).
  * Python Live gRPC Dueling-DDQN (`lambda=1.00 req/s`): Hit rate jumped from **27.40%** (`Part 1 Only`) to **40.40%** (`AdaLife-Pro`), and at `lambda=5.00 req/s` from **25.70%** to **33.50%**.
* **Table III Benchmark (`F=50, M=10,000 MiB, lambda=5.0 req/s, eta=1.0, 10k reqs`):**
  * Go Native Engine: Hit rate climbed from **58.39% (`CTD`)** and **62.67% (`Part 1 Only`)** to **75.84% (`AdaLife-Pro`)**, Byte Hit Rate reached **70.54%** (vs `56.94%`), and Total Utility surged from **3,615.11** to **5,494.44**.
  * Python Live gRPC Dueling-DDQN (`3,000 train steps, 1,000 eval reqs`): Achieved **74.10% Hit Rate** / **71.36% BHR** / **556.90 Utility** at $\lambda=5.0\text{ req/s}$, and **33.00% Hit Rate** at $\lambda=0.2\text{ req/s}$.
* **Low-Rate Source Domain (`F=50, M=10,000 MiB, lambda=0.2 req/s, eta=1.0, 10k reqs`):**
  * Go Native Engine: Broke the **15.94%** parity ceiling between `CTD` and `Part 1 Only`, lifting hit rate to **23.89%** (**+7.95 percentage points**, **+49.9% relative gain**) with `0.00%` evictions via lease renewals on active files.

---

## 2. Detailed Architectural Logic of the 5 Part 2 Innovations

All 5 mechanisms operate coherently inside `core.CacheEngine.ProcessAdaLifeStep` (`simulator-go/pkg/core/cache_state.go`), feeding live state and counterfactual reward signals across both the native Go benchmark wrapper (`cmd/baseline`) and the gRPC RL server (`pkg/server`).

### 2.1 Layer 1 — Dynamic Lifetime (`AdaLife` Adaptive Lease Extensions)
Instead of discarding active hot objects at a rigid stopwatch deadline ($w_l^f \in [10, 30]\text{s}$), validity lifetime acts as a renewable lease:
* **Initial Fetch (`ResetLease`):** Initializes `GenTime = currentTime`, `HitCount = 1`, and `EffectiveLifetime = Lifetime * scale` (where `scale = 1.0` in Protected tier and `ProbationTTLScale = 0.75` in Probationary tier).
* **Hit Renewal (`RenewLease`):** On every cache hit (when `EnableAdaLife == true`), `HitCount` increments and computes:
  $$\text{boost} = 1.0 + w_{\text{imp}} \cdot i_f + w_{\text{hit}} \cdot \min(5.0, \text{HitCount})$$
  $$\text{EffectiveLifetime} = \min\left(\text{MaxLifetimeCap},\; \text{Lifetime} \times \text{boost}\right)$$
  with `GenTime` reset to `currentTime` (`LeaseImportanceWeight = 0.5`, `LeaseHitWeight = 0.1`, `MaxLifetimeCap = 150.0`).
* **Unified Consumption:** Both `FileMetadata.Freshness` (`file.go`) and `CalculateFileUtility` (`utility.go`) query `FileMetadata.ActiveLifetime()`, seamlessly using `EffectiveLifetime` when active or falling back to base `Lifetime` in Part 1 mode.

### 2.2 Layer 2 — Elastic Capacity Headroom ($M_{\text{eff}}(t)$)
Fixed edge cache boxes choke during request surges while idling during quiet periods. Elastic capacity adapts $M_{\text{eff}}(t)$ smoothly to traffic demand and eviction pressure (`effectiveCapacityLocked`):
* **Unbiased Arrival Estimation (`UpdateTrafficRate`):** Tracks an Exponential Moving Average of inter-arrival times ($\text{TauEMA} = 0.05\tau + 0.95\,\text{TauEMA}$) and derives $\lambda_{\text{EMA}} = 1.0 / \max(0.01, \text{TauEMA})$. `RecentEvictions` decays by `0.98` per step.
* **Headroom Activation:** Expansion triggers above idle levels ($\lambda_{\text{EMA}} > 0.5\text{ req/s}$ or `RecentEvictions > 0.1`):
  $$\text{pressure} = \max\left(\min\left(1.0, \frac{\lambda_{\text{EMA}}}{\text{ElasticLambdaRef}}\right),\; \min(1.0, 0.25 \cdot \text{RecentEvictions})\right)$$
  $$M_{\text{eff}}(t) = M \cdot (1.0 + \text{ElasticMaxScale} \cdot \text{pressure})$$
  With defaults `ElasticMaxScale = 0.35` and `ElasticLambdaRef = 5.0`, peak surges expand cache capacity up to **1.35x $M$**.
* **Graceful Scale-Down Trimming:** At the start of `ProcessAdaLifeStep`, if demand drops and `UsedCapacity > effCap`, surplus objects are trimmed first from Probation (`evictProbationLocked`), then from lowest cost-utility density Protected entries (`evictProtectedLocked`), maintaining strict capacity invariants.

### 2.3 Layer 3 — Two-Tier Probationary Shield (`Probation` / `Protected`)
Prevents one-hit wonders from displacing multi-hit hot items in the main cache:
* **Partitioning (`tierCapacitiesLocked`):** Splits effective capacity into **10% Probation (`ProbationRatio = 0.10`)** and **90% Protected**, dynamically ensuring Probation can fit at least one incoming candidate file if total capacity allows ($\ge 2\times\text{reqSize}$).
* **Idle-RAM Overdraft (`probMaxBudget`):** First-time arrivals enter Probation with a shorter trial lease (`ProbationTTLScale = 0.75`) and evict strictly within Probation (`evictProbationLocked`, oldest FIFO victim) unless capacity is constrained by oversized Protected occupancy. Crucially, `probMaxBudget = max(probCap, effCap - ProtectedUsed)` allows Probation to borrow unoccupied RAM while Protected warms up, naturally clamping back down to 10% as Protected fills.
* **Promotion & Fast-Track:**
  * A cache hit inside Probation promotes the file directly into the Protected tier (`Promotions++`) and renews its lease.
  * Returning misses found in the Ghost Cache OR already popular in the sliding window (`popularity[reqFile] >= 2.0`) bypass Probation and fast-track directly into the Protected tier (if Cost-Utility density justifies eviction).

### 2.4 Layer 4 — Fetch-Cost & Latency Asymmetry ($C_{\text{fetch}}(f)$)
Differentiates expensive cloud fetches (large objects / high-latency backhaul) from cheap small items:
* **Origin Retrieval Coefficient (`FileMetadata.FetchCost`):**
  $$C_{\text{fetch}}(f) = \text{BaseOriginRTT} + \text{FetchBandwidthWeight} \cdot \frac{z_f}{1000.0}$$
  Calibrated defaults: `BaseOriginRTT = 0.8`, `FetchBandwidthWeight = 0.4` (returns `1.0` when `EnableFetchCost == false`).
* **Cost-Utility Density Eviction (`UtilityDensityEvictor.Evict`, `evictProtectedLocked`):** Victims are selected by minimizing $(y_f(t) \cdot C_{\text{fetch}}(f)) / \max(1.0, z_f)$.
* **RL State & Reward Alignment:** `GetWorth` / `getWorthLocked` and `GetCurrentState` multiply file utilities $y_f(t)$ by $C_{\text{fetch}}(f)$, guiding both heuristic admission and the Python Dueling-DDQN's cost-utility features (`cand_density`).

### 2.5 Layer 5 — Counterfactual Ghost Cache (Shadow Memory & Regret)
Solves the RL **Action-0 Blindspot**, where skipping a file previously produced zero negative feedback if the skipped file was requested again seconds later:
* **Bounded Shadow Log (`recordGhostLocked`):** Records lightweight `GhostEntry` structs (`Timestamp`, `Cost`, `WasRejected`, `FromProtected`, `WasExpired`) for the most recent `GhostMaxEntries = 1000` rejected (`action == 0`), evicted, or expired items without storing file bytes.
* **Regret Differentiation (`consumeGhostHitLocked`):** On a cache miss, if `fileID` matches an entry within `GhostHorizon = 60.0s`:
  * Increments `GhostHits`, triggers Protected fast-track eligibility on admission, and computes counterfactual regret based on decision culpability ($u = \text{UT}_{\max} \cdot i_f$):
    * **Avoidable miss** (`WasRejected || FromProtected`): full `GhostRegretWeight * u * Cost` (`GhostRegretWeight = 5.0`).
    * **Probation noise eviction** (`!WasRejected && !FromProtected && !WasExpired`): attenuated `0.25 * GhostRegretWeight * u * Cost`.
    * **Natural TTL expiration** (`WasExpired`): `0.0` regret (unavoidable staleness miss).
  * If the RL agent rejects a returning ghost hit *a second time* (`action == 0` in `ProcessAdaLifeStep`), an extra `0.5 * GhostRegretWeight * u * Cost` penalty is added.
* **Reward Subtraction (`ComputeReward`):** Returns `worth - mem * penaltyWeight - LastStepRegret` whenever `EnableGhostCache == true`.

---

## 3. Complete File-by-File Inventory

### Go Simulation Engine (`simulator-go`)
1. [`simulator-go/pkg/config/config.go`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/simulator-go/pkg/config/config.go#L58-L141): Added Part 2 `Config` struct fields and calibrated defaults inside `DefaultConfig()`.
2. [`simulator-go/pkg/core/file.go`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/simulator-go/pkg/core/file.go#L12-L121): Extended `FileMetadata` (`EffectiveLifetime`, `HitCount`); added `ActiveLifetime()`, `FetchCost()`, `ResetLease()`, `RenewLease()`; updated `Freshness()`.
3. [`simulator-go/pkg/core/utility.go`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/simulator-go/pkg/core/utility.go#L65-L69): Updated `CalculateFileUtility` to use `file.ActiveLifetime()`.
4. [`simulator-go/pkg/core/evictor.go`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/simulator-go/pkg/core/evictor.go#L84-L115): Updated `UtilityDensityEvictor.Evict` to rank victims by Cost-Utility Density (`Utility * FetchCost / Size`).
5. [`simulator-go/pkg/core/cache_state.go`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/simulator-go/pkg/core/cache_state.go#L10-L853): Implemented `GhostEntry`, dual-tier tracking (`InProbation`, `ProbationUsed`, `ProtectedUsed`), `EffectiveCapacity()`, `UpdateTrafficRate()`, `TierCapacities()`, `PeekGhost()`, `evictProbationLocked()`, `evictProtectedLocked()`, unified `ProcessAdaLifeStep()`, cost-weighted `GetWorth()` / `GetCurrentState()`, regret-aware `ComputeReward()`, and full cleanup in `Reset()`.
6. [`simulator-go/pkg/server/grpc_server.go`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/simulator-go/pkg/server/grpc_server.go#L107-L133): Wired `executeStepLocked` to call `s.cache.UpdateTrafficRate(tau)` and `s.cache.ProcessAdaLifeStep(reqFile, int(action), s.cache.CurrentTime)`.
7. [`simulator-go/cmd/server/main.go`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/simulator-go/cmd/server/main.go#L34-L132): Added `-adalife`, `-elastic`, `-probation`, `-fetch-cost`, `-ghost`, `-part1-only` flags and updated `runStandaloneSimulation` to invoke `UpdateTrafficRate` + `ProcessAdaLifeStep`.
8. [`simulator-go/cmd/baseline/main.go`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/simulator-go/cmd/baseline/main.go#L34-L455): Added matching Part 2 CLI flags, updated `smdpCacheWrapper` (`AccessAt`, `shouldAdmit` with Probation/Ghost/FetchCost logic), and introduced `utilityAwareCache` (`LastHitUtility()`) on `smdpCacheWrapper` and `ctdCacheWrapper`.
9. [`simulator-go/pkg/core/core_test.go`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/simulator-go/pkg/core/core_test.go#L168-L329): Added 4 unit tests covering lease extensions/caps, elastic capacity scaling, probation isolation + ghost fast-track/regret subtraction, and fetch-cost eviction ordering.

### Python DRL & Transfer Learning Agent (`agent-python`)
1. [`agent-python/config.py`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/agent-python/config.py#L24): Added `adalife_pro: bool = True` to `AgentConfig`.
2. [`agent-python/client/grpc_env_client.py`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/agent-python/client/grpc_env_client.py#L13-L98): Added `cache_capacity` parameter to `GrpcEnvClient` and `state_response_to_numpy`; fixed unit alignment for `can_fit_free` and `needed_space`.
3. [`agent-python/algorithms/ddql.py`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/agent-python/algorithms/ddql.py#L170-L221): Updated `compute_expert_action` to adapt between AdaLife-Pro Probationary/Ghost admission and strict Part 1 mode, and fixed the fallback non-candidate-feature branch capacity unit math.
4. [`agent-python/main.py`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/agent-python/main.py#L86-L280): Added `--part1-only` CLI flag, passed `cache_capacity=cfg.cache_capacity` to `GrpcEnvClient`, propagated `-part1-only` to auto-launched Go gRPC servers, and displayed active architecture in the startup banner.
5. [`agent-python/tests/test_adalife_agent.py`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/agent-python/tests/test_adalife_agent.py): New test file verifying capacity-scaled `can_fit_free` / `needed_space` feature math and `compute_expert_action` behavior across AdaLife-Pro vs Part 1 modes.

---

## 4. Engineering Bugs Caught & Resolved During Verification

| Issue Identified | Root Cause | Resolution Applied | Verified Outcome |
| :--- | :--- | :--- | :--- |
| **Jensen's Inequality Bias on `LambdaEMA`** | Averaging instantaneous arrival rate (`EMA(1 / tau)`) diverges upward for Exponential inter-arrival times because $\mathbb{E}[1/\tau] = +\infty$. | Changed [`UpdateTrafficRate`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/simulator-go/pkg/core/cache_state.go#L154-L166) to smooth inter-arrival time (`TauEMA = 0.05*tau + 0.95*TauEMA`) first, then derive `LambdaEMA = 1.0 / TauEMA`. | Accurately converges to true arrival rate (`5.0 req/s -> M_eff = 1,350 MiB` in `TestElasticCapacityScaling`). |
| **Probation Cold-Start Starvation** | Hard-capping Probation at 15% of capacity while Protected RAM sat empty caused Probation to thrash 1-2 files and stalled cache utilization at `14.47%` (`22.15%` HR when `Probation` ran alone). | Added [`probMaxBudget = math.Max(probCap, effCap - c.ProtectedUsed)`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/simulator-go/pkg/core/cache_state.go#L694-L720), calibrated 10%/90% tier split (`ProbationRatio = 0.10`), set `ProbationTTLScale = 0.75`, and fast-tracked `popularity >= 2.0`. | Single-layer `Probation=true` utilization jumped from `14.47%` to **95.92%**, lifting hit rate to **31.43%**. |
| **Python `can_fit_free` Unit Mismatch** | `state_response_to_numpy` compared `resp.mem` (a dimensionless `[0, 1]` proportion) directly against `cand_size = z / 1000.0` (in GiB). | Converted free space to GiB in [`state_response_to_numpy`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/agent-python/client/grpc_env_client.py#L62-L64) and fallback [`compute_expert_action`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/agent-python/algorithms/ddql.py#L210-L211): `free_space_gib = mem * (cache_capacity / 1000.0)`. | Exact `can_fit_free` and `needed_space` across `M = 5,000, 10,000, 15,000, 20,000 MiB`. |
| **Stale `GenTime=0` Utility Accumulation** | `cmd/baseline/main.go` computed served hit utility using an unmutated `files` slice whose `GenTime` stayed frozen at `0.0`, under-reporting true live hit utility after lease renewals or refreshes. | Added `utilityAwareCache` (`LastHitUtility()`) in [`cmd/baseline/main.go`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/simulator-go/cmd/baseline/main.go#L34-L37) for both `smdpCacheWrapper` and `ctdCacheWrapper`. | Reports exact live `Utility(t)` on every cache hit. |
| **Fetch-Cost Contention Calibration** | Initial bandwidth weight `0.5 + 1.0 * (z/1000)` overly shielded large cold items in tight `5,000 MiB` caches. | Calibrated [`DefaultConfig`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/simulator-go/pkg/config/config.go#L134-L136) to `BaseOriginRTT = 0.8` and `FetchBandwidthWeight = 0.4`. | Balanced origin cost savings with **40.20%** (contention) and **75.84%** (Table III) hit rates. |

---

## 5. CLI Flags Reference

| Binary / Entrypoint | New Flags (Default Behavior: All Part 2 Enabled) | Example Usage |
| :--- | :--- | :--- |
| **`bin/baseline.exe`** | `-adalife` (`true`), `-elastic` (`true`), `-probation` (`true`), `-fetch-cost` (`true`), `-ghost` (`true`), `-part1-only` (`false`) | `.\bin\baseline.exe -requests 10000 -files 100 -capacity 5000 -lambda 5.0 -eta 0.85` (add `-part1-only` for Part 1 baseline) |
| **`bin/server.exe`** | `-adalife` (`true`), `-elastic` (`true`), `-probation` (`true`), `-fetch-cost` (`true`), `-ghost` (`true`), `-part1-only` (`false`) | `.\bin\server.exe -port 50051 -files 50 -capacity 10000 -lambda 0.2 -eta 1.0` |
| **`agent-python/main.py`** | `--part1-only` (`False`; propagates `-part1-only` if auto-launching server and adjusts expert gate) | `python agent-python/main.py --mode full_experiment --source-steps 3000 --target-steps 1500` |

---

## 6. Verification & Empirical Benchmark Results

### 6.1 Automated Test Suites (100% Pass)
* **Go Engine (`cd simulator-go && go test -count=1 ./...`):** Passed all 7 test packages (`cmd/baseline`, `pkg/baselines`, `pkg/config`, `pkg/core`, `pkg/server`, `pkg/smdp`, `tests`), including all 4 new Part 2 unit tests (`TestDynamicLifetimeLeaseExtension`, `TestElasticCapacityScaling`, `TestProbationShieldAndGhostFastTrack`, `TestFetchCostAsymmetryEviction`).
* **Python Agent (`python -m pytest -v agent-python/tests`):** **9 / 9 passed** (7 existing buffer/loss/model tests + 2 new `test_adalife_agent.py` tests).

### 6.2 Go Native Comparative Benchmarks (10,000 Requests, Seed=42)

| Scenario | Algorithm / Configuration | Hit Rate | Byte Hit | Evict Rate | Cache Util | Total Utility | Final Miss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Contention**<br>`F=100, M=5k, lambda=5.0, eta=0.85` | `CTD` Baseline | 21.65% | 20.45% | 78.29% | 86.71% | 1,552.64 | 78.10% |
| **Contention** | `Part 1 Only` (`-part1-only`) | 32.08% | 27.17% | 19.21% | 93.16% | 1,851.63 | 67.30% |
| **Contention** | **`AdaLife-Pro` (All 5 Layers)** | **40.20%** | **32.44%** | 29.63% | **95.72%** | **3,114.75** | **57.40%** |
| **Contention** | `AdaLife-Pro` (`FetchCost=false`) | **41.25%** | **32.89%** | 28.12% | **99.44%** | **3,165.04** | **57.40%** |
| **Table III**<br>`F=50, M=10k, lambda=5.0, eta=1.0` | `CTD` Baseline | 58.39% | 56.94% | 41.42% | 95.83% | 3,615.11 | 42.70% |
| **Table III** | `Part 1 Only` (`-part1-only`) | 62.67% | 58.52% | 17.41% | 98.08% | 3,123.05 | 38.40% |
| **Table III** | **`AdaLife-Pro` (All 5 Layers)** | **75.84%** | **70.54%** | **12.42%** | 91.37% | **5,494.44** | **25.10%** |
| **Table III** | `AdaLife-Pro` (`FetchCost=false`) | **76.55%** | **70.99%** | **12.06%** | **99.99%** | **5,525.37** | **25.00%** |
| **Low-Rate Source**<br>`F=50, M=10k, lambda=0.2, eta=1.0` | `CTD` Baseline | 15.94% | 15.15% | 65.45% | 94.81% | 733.64 | 82.90% |
| **Low-Rate Source** | `Part 1 Only` (`-part1-only`) | 15.94% | 15.15% | 0.00% | 19.66% | 733.64 | 82.90% |
| **Low-Rate Source** | **`AdaLife-Pro` (All 5 Layers)** | **23.89%** | **22.46%** | **0.00%** | 19.66% | **1,202.55** | **74.60%** |

#### Table III Discrete MDP vs. Continuous SMDP Hit Count (1,000 Trials, `F=50, M=10,000`)
* **`lambda = 5.00 req/s`:** Part 1 SMDP = **599 hits** $\rightarrow$ **`AdaLife-Pro` SMDP = 664 hits** (MDP `0.20s` quota = 675 hits).
* **`lambda = 1.66 req/s`:** Part 1 SMDP = **484 hits** $\rightarrow$ **`AdaLife-Pro` SMDP = 489 hits**.
* **`lambda = 1.00 req/s`:** Part 1 SMDP = **398 hits** $\rightarrow$ **`AdaLife-Pro` SMDP = 401 hits**.

### 6.3 Single-Feature Ablation Analysis (Contention: `F=100, M=5k, lambda=5.0, eta=0.85`)

| Configuration | Hit Rate | Byte Hit | Evict Rate | Cache Util | Total Utility | Primary Behavioral Effect |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `Part 1 Only` (Baseline) | 32.08% | 27.17% | 19.21% | 93.16% | 1,851.63 | Utility-density eviction without leases/tiers |
| `+ AdaLife` only | **38.48%** | **31.43%** | **7.89%** | **98.57%** | **2,900.28** | Largest single-feature driver; eliminates hot expirations |
| `+ Elastic` only | 35.52% | 30.94% | 22.85% | 95.39% | 2,030.82 | Expands peak capacity to $1.35M$, easing memory pressure |
| `+ Probation` only | 31.43% | 25.63% | 44.88% | 95.92% | 1,903.54 | Isolates 1-hit churn to 10% Probation; raises Util over Part 1 |
| `+ FetchCost` only | 30.66% | 26.63% | 19.50% | 88.29% | 1,800.86 | Shifts protection toward heavy high-RTT origin objects |
| `+ Probation + Ghost` | 30.57% | 24.54% | 38.93% | 97.31% | 1,873.91 | Fast-tracks ghost misses to Protected (RL regret activates via gRPC) |

### 6.4 Python Live gRPC Dueling-DDQN & Transfer Learning Results

#### Multi-Rate Evaluation (`F=50, M=10,000, eta=1.0`, `3,000 train steps`, `1,000 eval reqs`, `AdaLife-Pro`)
| Arrival Rate ($\lambda$) | Hits / 1,000 | Hit Rate | Byte Hit Rate | Total Utility | Avg Reward |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `0.20 req/s` | 330 | **33.00%** | 33.30% | 171.63 | 14.90 |
| `0.50 req/s` | 526 | **52.60%** | 52.29% | 301.21 | 29.14 |
| `1.00 req/s` | 610 | **61.00%** | 59.87% | 390.33 | 40.07 |
| `1.66 req/s` | 674 | **67.40%** | 65.67% | 444.52 | 48.82 |
| `2.50 req/s` | 717 | **71.70%** | 70.24% | 493.40 | 54.66 |
| `5.00 req/s` | 741 | **74.10%** | 71.36% | 556.90 | 60.79 |
| `6.50 req/s` | 741 | **74.10%** | 71.18% | 571.31 | 61.83 |

#### Multi-Rate Contention Comparison (`F=100, M=5,000, eta=0.85`, `2,000 train steps`, `1,000 eval reqs`)
| Arrival Rate ($\lambda$) | `Part 1 Only` HR | `AdaLife-Pro` HR | Absolute Gain | `Part 1 Only` Util | `AdaLife-Pro` Util |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `0.20 req/s` | 12.90% | **19.20%** | **+6.30%** | 61.16 | **97.38** |
| `0.50 req/s` | 22.60% | **32.70%** | **+10.10%** | 113.09 | **185.10** |
| `1.00 req/s` | 27.40% | **40.40%** | **+13.00%** | 160.23 | **259.56** |
| `1.66 req/s` | 26.50% | **39.70%** | **+13.20%** | 172.46 | **290.46** |
| `2.50 req/s` | 25.50% | **36.10%** | **+10.60%** | 184.99 | **286.96** |
| `5.00 req/s` | 25.70% | **33.50%** | **+7.80%** | 203.02 | **288.38** |
| `6.50 req/s` | 25.30% | **34.20%** | **+8.90%** | 206.24 | **300.78** |

#### Transfer Learning Target Domain Comparison (`lambda_T = 0.30, 1,500 target steps`, `F=50, M=10,000`)
| Method | Hit Rate | Total Utility | Initial AvgR | Final AvgR | $\Delta$ AvgR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `PROPOSED` | 36.00% | 392.51 | 6.12 | 19.45 | +13.33 |
| `DQFD` | 36.67% | 394.98 | 6.63 | 19.85 | +13.22 |
| `LFS` | 35.93% | 389.46 | 6.36 | 19.88 | +13.52 |
| `DPR` | **37.47%** | **403.50** | **8.23** | **21.13** | +12.90 |
| `DQFD_OVERWRITE` | 36.40% | 395.07 | 6.52 | 19.82 | +13.30 |
| `DQFD_NEW_PRIO` | 35.80% | 388.55 | 6.13 | 19.83 | **+13.70** |

---

## 7. Verification & Hands-On Testing Commands ($\lambda = 5.0\text{ req/s}$)

All commands are ready to copy-paste into Windows PowerShell from the repository root:

### 7.1 Instant Go Native Comparative Benchmarks
Quick side-by-side execution against CTD baseline in ~1 second:

```powershell
# --- Standard Table III Setup (F=50, M=10,000 MiB, eta=1.0, lambda=5.0) ---
# 1. Full AdaLife-Pro (Part 1 + Part 2 full engine)
.\bin\baseline.exe -requests 10000 -files 50 -capacity 10000 -lambda 5.0 -eta 1.0 -mdp-table

# 2. Part 1 Baseline Only (AdaLife-Pro disabled)
.\bin\baseline.exe -requests 10000 -files 50 -capacity 10000 -lambda 5.0 -eta 1.0 -part1-only -mdp-table


# --- Strict Memory Contention Stress Test (F=100, M=5,000 MiB, eta=0.85, lambda=5.0) ---
# 1. Full AdaLife-Pro
.\bin\baseline.exe -requests 10000 -files 100 -capacity 5000 -lambda 5.0 -eta 0.85

# 2. Part 1 Baseline Only
.\bin\baseline.exe -requests 10000 -files 100 -capacity 5000 -lambda 5.0 -eta 0.85 -part1-only
```

### 7.2 Single-Feature Layer Ablations ($\lambda = 5.0\text{ req/s}$)
Isolate each Part 2 architectural layer independently under memory contention:

```powershell
# 1. Dynamic Lifetime only (Renewable leases on hit)
.\bin\baseline.exe -requests 10000 -files 100 -capacity 5000 -lambda 5.0 -eta 0.85 -adalife=true -elastic=false -probation=false -fetch-cost=false -ghost=false

# 2. Elastic Capacity only (Load-adaptive headroom)
.\bin\baseline.exe -requests 10000 -files 100 -capacity 5000 -lambda 5.0 -eta 0.85 -adalife=false -elastic=true -probation=false -fetch-cost=false -ghost=false

# 3. Two-Tier Probationary Shield only (10% probation with idle RAM overdraft)
.\bin\baseline.exe -requests 10000 -files 100 -capacity 5000 -lambda 5.0 -eta 0.85 -adalife=false -elastic=false -probation=true -fetch-cost=false -ghost=false

# 4. Fetch-Cost & Latency Asymmetry only
.\bin\baseline.exe -requests 10000 -files 100 -capacity 5000 -lambda 5.0 -eta 0.85 -adalife=false -elastic=false -probation=false -fetch-cost=true -ghost=false

# 5. Probation + Counterfactual Ghost Cache together (Fast-track promotion)
.\bin\baseline.exe -requests 10000 -files 100 -capacity 5000 -lambda 5.0 -eta 0.85 -adalife=false -elastic=false -probation=true -fetch-cost=false -ghost=true

# 6. AdaLife-Pro tuned for maximum Object Hit Rate (FetchCost disabled)
.\bin\baseline.exe -requests 10000 -files 100 -capacity 5000 -lambda 5.0 -eta 0.85 -fetch-cost=false
```

### 7.3 Live Python Dueling-DDQN Agent (Auto-Launches Go gRPC Server)
The Python runner automatically boots `bin\server.exe` in the background if port `50051` is unallocated, and shuts it down upon completion:

```powershell
# A. Fast evaluation of trained checkpoint at lambda = 5.0 (1,000 test requests):
python agent-python/main.py --mode eval --eval-lambda 5.0 --eval-requests 1000

# B. Train and evaluate at lambda = 5.0 on Standard Config (F=50, M=10,000 MiB):
# AdaLife-Pro (Part 1 + Part 2)
python agent-python/main.py --mode train_source --files 50 --capacity 10000 --eta 1.0 --lambda-source 5.0 --eval-lambda 5.0 --source-steps 3000 --eval-requests 1000

# Part 1 Only baseline
python agent-python/main.py --mode train_source --files 50 --capacity 10000 --eta 1.0 --lambda-source 5.0 --eval-lambda 5.0 --source-steps 3000 --eval-requests 1000 --part1-only

# C. Train and evaluate at lambda = 5.0 under Contention (F=100, M=5,000 MiB, eta=0.85):
# AdaLife-Pro
python agent-python/main.py --mode train_source --files 100 --capacity 5000 --eta 0.85 --lambda-source 5.0 --eval-lambda 5.0 --source-steps 2000 --eval-requests 1000

# Part 1 Only
python agent-python/main.py --mode train_source --files 100 --capacity 5000 --eta 0.85 --lambda-source 5.0 --eval-lambda 5.0 --source-steps 2000 --eval-requests 1000 --part1-only

# D. Full Automated Pipeline (Baselines + Python DRL + Transfer Learning + Graph Plots):
.\scripts\run_full_experiment.ps1 -EvalRequests 1000 -SourceSteps 3000 -TargetSteps 1500 -Capacity 10000 -EvalLambda 5.0 -ZipfEta 1.0 -Files 50
```

### 7.4 Standalone Go Simulation & Automated Test Suites

```powershell
# Standalone Go engine simulation at lambda = 5.0 (no gRPC overhead)
.\bin\server.exe -standalone -trials 5000 -files 50 -capacity 10000 -lambda 5.0 -eta 1.0

# Run all Go and Python unit tests
cd simulator-go ; go test -count=1 ./... ; cd .. ; python -m pytest -v agent-python/tests
```

---

## 8. Consolidated Master Command Reference (Top 6 Purpose-Driven Commands)

To eliminate workflow fragmentation and provide an intuitive evaluation interface, the command set has been consolidated into **6 purpose-driven commands** covering all benchmark and research needs:

```powershell
# [CMD 1] Quick 3-Way Benchmark & MDP Table (1k trials, executes in ~1 second):
.\bin\baseline.exe -requests 1000 -lambda 5.0 -mdp-table

# [CMD 2] High-Volume 3-Way Benchmark (10k requests, lambda = 5.0, Paper Standard Config):
.\bin\baseline.exe -requests 10000 -lambda 5.0

# [CMD 3] Severe Contention Stress Benchmark (100 files, M = 5,000 MiB, eta = 0.85):
.\bin\baseline.exe -requests 10000 -files 100 -capacity 5000 -lambda 5.0 -eta 0.85

# [CMD 4] Algorithm Ablation Comparison (AdaLife-Pro vs Part 1 Only baseline):
.\bin\baseline.exe -requests 10000 -lambda 5.0 -part1-only

# [CMD 5] Generate Visual Plots & Analytical Artifacts (SVGs, CSVs, and Sweeps):
.\bin\baseline.exe -requests 5000 -lambda 5.0 -g -results-dir results

# [CMD 6] Complete End-to-End Co-Simulation Pipeline (Go gRPC + PyTorch DDQN + Sweeps):
.\scripts\run_full_experiment.ps1 -EvalRequests 1000 -SourceSteps 3000 -TargetSteps 1500 -EvalLambda 5.0
```

---

## 9. Cleaned Workspace Documentation Architecture

To maintain a focused codebase for ongoing research and publication, redundant pre-implementation drafts have been cleaned up. The repository root retains 5 authoritative documents:

1. [`README.md`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/README.md) - Plain English user guide, architecture summary, installation, and quickstart.
2. [`PART2_ADALIFE_PRO_IMPLEMENTATION_REPORT.md`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/PART2_ADALIFE_PRO_IMPLEMENTATION_REPORT.md) - Authoritative Part 1 + Part 2 engineering implementation report, mathematical formulations, and 3-way benchmark results.
3. [`DATA_FLOW.md`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/DATA_FLOW.md) - Deep architectural breakdown of the Go IPC server, Python DRL agent, and memory telemetry.
4. [`RESEARCH_GAPS_AND_IMPROVEMENTS.md`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/RESEARCH_GAPS_AND_IMPROVEMENTS.md) - Formal identification of research gaps in Niknia et al. (2024) and foundational thesis for improvements.
5. [`REAL_WORLD_TRACE_EXPERIMENTATION_PROPOSAL.md`](file:///c:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/REAL_WORLD_TRACE_EXPERIMENTATION_PROPOSAL.md) - Roadmap for Part 3 evaluation against real-world production CDN traces (Akamai, Wikimedia).


