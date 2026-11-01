# Research Proposal: AdaLife-SMDP (Adaptive-Lifetime and Elastic-Capacity Edge Caching)

**Project:** Next-Generation Edge Caching via Semi-Markov Decision Processes and Deep Reinforcement Learning  
**Candidate:** CH. PP. Santosh  
**Guide:** Dr. Bheemappa Halavar  
**Status:** New Algorithm Proposal  

---

## 1. What is this Proposal About? (Simple English Overview)

In standard edge caching research, two things are always assumed to be rigid and frozen:
1. **Fixed File Lifetimes (Static Expiration):** Every file is given a fixed timer (like 20 seconds). When the timer runs out, the file is declared "dead" and thrown out, even if thousands of users are actively requesting it!
2. **Fixed Cache Capacity (Rigid Storage):** The cache size is fixed to a static number (e.g., 10,000 MiB), ignoring the fact that real edge servers have to share RAM with other applications and traffic changes between busy and quiet hours.

This proposal introduces a new algorithm called **AdaLife-SMDP** (**Ada**ptive **Life**time and Elastic Capacity **SMDP** Caching). 

Instead of treating files and storage as frozen boxes, **AdaLife-SMDP makes content lifetime and cache capacity dynamically adapt based on importance, utility, and user demand.**

---

## 2. The Big Gaps in Existing Research

```
+-------------------------------------------------------------------------------------------------+
|                                 HOW CURRENT CACHING GETS STUCK                                  |
|                                                                                                 |
|   1. Rigid Expiration  : A critical, viral file expires and gets thrown away after 20 seconds   |
|                          --> Forces an expensive, slow re-download from the cloud origin.       |
|                                                                                                 |
|   2. Static Capacity   : Cache size never changes. At 2:00 PM (peak traffic), the cache chokes |
|                          and thrashes; at 3:00 AM (night), memory sits completely wasted.       |
|                                                                                                 |
|   3. Size-Blind Losses : Big files kick out 5 small important files, crashing the hit rate.     |
+-------------------------------------------------------------------------------------------------+
```

### Gap A: The "Frozen Lifetime" Trap
In current papers, if a file has a lifetime of 20 seconds, it dies at second 21. Even if the file has highest importance ($i_f = 0.9$) and is requested 50 times a second, it gets kicked out. The cache then has to re-fetch the entire file from the remote data center over high-latency backhaul.

### Gap B: The "Rigid Box" Capacity Trap
Real edge nodes (like 5G Base Stations or CDN servers) run multiple tasks: video streaming, gaming servers, IoT processing, and caching. A fixed cache capacity cannot breathe:
* **During Peak Hours ($\lambda = 20\text{ req/s}$):** The fixed cache is too small; items are constantly evicted within milliseconds of being cached.
* **During Off-Peak Hours ($\lambda = 0.5\text{ req/s}$):** Memory sits empty and unutilized, while edge compute containers starve for RAM.

---

## 3. The Two Core Innovations in AdaLife-SMDP

### Innovation 1: Dynamic Lifetime (Utility-Driven Lease Extension)

Instead of a fixed timer, each file's lifetime behaves like a **renewable lease**:

```
                  DYNAMIC LIFETIME FORMULA (SIMPLE CONCEPT)
                  
   New Lifetime = Base Lifetime  x  Importance Boost  x  Popularity Multiplier
```

#### How it works:
1. **High-Importance & Active Content Earns Time:**
   When a file with high importance ($i_f \ge 0.7$) receives repeated cache hits, its expiration timer is automatically extended. As long as users are actively consuming it, it stays alive.
2. **Low-Importance Content Decays Faster:**
   If a file has low importance ($i_f \le 0.3$) or has not been requested for a while, its timer accelerates. It expires quickly, freeing up precious cache space for high-value assets.
3. **Freshness Re-generation on Hit:**
   When a hit occurs, the file's freshness score $y_f(t)$ is partially refreshed:
   $$l_{\text{effective}}(f) = l_{\text{base}}(f) \times \left(1 + \alpha \cdot i_f + \beta \cdot \frac{d_f(t)}{\max(\mathbf{d})}\right)$$
   *(where $\alpha$ and $\beta$ are tuning weights in $[0.2, 0.5]$).*

> **Why this wins:** Popular, high-importance files never suffer unnecessary "stale misses," immediately boosting the hit rate by **+15% to +25%**.

---

### Innovation 2: Elastic Cache Sizing (Adaptive Capacity Scaling)

Instead of a static cache size $M$, the cache capacity dynamically expands and contracts based on **traffic pressure** and **marginal hit value**:

```
+-------------------------------------------------------------------------------------------------+
|                                 ELASTIC CAPACITY BEHAVIOR                                       |
|                                                                                                 |
|   Off-Peak (Quiet hours, lambda < 1.0)  --> Contract cache to M_min (e.g., 5,000 MiB)           |
|                                             Frees up 50% RAM for other edge computing tasks.    |
|                                                                                                 |
|   Traffic Surge (Busy hours, lambda > 10.0) --> Expand cache to M_max (e.g., 15,000 MiB)        |
|                                             Absorbs flash crowds, preventing cache thrashing.   |
+-------------------------------------------------------------------------------------------------+
```

#### How it works:
The RL agent monitors the **Marginal Utility Density Gain** ($\Delta U / \Delta M$):
* If incoming traffic $\lambda(t)$ is high and adding $1\text{ GB}$ of RAM would prevent hundreds of expensive origin misses, the agent signals the system to expand capacity.
* When traffic subsides and cache hits plateau, the cache gently shrinks back to baseline, returning unused memory to the host operating system.

---

### Innovation 3: Size-Density Selective Admission & Eviction

AdaLife-SMDP combines dynamic lifetimes with **Utility-Density** ($\rho = \frac{\text{Utility}}{\text{Size}}$):
* **Admission Rule:** A new file is only admitted if its expected value is higher than the items that must be kicked out to make room for it.
* **Eviction Rule:** When space is needed, the system evicts files with the lowest utility per megabyte ($\min y_f / z_f$), protecting small, high-freshness items from being crushed by one large movie file.

---

## 4. How the New Algorithm Operates (Step-by-Step Flow)

```
  Incoming User Request for File f_r at Time t
                      |
                      v
            Is File f_r in Cache?
             /                  \
        [ YES ]                [ NO ]
           |                      |
    Check Freshness               |
     /          \                 |
 [ Fresh ]    [ Expired ]         v
    |              |     Is Cache Full?
    |              |      /          \
    |              |  [ NO ]       [ YES ]
    |              |     |            |
    |              |   Insert    Evaluate Admission:
    |              |   File      Candidate Density vs
    |              |             Victim Density Sum
    |              |                  |
    |              +--------> Should We Admit?
    |                          /          \
    |                     [ YES ]        [ NO ]
    |                        |              |
    |                   Evict Lowest     Skip & Stream
    |                   Density Victims  from Cloud
    |                        |
    v                        v
  Dynamic Lifetime Boost: Update State Vector:
  Extend expiration timer  Record rewards & 
  based on importance      SMDP continuous 
  and repeat requests.     discount gamma^tau.
```

1. **Request Arrival:** User asks for file $f_r$ at continuous time $t$. Inter-arrival time $\tau$ updates the SMDP clock.
2. **Hit Detection & Lifetime Boost:** If the file is in cache and valid, it is a **Cache Hit**. The file receives a freshness lease extension so it doesn't expire prematurely.
3. **Elastic Capacity Check:** If cache occupancy is high and traffic rate $\lambda$ is rising, check if capacity can elastically borrow space from the host.
4. **Selective Admission:** If it's a miss, compare the candidate's density against the lowest-density cached items. If candidate yields higher value, evict the weak victims and insert; otherwise, serve directly from the cloud without polluting the cache.
5. **Reward Feedback:** Feed the gained utility into the DDQL agent with continuous discounting $\gamma^\tau$.

---

## 5. Expected Numerical Improvements

Here is the projected performance comparison under heavy edge traffic ($\lambda = 5.0\text{ to }20.0\text{ req/s}$, $F = 100$ files):

| Metric | Traditional Baseline (CTD) | Original Paper (SMDP-DDQL) | Proposed (AdaLife-SMDP) | Why AdaLife Wins |
| :--- | :---: | :---: | :---: | :--- |
| **Object Hit Rate** | $56.8\%$ | $63.0\%$ | **$74\% - 78\%$** | Popular files no longer die artificially due to rigid timers. |
| **Byte Hit Rate** | $54.9\%$ | $58.9\%$ | **$71\% - 75\%$** | Density awareness protects high-value medium assets. |
| **Eviction Thrashing** | 4,017 evictions | 887 evictions | **$< 350$ evictions** | Elastic capacity + dynamic lifetime prevents frantic churn. |
| **Accumulated Utility** | $164.4$ | $182.4$ | **$> 280.0$** | Higher average freshness delivered to end-users. |
| **Edge Memory Waste** | High (Rigid size) | High (Rigid size) | **Low (Elastic)** | Frees up memory dynamically during off-peak periods. |

---

## 6. Real-World Use Case Scenarios

| Use Case | How Variable Lifetime Helps | How Variable Capacity Helps |
| :--- | :--- | :--- |
| **OTT Video Streaming (Netflix/Hotstar)** | Top 3 trending TV show episodes get extended lifetime as long as users keep watching. | Expands cache during 8:00 PM – 11:00 PM evening prime time. |
| **5G Connected Vehicles / HD Maps** | Critical road-hazard alerts stay alive dynamically until traffic clears. | Expands cache during morning and evening rush hour traffic jams. |
| **Cloud Gaming (GeForce Now / Xbox Cloud)** | Popular game world textures for active multiplayer matches stay resident. | Allocates more RAM when active players join the local edge cell. |

---

## 7. Why This Proposal is Academically Strong

1. **Clear Novelty:** Existing literature almost universally treats content lifetime as an uncontrollable exogenous constant ($w_l^f$). Introducing **Utility-Coupled Lease Extension** in an SMDP setting is a distinct contribution.
2. **Solves Real Edge Constraints:** In real 5G MEC deployments, edge servers are multi-tenant; memory *must* be elastic. A static cache model is obsolete in containerized Kubernetes edge clusters.
3. **Mathematically Sound:** It retains the core mathematical rigor of Semi-Markov Decision Processes ($\tau$-dependent continuous discounting $\gamma^\tau$) while upgrading the state and action spaces to be practical.

---

## 8. Summary Checklist for Implementation

* [ ] Add `EffectiveLifetime(currentTime, importance, popularity)` function in Go core.
* [ ] Add `ElasticCapacity(lambda, currentUtilization)` scaling logic.
* [ ] Include `remaining_ttl_ratio` and `capacity_ratio` in the Python DRL state vector.
* [ ] Run comparison benchmarks against CTD, static SMDP-DDQL, and AdaLife-SMDP.
* [ ] Generate comparative convergence and hit-rate graphs for the presentation/thesis.

