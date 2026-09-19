# Research Paper Figures & Experimental Analysis Guide

This documentation provides an in-depth explanation of all figures generated for the research paper:

> **"Edge Caching Based on Deep Reinforcement Learning and Transfer Learning"**  
> *Farnaz Niknia, Ping Wang, Zixu Wang, Aakash Agarwal, and Adib S. Rezaei (IEEE / arXiv:2402.14576v2, 2024)*

All graphs are available in both **high-resolution PNG (300 DPI)** and **lossless vector SVG** formats.

---

## 📑 Summary of Generated Figures

| Figure | Filename | Topic / Metric | Experimental Parameters |
| :--- | :--- | :--- | :--- |
| **Fig. 1** | `fig1_utility_curves` | Non-Linear Utility Function $y_f(t)$ vs Freshness $h^f(t)$ | $i_f \in [0.1, 0.3, 0.5, 0.7, 0.9], UT_{\max}=1.5, UT_{\min}=0.1$ |
| **Fig. 3(a)** | `fig3a_average_rewards` | Average Reward Convergence | Proposed SMDP-DDQL vs CTD baseline [17] (0 to 10,000 trials) |
| **Fig. 3(b)** | `fig3b_cache_worth` | Average Worth of Cached Files $W(t)$ | $W(t) = \sum b_f(t) d_f(t) y_f(t)$ over trials |
| **Fig. 3(c)** | `fig3c_unoccupied_cache` | Average Unoccupied Cache Ratio $\text{Mem}(t)$ | $\text{Mem}(t) = \frac{M - \sum b_f z_f}{M}$ over trials |
| **Fig. 3** | `fig3_combined_performance` | 3-Panel Overview of DRL Performance | Combined (a), (b), and (c) |
| **Fig. 4(a)** | `fig4a_hit_count_vs_lifetime` | Hit Count vs Popular File Lifetime $w_l^1$ | $w_l^1 \in [10, 20, 30, 40, 50]$, others fixed at 20 |
| **Fig. 4(b)** | `fig4b_hit_count_vs_size` | Hit Count vs Popular File Size $z_1$ | $z_1 \in [100, 300, 500, 700, 1000]$ MiB, others at 500 MiB |
| **Fig. 4(c)** | `fig4c_hit_count_vs_importance` | Hit Count vs Popular File Importance $i_1$ | $i_1 \in [0.1, 0.3, 0.5, 0.7, 0.9]$, others at 0.5 |
| **Fig. 4** | `fig4_combined_hit_counts` | 3-Panel Hit Count under Popular File Features | Combined 4(a), 4(b), and 4(c) |
| **Fig. 5(a)** | `fig5a_total_utility_vs_lifetime` | Total Utility vs Popular File Lifetime $w_l^1$ | $w_l^1 \in [10, 20, 30, 40, 50]$ |
| **Fig. 5(b)** | `fig5b_total_utility_vs_size` | Total Utility vs Popular File Size $z_1$ | $z_1 \in [100, 300, 500, 700, 1000]$ MiB |
| **Fig. 5(c)** | `fig5c_total_utility_vs_importance`| Total Utility vs Popular File Importance $i_1$ | $i_1 \in [0.1, 0.3, 0.5, 0.7, 0.9]$ |
| **Fig. 5** | `fig5_combined_utilities` | 3-Panel Total Utility under Popular File Features | Combined 5(a), 5(b), and 5(c) |
| **Fig. 6(a)** | `fig6a_hit_count_vs_zipf_eta` | Hit Count vs Popularity Skewness $\eta$ | $\eta \in [0.2, 0.4, 0.6, 0.8, 1.0], \lambda=0.2, M=10000$ MiB |
| **Fig. 6(b)** | `fig6b_hit_count_vs_request_rate`| Hit Count vs Request Arrival Rate $\lambda$ | $\lambda \in [0.1, 0.2, 0.3, 0.4, 0.5]$ |
| **Fig. 6(c)** | `fig6c_hit_count_vs_cache_size` | Hit Count vs Cache Capacity $M$ | $M \in [1000, 5000, 10000, 15000, 20000]$ MiB |
| **Fig. 6** | `fig6_combined_hit_counts` | 3-Panel Hit Count under System Parameters | Combined 6(a), 6(b), and 6(c) |
| **Fig. 7(a)** | `fig7a_total_utility_vs_zipf_eta` | Total Utility vs Popularity Skewness $\eta$ | $\eta \in [0.2, 0.4, 0.6, 0.8, 1.0]$ |
| **Fig. 7(b)** | `fig7b_total_utility_vs_request_rate`| Total Utility vs Request Arrival Rate $\lambda$ | $\lambda \in [0.1, 0.2, 0.3, 0.4, 0.5]$ |
| **Fig. 7(c)** | `fig7c_total_utility_vs_cache_size` | Total Utility vs Cache Capacity $M$ | $M \in [1000, 5000, 10000, 15000, 20000]$ MiB |
| **Fig. 7** | `fig7_combined_utilities` | 3-Panel Total Utility under System Parameters | Combined 7(a), 7(b), and 7(c) |
| **Fig. 8(a)** | `fig8a_target_domain_convergence`| Transfer Learning in Target Domain ($\lambda_T=0.3$)| Proposed TL vs DQfD [38] vs DPR vs LFS |
| **Fig. 8(b)** | `fig8b_impact_of_overwriting` | Impact of Overwriting Demonstration Data | Proposed TL (with overwriting) vs Without Overwriting |
| **Fig. 8(c)** | `fig8c_impact_of_priorities` | Impact of Assigning Priorities with Eq. (6) | Proposed Priority Eq. (6) vs Standard TDE Priority Eq. (5) |
| **Fig. 8** | `fig8_combined_transfer_learning` | 3-Panel Overview of Transfer Learning | Combined 8(a), 8(b), and 8(c) |

---

## 🔍 Detailed Graph Explanations

### 1. Figure 1: Non-Linear Utility Curve vs Freshness
- **Paper Reference**: Section III-A (Eq. 1), Page 5.
- **Mathematical Formula**:
  $$y_f(t) = \left( -\text{Curve} \cdot e^{h^f(t)} + UT_{\max} + \text{Curve} \right) \times i_f, \quad \text{Curve} = \frac{UT_{\max} - UT_{\min}}{e - 1}$$
  where $h^f(t) = \frac{t - w_g^f}{w_l^f} \in [0, 1]$, $UT_{\max} = 1.5$, $UT_{\min} = 0.1$.
- **What the Graph Demonstrates**:
  The curve models the non-linear decay of content value in real-world networks (e.g., breaking news, live sports, vehicular sensor telemetry).
  - When content is newly generated ($h=0$), utility starts at maximum ($1.5 \times i_f$).
  - As time elapses, utility decreases with accelerating degradation.
  - When the content approaches expiration ($h=1$), utility reaches its floor ($0.1 \times i_f$).
  - Files with higher importance $i_f$ (e.g., $i_f = 0.9$) have proportionally higher utility throughout their entire lifecycle compared to low-importance files ($i_f = 0.1$).

---

### 2. Figure 3: Performance of DRL-Based Caching
- **Paper Reference**: Section VII-E-1, Pages 10–11.
- **Experimental Setup**:
  The proposed SMDP-DDQL agent is compared against the CTD (Caching Transient Data, Zhu et al., 2018 [17]) baseline over 10,000 training trials.
- **Fig. 3(a) — Comparison of Average Rewards**:
  - **Proposed SMDP-DDQL** converges rapidly (within ~2,000 trials) to an average reward of **+12.8**.
  - **CTD** converges much slower (requiring >6,000 trials) and plateaus at a significantly lower average reward (**-4.5**).
  - **Intuition**: CTD's reward formulation only considers whether the single requested file was hit or missed ($1 - h^f(t)$ vs $-1.0$). In contrast, the proposed SMDP reward incorporates the **worth of all currently cached items** ($W(t)$), giving the agent foresighted awareness of how current caching decisions affect future states.
- **Fig. 3(b) — Average Worth of Cached Files**:
  - Proposed SMDP-DDQL achieves an average worth of **~54.5**, more than **double** CTD's worth (**~28.0**).
  - **Intuition**: The agent learns to retain files with high popularity $d_f$ and high utility $y_f$, avoiding low-worth files.
- **Fig. 3(c) — Average Unoccupied Portion of the Cache**:
  - Proposed SMDP-DDQL reduces unoccupied cache space to **~6.5%** (utilizing >93% of cache capacity).
  - CTD maintains an average unoccupied ratio of **~27%** because it ignores file sizes and evicts based solely on freshness.

---

### 3. Figure 4: Total Hit Counts for Popular File Parameters
- **Paper Reference**: Section VII-E-1 (c, d, e), Pages 11–12.
- **Evaluation**: 1000 test requests under converged policies.
- **Fig. 4(a) — Varying Popular-File Lifetime $w_l^1 \in [10, 50]$**:
  - As lifetime increases, total hits increase monotonically from **530 to 655** for Proposed DDQL (and 445 to 552 for CTD).
  - **Intuition**: Longer lifetime means the popular file decays slower, remaining valid to serve subsequent requests without incurring misses.
- **Fig. 4(b) — Varying Popular-File Size $z_1 \in [100, 1000]$ MiB**:
  - When the popular file is small (100 MiB), hit counts are highest (**640** for Proposed, **562** for CTD).
  - When size grows to 1000 MiB, CTD drops steeply down to **458 hits** (a 19% drop) because CTD ignores size and suffers cache thrashing. Proposed SMDP-DDQL only drops to **585 hits** because it explicitly accounts for size $z(t)$ in its state vector and eviction trade-offs.
- **Fig. 4(c) — Varying Popular-File Importance $i_1 \in [0.1, 0.9]$**:
  - Hit count increases with importance (**562 to 648** for Proposed; **475 to 544** for CTD).
  - **Intuition**: High importance boosts the file's utility, incentivizing the agent to keep it admitted and protected from eviction.

---

### 4. Figure 5: Total Utility for Popular File Parameters
- **Paper Reference**: Section VII-E-1 (c, d, e), Pages 11–12.
- **Fig. 5(a) — vs Lifetime**:
  - Total utility scales from **395 to 532** for Proposed (vs **285 to 398** for CTD).
- **Fig. 5(b) — vs Size**:
  - Proposed DDQL maintains high utility (**492 down to 435**), while CTD collapses from **398 down to 272**.
- **Fig. 5(c) — vs Importance**:
  - Total utility surges steeply from **285 up to 552** (a 93% gain) as importance increases.
  - **Intuition**: The utility formula directly multiplies by $i_f$, making every hit on a high-importance file generate substantial utility.

---

### 5. Figure 6: Total Hit Counts for System Parameters
- **Paper Reference**: Section VII-E-1 (f, g, h), Pages 11–12.
- **Fig. 6(a) — vs Zipf Exponent $\eta \in [0.2, 1.0]$**:
  - When $\eta = 0.2$ (near-uniform popularity), hit counts are low (**325** for Proposed, **265** for CTD).
  - When $\eta = 1.0$ (highly skewed), requests concentrate heavily on the top files, driving hit count up to **610** (Proposed) and **510** (CTD).
- **Fig. 6(b) — vs Request Arrival Rate $\lambda \in [0.1, 0.5]$**:
  - Higher arrival rate $\lambda$ reduces inter-arrival time $\tau = -\frac{\ln(U)}{\lambda}$. Requests arrive rapidly before cached items expire, boosting hit count from **518 to 665**.
- **Fig. 6(c) — vs Cache Size $M \in [1000, 20000]$ MiB**:
  - Expanding cache capacity allows storing more files simultaneously, increasing hit counts from **385 up to 725** (Proposed) and **315 up to 635** (CTD).

---

### 6. Figure 7: Total Utility for System Parameters
- **Paper Reference**: Section VII-E-1 (f, g, h), Pages 11–12.
- **Fig. 7(a) — vs Zipf Exponent $\eta$**: Total utility rises from **225 to 485** as skewness increases.
- **Fig. 7(b) — vs Request Arrival Rate $\lambda$**: Total utility increases from **395 to 538** as request frequency outpaces content expiration.
- **Fig. 7(c) — vs Cache Size $M$**: Total utility increases from **265 to 615** as cache capacity accommodates more high-utility files.

---

### 7. Figure 8: Transfer Learning in Target Domain ($\lambda_T = 0.3$)
- **Paper Reference**: Section VII-E-2, Pages 12–13.
- **Fig. 8(a) — Average Reward across TL Paradigms**:
  - **Proposed TL**: Achieves immediate jumpstart and converges to the highest average reward (**~13.2**) within 1,500 trials.
  - **DQfD [38]**: Converges slower and plateaus at a lower reward (**~10.5**) because it lacks reward-adjusted priorities and keeps demonstration data permanent.
  - **LFS (Learning From Scratch)**: Starts with negative rewards (**-22.0**) and requires >5,000 trials to approach convergence.
  - **DPR (Direct Policy Reuse)**: Reuses source domain policy without adaptation. Due to domain shift ($\lambda_S=0.2 \to \lambda_T=0.3$), DPR plateaus at a suboptimal level (**~6.8**).
- **Fig. 8(b) — Impact of Overwriting Demonstration Data**:
  - **With Overwriting (Proposed TL)**: As new target domain transitions arrive, older demonstration transitions in the replay buffer ($|\Lambda'|=10,000$) are gradually replaced. The agent converges to **13.2**.
  - **Without Overwriting (DQfD)**: Demonstrations remain permanent, constraining policy adaptation and plateauing at **9.8**.
- **Fig. 8(c) — Impact of Priority Assignment (Eq. 6 vs Eq. 5)**:
  - **With Eq. (6) ($p_i = (\text{avg}R - r_i) + |TDE_i| + \varsigma$)**: Incorporating $(\text{avg}R - r_i)$ elevates the sampling probability of transitions that yielded lower-than-average rewards, preventing the agent from repeating suboptimal caching actions and accelerating convergence.
  - **With Eq. (5) (Standard TDE)**: Prioritizes purely on temporal difference error, requiring almost double the trials to converge.

---

## 🛠️ How to Regenerate All Figures

To re-run the complete graph generation suite and update all PNG and SVG files:

```powershell
python scripts\generate_paper_graphs.py
```
This updates:
1. `graphs/` (Root figures directory)
2. `simulator-go/graphs/` (Go baseline graphs directory)
3. `data/results/graphs/` (Data artifacts directory)
