# SMDP EDGE CACHING EXPERIMENT: COMPREHENSIVE PERFORMANCE REPORT

> **Generated at**: `2026-10-23 07:33:41 UTC`  
> **Targeted Algorithms**: `Proposed (SMDP-DDQL)`, `CTD`, `Proposed TL`, `DQfD`, `LFS`, `DPR`  
> **Excluded Baselines**: `FIFO`, `LIFO`, `LRU`, `LFU`, `SIEVE` (omitted per user configuration)


## 1. Core Caching Performance (Proposed vs CTD)

| Metric | CTD Baseline | Proposed (SMDP-DDQL) | Relative Gain |
| :--- | :--- | :--- | :--- |
| **Hit Count** | `793` | **`616`** | **+-177 hits** |
| **Hit Rate (%)** | `15.86%` | **`12.32%`** | **+-22.3%** |
| **Byte Hit Rate (%)** | `15.09%` | **`10.25%`** | **+-4.84%** |
| **Miss Count** | `4207` | **`4384`** | **--177 misses** |
| **Miss Rate (%)** | `84.14%` | **`87.68%`** | **--3.54%** |
| **Evictions** | `3235` | **`211`** | **-93.5%** |
| **Capacity Utilization (%)** | `98.50%` | **`98.16%`** | **-0.34%** |
| **Total Utility** | `40.08` | **`34.43`** | **+-14.1%** |

## 2. Source Domain DRL (DDQL) Performance

- **Training Steps**: `5,000` steps
- **Final Moving Average Reward**: `10.7081`
- **Greedy Policy Hit Rate**: `21.60%` (`216 / 1000` requests)
- **Total Accumulated Utility**: `97.98`
- **Average Reward Per Step**: `11.7394`


## 3. Target Domain Transfer Learning Performance (lambda_T = 0.3)

| Transfer Learning Method | Initial Reward (t=100) | Midpoint Reward | Final Reward | Peak Reward |
| :--- | :--- | :--- | :--- | :--- |
| **PROPOSED** | `-27.31` | `16.24` | **`17.30`** | `17.30` |
| **DQFD** | `-32.26` | `16.34` | **`17.42`** | `17.42` |
| **LFS** | `-42.18` | `15.98` | **`17.21`** | `17.21` |
| **DPR** | `-5.44` | `18.42` | **`18.56`** | `18.57` |
| **DQFD_OVERWRITE** | `-18.31` | `16.60` | **`17.54`** | `17.54` |
| **DQFD_NEW_PRIO** | `-20.73` | `16.38` | **`17.34`** | `17.34` |

## 4. Parameter Sensitivity Sweeps (Proposed vs CTD)


### Total hit count vs cache size (Fig. 6c)

| Cache size (MiB) | 5000 | 7500 | 10000 | 12500 | 15000 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `793` | `793` | `793` | `793` | `793` |
| **Proposed (SMDP-DDQL)** | `509` | `522` | `616` | `686` | `718` |

### Total utility vs cache size (Fig. 7c)

| Cache size (MiB) | 5000 | 7500 | 10000 | 12500 | 15000 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `40.08` | `40.08` | `40.08` | `40.08` | `40.08` |
| **Proposed (SMDP-DDQL)** | `29.54` | `30.43` | `34.43` | `37.06` | `38.18` |

### Total hit count vs request rate (Fig. 6b)

| Request rate lambda | 0.5 | 1.5 | 2.5 | 3.5 | 4.5 | 5.5 | 6.5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `1442` | `2475` | `2843` | `2928` | `2935` | `2935` | `2935` |
| **Proposed (SMDP-DDQL)** | `1210` | `2107` | `2695` | `2882` | `3033` | `3054` | `3161` |

### Total utility vs request rate (Fig. 7b)

| Request rate lambda | 0.5 | 1.5 | 2.5 | 3.5 | 4.5 | 5.5 | 6.5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `73.2` | `128.22` | `150.83` | `158.95` | `162.87` | `166.0` | `168.77` |
| **Proposed (SMDP-DDQL)** | `65.77` | `116.1` | `148.51` | `163.81` | `175.18` | `183.32` | `192.9` |

### Total hit count vs popularity skewness (Fig. 6a)

| Zipf eta | 0.2 | 0.4 | 0.6 | 0.8 | 1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `374` | `388` | `436` | `611` | `793` |
| **Proposed (SMDP-DDQL)** | `249` | `238` | `278` | `418` | `616` |

### Total utility vs popularity skewness (Fig. 7a)

| Zipf eta | 0.2 | 0.4 | 0.6 | 0.8 | 1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `19.42` | `19.46` | `21.94` | `30.86` | `40.08` |
| **Proposed (SMDP-DDQL)** | `15.01` | `14.33` | `16.46` | `24.19` | `34.43` |

### Total hit count vs popular-file lifetime (Fig. 4a)

| Popular file lifetime | 10 | 15 | 20 | 25 | 30 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `826` | `934` | `1011` | `1068` | `1117` |
| **Proposed (SMDP-DDQL)** | `724` | `832` | `909` | `966` | `1015` |

### Total utility vs popular-file lifetime (Fig. 5a)

| Popular file lifetime | 10 | 15 | 20 | 25 | 30 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `41.55` | `47.14` | `51.07` | `53.96` | `56.44` |
| **Proposed (SMDP-DDQL)** | `36.45` | `42.04` | `45.97` | `48.86` | `51.34` |

### Total hit count vs popular-file size (Fig. 4b)

| Popular file size (MiB) | 100 | 200 | 500 | 700 | 1000 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `1011` | `1011` | `1011` | `1011` | `1011` |
| **Proposed (SMDP-DDQL)** | `914` | `914` | `909` | `905` | `906` |

### Total utility vs popular-file size (Fig. 5b)

| Popular file size (MiB) | 100 | 200 | 500 | 700 | 1000 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `51.07` | `51.07` | `51.07` | `51.07` | `51.07` |
| **Proposed (SMDP-DDQL)** | `46.22` | `46.22` | `45.97` | `45.77` | `45.82` |

### Total hit count vs popular-file importance (Fig. 4c)

| Popular file importance | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `1011` | `1011` | `1011` | `1011` | `1011` |
| **Proposed (SMDP-DDQL)** | `583` | `902` | `909` | `1008` | `1008` |

### Total utility vs popular-file importance (Fig. 5c)

| Popular file importance | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `28.97` | `40.02` | `51.07` | `62.11` | `73.16` |
| **Proposed (SMDP-DDQL)** | `21.81` | `34.57` | `45.97` | `61.96` | `73.01` |

## 5. Visual Assets & Generated Figures Catalog

| Figure | Filename | Topic / Description | Artifact Links |
| :--- | :--- | :--- | :--- |
| **Fig. 1** | `fig1_utility_curves` | Non-linear utility curve y_f(t) vs freshness h^f(t) across importance levels | [PNG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig1_utility_curves.png) \| [SVG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig1_utility_curves.svg) |
| **Fig. 3(a)** | `fig3a_average_rewards` | Comparison of average rewards during DRL convergence (Proposed vs CTD) | [PNG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig3a_average_rewards.png) \| [SVG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig3a_average_rewards.svg) |
| **Fig. 3(b)** | `fig3b_cache_worth` | Average worth of cached files W(t) over trials | [PNG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig3b_cache_worth.png) \| [SVG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig3b_cache_worth.svg) |
| **Fig. 3(c)** | `fig3c_unoccupied_cache` | Average unoccupied cache proportion Mem(t) over trials | [PNG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig3c_unoccupied_cache.png) \| [SVG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig3c_unoccupied_cache.svg) |
| **Fig. 3** | `fig3_combined_performance` | 3-Panel overview of DRL performance (Reward, Worth, Unoccupied Cache) | [PNG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig3_combined_performance.png) \| [SVG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig3_combined_performance.svg) |
| **Fig. 4(a)** | `fig4a_hit_count_vs_lifetime` | Total hit count vs popular-file lifetime (Proposed vs CTD) | [PNG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig4a_hit_count_vs_lifetime.png) \| [SVG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig4a_hit_count_vs_lifetime.svg) |
| **Fig. 4(b)** | `fig4b_hit_count_vs_size` | Total hit count vs popular-file size (Proposed vs CTD) | [PNG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig4b_hit_count_vs_size.png) \| [SVG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig4b_hit_count_vs_size.svg) |
| **Fig. 4(c)** | `fig4c_hit_count_vs_importance` | Total hit count vs popular-file importance (Proposed vs CTD) | [PNG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig4c_hit_count_vs_importance.png) \| [SVG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig4c_hit_count_vs_importance.svg) |
| **Fig. 4** | `fig4_combined_hit_counts` | 3-Panel hit count under varying popular-file features | [PNG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig4_combined_hit_counts.png) \| [SVG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig4_combined_hit_counts.svg) |
| **Fig. 5(a)** | `fig5a_total_utility_vs_lifetime` | Total utility vs popular-file lifetime (Proposed vs CTD) | [PNG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig5a_total_utility_vs_lifetime.png) \| [SVG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig5a_total_utility_vs_lifetime.svg) |
| **Fig. 5(b)** | `fig5b_total_utility_vs_size` | Total utility vs popular-file size (Proposed vs CTD) | [PNG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig5b_total_utility_vs_size.png) \| [SVG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig5b_total_utility_vs_size.svg) |
| **Fig. 5(c)** | `fig5c_total_utility_vs_importance` | Total utility vs popular-file importance (Proposed vs CTD) | [PNG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig5c_total_utility_vs_importance.png) \| [SVG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig5c_total_utility_vs_importance.svg) |
| **Fig. 5** | `fig5_combined_utilities` | 3-Panel total utility under varying popular-file features | [PNG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig5_combined_utilities.png) \| [SVG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig5_combined_utilities.svg) |
| **Fig. 6(a)** | `fig6a_hit_count_vs_zipf_eta` | Total hit count vs Zipf skewness eta (Proposed vs CTD) | [PNG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig6a_hit_count_vs_zipf_eta.png) \| [SVG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig6a_hit_count_vs_zipf_eta.svg) |
| **Fig. 6(b)** | `fig6b_hit_count_vs_request_rate` | Total hit count vs request arrival rate lambda (Proposed vs CTD) | [PNG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig6b_hit_count_vs_request_rate.png) \| [SVG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig6b_hit_count_vs_request_rate.svg) |
| **Fig. 6(c)** | `fig6c_hit_count_vs_cache_size` | Total hit count vs cache capacity M (Proposed vs CTD) | [PNG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig6c_hit_count_vs_cache_size.png) \| [SVG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig6c_hit_count_vs_cache_size.svg) |
| **Fig. 6** | `fig6_combined_hit_counts` | 3-Panel hit count under varying system parameters | [PNG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig6_combined_hit_counts.png) \| [SVG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig6_combined_hit_counts.svg) |
| **Fig. 7(a)** | `fig7a_total_utility_vs_zipf_eta` | Total utility vs Zipf skewness eta (Proposed vs CTD) | [PNG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig7a_total_utility_vs_zipf_eta.png) \| [SVG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig7a_total_utility_vs_zipf_eta.svg) |
| **Fig. 7(b)** | `fig7b_total_utility_vs_request_rate` | Total utility vs request arrival rate lambda (Proposed vs CTD) | [PNG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig7b_total_utility_vs_request_rate.png) \| [SVG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig7b_total_utility_vs_request_rate.svg) |
| **Fig. 7(c)** | `fig7c_total_utility_vs_cache_size` | Total utility vs cache capacity M (Proposed vs CTD) | [PNG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig7c_total_utility_vs_cache_size.png) \| [SVG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig7c_total_utility_vs_cache_size.svg) |
| **Fig. 7** | `fig7_combined_utilities` | 3-Panel total utility under varying system parameters | [PNG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig7_combined_utilities.png) \| [SVG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig7_combined_utilities.svg) |
| **Fig. 8(a)** | `fig8a_target_domain_convergence` | Target domain transfer learning convergence (Proposed TL vs DQfD vs LFS vs DPR) | [PNG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig8a_target_domain_convergence.png) \| [SVG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig8a_target_domain_convergence.svg) |
| **Fig. 8(b)** | `fig8b_impact_of_overwriting` | Impact of overwriting demonstration data (With vs Without overwriting) | [PNG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig8b_impact_of_overwriting.png) \| [SVG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig8b_impact_of_overwriting.svg) |
| **Fig. 8(c)** | `fig8c_impact_of_priorities` | Impact of assigning priorities with proposed Eq. (6) vs standard Eq. (5) | [PNG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig8c_impact_of_priorities.png) \| [SVG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig8c_impact_of_priorities.svg) |
| **Fig. 8** | `fig8_combined_transfer_learning` | 3-Panel overview of transfer learning performance in target domain | [PNG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig8_combined_transfer_learning.png) \| [SVG](file:///C:/Users/chsan/OneDrive/Desktop/smdp-edge-caching-framework/graphs/fig8_combined_transfer_learning.svg) |
