# SMDP EDGE CACHING EXPERIMENT: COMPREHENSIVE PERFORMANCE REPORT

> **Generated at**: `2026-11-09 18:17:10 UTC`  
> **Targeted Algorithms**: `Proposed (SMDP-DDQL)`, `CTD`, `Proposed TL`, `DQfD`, `LFS`, `DPR`  
> **Excluded Baselines**: `FIFO`, `LIFO`, `LRU`, `LFU`, `SIEVE` (omitted per user configuration)


## 1. Core Caching Performance (Proposed vs CTD)

| Metric | CTD Baseline | Proposed (SMDP-DDQL) | Relative Gain |
| :--- | :--- | :--- | :--- |

## 2. Source Domain DRL (DDQL) Performance

- **Training Steps**: `3,000` steps
- **Final Moving Average Reward**: `10.1333`
- **Greedy Policy Hit Rate**: `33.00%` (`330 / 1000` requests)
- **Total Accumulated Utility**: `171.63`
- **Average Reward Per Step**: `14.9911`


## 3. Target Domain Transfer Learning Performance (lambda_T = 0.3)

| Transfer Learning Method | Initial Reward (t=100) | Midpoint Reward | Final Reward | Peak Reward |
| :--- | :--- | :--- | :--- | :--- |
| **PROPOSED** | `6.03` | `18.65` | **`19.58`** | `19.58` |
| **DQFD** | `5.88` | `19.53` | **`20.15`** | `20.15` |
| **LFS** | `4.95` | `17.98` | **`19.32`** | `19.32` |
| **DPR** | `8.46` | `21.36` | **`21.36`** | `21.59` |
| **DQFD_OVERWRITE** | `4.92` | `18.93` | **`19.84`** | `19.84` |
| **DQFD_NEW_PRIO** | `6.35` | `18.17` | **`19.42`** | `19.42` |

## 4. Parameter Sensitivity Sweeps (Proposed vs CTD)


### Total hit count vs cache size (Fig. 6c)

| Cache size (MiB) | 5000 | 7500 | 10000 | 12500 | 15000 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `370` | `480` | `568` | `646` | `690` |

### Total utility vs cache size (Fig. 7c)

| Cache size (MiB) | 5000 | 7500 | 10000 | 12500 | 15000 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `267.57` | `330.51` | `361.94` | `356.96` | `326.79` |

### Total hit count vs request rate (Fig. 6b)

| Request rate lambda | 0.5 | 1.5 | 2.5 | 3.5 | 4.5 | 5.5 | 6.5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `274` | `477` | `550` | `568` | `568` | `568` | `568` |

### Total utility vs request rate (Fig. 7b)

| Request rate lambda | 0.5 | 1.5 | 2.5 | 3.5 | 4.5 | 5.5 | 6.5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `131.31` | `225.82` | `275.42` | `324.3` | `352.35` | `369.49` | `380.61` |

### Total hit count vs popularity skewness (Fig. 6a)

| Zipf eta | 0.2 | 0.4 | 0.6 | 0.8 | 1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `335` | `355` | `424` | `495` | `568` |

### Total utility vs popularity skewness (Fig. 7a)

| Zipf eta | 0.2 | 0.4 | 0.6 | 0.8 | 1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `236.3` | `242.96` | `279.77` | `310.66` | `361.94` |

### Total hit count vs popular-file lifetime (Fig. 4a)

| Popular file lifetime | 10 | 15 | 20 | 25 | 30 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `625` | `629` | `639` | `634` | `637` |

### Total utility vs popular-file lifetime (Fig. 5a)

| Popular file lifetime | 10 | 15 | 20 | 25 | 30 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `358.23` | `363.25` | `369.95` | `373.22` | `369.59` |

### Total hit count vs popular-file size (Fig. 4b)

| Popular file size (MiB) | 100 | 200 | 500 | 700 | 1000 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `639` | `639` | `639` | `617` | `617` |

### Total utility vs popular-file size (Fig. 5b)

| Popular file size (MiB) | 100 | 200 | 500 | 700 | 1000 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `369.95` | `369.95` | `369.95` | `370.56` | `370.56` |

### Total hit count vs popular-file importance (Fig. 4c)

| Popular file importance | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `639` | `639` | `639` | `639` | `639` |

### Total utility vs popular-file importance (Fig. 5c)

| Popular file importance | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `271.11` | `320.53` | `369.95` | `419.36` | `468.78` |

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
