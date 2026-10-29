# SMDP EDGE CACHING EXPERIMENT: COMPREHENSIVE PERFORMANCE REPORT

> **Generated at**: `2026-10-29 19:05:27 UTC`  
> **Targeted Algorithms**: `Proposed (SMDP-DDQL)`, `CTD`, `Proposed TL`, `DQfD`, `LFS`, `DPR`  
> **Excluded Baselines**: `FIFO`, `LIFO`, `LRU`, `LFU`, `SIEVE` (omitted per user configuration)


## 1. Core Caching Performance (Proposed vs CTD)

| Metric | CTD Baseline | Proposed (SMDP-DDQL) | Relative Gain |
| :--- | :--- | :--- | :--- |
| **Hit Count** | `278` | **`298`** | **+20 hits** |
| **Hit Rate (%)** | `55.60%` | **`59.60%`** | **+7.2%** |
| **Byte Hit Rate (%)** | `54.16%` | **`56.48%`** | **+2.32%** |
| **Miss Count** | `222` | **`202`** | **-20 misses** |
| **Miss Rate (%)** | `44.40%` | **`40.40%`** | **-4.00%** |
| **Evictions** | `204` | **`115`** | **-43.6%** |
| **Capacity Utilization (%)** | `94.91%` | **`98.22%`** | **3.31%** |
| **Total Utility** | `30.88` | **`33.50`** | **+8.5%** |

## 2. Source Domain DRL (DDQL) Performance

- **Training Steps**: `1,000` steps
- **Final Moving Average Reward**: `-78.1040`
- **Greedy Policy Hit Rate**: `22.80%` (`114 / 500` requests)
- **Total Accumulated Utility**: `49.06`
- **Average Reward Per Step**: `-64.2591`


## 3. Target Domain Transfer Learning Performance (lambda_T = 0.3)

| Transfer Learning Method | Initial Reward (t=100) | Midpoint Reward | Final Reward | Peak Reward |
| :--- | :--- | :--- | :--- | :--- |
| **PROPOSED** | `-81.28` | `-78.13` | **`-71.63`** | `-71.63` |
| **DQFD** | `-84.27` | `-76.55` | **`-70.82`** | `-70.82` |
| **LFS** | `-81.09` | `-84.89` | **`-82.56`** | `-80.59` |
| **DPR** | `-70.62` | `-60.99` | **`-60.59`** | `-60.59` |
| **DQFD_OVERWRITE** | `-84.05` | `-71.19` | **`-68.51`** | `-68.51` |
| **DQFD_NEW_PRIO** | `-81.77` | `-73.98` | **`-69.37`** | `-69.37` |

## 4. Parameter Sensitivity Sweeps (Proposed vs CTD)


### Total hit count vs cache size (Fig. 6c)

| Cache size (MiB) | 5000 | 7500 | 10000 | 12500 | 15000 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `175` | `230` | `278` | `307` | `333` |
| **Proposed (SMDP-DDQL)** | `211` | `258` | `298` | `320` | `333` |

### Total utility vs cache size (Fig. 7c)

| Cache size (MiB) | 5000 | 7500 | 10000 | 12500 | 15000 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `20.9` | `25.77` | `30.88` | `33.51` | `35.19` |
| **Proposed (SMDP-DDQL)** | `26.26` | `29.43` | `33.5` | `34.92` | `35.59` |

### Total hit count vs request rate (Fig. 6b)

| Request rate lambda | 0.5 | 1.5 | 2.5 | 3.5 | 4.5 | 5.5 | 6.5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `132` | `233` | `272` | `278` | `278` | `278` | `278` |
| **Proposed (SMDP-DDQL)** | `132` | `233` | `268` | `282` | `290` | `295` | `302` |

### Total utility vs request rate (Fig. 7b)

| Request rate lambda | 0.5 | 1.5 | 2.5 | 3.5 | 4.5 | 5.5 | 6.5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `7.73` | `16.41` | `21.89` | `25.68` | `29.34` | `32.47` | `35.24` |
| **Proposed (SMDP-DDQL)** | `7.73` | `16.41` | `21.9` | `26.24` | `31.4` | `34.97` | `38.75` |

### Total hit count vs popularity skewness (Fig. 6a)

| Zipf eta | 0.2 | 0.4 | 0.6 | 0.8 | 1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `168` | `159` | `204` | `239` | `278` |
| **Proposed (SMDP-DDQL)** | `196` | `182` | `235` | `253` | `298` |

### Total utility vs popularity skewness (Fig. 7a)

| Zipf eta | 0.2 | 0.4 | 0.6 | 0.8 | 1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `22.33` | `22.51` | `21.81` | `25.94` | `30.88` |
| **Proposed (SMDP-DDQL)** | `28.86` | `27.16` | `27.87` | `28.6` | `33.5` |

### Total hit count vs popular-file lifetime (Fig. 4a)

| Popular file lifetime | 10 | 15 | 20 | 25 | 30 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `294` | `298` | `301` | `302` | `304` |
| **Proposed (SMDP-DDQL)** | `311` | `313` | `312` | `306` | `316` |

### Total utility vs popular-file lifetime (Fig. 5a)

| Popular file lifetime | 10 | 15 | 20 | 25 | 30 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `31.68` | `33.33` | `35.47` | `37.52` | `39.45` |
| **Proposed (SMDP-DDQL)** | `32.85` | `35.13` | `37.31` | `38.73` | `41.2` |

### Total hit count vs popular-file size (Fig. 4b)

| Popular file size (MiB) | 100 | 200 | 500 | 700 | 1000 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `301` | `301` | `301` | `298` | `298` |
| **Proposed (SMDP-DDQL)** | `312` | `312` | `312` | `293` | `288` |

### Total utility vs popular-file size (Fig. 5b)

| Popular file size (MiB) | 100 | 200 | 500 | 700 | 1000 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `35.47` | `35.47` | `35.47` | `34.68` | `34.68` |
| **Proposed (SMDP-DDQL)** | `37.31` | `37.31` | `37.31` | `36.36` | `36.11` |

### Total hit count vs popular-file importance (Fig. 4c)

| Popular file importance | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `301` | `301` | `301` | `301` | `301` |
| **Proposed (SMDP-DDQL)** | `317` | `314` | `312` | `319` | `317` |

### Total utility vs popular-file importance (Fig. 5c)

| Popular file importance | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `24.76` | `30.12` | `35.47` | `40.83` | `46.19` |
| **Proposed (SMDP-DDQL)** | `26.17` | `31.36` | `37.31` | `43.26` | `48.75` |

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
