# SMDP EDGE CACHING EXPERIMENT: COMPREHENSIVE PERFORMANCE REPORT

> **Generated at**: `2026-10-26 16:08:30 UTC`  
> **Targeted Algorithms**: `Proposed (SMDP-DDQL)`, `CTD`, `Proposed TL`, `DQfD`, `LFS`, `DPR`  
> **Excluded Baselines**: `FIFO`, `LIFO`, `LRU`, `LFU`, `SIEVE` (omitted per user configuration)


## 1. Core Caching Performance (Proposed vs CTD)

| Metric | CTD Baseline | Proposed (SMDP-DDQL) | Relative Gain |
| :--- | :--- | :--- | :--- |
| **Hit Count** | `15791` | **`14262`** | **+-1529 hits** |
| **Hit Rate (%)** | `15.79%` | **`14.26%`** | **+-9.7%** |
| **Byte Hit Rate (%)** | `14.99%` | **`12.85%`** | **+-2.14%** |
| **Miss Count** | `84209` | **`85738`** | **--1529 misses** |
| **Miss Rate (%)** | `84.21%` | **`85.74%`** | **--1.53%** |
| **Evictions** | `57733` | **`7312`** | **-87.3%** |
| **Capacity Utilization (%)** | `98.19%` | **`99.44%`** | **1.25%** |
| **Total Utility** | `793.99` | **`754.40`** | **+-5.0%** |

## 2. Source Domain DRL (DDQL) Performance

- **Training Steps**: `1,000` steps
- **Final Moving Average Reward**: `-76.7211`
- **Greedy Policy Hit Rate**: `22.80%` (`114 / 500` requests)
- **Total Accumulated Utility**: `49.06`
- **Average Reward Per Step**: `-64.2591`


## 3. Target Domain Transfer Learning Performance (lambda_T = 0.3)

| Transfer Learning Method | Initial Reward (t=100) | Midpoint Reward | Final Reward | Peak Reward |
| :--- | :--- | :--- | :--- | :--- |
| **PROPOSED** | `-82.30` | `-76.35` | **`-71.50`** | `-71.50` |
| **DQFD** | `-83.33` | `-77.73` | **`-72.15`** | `-72.15` |
| **LFS** | `-80.97` | `-72.66` | **`-70.65`** | `-70.51` |
| **DPR** | `-70.62` | `-60.99` | **`-60.59`** | `-60.59` |
| **DQFD_OVERWRITE** | `-84.45` | `-72.33` | **`-68.82`** | `-68.82` |
| **DQFD_NEW_PRIO** | `-81.24` | `-71.62` | **`-67.97`** | `-67.97` |

## 4. Parameter Sensitivity Sweeps (Proposed vs CTD)


### Total hit count vs cache size (Fig. 6c)

| Cache size (MiB) | 5000 | 7500 | 10000 | 12500 | 15000 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `15782` | `15791` | `15791` | `15791` | `15791` |
| **Proposed (SMDP-DDQL)** | `9654` | `10222` | `11693` | `13665` | `14262` |

### Total utility vs cache size (Fig. 7c)

| Cache size (MiB) | 5000 | 7500 | 10000 | 12500 | 15000 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `793.63` | `793.99` | `793.99` | `793.99` | `793.99` |
| **Proposed (SMDP-DDQL)** | `556.68` | `590.74` | `657.97` | `733.91` | `754.4` |

### Total hit count vs request rate (Fig. 6b)

| Request rate lambda | 0.5 | 1.5 | 2.5 | 3.5 | 4.5 | 5.5 | 6.5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `29225` | `50016` | `59768` | `65680` | `69385` | `71591` | `72679` |
| **Proposed (SMDP-DDQL)** | `26942` | `46957` | `56940` | `62841` | `67118` | `69966` | `71936` |

### Total utility vs request rate (Fig. 7b)

| Request rate lambda | 0.5 | 1.5 | 2.5 | 3.5 | 4.5 | 5.5 | 6.5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `1463.22` | `2496.56` | `2991.78` | `3300.22` | `3496.22` | `3615.48` | `3675.79` |
| **Proposed (SMDP-DDQL)** | `1404.04` | `2412.81` | `2914.09` | `3222.35` | `3445.72` | `3607.88` | `3725.67` |

### Total hit count vs popularity skewness (Fig. 6a)

| Zipf eta | 0.2 | 0.4 | 0.6 | 0.8 | 1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `7265` | `7770` | `9232` | `11967` | `15791` |
| **Proposed (SMDP-DDQL)** | `5677` | `6138` | `7446` | `10293` | `14262` |

### Total utility vs popularity skewness (Fig. 7a)

| Zipf eta | 0.2 | 0.4 | 0.6 | 0.8 | 1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `372.76` | `394.53` | `453.26` | `593.09` | `793.99` |
| **Proposed (SMDP-DDQL)** | `328.28` | `349.29` | `404.07` | `548.77` | `754.4` |

### Total hit count vs popular-file lifetime (Fig. 4a)

| Popular file lifetime | 10 | 15 | 20 | 25 | 30 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `16580` | `18656` | `20140` | `21352` | `22382` |
| **Proposed (SMDP-DDQL)** | `14999` | `17241` | `18553` | `20104` | `20967` |

### Total utility vs popular-file lifetime (Fig. 5a)

| Popular file lifetime | 10 | 15 | 20 | 25 | 30 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `829.25` | `933.24` | `1007.52` | `1068.16` | `1119.69` |
| **Proposed (SMDP-DDQL)** | `750.2` | `862.49` | `928.17` | `1005.76` | `1048.94` |

### Total hit count vs popular-file size (Fig. 4b)

| Popular file size (MiB) | 100 | 200 | 500 | 700 | 1000 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `20140` | `20140` | `20140` | `20140` | `20140` |
| **Proposed (SMDP-DDQL)** | `18953` | `18786` | `18553` | `16568` | `16398` |

### Total utility vs popular-file size (Fig. 5b)

| Popular file size (MiB) | 100 | 200 | 500 | 700 | 1000 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `1007.52` | `1007.52` | `1007.52` | `1007.52` | `1007.52` |
| **Proposed (SMDP-DDQL)** | `948.17` | `939.82` | `928.17` | `828.92` | `820.42` |

### Total hit count vs popular-file importance (Fig. 4c)

| Popular file importance | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `20140` | `20140` | `20140` | `20140` | `20140` |
| **Proposed (SMDP-DDQL)** | `9784` | `18200` | `18553` | `20072` | `20072` |

### Total utility vs popular-file importance (Fig. 5c)

| Popular file importance | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CTD** | `591.1` | `799.31` | `1007.52` | `1215.72` | `1423.93` |
| **Proposed (SMDP-DDQL)** | `384.42` | `704.97` | `928.17` | `1212.32` | `1420.53` |

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
