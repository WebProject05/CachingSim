# SMDP EDGE CACHING EXPERIMENT: COMPREHENSIVE THREE-WAY PERFORMANCE REPORT

> **Generated at**: `2026-11-14 17:56:49 UTC`  
> **Targeted Algorithms**: `CTD (Baseline)`, `Paper (Niknia '24 DDQL)`, `AdaLife-Pro (Ours)`  
> **Excluded Baselines**: `FIFO`, `LIFO`, `LRU`, `LFU`, `SIEVE` (omitted per user configuration)


## 1. Three-Way Caching Performance (CTD vs Paper DDQL vs AdaLife-Pro)

| Metric | CTD Baseline | Paper (Niknia '24) | AdaLife-Pro (Ours) | Gain vs Paper | Gain vs CTD |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hit Count** | `332` | `404` | **`697`** | **+293 hits** | **+365 hits** |
| **Hit Rate (%)** | `13.28%` | `16.16%` | **`27.88%`** | **+11.72%** | **+14.60%** |
| **Byte Hit Rate (%)** | `12.51%` | `11.22%` | **`21.51%`** | **+10.29%** | **+9.00%** |
| **Miss Count** | `2168` | `2096` | **`1803`** | **-293 misses** | **-365 misses** |
| **Miss Rate (%)** | `86.72%` | `83.84%` | **`72.12%`** | **-11.72%** | **-14.60%** |
| **Evictions** | `2159` | `1631` | **`696`** | **-935 (-57.3%)** | **-1463 (-67.8%)** |
| **Capacity Utilization (%)** | `94.88%` | `93.09%` | **`89.14%`** | **-3.95%** | **-5.74%** |
| **Total Utility** | `246.35` | `321.30` | **`571.61`** | **+77.9%** | **+132.0%** |

## 2. Source Domain DRL (DDQL) Performance

- **Training Steps**: `2,500` steps
- **Final Moving Average Reward**: `1.9707`
- **Greedy Policy Hit Rate**: `11.36%` (`284 / 2500` requests)
- **Total Accumulated Utility**: `138.87`
- **Average Reward Per Step**: `5.1658`


## 3. Target Domain Transfer Learning Performance (lambda_T = 0.3)

| Method | Initial Reward (t=100) | Midpoint Reward | Final Reward | Peak Reward |
| :--- | :--- | :--- | :--- | :--- |
| **Proposed TL (AdaLife-Pro)** | `0.2305` | `2.5215` | `4.5630` | **`4.5630`** |
| **Standard DQfD (Eq. 5)** | `-0.1434` | `2.1520` | `4.1875` | **`5.5401`** |
| **Learning From Scratch (LFS)** | `0.9640` | `2.9129` | `4.6193` | **`4.6193`** |
| **Direct Policy Reuse (DPR)** | `2.9389` | `6.6728` | `7.3309` | **`8.0699`** |

## 4. Research Evaluation Figures (PNG + SVG)

- **Fig. 1**: Non-linear utility curve y_f(t) vs freshness h^f(t) across importance levels  
  *Formats*: [fig1_utility_curves.png](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig1_utility_curves.png) | [fig1_utility_curves.svg](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig1_utility_curves.svg)
- **Fig. 3(a)**: Average reward convergence (AdaLife-Pro vs Paper DDQL)  
  *Formats*: [fig3a_average_rewards.png](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig3a_average_rewards.png) | [fig3a_average_rewards.svg](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig3a_average_rewards.svg)
- **Fig. 3(b)**: Average worth of cached files W(t) over trials  
  *Formats*: [fig3b_cache_worth.png](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig3b_cache_worth.png) | [fig3b_cache_worth.svg](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig3b_cache_worth.svg)
- **Fig. 3(c)**: Average unoccupied cache proportion Mem(t) over trials  
  *Formats*: [fig3c_unoccupied_cache.png](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig3c_unoccupied_cache.png) | [fig3c_unoccupied_cache.svg](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig3c_unoccupied_cache.svg)
- **Fig. 3**: 3-Panel overview of DRL performance (Reward, Worth, Unoccupied Cache)  
  *Formats*: [fig3_combined_performance.png](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig3_combined_performance.png) | [fig3_combined_performance.svg](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig3_combined_performance.svg)
- **Fig. 4(a)**: Total hit count vs popular-file lifetime (AdaLife-Pro vs Paper DDQL)  
  *Formats*: [fig4a_hit_count_vs_lifetime.png](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig4a_hit_count_vs_lifetime.png) | [fig4a_hit_count_vs_lifetime.svg](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig4a_hit_count_vs_lifetime.svg)
- **Fig. 4(b)**: Total hit count vs popular-file size (AdaLife-Pro vs Paper DDQL)  
  *Formats*: [fig4b_hit_count_vs_size.png](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig4b_hit_count_vs_size.png) | [fig4b_hit_count_vs_size.svg](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig4b_hit_count_vs_size.svg)
- **Fig. 4(c)**: Total hit count vs popular-file importance (AdaLife-Pro vs Paper DDQL)  
  *Formats*: [fig4c_hit_count_vs_importance.png](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig4c_hit_count_vs_importance.png) | [fig4c_hit_count_vs_importance.svg](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig4c_hit_count_vs_importance.svg)
- **Fig. 4**: 3-Panel hit count under varying popular-file features  
  *Formats*: [fig4_combined_hit_counts.png](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig4_combined_hit_counts.png) | [fig4_combined_hit_counts.svg](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig4_combined_hit_counts.svg)
- **Fig. 5(a)**: Total utility vs popular-file lifetime (AdaLife-Pro vs Paper DDQL)  
  *Formats*: [fig5a_total_utility_vs_lifetime.png](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig5a_total_utility_vs_lifetime.png) | [fig5a_total_utility_vs_lifetime.svg](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig5a_total_utility_vs_lifetime.svg)
- **Fig. 5(b)**: Total utility vs popular-file size (AdaLife-Pro vs Paper DDQL)  
  *Formats*: [fig5b_total_utility_vs_size.png](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig5b_total_utility_vs_size.png) | [fig5b_total_utility_vs_size.svg](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig5b_total_utility_vs_size.svg)
- **Fig. 5(c)**: Total utility vs popular-file importance (AdaLife-Pro vs Paper DDQL)  
  *Formats*: [fig5c_total_utility_vs_importance.png](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig5c_total_utility_vs_importance.png) | [fig5c_total_utility_vs_importance.svg](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig5c_total_utility_vs_importance.svg)
- **Fig. 5**: 3-Panel total utility under varying popular-file features  
  *Formats*: [fig5_combined_utilities.png](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig5_combined_utilities.png) | [fig5_combined_utilities.svg](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig5_combined_utilities.svg)
- **Fig. 6(a)**: Total hit count vs Zipf skewness eta (AdaLife-Pro vs Paper DDQL)  
  *Formats*: [fig6a_hit_count_vs_zipf_eta.png](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig6a_hit_count_vs_zipf_eta.png) | [fig6a_hit_count_vs_zipf_eta.svg](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig6a_hit_count_vs_zipf_eta.svg)
- **Fig. 6(b)**: Total hit count vs request arrival rate lambda (AdaLife-Pro vs Paper DDQL)  
  *Formats*: [fig6b_hit_count_vs_request_rate.png](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig6b_hit_count_vs_request_rate.png) | [fig6b_hit_count_vs_request_rate.svg](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig6b_hit_count_vs_request_rate.svg)
- **Fig. 6(c)**: Total hit count vs cache capacity M (AdaLife-Pro vs Paper DDQL)  
  *Formats*: [fig6c_hit_count_vs_cache_size.png](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig6c_hit_count_vs_cache_size.png) | [fig6c_hit_count_vs_cache_size.svg](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig6c_hit_count_vs_cache_size.svg)
- **Fig. 6**: 3-Panel hit count under varying system parameters  
  *Formats*: [fig6_combined_hit_counts.png](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig6_combined_hit_counts.png) | [fig6_combined_hit_counts.svg](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig6_combined_hit_counts.svg)
- **Fig. 7(a)**: Total utility vs Zipf skewness eta (AdaLife-Pro vs Paper DDQL)  
  *Formats*: [fig7a_total_utility_vs_zipf_eta.png](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig7a_total_utility_vs_zipf_eta.png) | [fig7a_total_utility_vs_zipf_eta.svg](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig7a_total_utility_vs_zipf_eta.svg)
- **Fig. 7(b)**: Total utility vs request arrival rate lambda (AdaLife-Pro vs Paper DDQL)  
  *Formats*: [fig7b_total_utility_vs_request_rate.png](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig7b_total_utility_vs_request_rate.png) | [fig7b_total_utility_vs_request_rate.svg](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig7b_total_utility_vs_request_rate.svg)
- **Fig. 7(c)**: Total utility vs cache capacity M (AdaLife-Pro vs Paper DDQL)  
  *Formats*: [fig7c_total_utility_vs_cache_size.png](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig7c_total_utility_vs_cache_size.png) | [fig7c_total_utility_vs_cache_size.svg](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig7c_total_utility_vs_cache_size.svg)
- **Fig. 7**: 3-Panel total utility under varying system parameters  
  *Formats*: [fig7_combined_utilities.png](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig7_combined_utilities.png) | [fig7_combined_utilities.svg](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig7_combined_utilities.svg)
- **Fig. 8(a)**: Target domain transfer learning convergence (AdaLife-Pro TL vs Paper DDQL TL vs LFS vs DPR)  
  *Formats*: [fig8a_target_domain_convergence.png](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig8a_target_domain_convergence.png) | [fig8a_target_domain_convergence.svg](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig8a_target_domain_convergence.svg)
- **Fig. 8(b)**: Impact of overwriting demonstration data (With vs Without overwriting)  
  *Formats*: [fig8b_impact_of_overwriting.png](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig8b_impact_of_overwriting.png) | [fig8b_impact_of_overwriting.svg](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig8b_impact_of_overwriting.svg)
- **Fig. 8(c)**: Impact of assigning priorities with proposed Eq. (6) vs standard Eq. (5)  
  *Formats*: [fig8c_impact_of_priorities.png](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig8c_impact_of_priorities.png) | [fig8c_impact_of_priorities.svg](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig8c_impact_of_priorities.svg)
- **Fig. 8**: 3-Panel overview of transfer learning performance in target domain  
  *Formats*: [fig8_combined_transfer_learning.png](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig8_combined_transfer_learning.png) | [fig8_combined_transfer_learning.svg](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig8_combined_transfer_learning.svg)
- **Fig. 9**: Eviction churn and cache thrashing reduction across cache sizes and request rates  
  *Formats*: [fig9_eviction_churn_comparison.png](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig9_eviction_churn_comparison.png) | [fig9_eviction_churn_comparison.svg](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig9_eviction_churn_comparison.svg)
- **Fig. 10**: Byte hit rate and bandwidth savings comparison from utility-density eviction  
  *Formats*: [fig10_byte_hit_rate_comparison.png](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig10_byte_hit_rate_comparison.png) | [fig10_byte_hit_rate_comparison.svg](C:\Users\chsan\OneDrive\Desktop\smdp-edge-caching-framework\graphs\fig10_byte_hit_rate_comparison.svg)
