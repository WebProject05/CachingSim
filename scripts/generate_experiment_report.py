"""
Comprehensive Experiment Report Generator for SMDP Edge Caching Framework.

Extracts and compiles all recorded metrics and numerical values from:
- data/results/baseline_results.json (AdaLife-Pro vs Paper DDQL vs CTD)
- data/results/parameter_sweeps.json (Sensitivity sweeps)
- data/results/drl_source_training.json (DDQL Source domain training & evaluation)
- data/results/transfer_learning_comparison.json (Target domain TL trajectories)
- data/results/experiment_summary.json (High-level summary)

Produces:
1. data/results/experiment_report.json (machine-readable structured JSON)
2. data/results/experiment_report.txt  (human-readable ASCII formatted report)
3. data/results/experiment_report.md   (GitHub-flavored Markdown report with graph links)
"""

import os
import sys
import json
from datetime import datetime


def load_json(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[!] Warning: Could not read {filepath}: {e}")
    return None


def format_table(headers, rows):
    """Formats a neat ASCII table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    border = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_str = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"

    lines = [border, header_str, border]
    for row in rows:
        row_str = "| " + " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row)) + " |"
        lines.append(row_str)
    lines.append(border)
    return "\n".join(lines)


def generate_reports():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(root_dir, "data", "results")
    graphs_dir = os.path.join(root_dir, "graphs")

    # Load recorded artifacts
    baseline_data = load_json(os.path.join(results_dir, "baseline_results.json"))
    sweeps_data = load_json(os.path.join(results_dir, "parameter_sweeps.json"))
    source_training_data = load_json(os.path.join(results_dir, "drl_source_training.json"))
    tl_comparison_data = load_json(os.path.join(results_dir, "transfer_learning_comparison.json"))
    summary_data = load_json(os.path.join(results_dir, "experiment_summary.json"))

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    # =========================================================================
    # 1. System Parameters & Configuration
    # =========================================================================
    metadata = {
        "report_generated_at": timestamp,
        "environment": "SMDP Edge Caching (Go Simulation Engine + Python PyTorch DDQL)",
        "evaluated_algorithms": [
            "AdaLife-Pro (Our Proposed Algorithm)",
            "Paper (Niknia et al., 2024 SMDP-DDQL)",
            "CTD (Continuous Time Delay Baseline)",
            "Proposed TL (Transfer Learning with Eq. 6 Priority & Demonstrations)",
            "DQfD (Standard Deep Q-learning from Demonstrations, Eq. 5 Priority)",
            "LFS (Learning From Scratch)",
            "DPR (Direct Policy Reuse)"
        ],
        "excluded_classical_baselines": [
            "FIFO", "LIFO", "LRU", "LFU", "SIEVE"
        ],
        "system_configuration": {
            "total_file_types": 50,
            "cache_capacity_mib": 10000.0,
            "zipf_popularity_eta": 1.0,
            "lambda_source": 5.0,
            "lambda_target": 0.30,
            "gamma_continuous": "gamma^tau (0.99^tau)",
            "neural_network": "2-Layer MLP (16-16, ReLU)",
            "per_buffer_capacity": 5000,
            "target_buffer_capacity": 10000,
            "n_step_horizon": 3
        }
    }

    # =========================================================================
    # 2. Comparative Caching Metrics (AdaLife-Pro vs Paper DDQL vs CTD)
    # =========================================================================
    caching_metrics = {}
    if baseline_data and "algorithms" in baseline_data:
        for algo in baseline_data["algorithms"]:
            name = algo.get("name", "")
            if name in ["AdaLife-Pro", "SMDP-DDQL", "Paper-Proposed", "CTD"]:
                display_name = name
                if name in ["AdaLife-Pro", "SMDP-DDQL"]:
                    display_name = "AdaLife-Pro (Ours)"
                elif name == "Paper-Proposed":
                    display_name = "Paper (Niknia '24 DDQL)"
                elif name == "CTD":
                    display_name = "CTD (Baseline)"
                
                caching_metrics[display_name] = {
                    "requests": algo.get("requests", 0),
                    "hits": algo.get("hits", 0),
                    "misses": algo.get("misses", 0),
                    "hit_rate_pct": round(algo.get("hit_rate_pct", 0.0), 2),
                    "byte_hit_rate_pct": round(algo.get("byte_hit_rate_pct", 0.0), 2),
                    "miss_rate_pct": round(algo.get("miss_rate_pct", 0.0), 2),
                    "evictions": algo.get("evictions", 0),
                    "eviction_rate_pct": round(algo.get("eviction_rate_pct", 0.0), 2),
                    "cached_files": algo.get("cached_files", 0),
                    "used_capacity_mib": round(algo.get("used_capacity_mib", 0.0), 2),
                    "utilization_pct": round(algo.get("utilization_pct", 0.0), 2),
                    "total_utility": round(algo.get("total_utility", 0.0), 4),
                    "warmup_miss_rate_pct": round(algo.get("warmup_miss_rate_pct", 0.0), 2),
                    "final_miss_rate_pct": round(algo.get("final_miss_rate_pct", 0.0), 2),
                }

    # Calculate relative gains
    relative_gains = {}
    ada = caching_metrics.get("AdaLife-Pro (Ours)")
    paper = caching_metrics.get("Paper (Niknia '24 DDQL)")
    ctd = caching_metrics.get("CTD (Baseline)")

    if ada and paper:
        relative_gains["vs_paper"] = {
            "hit_rate_gain_pct": round(ada["hit_rate_pct"] - paper["hit_rate_pct"], 2),
            "byte_hit_rate_gain_pct": round(ada["byte_hit_rate_pct"] - paper["byte_hit_rate_pct"], 2),
            "eviction_reduction_pct": round(((paper["evictions"] - ada["evictions"]) / max(1, paper["evictions"])) * 100.0, 2),
            "total_utility_gain_pct": round(((ada["total_utility"] - paper["total_utility"]) / max(0.1, paper["total_utility"])) * 100.0, 2),
        }
    if ada and ctd:
        relative_gains["vs_ctd"] = {
            "hit_rate_gain_pct": round(ada["hit_rate_pct"] - ctd["hit_rate_pct"], 2),
            "byte_hit_rate_gain_pct": round(ada["byte_hit_rate_pct"] - ctd["byte_hit_rate_pct"], 2),
            "eviction_reduction_pct": round(((ctd["evictions"] - ada["evictions"]) / max(1, ctd["evictions"])) * 100.0, 2),
            "total_utility_gain_pct": round(((ada["total_utility"] - ctd["total_utility"]) / max(0.1, ctd["total_utility"])) * 100.0, 2),
        }

    # =========================================================================
    # 3. Source Domain DRL Training & Evaluation
    # =========================================================================
    source_results = {}
    if source_training_data:
        eval_m = source_training_data.get("evaluation_metrics", {})
        traj = source_training_data.get("training_trajectory", [])
        final_train_r = traj[-1]["avg_reward"] if traj else 0.0
        source_results = {
            "training_steps": traj[-1]["step"] if traj else 0,
            "final_training_average_reward": round(final_train_r, 4),
            "policy_evaluation_1000_requests": {
                "requests": eval_m.get("requests", 0),
                "hits": eval_m.get("hits", 0),
                "hit_rate_pct": round(eval_m.get("hit_rate", eval_m.get("hit_rate_pct", 0.0)), 2),
                "total_utility": round(eval_m.get("total_utility", 0.0), 2),
                "total_reward": round(eval_m.get("total_reward", 0.0), 2),
                "average_reward_per_step": round(eval_m.get("avg_reward", 0.0), 4),
            }
        }

    # =========================================================================
    # 4. Target Domain Transfer Learning
    # =========================================================================
    tl_results = {}
    if tl_comparison_data and "trajectories" in tl_comparison_data:
        trajs = tl_comparison_data["trajectories"]
        for method_key, d_name in [
            ("PROPOSED", "Proposed TL (AdaLife-Pro)"),
            ("DQFD", "Standard DQfD (Eq. 5)"),
            ("LFS", "Learning From Scratch (LFS)"),
            ("DPR", "Direct Policy Reuse (DPR)")
        ]:
            if method_key in trajs and len(trajs[method_key]) > 0:
                series = trajs[method_key]
                r_vals = [e["avg_reward"] for e in series]
                tl_results[d_name] = {
                    "initial_reward": round(r_vals[0], 4),
                    "midpoint_reward": round(r_vals[len(r_vals) // 2], 4),
                    "final_reward": round(r_vals[-1], 4),
                    "peak_reward": round(max(r_vals), 4)
                }

    # =========================================================================
    # 5. Parameter Sweeps (AdaLife-Pro vs Paper DDQL vs CTD)
    # =========================================================================
    filtered_sweeps = []
    if sweeps_data and "sweeps" in sweeps_data:
        for sw in sweeps_data["sweeps"]:
            paper_series = []
            for s in sw.get("series", []):
                algo_name = s.get("algorithm", "")
                if algo_name in ["AdaLife-Pro", "SMDP-DDQL", "Paper-Proposed", "CTD"]:
                    d_name = algo_name
                    if algo_name in ["AdaLife-Pro", "SMDP-DDQL"]:
                        d_name = "AdaLife-Pro (Ours)"
                    elif algo_name == "Paper-Proposed":
                        d_name = "Paper (Niknia '24 DDQL)"
                    elif algo_name == "CTD":
                        d_name = "CTD (Baseline)"
                    paper_series.append({
                        "algorithm": d_name,
                        "values": [round(v, 2) for v in s.get("y_values", [])]
                    })
            filtered_sweeps.append({
                "chart_file": sw.get("file", ""),
                "title": sw.get("title", ""),
                "x_parameter": sw.get("x_parameter", ""),
                "y_metric": sw.get("y_metric", ""),
                "x_values": sw.get("x_values", []),
                "series": paper_series
            })

    # =========================================================================
    # 6. Figures Catalog
    # =========================================================================
    figures_catalog = [
        {"figure": "Fig. 1", "file": "fig1_utility_curves", "description": "Non-linear utility curve y_f(t) vs freshness h^f(t) across importance levels"},
        {"figure": "Fig. 3(a)", "file": "fig3a_average_rewards", "description": "Average reward convergence (AdaLife-Pro vs Paper DDQL)"},
        {"figure": "Fig. 3(b)", "file": "fig3b_cache_worth", "description": "Average worth of cached files W(t) over trials"},
        {"figure": "Fig. 3(c)", "file": "fig3c_unoccupied_cache", "description": "Average unoccupied cache proportion Mem(t) over trials"},
        {"figure": "Fig. 3", "file": "fig3_combined_performance", "description": "3-Panel overview of DRL performance (Reward, Worth, Unoccupied Cache)"},
        {"figure": "Fig. 4(a)", "file": "fig4a_hit_count_vs_lifetime", "description": "Total hit count vs popular-file lifetime (AdaLife-Pro vs Paper DDQL)"},
        {"figure": "Fig. 4(b)", "file": "fig4b_hit_count_vs_size", "description": "Total hit count vs popular-file size (AdaLife-Pro vs Paper DDQL)"},
        {"figure": "Fig. 4(c)", "file": "fig4c_hit_count_vs_importance", "description": "Total hit count vs popular-file importance (AdaLife-Pro vs Paper DDQL)"},
        {"figure": "Fig. 4", "file": "fig4_combined_hit_counts", "description": "3-Panel hit count under varying popular-file features"},
        {"figure": "Fig. 5(a)", "file": "fig5a_total_utility_vs_lifetime", "description": "Total utility vs popular-file lifetime (AdaLife-Pro vs Paper DDQL)"},
        {"figure": "Fig. 5(b)", "file": "fig5b_total_utility_vs_size", "description": "Total utility vs popular-file size (AdaLife-Pro vs Paper DDQL)"},
        {"figure": "Fig. 5(c)", "file": "fig5c_total_utility_vs_importance", "description": "Total utility vs popular-file importance (AdaLife-Pro vs Paper DDQL)"},
        {"figure": "Fig. 5", "file": "fig5_combined_utilities", "description": "3-Panel total utility under varying popular-file features"},
        {"figure": "Fig. 6(a)", "file": "fig6a_hit_count_vs_zipf_eta", "description": "Total hit count vs Zipf skewness eta (AdaLife-Pro vs Paper DDQL)"},
        {"figure": "Fig. 6(b)", "file": "fig6b_hit_count_vs_request_rate", "description": "Total hit count vs request arrival rate lambda (AdaLife-Pro vs Paper DDQL)"},
        {"figure": "Fig. 6(c)", "file": "fig6c_hit_count_vs_cache_size", "description": "Total hit count vs cache capacity M (AdaLife-Pro vs Paper DDQL)"},
        {"figure": "Fig. 6", "file": "fig6_combined_hit_counts", "description": "3-Panel hit count under varying system parameters"},
        {"figure": "Fig. 7(a)", "file": "fig7a_total_utility_vs_zipf_eta", "description": "Total utility vs Zipf skewness eta (AdaLife-Pro vs Paper DDQL)"},
        {"figure": "Fig. 7(b)", "file": "fig7b_total_utility_vs_request_rate", "description": "Total utility vs request arrival rate lambda (AdaLife-Pro vs Paper DDQL)"},
        {"figure": "Fig. 7(c)", "file": "fig7c_total_utility_vs_cache_size", "description": "Total utility vs cache capacity M (AdaLife-Pro vs Paper DDQL)"},
        {"figure": "Fig. 7", "file": "fig7_combined_utilities", "description": "3-Panel total utility under varying system parameters"},
        {"figure": "Fig. 8(a)", "file": "fig8a_target_domain_convergence", "description": "Target domain transfer learning convergence (AdaLife-Pro TL vs Paper DDQL TL vs LFS vs DPR)"},
        {"figure": "Fig. 8(b)", "file": "fig8b_impact_of_overwriting", "description": "Impact of overwriting demonstration data (With vs Without overwriting)"},
        {"figure": "Fig. 8(c)", "file": "fig8c_impact_of_priorities", "description": "Impact of assigning priorities with proposed Eq. (6) vs standard Eq. (5)"},
        {"figure": "Fig. 8", "file": "fig8_combined_transfer_learning", "description": "3-Panel overview of transfer learning performance in target domain"},
        {"figure": "Fig. 9", "file": "fig9_eviction_churn_comparison", "description": "Eviction churn and cache thrashing reduction across cache sizes and request rates"},
        {"figure": "Fig. 10", "file": "fig10_byte_hit_rate_comparison", "description": "Byte hit rate and bandwidth savings comparison from utility-density eviction"},
    ]

    for f_info in figures_catalog:
        f_info["png_path"] = os.path.join(graphs_dir, f"{f_info['file']}.png")
        f_info["svg_path"] = os.path.join(graphs_dir, f"{f_info['file']}.svg")
        f_info["exists"] = os.path.exists(f_info["png_path"])

    # Build Master JSON
    master_report = {
        "metadata": metadata,
        "caching_performance": caching_metrics,
        "relative_improvements": relative_gains,
        "source_domain_drl": source_results,
        "target_domain_transfer_learning": tl_results,
        "parameter_sensitivity_sweeps": filtered_sweeps,
        "figures_catalog": figures_catalog
    }

    # Save JSON Report
    json_path = os.path.join(results_dir, "experiment_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(master_report, f, indent=2)

    # Build Text & Markdown Reports
    txt_lines = []
    md_lines = []

    header_title = "SMDP EDGE CACHING EXPERIMENT: COMPREHENSIVE THREE-WAY PERFORMANCE REPORT"
    txt_lines.append("=" * 80)
    txt_lines.append(header_title)
    txt_lines.append("=" * 80)
    txt_lines.append(f"Generated at: {timestamp}")
    txt_lines.append("Targeted Algorithms : CTD (Baseline), Paper (Niknia '24 DDQL), AdaLife-Pro (Ours)")
    txt_lines.append("Excluded Baselines  : FIFO, LIFO, LRU, LFU, SIEVE (omitted per research specification)")
    txt_lines.append("-" * 80)

    md_lines.append(f"# {header_title}\n")
    md_lines.append(f"> **Generated at**: `{timestamp}`  \n> **Targeted Algorithms**: `CTD (Baseline)`, `Paper (Niknia '24 DDQL)`, `AdaLife-Pro (Ours)`  \n> **Excluded Baselines**: `FIFO`, `LIFO`, `LRU`, `LFU`, `SIEVE` (omitted per user configuration)\n")

    # Section 1: Caching Metrics Table
    txt_lines.append("\n1. THREE-WAY CACHING PERFORMANCE (CTD vs PAPER DDQL vs ADALIFE-PRO)\n" + "-" * 80)
    md_lines.append("\n## 1. Three-Way Caching Performance (CTD vs Paper DDQL vs AdaLife-Pro)\n")

    caching_headers = ["Metric", "CTD Baseline", "Paper (Niknia '24)", "AdaLife-Pro (Ours)", "Gain vs Paper", "Gain vs CTD"]
    caching_rows = []
    
    if ada and paper and ctd:
        caching_rows.append(["Hit Count", f"{ctd['hits']}", f"{paper['hits']}", f"{ada['hits']}", f"+{ada['hits'] - paper['hits']} hits", f"+{ada['hits'] - ctd['hits']} hits"])
        caching_rows.append(["Hit Rate (%)", f"{ctd['hit_rate_pct']:.2f}%", f"{paper['hit_rate_pct']:.2f}%", f"{ada['hit_rate_pct']:.2f}%", f"+{ada['hit_rate_pct'] - paper['hit_rate_pct']:.2f}%", f"+{ada['hit_rate_pct'] - ctd['hit_rate_pct']:.2f}%"])
        caching_rows.append(["Byte Hit Rate (%)", f"{ctd['byte_hit_rate_pct']:.2f}%", f"{paper['byte_hit_rate_pct']:.2f}%", f"{ada['byte_hit_rate_pct']:.2f}%", f"+{ada['byte_hit_rate_pct'] - paper['byte_hit_rate_pct']:.2f}%", f"+{ada['byte_hit_rate_pct'] - ctd['byte_hit_rate_pct']:.2f}%"])
        caching_rows.append(["Miss Count", f"{ctd['misses']}", f"{paper['misses']}", f"{ada['misses']}", f"-{paper['misses'] - ada['misses']} misses", f"-{ctd['misses'] - ada['misses']} misses"])
        caching_rows.append(["Miss Rate (%)", f"{ctd['miss_rate_pct']:.2f}%", f"{paper['miss_rate_pct']:.2f}%", f"{ada['miss_rate_pct']:.2f}%", f"-{paper['miss_rate_pct'] - ada['miss_rate_pct']:.2f}%", f"-{ctd['miss_rate_pct'] - ada['miss_rate_pct']:.2f}%"])
        caching_rows.append(["Evictions", f"{ctd['evictions']}", f"{paper['evictions']}", f"{ada['evictions']}", f"-{paper['evictions'] - ada['evictions']} (-{relative_gains.get('vs_paper', {}).get('eviction_reduction_pct', 0.0):.1f}%)", f"-{ctd['evictions'] - ada['evictions']} (-{relative_gains.get('vs_ctd', {}).get('eviction_reduction_pct', 0.0):.1f}%)"])
        caching_rows.append(["Capacity Utilization (%)", f"{ctd['utilization_pct']:.2f}%", f"{paper['utilization_pct']:.2f}%", f"{ada['utilization_pct']:.2f}%", f"{ada['utilization_pct'] - paper['utilization_pct']:.2f}%", f"{ada['utilization_pct'] - ctd['utilization_pct']:.2f}%"])
        caching_rows.append(["Total Utility", f"{ctd['total_utility']:.2f}", f"{paper['total_utility']:.2f}", f"{ada['total_utility']:.2f}", f"+{relative_gains.get('vs_paper', {}).get('total_utility_gain_pct', 0.0):.1f}%", f"+{relative_gains.get('vs_ctd', {}).get('total_utility_gain_pct', 0.0):.1f}%"])

    caching_table_str = format_table(caching_headers, caching_rows)
    txt_lines.append(caching_table_str)

    md_lines.append("| Metric | CTD Baseline | Paper (Niknia '24) | AdaLife-Pro (Ours) | Gain vs Paper | Gain vs CTD |")
    md_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
    for row in caching_rows:
        md_lines.append(f"| **{row[0]}** | `{row[1]}` | `{row[2]}` | **`{row[3]}`** | **{row[4]}** | **{row[5]}** |")

    # Section 2: Source Domain DRL
    txt_lines.append("\n2. SOURCE DOMAIN DRL (DDQL) PERFORMANCE\n" + "-" * 80)
    md_lines.append("\n## 2. Source Domain DRL (DDQL) Performance\n")
    if source_results:
        pe = source_results.get("policy_evaluation_1000_requests", {})
        txt_lines.append(f"  Training Steps Completed      : {source_results.get('training_steps', 0):,}")
        txt_lines.append(f"  Final Moving Average Reward   : {source_results.get('final_training_average_reward', 0.0):.4f}")
        txt_lines.append(f"  Evaluation Hit Rate           : {pe.get('hit_rate_pct', 0.0):.2f}% ({pe.get('hits', 0)} / {pe.get('requests', 0)} requests)")
        txt_lines.append(f"  Evaluation Total Utility      : {pe.get('total_utility', 0.0):.2f}")
        txt_lines.append(f"  Evaluation Avg Reward / Step  : {pe.get('average_reward_per_step', 0.0):.4f}")

        md_lines.append(f"- **Training Steps**: `{source_results.get('training_steps', 0):,}` steps")
        md_lines.append(f"- **Final Moving Average Reward**: `{source_results.get('final_training_average_reward', 0.0):.4f}`")
        md_lines.append(f"- **Greedy Policy Hit Rate**: `{pe.get('hit_rate_pct', 0.0):.2f}%` (`{pe.get('hits', 0)} / {pe.get('requests', 0)}` requests)")
        md_lines.append(f"- **Total Accumulated Utility**: `{pe.get('total_utility', 0.0):.2f}`")
        md_lines.append(f"- **Average Reward Per Step**: `{pe.get('average_reward_per_step', 0.0):.4f}`\n")

    # Section 3: Target Domain Transfer Learning
    txt_lines.append("\n3. TARGET DOMAIN TRANSFER LEARNING PERFORMANCE (lambda_T = 0.3)\n" + "-" * 80)
    md_lines.append("\n## 3. Target Domain Transfer Learning Performance (lambda_T = 0.3)\n")
    tl_headers = ["Method", "Initial Reward (t=100)", "Midpoint Reward", "Final Reward", "Peak Reward"]
    tl_rows = []
    for method, res in tl_results.items():
        tl_rows.append([method, f"{res['initial_reward']:.4f}", f"{res['midpoint_reward']:.4f}", f"{res['final_reward']:.4f}", f"{res['peak_reward']:.4f}"])

    txt_lines.append(format_table(tl_headers, tl_rows))
    md_lines.append("| Method | Initial Reward (t=100) | Midpoint Reward | Final Reward | Peak Reward |")
    md_lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for r in tl_rows:
        md_lines.append(f"| **{r[0]}** | `{r[1]}` | `{r[2]}` | `{r[3]}` | **`{r[4]}`** |")

    # Section 4: Figures Catalog
    txt_lines.append("\n4. RESEARCH EVALUATION FIGURES (PNG + SVG)\n" + "-" * 80)
    md_lines.append("\n## 4. Research Evaluation Figures (PNG + SVG)\n")
    for f in figures_catalog:
        status_str = "[OK]" if f["exists"] else "[MISSING]"
        txt_lines.append(f"  {status_str} {f['figure']:<12}: {f['description']}")
        txt_lines.append(f"         PNG: {f['png_path']}")
        txt_lines.append(f"         SVG: {f['svg_path']}")

        md_link = f"[{f['file']}.png]({f['png_path']})"
        svg_link = f"[{f['file']}.svg]({f['svg_path']})"
        md_lines.append(f"- **{f['figure']}**: {f['description']}  \n  *Formats*: {md_link} | {svg_link}")

    # Save Text Report
    txt_path = os.path.join(results_dir, "experiment_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines) + "\n")

    # Save Markdown Report
    md_path = os.path.join(results_dir, "experiment_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"[+] Master JSON Report saved to : {json_path}")
    print(f"[+] Formatted Text Report saved to: {txt_path}")
    print(f"[+] Markdown Report saved to      : {md_path}")


if __name__ == "__main__":
    generate_reports()
