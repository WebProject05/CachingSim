"""
Comprehensive Experiment Report Generator for SMDP Edge Caching Framework.

Extracts and compiles all recorded metrics and numerical values from:
- data/results/baseline_results.json (Proposed vs CTD)
- data/results/parameter_sweeps.json (Sensitivity sweeps for Proposed and CTD)
- data/results/drl_source_training.json (DDQL Source domain training & evaluation)
- data/results/transfer_learning_comparison.json (Target domain TL trajectories)
- data/results/experiment_summary.json (High-level summary)

STRICTLY EXCLUDES non-paper classical baselines (FIFO, LIFO, LRU, LFU, SIEVE).
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
            "Proposed (SMDP-DDQL)",
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
            "lambda_source": 0.20,
            "lambda_target": 0.30,
            "gamma_continuous": "gamma^tau (0.99^tau)",
            "neural_network": "2-Layer MLP (16-16, ReLU)",
            "per_buffer_capacity": 5000,
            "target_buffer_capacity": 10000,
            "n_step_horizon": 3
        }
    }

    # =========================================================================
    # 2. Comparative Caching Metrics (Proposed vs CTD)
    # =========================================================================
    caching_metrics = {}
    if baseline_data and "algorithms" in baseline_data:
        for algo in baseline_data["algorithms"]:
            name = algo.get("name", "")
            if name in ["SMDP-DDQL", "CTD"]:
                display_name = "Proposed (SMDP-DDQL)" if name == "SMDP-DDQL" else "CTD"
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
    if "Proposed (SMDP-DDQL)" in caching_metrics and "CTD" in caching_metrics:
        prop = caching_metrics["Proposed (SMDP-DDQL)"]
        ctd = caching_metrics["CTD"]
        hit_gain = ((prop["hit_rate_pct"] - ctd["hit_rate_pct"]) / max(0.1, ctd["hit_rate_pct"])) * 100.0
        util_gain = ((prop["total_utility"] - ctd["total_utility"]) / max(0.1, ctd["total_utility"])) * 100.0
        evict_reduction = ((ctd["evictions"] - prop["evictions"]) / max(1, ctd["evictions"])) * 100.0
        relative_gains = {
            "hit_rate_absolute_gain_pct": round(prop["hit_rate_pct"] - ctd["hit_rate_pct"], 2),
            "hit_rate_relative_improvement_pct": round(hit_gain, 2),
            "total_utility_gain_pct": round(util_gain, 2),
            "eviction_reduction_pct": round(evict_reduction, 2),
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
    # 4. Target Domain Transfer Learning Performance (lambda_T = 0.3)
    # =========================================================================
    tl_results = {}
    if tl_comparison_data and "trajectories" in tl_comparison_data:
        for method, traj in tl_comparison_data["trajectories"].items():
            if not traj:
                continue
            init_r = traj[0]["avg_reward"]
            mid_idx = len(traj) // 2
            mid_r = traj[mid_idx]["avg_reward"]
            final_r = traj[-1]["avg_reward"]
            max_r = max(entry["avg_reward"] for entry in traj)
            tl_results[method] = {
                "initial_reward_step_100": round(init_r, 4),
                "midpoint_reward": round(mid_r, 4),
                "final_asymptotic_reward": round(final_r, 4),
                "peak_reward": round(max_r, 4),
                "recorded_steps": traj[-1]["step"],
            }

    # =========================================================================
    # 5. Parameter Sweeps (Proposed vs CTD)
    # =========================================================================
    filtered_sweeps = []
    if sweeps_data and "sweeps" in sweeps_data:
        for sw in sweeps_data["sweeps"]:
            paper_series = []
            for s in sw.get("series", []):
                algo_name = s.get("algorithm", "")
                if algo_name in ["SMDP-DDQL", "CTD"]:
                    d_name = "Proposed (SMDP-DDQL)" if algo_name == "SMDP-DDQL" else "CTD"
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
    # 6. Figures Catalog (25 Publication Figures)
    # =========================================================================
    figures_catalog = [
        {"figure": "Fig. 1", "file": "fig1_utility_curves", "description": "Non-linear utility curve y_f(t) vs freshness h^f(t) across importance levels"},
        {"figure": "Fig. 3(a)", "file": "fig3a_average_rewards", "description": "Comparison of average rewards during DRL convergence (Proposed vs CTD)"},
        {"figure": "Fig. 3(b)", "file": "fig3b_cache_worth", "description": "Average worth of cached files W(t) over trials"},
        {"figure": "Fig. 3(c)", "file": "fig3c_unoccupied_cache", "description": "Average unoccupied cache proportion Mem(t) over trials"},
        {"figure": "Fig. 3", "file": "fig3_combined_performance", "description": "3-Panel overview of DRL performance (Reward, Worth, Unoccupied Cache)"},
        {"figure": "Fig. 4(a)", "file": "fig4a_hit_count_vs_lifetime", "description": "Total hit count vs popular-file lifetime (Proposed vs CTD)"},
        {"figure": "Fig. 4(b)", "file": "fig4b_hit_count_vs_size", "description": "Total hit count vs popular-file size (Proposed vs CTD)"},
        {"figure": "Fig. 4(c)", "file": "fig4c_hit_count_vs_importance", "description": "Total hit count vs popular-file importance (Proposed vs CTD)"},
        {"figure": "Fig. 4", "file": "fig4_combined_hit_counts", "description": "3-Panel hit count under varying popular-file features"},
        {"figure": "Fig. 5(a)", "file": "fig5a_total_utility_vs_lifetime", "description": "Total utility vs popular-file lifetime (Proposed vs CTD)"},
        {"figure": "Fig. 5(b)", "file": "fig5b_total_utility_vs_size", "description": "Total utility vs popular-file size (Proposed vs CTD)"},
        {"figure": "Fig. 5(c)", "file": "fig5c_total_utility_vs_importance", "description": "Total utility vs popular-file importance (Proposed vs CTD)"},
        {"figure": "Fig. 5", "file": "fig5_combined_utilities", "description": "3-Panel total utility under varying popular-file features"},
        {"figure": "Fig. 6(a)", "file": "fig6a_hit_count_vs_zipf_eta", "description": "Total hit count vs Zipf skewness eta (Proposed vs CTD)"},
        {"figure": "Fig. 6(b)", "file": "fig6b_hit_count_vs_request_rate", "description": "Total hit count vs request arrival rate lambda (Proposed vs CTD)"},
        {"figure": "Fig. 6(c)", "file": "fig6c_hit_count_vs_cache_size", "description": "Total hit count vs cache capacity M (Proposed vs CTD)"},
        {"figure": "Fig. 6", "file": "fig6_combined_hit_counts", "description": "3-Panel hit count under varying system parameters"},
        {"figure": "Fig. 7(a)", "file": "fig7a_total_utility_vs_zipf_eta", "description": "Total utility vs Zipf skewness eta (Proposed vs CTD)"},
        {"figure": "Fig. 7(b)", "file": "fig7b_total_utility_vs_request_rate", "description": "Total utility vs request arrival rate lambda (Proposed vs CTD)"},
        {"figure": "Fig. 7(c)", "file": "fig7c_total_utility_vs_cache_size", "description": "Total utility vs cache capacity M (Proposed vs CTD)"},
        {"figure": "Fig. 7", "file": "fig7_combined_utilities", "description": "3-Panel total utility under varying system parameters"},
        {"figure": "Fig. 8(a)", "file": "fig8a_target_domain_convergence", "description": "Target domain transfer learning convergence (Proposed TL vs DQfD vs LFS vs DPR)"},
        {"figure": "Fig. 8(b)", "file": "fig8b_impact_of_overwriting", "description": "Impact of overwriting demonstration data (With vs Without overwriting)"},
        {"figure": "Fig. 8(c)", "file": "fig8c_impact_of_priorities", "description": "Impact of assigning priorities with proposed Eq. (6) vs standard Eq. (5)"},
        {"figure": "Fig. 8", "file": "fig8_combined_transfer_learning", "description": "3-Panel overview of transfer learning performance in target domain"},
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

    header_title = "SMDP EDGE CACHING EXPERIMENT: COMPREHENSIVE PERFORMANCE REPORT"
    txt_lines.append("=" * 80)
    txt_lines.append(header_title)
    txt_lines.append("=" * 80)
    txt_lines.append(f"Generated at: {timestamp}")
    txt_lines.append("Included Algorithms : Proposed (SMDP-DDQL), CTD, Proposed TL, DQfD, LFS, DPR")
    txt_lines.append("Excluded Baselines  : FIFO, LIFO, LRU, LFU, SIEVE (omitted per research specification)")
    txt_lines.append("-" * 80)

    md_lines.append(f"# {header_title}\n")
    md_lines.append(f"> **Generated at**: `{timestamp}`  \n> **Targeted Algorithms**: `Proposed (SMDP-DDQL)`, `CTD`, `Proposed TL`, `DQfD`, `LFS`, `DPR`  \n> **Excluded Baselines**: `FIFO`, `LIFO`, `LRU`, `LFU`, `SIEVE` (omitted per user configuration)\n")

    # Section 1: Caching Metrics Table
    txt_lines.append("\n1. CORE CACHING PERFORMANCE (PROPOSED vs CTD)\n" + "-" * 80)
    md_lines.append("\n## 1. Core Caching Performance (Proposed vs CTD)\n")

    caching_headers = ["Metric", "CTD Baseline", "Proposed (SMDP-DDQL)", "Relative Gain"]
    caching_rows = []
    if "Proposed (SMDP-DDQL)" in caching_metrics and "CTD" in caching_metrics:
        p = caching_metrics["Proposed (SMDP-DDQL)"]
        c = caching_metrics["CTD"]
        caching_rows.append(["Hit Count", f"{c['hits']}", f"{p['hits']}", f"+{p['hits'] - c['hits']} hits"])
        caching_rows.append(["Hit Rate (%)", f"{c['hit_rate_pct']:.2f}%", f"{p['hit_rate_pct']:.2f}%", f"+{relative_gains.get('hit_rate_relative_improvement_pct', 0.0):.1f}%"])
        caching_rows.append(["Byte Hit Rate (%)", f"{c['byte_hit_rate_pct']:.2f}%", f"{p['byte_hit_rate_pct']:.2f}%", f"+{p['byte_hit_rate_pct'] - c['byte_hit_rate_pct']:.2f}%"])
        caching_rows.append(["Miss Count", f"{c['misses']}", f"{p['misses']}", f"-{c['misses'] - p['misses']} misses"])
        caching_rows.append(["Miss Rate (%)", f"{c['miss_rate_pct']:.2f}%", f"{p['miss_rate_pct']:.2f}%", f"-{c['miss_rate_pct'] - p['miss_rate_pct']:.2f}%"])
        caching_rows.append(["Evictions", f"{c['evictions']}", f"{p['evictions']}", f"-{relative_gains.get('eviction_reduction_pct', 0.0):.1f}%"])
        caching_rows.append(["Capacity Utilization (%)", f"{c['utilization_pct']:.2f}%", f"{p['utilization_pct']:.2f}%", f"{p['utilization_pct'] - c['utilization_pct']:.2f}%"])
        caching_rows.append(["Total Utility", f"{c['total_utility']:.2f}", f"{p['total_utility']:.2f}", f"+{relative_gains.get('total_utility_gain_pct', 0.0):.1f}%"])

    caching_table_str = format_table(caching_headers, caching_rows)
    txt_lines.append(caching_table_str)

    md_lines.append("| Metric | CTD Baseline | Proposed (SMDP-DDQL) | Relative Gain |")
    md_lines.append("| :--- | :--- | :--- | :--- |")
    for row in caching_rows:
        md_lines.append(f"| **{row[0]}** | `{row[1]}` | **`{row[2]}`** | **{row[3]}** |")

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
        tl_rows.append([
            method,
            f"{res['initial_reward_step_100']:.2f}",
            f"{res['midpoint_reward']:.2f}",
            f"{res['final_asymptotic_reward']:.2f}",
            f"{res['peak_reward']:.2f}"
        ])
    txt_lines.append(format_table(tl_headers, tl_rows))

    md_lines.append("| Transfer Learning Method | Initial Reward (t=100) | Midpoint Reward | Final Reward | Peak Reward |")
    md_lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for r in tl_rows:
        md_lines.append(f"| **{r[0]}** | `{r[1]}` | `{r[2]}` | **`{r[3]}`** | `{r[4]}` |")

    # Section 4: Parameter Sensitivity Sweeps
    txt_lines.append("\n4. PARAMETER SENSITIVITY SWEEPS (PROPOSED vs CTD)\n" + "-" * 80)
    md_lines.append("\n## 4. Parameter Sensitivity Sweeps (Proposed vs CTD)\n")

    for sw in filtered_sweeps:
        txt_lines.append(f"\n>> {sw['title']} [{sw['chart_file']}]")
        md_lines.append(f"\n### {sw['title']}\n")
        x_vals = [str(x) for x in sw["x_values"]]
        sw_headers = [sw["x_parameter"]] + x_vals
        sw_rows = []
        for s in sw["series"]:
            row_vals = [s["algorithm"]] + [str(v) for v in s["values"]]
            sw_rows.append(row_vals)
        txt_lines.append(format_table(sw_headers, sw_rows))

        md_header_row = f"| {sw['x_parameter']} | " + " | ".join(x_vals) + " |"
        md_sep_row = "| :--- | " + " | ".join([":---:" for _ in x_vals]) + " |"
        md_lines.append(md_header_row)
        md_lines.append(md_sep_row)
        for s in sw["series"]:
            val_strs = [f"`{v}`" for v in s["values"]]
            md_lines.append(f"| **{s['algorithm']}** | " + " | ".join(val_strs) + " |")

    # Section 5: Figures Catalog
    txt_lines.append("\n5. PUBLICATION-QUALITY FIGURES CATALOG\n" + "-" * 80)
    md_lines.append("\n## 5. Visual Assets & Generated Figures Catalog\n")
    fig_headers = ["Figure", "Filename", "Description", "Status"]
    fig_rows = []
    for f in figures_catalog:
        status = "Available (PNG+SVG)" if f["exists"] else "Not Generated"
        fig_rows.append([f["figure"], f["file"], f["description"][:55] + "...", status])

    txt_lines.append(format_table(fig_headers, fig_rows))

    md_lines.append("| Figure | Filename | Topic / Description | Artifact Links |")
    md_lines.append("| :--- | :--- | :--- | :--- |")
    for f in figures_catalog:
        md_lines.append(f"| **{f['figure']}** | `{f['file']}` | {f['description']} | [PNG](file:///{f['png_path'].replace(os.sep, '/')}) \\| [SVG](file:///{f['svg_path'].replace(os.sep, '/')}) |")

    txt_lines.append("\n" + "=" * 80)
    txt_lines.append("END OF EXPERIMENT REPORT")
    txt_lines.append("=" * 80)

    # Write Text and Markdown files
    txt_path = os.path.join(results_dir, "experiment_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines) + "\n")

    md_path = os.path.join(results_dir, "experiment_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\n[+] Comprehensive Experiment Reports Generated Successfully:")
    print(f"    - JSON Format : {json_path}")
    print(f"    - Text Format : {txt_path}")
    print(f"    - Markdown    : {md_path}")


if __name__ == "__main__":
    generate_reports()
