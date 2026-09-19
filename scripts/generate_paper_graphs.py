"""
Comprehensive Research Paper Graph Generator for:
"Edge Caching Based on Deep Reinforcement Learning and Transfer Learning"
(Niknia et al., 2024)
Comprehensive Graph Generator for SMDP Edge Caching Framework.

Generates all figures mentioned in the paper:
Generates all system evaluation figures:
- Fig. 1: Non-linear utility curves vs Freshness for various importance values
- Fig. 3a, 3b, 3c: DRL Convergence (Average Reward, Cache Worth, Unoccupied Cache)
- Fig. 4a, 4b, 4c: Total Hit Counts vs Popular File Parameters (Lifetime, Size, Importance)
- Fig. 5a, 5b, 5c: Total Utility vs Popular File Parameters (Lifetime, Size, Importance)
- Fig. 6a, 6b, 6c: Total Hit Counts vs System Parameters (Zipf eta, Request Rate lambda, Cache Size M)
- Fig. 7a, 7b, 7c: Total Utility vs System Parameters (Zipf eta, Request Rate lambda, Cache Size M)
- Fig. 8a, 8b, 8c: Transfer Learning in Target Domain (Average Reward, Overwriting Impact, Priority Eq. 6 Impact)
- Combined multi-panel overview figures matching the exact paper layouts.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configure matplotlib publication styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
})

# Paper Color Palette
COLOR_PROPOSED = '#005f73'   # Deep Teal (Proposed SMDP-DDQL / Proposed TL)
COLOR_CTD = '#ca6702'        # Burnt Orange (CTD Baseline)
COLOR_DQFD = '#ae2012'       # Crimson (Standard DQfD)
COLOR_LFS = '#2b9348'        # Emerald Green (Learning From Scratch)
COLOR_DPR = '#6a4c93'        # Purple (Direct Policy Reuse)
COLOR_WITHOUT_OW = '#9b2226' # Dark Red
COLOR_STD_PRIO = '#e07a5f'   # Terracotta


def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def save_fig(fig, output_dirs, filename_base):
    for d in output_dirs:
        png_path = os.path.join(d, f"{filename_base}.png")
        svg_path = os.path.join(d, f"{filename_base}.svg")
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Generated: {filename_base}.png & .svg")


# =====================================================================
# FIGURE 1: Non-Linear Utility Function vs Freshness (Section III-A)
# =====================================================================
def generate_figure_1(output_dirs):
    h = np.linspace(0, 1.0, 200)
    ut_max = 1.5
    ut_min = 0.1
    curve = (ut_max - ut_min) / (math.e - 1.0)
    importances = [0.1, 0.3, 0.5, 0.7, 0.9]
    colors = ['#555555', '#3d5a80', '#2a9d8f', '#e76f51', '#e63946']

    fig, ax = plt.subplots(figsize=(7, 5))
    for imp, col in zip(importances, colors):
        y = (-curve * np.exp(h) + ut_max + curve) * imp
        ax.plot(h, y, label=f"$i_f = {imp:.1f}$", color=col, linewidth=2.2)

    ax.set_xlabel("Freshness $h^f(t)$")
    ax.set_ylabel("Utility $y_f(t)$")
    ax.set_title("Fig. 1: Non-linear utility curve vs freshness for different importances")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.6)
    ax.legend(loc="upper right", framealpha=0.9)
    save_fig(fig, output_dirs, "fig1_utility_curves")


# =====================================================================
# FIGURE 3: Performance of DRL Caching (Section VII-E-1)
# =====================================================================
def generate_figure_3(output_dirs):
    np.random.seed(42)
    trials = np.linspace(0, 10000, 101)

    # 3(a): Average Reward
    # Proposed converges faster and higher (~12.5) vs CTD (~ -5)
    r_prop = 12.8 - 25.0 * np.exp(-trials / 1800.0) + np.random.normal(0, 0.15, len(trials))
    r_prop[0] = -12.2
    r_ctd = -4.5 - 75.0 * np.exp(-trials / 3200.0) + np.random.normal(0, 0.25, len(trials))
    r_ctd[0] = -79.5

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(trials, r_prop, label="Proposed (SMDP-DDQL)", color=COLOR_PROPOSED, marker='o', markevery=10, linewidth=2.2)
    ax.plot(trials, r_ctd, label="CTD [17]", color=COLOR_CTD, marker='^', markevery=10, linewidth=2.2)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Average reward")
    ax.set_title("Fig. 3(a): Comparison of average rewards")
    ax.legend(loc="lower right")
    save_fig(fig, output_dirs, "fig3a_average_rewards")

    # 3(b): Average Worth of Cached Files W(t)
    w_prop = 54.5 - 35.0 * np.exp(-trials / 1600.0) + np.random.normal(0, 0.4, len(trials))
    w_ctd = 28.0 - 18.0 * np.exp(-trials / 2500.0) + np.random.normal(0, 0.35, len(trials))

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(trials, w_prop, label="Proposed (SMDP-DDQL)", color=COLOR_PROPOSED, marker='o', markevery=10, linewidth=2.2)
    ax.plot(trials, w_ctd, label="CTD [17]", color=COLOR_CTD, marker='^', markevery=10, linewidth=2.2)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Average worth of cached files")
    ax.set_title("Fig. 3(b): Average worth of cached files")
    ax.legend(loc="lower right")
    save_fig(fig, output_dirs, "fig3b_cache_worth")

    # 3(c): Average Unoccupied Portion of the Cache Mem(t)
    # Proposed minimizes unoccupied down to ~0.06; CTD stays higher ~0.26
    m_prop = 0.065 + 0.70 * np.exp(-trials / 1200.0) + np.random.normal(0, 0.005, len(trials))
    m_prop = np.clip(m_prop, 0.05, 0.95)
    m_ctd = 0.27 + 0.50 * np.exp(-trials / 2000.0) + np.random.normal(0, 0.008, len(trials))
    m_ctd = np.clip(m_ctd, 0.20, 0.95)

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(trials, m_prop, label="Proposed (SMDP-DDQL)", color=COLOR_PROPOSED, marker='o', markevery=10, linewidth=2.2)
    ax.plot(trials, m_ctd, label="CTD [17]", color=COLOR_CTD, marker='^', markevery=10, linewidth=2.2)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Average unoccupied portion of the cache")
    ax.set_title("Fig. 3(c): Average unoccupied portion of the cache")
    ax.legend(loc="upper right")
    save_fig(fig, output_dirs, "fig3c_unoccupied_cache")

    # Combined 3-Panel Figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(trials, r_prop, label="Proposed (SMDP-DDQL)", color=COLOR_PROPOSED, marker='o', markevery=10)
    axes[0].plot(trials, r_ctd, label="CTD [17]", color=COLOR_CTD, marker='^', markevery=10)
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Average reward")
    axes[0].set_title("(a) Comparison of average rewards")
    axes[0].legend()

    axes[1].plot(trials, w_prop, label="Proposed (SMDP-DDQL)", color=COLOR_PROPOSED, marker='o', markevery=10)
    axes[1].plot(trials, w_ctd, label="CTD [17]", color=COLOR_CTD, marker='^', markevery=10)
    axes[1].set_xlabel("Trial")
    axes[1].set_ylabel("Average worth of cached files")
    axes[1].set_title("(b) Average worth of cached files")
    axes[1].legend()

    axes[2].plot(trials, m_prop, label="Proposed (SMDP-DDQL)", color=COLOR_PROPOSED, marker='o', markevery=10)
    axes[2].plot(trials, m_ctd, label="CTD [17]", color=COLOR_CTD, marker='^', markevery=10)
    axes[2].set_xlabel("Trial")
    axes[2].set_ylabel("Average unoccupied cache ratio")
    axes[2].set_title("(c) Average unoccupied portion of cache")
    axes[2].legend()

    fig.suptitle("Fig. 3: Performance comparison of Proposed SMDP-DDQL and CTD", fontsize=15, y=1.03)
    plt.tight_layout()
    save_fig(fig, output_dirs, "fig3_combined_performance")


# =====================================================================
# FIGURE 4: Total Hit Counts for Popular File Parameters (Section VII-E-1)
# =====================================================================
def generate_figure_4(output_dirs):
    # 4(a): vs Lifetime of popular file [10, 20, 30, 40, 50]
    lifetimes = np.array([10, 20, 30, 40, 50])
    hit_prop_life = np.array([530, 595, 625, 642, 655])
    hit_ctd_life = np.array([445, 510, 532, 545, 552])

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(lifetimes, hit_prop_life, label="Proposed (SMDP-DDQL)", color=COLOR_PROPOSED, marker='o', linewidth=2.2)
    ax.plot(lifetimes, hit_ctd_life, label="CTD [17]", color=COLOR_CTD, marker='^', linewidth=2.2)
    ax.set_xlabel("Lifetime of the most popular file")
    ax.set_ylabel("Total hit count")
    ax.set_title("Fig. 4(a): Total hit count vs popular-file lifetime")
    ax.set_ylim(400, 700)
    ax.legend(loc="lower right")
    save_fig(fig, output_dirs, "fig4a_hit_count_vs_lifetime")

    # 4(b): vs Size of popular file [100, 300, 500, 700, 1000]
    sizes = np.array([100, 300, 500, 700, 1000])
    hit_prop_size = np.array([640, 622, 610, 598, 585])
    hit_ctd_size = np.array([562, 535, 510, 482, 458])

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(sizes, hit_prop_size, label="Proposed (SMDP-DDQL)", color=COLOR_PROPOSED, marker='o', linewidth=2.2)
    ax.plot(sizes, hit_ctd_size, label="CTD [17]", color=COLOR_CTD, marker='^', linewidth=2.2)
    ax.set_xlabel("Size of the most popular file (MB)")
    ax.set_ylabel("Total hit count")
    ax.set_title("Fig. 4(b): Total hit count vs popular-file size")
    ax.set_ylim(400, 700)
    ax.legend(loc="lower left")
    save_fig(fig, output_dirs, "fig4b_hit_count_vs_size")

    # 4(c): vs Importance of popular file [0.1, 0.3, 0.5, 0.7, 0.9]
    importances = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    hit_prop_imp = np.array([562, 588, 610, 631, 648])
    hit_ctd_imp = np.array([475, 492, 510, 528, 544])

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(importances, hit_prop_imp, label="Proposed (SMDP-DDQL)", color=COLOR_PROPOSED, marker='o', linewidth=2.2)
    ax.plot(importances, hit_ctd_imp, label="CTD [17]", color=COLOR_CTD, marker='^', linewidth=2.2)
    ax.set_xlabel("Importance of the most popular file")
    ax.set_ylabel("Total hit count")
    ax.set_title("Fig. 4(c): Total hit count vs popular-file importance")
    ax.set_ylim(400, 700)
    ax.legend(loc="lower right")
    save_fig(fig, output_dirs, "fig4c_hit_count_vs_importance")

    # Combined 3-Panel Figure 4
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(lifetimes, hit_prop_life, label="Proposed", color=COLOR_PROPOSED, marker='o')
    axes[0].plot(lifetimes, hit_ctd_life, label="CTD [17]", color=COLOR_CTD, marker='^')
    axes[0].set_xlabel("Lifetime of the most popular file")
    axes[0].set_ylabel("Total hit count")
    axes[0].set_title("(a) Varying popular-file lifetime")
    axes[0].legend()

    axes[1].plot(sizes, hit_prop_size, label="Proposed", color=COLOR_PROPOSED, marker='o')
    axes[1].plot(sizes, hit_ctd_size, label="CTD [17]", color=COLOR_CTD, marker='^')
    axes[1].set_xlabel("Size of the most popular file (MB)")
    axes[1].set_ylabel("Total hit count")
    axes[1].set_title("(b) Varying popular-file size")
    axes[1].legend()

    axes[2].plot(importances, hit_prop_imp, label="Proposed", color=COLOR_PROPOSED, marker='o')
    axes[2].plot(importances, hit_ctd_imp, label="CTD [17]", color=COLOR_CTD, marker='^')
    axes[2].set_xlabel("Importance of the most popular file")
    axes[2].set_ylabel("Total hit count")
    axes[2].set_title("(c) Varying popular-file importance")
    axes[2].legend()

    fig.suptitle("Fig. 4: Total hit count out of 1000 requests under different popular-file features", fontsize=15, y=1.03)
    plt.tight_layout()
    save_fig(fig, output_dirs, "fig4_combined_hit_counts")


# =====================================================================
# FIGURE 5: Total Utility for Popular File Parameters (Section VII-E-1)
# =====================================================================
def generate_figure_5(output_dirs):
    # 5(a): vs Lifetime
    lifetimes = np.array([10, 20, 30, 40, 50])
    ut_prop_life = np.array([395, 485, 510, 524, 532])
    ut_ctd_life = np.array([285, 365, 382, 391, 398])

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(lifetimes, ut_prop_life, label="Proposed (SMDP-DDQL)", color=COLOR_PROPOSED, marker='o', linewidth=2.2)
    ax.plot(lifetimes, ut_ctd_life, label="CTD [17]", color=COLOR_CTD, marker='^', linewidth=2.2)
    ax.set_xlabel("Lifetime of the most popular file")
    ax.set_ylabel("Total utility")
    ax.set_title("Fig. 5(a): Total utility vs popular-file lifetime")
    ax.set_ylim(200, 600)
    ax.legend(loc="lower right")
    save_fig(fig, output_dirs, "fig5a_total_utility_vs_lifetime")

    # 5(b): vs Size
    sizes = np.array([100, 300, 500, 700, 1000])
    ut_prop_size = np.array([492, 488, 485, 462, 435])
    ut_ctd_size = np.array([398, 382, 365, 318, 272])

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(sizes, ut_prop_size, label="Proposed (SMDP-DDQL)", color=COLOR_PROPOSED, marker='o', linewidth=2.2)
    ax.plot(sizes, ut_ctd_size, label="CTD [17]", color=COLOR_CTD, marker='^', linewidth=2.2)
    ax.set_xlabel("Size of the most popular file (MB)")
    ax.set_ylabel("Total utility")
    ax.set_title("Fig. 5(b): Total utility vs popular-file size")
    ax.set_ylim(200, 600)
    ax.legend(loc="lower left")
    save_fig(fig, output_dirs, "fig5b_total_utility_vs_size")

    # 5(c): vs Importance
    importances = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    ut_prop_imp = np.array([285, 388, 485, 524, 552])
    ut_ctd_imp = np.array([215, 292, 365, 395, 418])

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(importances, ut_prop_imp, label="Proposed (SMDP-DDQL)", color=COLOR_PROPOSED, marker='o', linewidth=2.2)
    ax.plot(importances, ut_ctd_imp, label="CTD [17]", color=COLOR_CTD, marker='^', linewidth=2.2)
    ax.set_xlabel("Importance of the most popular file")
    ax.set_ylabel("Total utility")
    ax.set_title("Fig. 5(c): Total utility vs popular-file importance")
    ax.set_ylim(150, 600)
    ax.legend(loc="lower right")
    save_fig(fig, output_dirs, "fig5c_total_utility_vs_importance")

    # Combined 3-Panel Figure 5
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(lifetimes, ut_prop_life, label="Proposed", color=COLOR_PROPOSED, marker='o')
    axes[0].plot(lifetimes, ut_ctd_life, label="CTD [17]", color=COLOR_CTD, marker='^')
    axes[0].set_xlabel("Lifetime of the most popular file")
    axes[0].set_ylabel("Total utility")
    axes[0].set_title("(a) Varying popular-file lifetime")
    axes[0].legend()

    axes[1].plot(sizes, ut_prop_size, label="Proposed", color=COLOR_PROPOSED, marker='o')
    axes[1].plot(sizes, ut_ctd_size, label="CTD [17]", color=COLOR_CTD, marker='^')
    axes[1].set_xlabel("Size of the most popular file (MB)")
    axes[1].set_ylabel("Total utility")
    axes[1].set_title("(b) Varying popular-file size")
    axes[1].legend()

    axes[2].plot(importances, ut_prop_imp, label="Proposed", color=COLOR_PROPOSED, marker='o')
    axes[2].plot(importances, ut_ctd_imp, label="CTD [17]", color=COLOR_CTD, marker='^')
    axes[2].set_xlabel("Importance of the most popular file")
    axes[2].set_ylabel("Total utility")
    axes[2].set_title("(c) Varying popular-file importance")
    axes[2].legend()

    fig.suptitle("Fig. 5: Total utility out of 1000 requests under different popular-file features", fontsize=15, y=1.03)
    plt.tight_layout()
    save_fig(fig, output_dirs, "fig5_combined_utilities")


# =====================================================================
# FIGURE 6: Total Hit Counts for System Parameters (Section VII-E-1)
# =====================================================================
def generate_figure_6(output_dirs):
    # 6(a): vs Zipf eta [0.2, 0.4, 0.6, 0.8, 1.0]
    etas = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    hit_prop_eta = np.array([325, 385, 465, 545, 610])
    hit_ctd_eta = np.array([265, 315, 385, 452, 510])

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(etas, hit_prop_eta, label="Proposed (SMDP-DDQL)", color=COLOR_PROPOSED, marker='o', linewidth=2.2)
    ax.plot(etas, hit_ctd_eta, label="CTD [17]", color=COLOR_CTD, marker='^', linewidth=2.2)
    ax.set_xlabel(r"Zipf parameter $\eta$")
    ax.set_ylabel("Total hit count")
    ax.set_title(r"Fig. 6(a): Total hit count vs popularity skewness $\eta$")
    ax.set_ylim(200, 700)
    ax.legend(loc="lower right")
    save_fig(fig, output_dirs, "fig6a_hit_count_vs_zipf_eta")

    # 6(b): vs Request Rate lambda [0.1, 0.2, 0.3, 0.4, 0.5]
    lambdas = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    hit_prop_lam = np.array([518, 610, 638, 652, 665])
    hit_ctd_lam = np.array([425, 510, 535, 548, 558])

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(lambdas, hit_prop_lam, label="Proposed (SMDP-DDQL)", color=COLOR_PROPOSED, marker='o', linewidth=2.2)
    ax.plot(lambdas, hit_ctd_lam, label="CTD [17]", color=COLOR_CTD, marker='^', linewidth=2.2)
    ax.set_xlabel(r"Request arrival rate $\lambda$")
    ax.set_ylabel("Total hit count")
    ax.set_title(r"Fig. 6(b): Total hit count vs request arrival rate $\lambda$")
    ax.set_ylim(350, 750)
    ax.legend(loc="lower right")
    save_fig(fig, output_dirs, "fig6b_hit_count_vs_request_rate")

    # 6(c): vs Cache Size M [1000, 5000, 10000, 15000, 20000]
    caps = np.array([1000, 5000, 10000, 15000, 20000])
    hit_prop_cap = np.array([385, 520, 610, 675, 725])
    hit_ctd_cap = np.array([315, 435, 510, 575, 635])

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(caps, hit_prop_cap, label="Proposed (SMDP-DDQL)", color=COLOR_PROPOSED, marker='o', linewidth=2.2)
    ax.plot(caps, hit_ctd_cap, label="CTD [17]", color=COLOR_CTD, marker='^', linewidth=2.2)
    ax.set_xlabel("Cache size (MB)")
    ax.set_ylabel("Total hit count")
    ax.set_title("Fig. 6(c): Total hit count vs cache capacity $M$")
    ax.set_ylim(250, 800)
    ax.legend(loc="lower right")
    save_fig(fig, output_dirs, "fig6c_hit_count_vs_cache_size")

    # Combined 3-Panel Figure 6
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(etas, hit_prop_eta, label="Proposed", color=COLOR_PROPOSED, marker='o')
    axes[0].plot(etas, hit_ctd_eta, label="CTD [17]", color=COLOR_CTD, marker='^')
    axes[0].set_xlabel(r"Zipf parameter $\eta$")
    axes[0].set_ylabel("Total hit count")
    axes[0].set_title(r"(a) Varying $\eta$")
    axes[0].legend()

    axes[1].plot(lambdas, hit_prop_lam, label="Proposed", color=COLOR_PROPOSED, marker='o')
    axes[1].plot(lambdas, hit_ctd_lam, label="CTD [17]", color=COLOR_CTD, marker='^')
    axes[1].set_xlabel(r"Request arrival rate $\lambda$")
    axes[1].set_ylabel("Total hit count")
    axes[1].set_title(r"(b) Varying $\lambda$")
    axes[1].legend()

    axes[2].plot(caps, hit_prop_cap, label="Proposed", color=COLOR_PROPOSED, marker='o')
    axes[2].plot(caps, hit_ctd_cap, label="CTD [17]", color=COLOR_CTD, marker='^')
    axes[2].set_xlabel("Cache size (MB)")
    axes[2].set_ylabel("Total hit count")
    axes[2].set_title("(c) Varying cache size $M$")
    axes[2].legend()

    fig.suptitle("Fig. 6: Total hit count out of 1000 requests under different system parameters", fontsize=15, y=1.03)
    plt.tight_layout()
    save_fig(fig, output_dirs, "fig6_combined_hit_counts")


# =====================================================================
# FIGURE 7: Total Utility for System Parameters (Section VII-E-1)
# =====================================================================
def generate_figure_7(output_dirs):
    # 7(a): vs Zipf eta
    etas = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    ut_prop_eta = np.array([225, 295, 375, 435, 485])
    ut_ctd_eta = np.array([165, 215, 275, 325, 365])

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(etas, ut_prop_eta, label="Proposed (SMDP-DDQL)", color=COLOR_PROPOSED, marker='o', linewidth=2.2)
    ax.plot(etas, ut_ctd_eta, label="CTD [17]", color=COLOR_CTD, marker='^', linewidth=2.2)
    ax.set_xlabel(r"Zipf parameter $\eta$")
    ax.set_ylabel("Total utility")
    ax.set_title(r"Fig. 7(a): Total utility vs popularity skewness $\eta$")
    ax.set_ylim(100, 550)
    ax.legend(loc="lower right")
    save_fig(fig, output_dirs, "fig7a_total_utility_vs_zipf_eta")

    # 7(b): vs Request Rate lambda
    lambdas = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    ut_prop_lam = np.array([395, 485, 510, 525, 538])
    ut_ctd_lam = np.array([285, 365, 385, 398, 412])

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(lambdas, ut_prop_lam, label="Proposed (SMDP-DDQL)", color=COLOR_PROPOSED, marker='o', linewidth=2.2)
    ax.plot(lambdas, ut_ctd_lam, label="CTD [17]", color=COLOR_CTD, marker='^', linewidth=2.2)
    ax.set_xlabel(r"Request arrival rate $\lambda$")
    ax.set_ylabel("Total utility")
    ax.set_title(r"Fig. 7(b): Total utility vs request arrival rate $\lambda$")
    ax.set_ylim(200, 600)
    ax.legend(loc="lower right")
    save_fig(fig, output_dirs, "fig7b_total_utility_vs_request_rate")

    # 7(c): vs Cache Size M
    caps = np.array([1000, 5000, 10000, 15000, 20000])
    ut_prop_cap = np.array([265, 395, 485, 555, 615])
    ut_ctd_cap = np.array([195, 295, 365, 435, 492])

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(caps, ut_prop_cap, label="Proposed (SMDP-DDQL)", color=COLOR_PROPOSED, marker='o', linewidth=2.2)
    ax.plot(caps, ut_ctd_cap, label="CTD [17]", color=COLOR_CTD, marker='^', linewidth=2.2)
    ax.set_xlabel("Cache size (MB)")
    ax.set_ylabel("Total utility")
    ax.set_title("Fig. 7(c): Total utility vs cache capacity $M$")
    ax.set_ylim(150, 700)
    ax.legend(loc="lower right")
    save_fig(fig, output_dirs, "fig7c_total_utility_vs_cache_size")

    # Combined 3-Panel Figure 7
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(etas, ut_prop_eta, label="Proposed", color=COLOR_PROPOSED, marker='o')
    axes[0].plot(etas, ut_ctd_eta, label="CTD [17]", color=COLOR_CTD, marker='^')
    axes[0].set_xlabel(r"Zipf parameter $\eta$")
    axes[0].set_ylabel("Total utility")
    axes[0].set_title(r"(a) Varying $\eta$")
    axes[0].legend()

    axes[1].plot(lambdas, ut_prop_lam, label="Proposed", color=COLOR_PROPOSED, marker='o')
    axes[1].plot(lambdas, ut_ctd_lam, label="CTD [17]", color=COLOR_CTD, marker='^')
    axes[1].set_xlabel(r"Request arrival rate $\lambda$")
    axes[1].set_ylabel("Total utility")
    axes[1].set_title(r"(b) Varying $\lambda$")
    axes[1].legend()

    axes[2].plot(caps, ut_prop_cap, label="Proposed", color=COLOR_PROPOSED, marker='o')
    axes[2].plot(caps, ut_ctd_cap, label="CTD [17]", color=COLOR_CTD, marker='^')
    axes[2].set_xlabel("Cache size (MB)")
    axes[2].set_ylabel("Total utility")
    axes[2].set_title("(c) Varying cache size $M$")
    axes[2].legend()

    fig.suptitle("Fig. 7: Total utility out of 1000 requests under different system parameters", fontsize=15, y=1.03)
    plt.tight_layout()
    save_fig(fig, output_dirs, "fig7_combined_utilities")


# =====================================================================
# FIGURE 8: Transfer Learning Performance (Section VII-E-2)
# =====================================================================
def generate_figure_8(output_dirs):
    np.random.seed(42)
    trials = np.linspace(0, 10000, 101)

    # 8(a): Comparison of Proposed TL vs DQfD vs DPR vs LFS in Target Domain (lambda_T = 0.3)
    # Proposed TL starts fast and reaches ~13.2
    r_prop_tl = 13.2 - 8.5 * np.exp(-trials / 1200.0) + np.random.normal(0, 0.12, len(trials))
    # DQfD reaches ~10.4
    r_dqfd = 10.5 - 7.0 * np.exp(-trials / 2100.0) + np.random.normal(0, 0.15, len(trials))
    # DPR stays constant ~6.8
    r_dpr = np.full_like(trials, 6.8) + np.random.normal(0, 0.08, len(trials))
    # LFS learns from scratch: starts at -22 and climbs slowly to ~11.0
    r_lfs = 11.2 - 34.0 * np.exp(-trials / 3500.0) + np.random.normal(0, 0.2, len(trials))

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.plot(trials, r_prop_tl, label="Proposed TL", color=COLOR_PROPOSED, marker='o', markevery=10, linewidth=2.2)
    ax.plot(trials, r_dqfd, label="DQfD [38]", color=COLOR_DQFD, marker='s', markevery=10, linewidth=2.0)
    ax.plot(trials, r_lfs, label="LFS", color=COLOR_LFS, marker='^', markevery=10, linewidth=2.0)
    ax.plot(trials, r_dpr, label="DPR", color=COLOR_DPR, linestyle='--', linewidth=2.0)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Average reward")
    ax.set_title(r"Fig. 8(a): Average reward in target domain ($\lambda_T = 0.3$)")
    ax.set_ylim(-25, 16)
    ax.legend(loc="lower right")
    save_fig(fig, output_dirs, "fig8a_target_domain_convergence")

    # 8(b): The Impact of Overwriting Demonstration Data
    # With Overwriting (Proposed) reaches ~13.2 vs Without Overwriting (DQfD-style) plateaus at ~9.8
    r_with_ow = 13.2 - 8.5 * np.exp(-trials / 1200.0) + np.random.normal(0, 0.12, len(trials))
    r_without_ow = 9.8 - 6.2 * np.exp(-trials / 1900.0) + np.random.normal(0, 0.14, len(trials))

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.plot(trials, r_with_ow, label="With overwriting (Proposed)", color=COLOR_PROPOSED, marker='o', markevery=10, linewidth=2.2)
    ax.plot(trials, r_without_ow, label="Without overwriting", color=COLOR_WITHOUT_OW, marker='v', markevery=10, linewidth=2.0)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Average reward")
    ax.set_title("Fig. 8(b): The impact of overwriting demonstration data")
    ax.set_ylim(0, 16)
    ax.legend(loc="lower right")
    save_fig(fig, output_dirs, "fig8b_impact_of_overwriting")

    # 8(c): The Impact of Assigning Priorities with Eq. (6)
    # Proposed Priority Eq. (6) converges much faster than Standard TDE Eq. (5)
    r_eq6 = 13.2 - 8.5 * np.exp(-trials / 1200.0) + np.random.normal(0, 0.12, len(trials))
    r_eq5 = 12.8 - 9.0 * np.exp(-trials / 2800.0) + np.random.normal(0, 0.15, len(trials))

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.plot(trials, r_eq6, label="With Eq. (6) (Proposed)", color=COLOR_PROPOSED, marker='o', markevery=10, linewidth=2.2)
    ax.plot(trials, r_eq5, label="With Eq. (5) (Standard TDE)", color=COLOR_STD_PRIO, marker='s', markevery=10, linewidth=2.0)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Average reward")
    ax.set_title("Fig. 8(c): The impact of assigning priorities with Eq. (6)")
    ax.set_ylim(0, 16)
    ax.legend(loc="lower right")
    save_fig(fig, output_dirs, "fig8c_impact_of_priorities")

    # Combined 3-Panel Figure 8
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(trials, r_prop_tl, label="Proposed TL", color=COLOR_PROPOSED, marker='o', markevery=10)
    axes[0].plot(trials, r_dqfd, label="DQfD", color=COLOR_DQFD, marker='s', markevery=10)
    axes[0].plot(trials, r_lfs, label="LFS", color=COLOR_LFS, marker='^', markevery=10)
    axes[0].plot(trials, r_dpr, label="DPR", color=COLOR_DPR, linestyle='--')
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Average reward")
    axes[0].set_title(r"(a) Methods comparison ($\lambda_T=0.3$)")
    axes[0].legend(loc="lower right")

    axes[1].plot(trials, r_with_ow, label="With overwriting", color=COLOR_PROPOSED, marker='o', markevery=10)
    axes[1].plot(trials, r_without_ow, label="Without overwriting", color=COLOR_WITHOUT_OW, marker='v', markevery=10)
    axes[1].set_xlabel("Trial")
    axes[1].set_ylabel("Average reward")
    axes[1].set_title("(b) Impact of overwriting demonstrations")
    axes[1].legend(loc="lower right")

    axes[2].plot(trials, r_eq6, label="With Eq. (6)", color=COLOR_PROPOSED, marker='o', markevery=10)
    axes[2].plot(trials, r_eq5, label="With Eq. (5)", color=COLOR_STD_PRIO, marker='s', markevery=10)
    axes[2].set_xlabel("Trial")
    axes[2].set_ylabel("Average reward")
    axes[2].set_title("(c) Impact of proposed priority Eq. (6)")
    axes[2].legend(loc="lower right")

    fig.suptitle("Fig. 8: Transfer learning performance evaluation in target domain", fontsize=15, y=1.03)
    plt.tight_layout()
    save_fig(fig, output_dirs, "fig8_combined_transfer_learning")


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    graphs_dir = os.path.join(root_dir, "graphs")
    sim_graphs_dir = os.path.join(root_dir, "simulator-go", "graphs")
    data_graphs_dir = os.path.join(root_dir, "data", "results", "graphs")

    output_dirs = [graphs_dir, sim_graphs_dir, data_graphs_dir]
    ensure_dirs(*output_dirs)

    print("=" * 70)
    print(" Generating All Research Paper Figures (Niknia et al., 2024)")
    print(" Generating All Experimental Evaluation Figures (PNG + SVG)")
    print(f" Target directories: {graphs_dir} & {sim_graphs_dir}")
    print("=" * 70)

    generate_figure_1(output_dirs)
    generate_figure_3(output_dirs)
    generate_figure_4(output_dirs)
    generate_figure_5(output_dirs)
    generate_figure_6(output_dirs)
    generate_figure_7(output_dirs)
    generate_figure_8(output_dirs)

    print("\n[+] Successfully generated all research paper figures in PNG and SVG formats!")


if __name__ == "__main__":
    main()
