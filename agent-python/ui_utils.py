"""
Terminal UI Utilities for SMDP Edge Caching Simulation and RL Agents.
Pure standard Python without any third-party TUI libraries.
Provides clean, readable terminal progress tracking, milestone cards, and aligned tables.
"""

import sys
import time
from typing import List, Dict, Any, Optional, Tuple


def render_banner(title: str, details: Optional[Dict[str, Any]] = None, width: int = 82) -> None:
    """Prints a prominent header banner with '=' borders."""
    print("=" * width)
    print(f"  {title.upper()}")
    if details:
        print("-" * width)
        for key, val in details.items():
            print(f"    * {key:<24}: {val}")
    print("=" * width)


def render_section_header(title: str, width: int = 82) -> None:
    """Prints a clean section divider."""
    print("\n" + "=" * width)
    print(f"  >>> {title}")
    print("=" * width)


def render_milestone(
    domain: str,
    step: int,
    total_steps: int,
    hit_rate: float,
    hits: int,
    avg_reward: float,
    loss: Optional[float] = None,
    epsilon: Optional[float] = None,
    width: int = 82,
) -> None:
    """Prints a milestone summary card at checkpoint intervals."""
    # Clear current progress line first
    sys.stdout.write("\r" + " " * width + "\r")
    sys.stdout.flush()

    pct = (step / total_steps) * 100.0 if total_steps > 0 else 0.0
    print("-" * width)
    line1 = f"  [{domain.upper()} MILESTONE] Step {step:,}/{total_steps:,} ({pct:5.1f}%)"
    print(line1)
    
    parts = [
        f"Hits: {hits:,} ({hit_rate:5.2f}%)",
        f"Avg Reward: {avg_reward:6.2f}",
    ]
    if loss is not None:
        parts.append(f"Loss: {loss:6.4f}")
    if epsilon is not None:
        parts.append(f"Epsilon: {epsilon:5.3f}")
        
    print("    " + " | ".join(parts))
    print("-" * width)


def update_progress(
    prefix: str,
    current: int,
    total: int,
    stats: Optional[Dict[str, Any]] = None,
    bar_length: int = 24,
    width: int = 82,
) -> None:
    """
    Renders an in-place live progress bar using carriage return '\\r'.
    Example:
    [Source Training] [====================>-------] 72.0% | Step: 3600/5000 | Hits: 61.2% | Loss: 0.124
    """
    if total <= 0:
        total = 1
    fraction = min(1.0, max(0.0, current / total))
    filled_len = int(bar_length * fraction)

    if filled_len == bar_length:
        bar = "=" * bar_length
    elif filled_len > 0:
        bar = "=" * (filled_len - 1) + ">" + "-" * (bar_length - filled_len)
    else:
        bar = "-" * bar_length

    pct_str = f"{fraction * 100:5.1f}%"
    stat_parts = []
    if stats:
        for k, v in stats.items():
            if isinstance(v, float):
                stat_parts.append(f"{k}: {v:5.2f}")
            elif isinstance(v, int):
                stat_parts.append(f"{k}: {v:,}")
            else:
                stat_parts.append(f"{k}: {v}")
    stat_str = " | ".join(stat_parts)

    msg = f"\r  [{prefix}] [{bar}] {pct_str} | {stat_str}"
    # Pad to prevent leftover characters on shortening
    if len(msg) < width:
        msg = msg.ljust(width)
    sys.stdout.write(msg)
    sys.stdout.flush()


def end_progress() -> None:
    """Ends the progress bar line by emitting a newline."""
    sys.stdout.write("\n")
    sys.stdout.flush()


def render_card(title: str, metrics: Dict[str, Any], width: int = 82) -> None:
    """Renders a clean metrics card bordered by '=' and '-' characters."""
    print("=" * width)
    print(f"  {title.upper()}")
    print("-" * width)
    for k, v in metrics.items():
        if isinstance(v, float):
            val_str = f"{v:10.2f}"
        elif isinstance(v, int):
            val_str = f"{v:10,d}"
        else:
            val_str = str(v)
        print(f"    {k:<32}: {val_str}")
    print("=" * width)


def render_table(
    headers: List[str],
    rows: List[List[Any]],
    alignments: Optional[List[str]] = None,
) -> None:
    """
    Renders an ASCII table with '=', '-', and '|' separators.
    alignments: list of 'L' (left) or 'R' (right) for each column.
    """
    num_cols = len(headers)
    if alignments is None:
        alignments = ['L'] + ['R'] * (num_cols - 1)

    # Compute column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            s = f"{cell:.2f}" if isinstance(cell, float) else str(cell)
            if len(s) > col_widths[i]:
                col_widths[i] = len(s)

    # Pad widths slightly
    col_widths = [w + 2 for w in col_widths]

    def format_row(cells: List[Any], char: str = " ") -> str:
        formatted = []
        for i, c in enumerate(cells):
            s = f"{c:.2f}" if isinstance(c, float) else str(c)
            align = alignments[i] if i < len(alignments) else 'L'
            if align == 'R':
                formatted.append(f" {s.rjust(col_widths[i] - 1)} ")
            else:
                formatted.append(f" {s.ljust(col_widths[i] - 1)} ")
        return "|" + "|".join(formatted) + "|"

    total_table_width = sum(col_widths) + 2 * num_cols + (num_cols + 1)
    divider = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    outer_border = "=" * len(divider)

    print(outer_border)
    print(format_row(headers))
    print(divider)
    for r in rows:
        print(format_row(r))
    print(outer_border)
