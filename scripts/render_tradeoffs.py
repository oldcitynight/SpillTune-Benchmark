#!/usr/bin/env python3
"""Render the R112 summary figure from the packaged machine-readable summary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/spilltune-benchmark-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NATIVE = "independent_native_resident"
STREAMING = "spilltune_streaming"
COLORS = {NATIVE: "#315a7d", STREAMING: "#148a7b"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "reports/summary.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/streaming_tradeoffs.svg")
    parser.add_argument("--preview", type=Path, help="also write a PNG preview to this path")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    resources = summary["sampled_resources"]
    times = summary["training_time"]["training_with_saves_host_seconds"]
    ratios = [
        (
            "Whole-device SMI peak",
            resources["whole_device_smi_memory_used_peak_bytes"][NATIVE],
            resources["whole_device_smi_memory_used_peak_bytes"][STREAMING],
        ),
        ("Training with saves", times[NATIVE], times[STREAMING]),
        (
            "Worker RSS peak",
            resources["worker_rss_peak_bytes"][NATIVE],
            resources["worker_rss_peak_bytes"][STREAMING],
        ),
        (
            "cgroup memory peak",
            resources["cgroup_memory_peak_bytes"][NATIVE],
            resources["cgroup_memory_peak_bytes"][STREAMING],
        ),
    ]

    fig = plt.figure(figsize=(14, 6.2), facecolor="white")
    fig.suptitle(
        "R112: lower sampled GPU memory with a longer training run",
        x=0.04,
        y=0.975,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.18, 1.0),
        left=0.04,
        right=0.98,
        top=0.77,
        bottom=0.21,
        wspace=0.20,
    )

    fig.text(0.055, 0.835, "Resource and time index", fontsize=13, fontweight="bold")
    fig.text(0.055, 0.805, "Native resident = 100; bars show the streaming ratio", fontsize=9.5, color="#52606d")
    resource_group = outer[0, 0].subgridspec(1, 2, width_ratios=(1.85, 4.2), wspace=0.02)
    resource_label_ax = fig.add_subplot(resource_group[0, 0])
    resource_ax = fig.add_subplot(resource_group[0, 1], sharey=resource_label_ax)
    labels = [row[0] for row in ratios]
    stream_index = [100.0 * row[2] / row[1] for row in ratios]
    y_positions = list(range(len(labels)))
    resource_label_ax.set_xlim(0, 1)
    resource_label_ax.set_ylim(len(labels) - 0.5, -0.5)
    resource_label_ax.set_yticks([])
    resource_label_ax.axis("off")
    for y, label in zip(y_positions, labels, strict=True):
        resource_label_ax.text(0, y, label, va="center", ha="left", fontsize=9.5)
    resource_ax.barh(y_positions, stream_index, height=0.54, color=COLORS[STREAMING], zorder=3)
    resource_ax.axvline(100, color="#66717c", linewidth=1.2, linestyle="--", zorder=2)
    resource_ax.set_yticks([])
    resource_ax.set_ylim(len(labels) - 0.5, -0.5)
    resource_ax.set_xlim(0, 160)
    resource_ax.set_xticks([0, 50, 100, 150])
    resource_ax.set_xlabel("Index (native resident = 100)")
    resource_ax.grid(axis="x", color="#d7dee5", linestyle=":", linewidth=0.8, zorder=0)
    resource_ax.set_axisbelow(True)
    resource_ax.spines[["top", "right", "left"]].set_visible(False)
    resource_ax.spines["bottom"].set_color("#aab4bf")
    resource_ax.tick_params(axis="y", length=0, labelleft=False)
    for y, value, (label, native, streaming) in zip(y_positions, stream_index, ratios, strict=True):
        change = 100.0 * (streaming / native - 1.0)
        change_text = f"({change:+.1f}%)" if change else "(0.0%)"
        if value < 100:
            resource_ax.text(value - 2.5, y, f"{value:.1f} {change_text}", va="center", ha="right", fontsize=8.5, color="white", fontweight="bold")
        else:
            resource_ax.text(value + 2.5, y, f"{value:.1f} {change_text}", va="center", ha="left", fontsize=8.5, color="#263440")

    fig.text(0.585, 0.835, "Stored judgment accuracy difference", fontsize=13, fontweight="bold")
    fig.text(0.585, 0.805, "Streaming − native resident, percentage points", fontsize=9.5, color="#52606d")
    right = outer[0, 1].subgridspec(1, 3, width_ratios=(1.08, 0.98, 3.0), wspace=0.02)
    label_ax = fig.add_subplot(right[0, 0])
    value_ax = fig.add_subplot(right[0, 1], sharey=label_ax)
    delta_ax = fig.add_subplot(right[0, 2], sharey=label_ax)

    scores = summary["judgments"]["score_summary"]
    short_names = {
        "codex_56_sol_max": "56 Sol",
        "codex_61_sol_max": "61 Sol",
        "codex_6_astra_max": "Astra",
        "codex_6_luna_max": "Luna",
        "numeric": "Numeric",
    }
    deltas = [float(row["difference_percentage_points"]) for row in scores]
    rows = list(range(len(scores)))
    label_ax.set_xlim(0, 1)
    value_ax.set_xlim(0, 1)
    for ax in (label_ax, value_ax, delta_ax):
        ax.set_ylim(len(scores) - 0.5, -0.5)
        ax.set_yticks([])
        ax.tick_params(left=False, labelleft=False)
    for y, row, delta in zip(rows, scores, deltas, strict=True):
        label_ax.text(0.0, y, short_names[row["channel"]], va="center", fontsize=10)
        value_ax.text(0.98, y, f"{delta:+.1f} pp", va="center", ha="right", fontsize=9, color="#263440")
    label_ax.axis("off")
    value_ax.axis("off")

    delta_ax.set_xlim(-0.72, 0.06)
    delta_ax.set_xticks([-0.6, -0.3, 0.0], labels=["−0.6", "−0.3", "0"])
    delta_ax.xaxis.tick_bottom()
    delta_ax.set_xlabel("Difference (percentage points)", labelpad=8)
    delta_ax.axvline(0, color="#66717c", linewidth=1.1, zorder=2)
    delta_ax.grid(axis="x", color="#d7dee5", linestyle=":", linewidth=0.8, zorder=0)
    delta_ax.set_axisbelow(True)
    delta_ax.spines[["top", "right", "left"]].set_visible(False)
    delta_ax.spines["bottom"].set_color("#aab4bf")
    delta_ax.tick_params(axis="x", labelsize=8.5)
    for y, delta in zip(rows, deltas, strict=True):
        if delta < 0:
            delta_ax.barh(y, delta, height=0.46, color="#b25448", zorder=3)
        else:
            delta_ax.scatter([0], [y], s=28, color="#aab4bf", zorder=4)

    fig.text(
        0.04,
        0.085,
        "All 500 paired generated outputs matched exactly in text, token IDs, and stop reason.\n"
        "The small score differences are rating records on those same outputs; this is one descriptive run pair.",
        fontsize=9.5,
        color="#394956",
        va="bottom",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams["svg.fonttype"] = "none"
    fig.savefig(args.output, format="svg", metadata={"Date": None, "Creator": "SpillTune Benchmark"})
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.preview, format="png", dpi=145)
    plt.close(fig)
    print(f"Rendered {args.output}")
    if args.preview:
        print(f"Rendered {args.preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
