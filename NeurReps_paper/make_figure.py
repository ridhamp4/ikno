"""Generate the main figure from the frozen-holdout extension results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_PATH = (Path(__file__).resolve().parent.parent /
                "results_neurreps_extensions/final_holdout/metrics.json")
with RESULTS_PATH.open() as handle:
    RESULTS = json.load(handle)

METHODS = [
    ("ikno", "IKNO", "#2468A2"),
    ("tfn", "TFN", "#2A9D8F"),
    ("gns", "GNS", "#6B7280"),
    ("gns_aug", "GNS+aug", "#A0A7B4"),
    ("component_egnn", "comp.-EGNN", "#D97706"),
    ("component_egnn_aug", "comp.-EGNN+aug", "#F2B35D"),
]


def seed_values(method, metric):
    return np.asarray([RESULTS[method][str(seed)][metric] for seed in range(3)])


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.75, 2.45))
x = np.arange(len(METHODS))
colors = [color for _, _, color in METHODS]
labels = [label for _, label, _ in METHODS]

eq_values = [seed_values(method, "equivariance_error") for method, _, _ in METHODS]
ax0.bar(x, [value.mean() for value in eq_values],
        yerr=[value.std(ddof=1) for value in eq_values], color=colors,
        width=0.72, capsize=2.0, linewidth=0.45, edgecolor="black")
ax0.set_yscale("log")
ax0.set_ylabel("Normalized equivariance error")
ax0.set_xticks(x, labels, rotation=28, ha="right")
ax0.set_title("(a) Operator consistency")
ax0.grid(axis="y", which="major", linewidth=0.4, alpha=0.35)

degradation = []
for method, _, _ in METHODS:
    values = []
    for seed in range(3):
        run = RESULTS[method][str(seed)]
        values.append(100.0 * run["rotation_delta_max"] / run["one_step"])
    degradation.append(np.asarray(values))
ax1.bar(x, [value.mean() for value in degradation],
        yerr=[value.std(ddof=1) for value in degradation], color=colors,
        width=0.72, capsize=2.0, linewidth=0.45, edgecolor="black")
ax1.set_yscale("log")
ax1.set_ylabel("Maximum one-step degradation (%)")
ax1.set_xticks(x, labels, rotation=28, ha="right")
ax1.set_title("(b) 1,000-rotation sweep")
ax1.grid(axis="y", which="major", linewidth=0.4, alpha=0.35)

for ax in (ax0, ax1):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout(w_pad=1.8)
fig.savefig(Path(__file__).with_name("result_figure.pdf"), bbox_inches="tight")
fig.savefig(Path(__file__).with_name("result_figure.png"), dpi=300,
            bbox_inches="tight")
