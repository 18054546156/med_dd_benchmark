"""Render the matched teacher-level contrasts used in the manuscript."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def mean_difference(classes, left, right):
    def values(field):
        array = np.array([entry[field] for entry in classes])
        return array[:, :2] if array.ndim == 2 else array[:, :, :2].mean(axis=1)

    return float((values(left) - values(right)).mean())


def render(source, out):
    data = json.loads(source.read_text(encoding="utf-8"))
    contrasts = (
        ("Real decoded − paired real originals", "real_decode_q10", "same_real_raw10_q10", "#a34b28", "s"),
        ("DD decoded − real originals", "dd_q10", "real_q10", "#1c596a", "o"),
        ("DD decoded − real decoded", "dd_q10", "real_decode_q10", "#6a4f79", "D"),
    )
    records = [(dd_id, teacher) for dd_id, run in data["runs"].items() for teacher in run["teachers"]]
    if len(records) != 6:
        raise ValueError("expected two DD sets with three teachers each")
    y = np.arange(len(records))
    fig, ax = plt.subplots(figsize=(7.1, 3.25), layout="constrained")
    for label, left, right, color, marker in contrasts:
        differences = [mean_difference(t["classes"], left, right) for _, t in records]
        ax.scatter(differences, y, color=color, marker=marker, s=34, label=label, zorder=3)
    ax.axvline(0, color="#555555", linewidth=.85)
    ax.axhline(2.5, color="#dadada", linewidth=.85)
    ax.set_yticks(y, [f"{dd}   /   Teacher {t['teacher']}" for dd, t in records])
    ax.invert_yaxis()
    ax.set_xlim(-.095, .095)
    ax.set_xticks([-.08, -.04, 0, .04, .08], ["−0.08", "−0.04", "0", "+0.04", "+0.08"])
    ax.set_xlabel("Difference in class-pair angular-margin q10")
    ax.tick_params(axis="both", length=0, labelsize=9)
    ax.grid(axis="x", color="#ededed", linewidth=.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), frameon=False, fontsize=8.5, ncol=2)
    for suffix in (".png", ".pdf"):
        fig.savefig(out.with_suffix(suffix), dpi=230, bbox_inches="tight", pad_inches=.08)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    render(args.source, args.out)
