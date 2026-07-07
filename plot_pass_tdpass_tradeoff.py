from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt


POINTS = [
    {
        "model": "Qwen3-30B",
        "pass5": 69.91,
        "tdpass5": 56.60,
        "marker": "o",
        "color": "#4E79A7",
        "label_offset": (-20, 12),
    },
    {
        "model": "Qwen3-80B",
        "pass5": 75.23,
        "tdpass5": 58.44,
        "marker": "s",
        "color": "#F28E2B",
        "label_offset": (-25, 12),
    },
    {
        "model": "GPT-5 Nano",
        "pass5": 67.16,
        "tdpass5": 56.63,
        "marker": "^",
        "color": "#59A14F",
        "label_offset": (-35, 12),
    },
    {
        "model": "GPT-5",
        "pass5": 83.85,
        "tdpass5": 67.02,
        "marker": "D",
        "color": "#8A63A6",
        "label_offset": (-15, 12),
    },
]


def main() -> None:
    output_dir = Path("imgs")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(5.8, 4.2))

    for point in POINTS:
        ax.scatter(
            point["pass5"],
            point["tdpass5"],
            s=78,
            marker=point["marker"],
            facecolor=point["color"],
            edgecolor="0.15",
            linewidth=0.9,
            zorder=3,
        )
        ax.annotate(
            point["model"],
            xy=(point["pass5"], point["tdpass5"]),
            xytext=point["label_offset"],
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=10,
        )

    ax.set_xlabel("Pass@5 (%)")
    ax.set_ylabel("TDPass@5 (%)")
    ax.set_xlim(65, 86)
    ax.set_ylim(55, 68)
    ax.set_xticks([65, 70, 75, 80, 85])
    ax.set_yticks([55, 58, 61, 64, 67])

    ax.grid(True, axis="both", color="0.86", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", width=0.8, length=3)

    fig.tight_layout()
    fig.savefig(output_dir / "pass_tdpass_tradeoff.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "pass_tdpass_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
