from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt


METRICS = ["Pass@3", "DPass@3", "TDPass@3"]
R_CLARIFY_FULL = {
    "Qwen3-80B": [66.97, 62.17, 56.40],
    "GPT-5": [79.08, 72.11, 66.03],
}
WITHOUT_TUR = {
    "Qwen3-80B": [54.68, 53.50, 50.58],
    "GPT-5": [64.95, 61.04, 59.57],
}
GAINS = {
    model: [round(full - baseline, 2) for full, baseline in zip(scores, WITHOUT_TUR[model])]
    for model, scores in R_CLARIFY_FULL.items()
}


def add_value_labels(ax: plt.Axes, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"+{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )


def main() -> None:
    output_dir = Path("imgs")
    output_dir.mkdir(parents=True, exist_ok=True)

    x = list(range(len(METRICS)))
    width = 0.34

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9.5,
        }
    )

    fig, ax = plt.subplots(figsize=(6.2, 3.6))

    qwen_bars = ax.bar(
        [pos - width / 2 for pos in x],
        GAINS["Qwen3-80B"],
        width,
        label="Qwen3-80B",
        color="#4E79A7",
        edgecolor="#2F3A45",
        linewidth=0.8,
    )
    gpt_bars = ax.bar(
        [pos + width / 2 for pos in x],
        GAINS["GPT-5"],
        width,
        label="GPT-5",
        color="#F28E2B",
        edgecolor="#2F3A45",
        linewidth=0.8,
    )

    ax.set_ylabel("Absolute gain over w/o TUR (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS)
    ax.set_ylim(0, 16)
    ax.set_yticks(range(0, 17, 4))

    ax.grid(axis="y", color="0.86", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", width=0.8, length=3)

    add_value_labels(ax, qwen_bars)
    add_value_labels(ax, gpt_bars)

    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        handlelength=1.8,
        columnspacing=1.6,
    )

    fig.tight_layout()
    fig.savefig(output_dir / "tur_gain.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "tur_gain.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
