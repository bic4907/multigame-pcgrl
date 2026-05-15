import matplotlib.pyplot as plt
import numpy as np

# Equal-width left/right layout
# Left: Seen Games (zero-shot)
# Right: Unseen Games (few-shot)
x_seen = np.array([-2.75, -1.8333333333, -0.9166666667, 0.0])
x_unseen = np.array([0.25, 1.1666666667, 2.0833333333, 3.0])
x_all = np.concatenate([x_seen, x_unseen])

tick_labels = ["1", "10", "50", "100", "1", "10", "50", "100"]

# Dummy mean/std values over 5 seeds
data = {
    "1 seen game": {
        "seen_mean":   np.array([0.608, 0.645, 0.627, 0.694]),
        "unseen_mean": np.array([0.699, 0.706, 0.714, 0.722]),
        "seen_std":    np.array([0.006, 0.004, 0.005, 0.004]),
        "unseen_std":  np.array([0.004, 0.004, 0.003, 0.004]),
        "marker": "o",
    },
    "2 seen games": {
        "seen_mean":   np.array([0.623, 0.610, 0.640, 0.671]),
        "unseen_mean": np.array([0.675, 0.681, 0.688, 0.697]),
        "seen_std":    np.array([0.005, 0.004, 0.004, 0.003]),
        "unseen_std":  np.array([0.004, 0.003, 0.003, 0.003]),
        "marker": "s",
    },
    "3 seen games": {
        "seen_mean":   np.array([0.557, 0.581, 0.628, 0.703]),
        "unseen_mean": np.array([0.707, 0.715, 0.723, 0.732]),
        "seen_std":    np.array([0.006, 0.005, 0.004, 0.004]),
        "unseen_std":  np.array([0.004, 0.003, 0.003, 0.003]),
        "marker": "^",
    },
    "4 seen games": {
        "seen_mean":   np.array([0.637, 0.608, 0.635, 0.726]),
        "unseen_mean": np.array([0.730, 0.738, 0.748, 0.760]),
        "seen_std":    np.array([0.005, 0.004, 0.004, 0.004]),
        "unseen_std":  np.array([0.003, 0.003, 0.003, 0.003]),
        "marker": "D",
    },
}

fig, ax = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)

for label, cfg in data.items():
    # Left: seen games
    line, = ax.plot(
        x_seen,
        cfg["seen_mean"],
        marker=cfg["marker"],
        linewidth=1.8,
        markersize=5.0,
        label=label,
    )
    color = line.get_color()

    ax.fill_between(
        x_seen,
        cfg["seen_mean"] - cfg["seen_std"],
        cfg["seen_mean"] + cfg["seen_std"],
        color=color,
        alpha=0.18,
    )

    # Right: unseen games, same color
    ax.plot(
        x_unseen,
        cfg["unseen_mean"],
        marker=cfg["marker"],
        linewidth=1.8,
        markersize=5.0,
        color=color,
    )

    ax.fill_between(
        x_unseen,
        cfg["unseen_mean"] - cfg["unseen_std"],
        cfg["unseen_mean"] + cfg["unseen_std"],
        color=color,
        alpha=0.18,
    )

# Center dashed line
ax.axvline(
    0,
    linestyle="--",
    linewidth=1.2,
    color="black",
    alpha=0.8,
)

# Text near center divider
ax.text(
    -0.18,
    0.565,
    "Seen Games\n(zero-shot)",
    ha="right",
    va="center",
    fontsize=12,
)

ax.text(
    0.18,
    0.565,
    "Unseen Games\n(few-shot)",
    ha="left",
    va="center",
    fontsize=12,
)

# Axis labels
ax.set_xlabel("Train Data Ratio (%)", fontsize=12)
ax.set_ylabel("Normalized Progress", fontsize=12)

# Ticks
ax.set_xticks(x_all)
ax.set_xticklabels(tick_labels, fontsize=9)
ax.tick_params(axis="y", labelsize=9)

# Limits
ax.set_xlim(-3.1, 3.1)
ax.set_ylim(0.55, 0.775)

# Style
ax.grid(True, axis="y", alpha=0.25)
ax.grid(True, axis="x", alpha=0.12)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    title="# Seen Games",
    fontsize=8.8,
    title_fontsize=9.2,
    loc="upper left",
    frameon=True,
)

plt.savefig(
    "seen_unseen_progress_errorband.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    "seen_unseen_progress_errorband.pdf",
    bbox_inches="tight",
)
plt.show()