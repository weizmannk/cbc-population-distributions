# coding: utf-8
"""
---------------------------------------------------------------------------------------------------
                                    ABOUT
@author         : Ramodgwendé Weizmann KIENDREBEOGO
@email          : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
@repo           : https://github.com/weizmannk/bns-inspiral-range
@createdOn      : February 2026
@description    : Computes the Binary Neutron Star (BNS) inspiral range for
                  observing runs O4 and O5 (a/b/c), using SNR=8 and a
                  1.4 solar mass binary system.
---------------------------------------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["font.size"] = 14
plt.rcParams["axes.linewidth"] = 1.2

bns_color, nsbh_color, bbh_color = sns.color_palette("rocket", 3)

MODEL_NAMES = {
    "Model_A": "GWTC-3",
    "Model_B": "GWTC-4",
    "Model_C": "Fullpop O4-HL",
    "Model_D": "BGP O4-HL",
}

# Data for 8. months O4a
data = {
    "Model_A": {
        "BNS": {"mid": 8, "lower_delta": -7, "upper_delta": 12},
        "NSBH": {"mid": 1, "lower_delta": -1, "upper_delta": 3},
        "BBH": {"mid": 84, "lower_delta": -49, "upper_delta": 109},
    },
    "Model_B": {
        "BNS": {"mid": 2, "lower_delta": -2, "upper_delta": 5},
        "NSBH": {"mid": 2, "lower_delta": -2, "upper_delta": 7},
        "BBH": {"mid": 40, "lower_delta": -26, "upper_delta": 59},
    },
    "Model_C": {
        "BNS": {"mid": 2, "lower_delta": -2, "upper_delta": 5},
        "NSBH": {"mid": 2, "lower_delta": -2, "upper_delta": 6},
        "BBH": {"mid": 41, "lower_delta": -26, "upper_delta": 59},
    },
    "Model_D": {
        "BNS": {"mid": 0, "lower_delta": -0, "upper_delta": 0},
        "NSBH": {"mid": 1, "lower_delta": -1, "upper_delta": 5},
        "BBH": {"mid": 141, "lower_delta": -79, "upper_delta": 171},
    },
    "Observed": {"BNS": 0, "NSBH": 1, "BBH": 84},
}

populations = ["BNS", "NSBH", "BBH"]
colors_pop = {"BNS": bns_color, "NSBH": nsbh_color, "BBH": bbh_color}

models = ["Model_A", "Model_C", "Model_D"]
# Bounds
for model in models:
    for pop in populations:
        mid = data[model][pop]["mid"]
        lower_delta = data[model][pop]["lower_delta"]
        upper_delta = data[model][pop]["upper_delta"]
        data[model][pop]["low"] = mid + lower_delta
        data[model][pop]["high"] = mid + upper_delta


# fig, axes = plt.subplots(1, 3, figsize=(14, 6), dpi=300)
fig, axes = plt.subplots(1, 3, figsize=(16, 7), dpi=300)

for idx, pop in enumerate(populations):
    ax = axes[idx]

    x_pos = np.arange(len(models))
    predictions = []
    err_low = []
    err_high = []

    for model in models:
        mid = data[model][pop]["mid"]
        low = data[model][pop]["low"]
        high = data[model][pop]["high"]
        predictions.append(mid)
        err_low.append(mid - low)
        err_high.append(high - mid)

    # Plot bars
    bars = ax.bar(
        x_pos,
        predictions,
        width=0.6,
        color=colors_pop[pop],
        alpha=1,
        edgecolor="black",
        linewidth=0.8,
    )

    # Error bars
    ax.errorbar(
        x_pos,
        predictions,
        yerr=[err_low, err_high],
        fmt="none",
        ecolor="black",
        capsize=6,
        capthick=1.8,
        linewidth=1.5,
        zorder=5,
    )

    # Observed line
    obs_value = data["Observed"][pop]
    ax.axhline(
        y=obs_value,
        color="#2A0492",
        linestyle="--",
        linewidth=2,
        label=f"Observed: {obs_value}",
        zorder=1,
        alpha=1,
    )

    # Shaded CI regions
    for i, model in enumerate(models):
        low = data[model][pop]["low"]
        high = data[model][pop]["high"]
        ax.axhspan(
            low,
            high,
            xmin=i / len(models),
            xmax=(i + 1) / len(models),
            alpha=0.12,
            color=colors_pop[pop],
            zorder=1,
        )

    # Add numerical labels: value (lower_delta, upper_delta)
    for i, (bar, val) in enumerate(zip(bars, predictions)):
        height = bar.get_height()
        model = models[i]
        lower_delta = data[model][pop]["lower_delta"]
        upper_delta = data[model][pop]["upper_delta"]

        text_y = height + err_high[i] + max(predictions) * 0.15
        label_text = f"{val}\n({lower_delta:+d}, {upper_delta:+d})"

        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            text_y,
            label_text,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            linespacing=1,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [MODEL_NAMES[m] for m in models],
        fontsize=14,
        rotation=15,
        ha="right",
        fontweight="bold",
    )
    ax.set_ylabel("Detections (8.23 months)", fontsize=14, fontweight="bold")
    ax.set_title(f"{pop}", fontsize=20, fontweight="bold", pad=30, color="navy")
    ax.legend(fontsize=14, framealpha=0.95, edgecolor="gray")
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
path = "./"
basename = "detection_rate_predictions_fullpop_vs_bgp"
for ext in ("png", "pdf"):
    path = f"{basename}.{ext}"
    fig.savefig(path, dpi=600, bbox_inches="tight")
plt.close()
