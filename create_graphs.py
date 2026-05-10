import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# DATA
# ============================================================

models = [
    "DeepCF",
    "Static Mask",
    "RP-UCB",
    "RP-UCB + Attn (U)",
    "RP-UCB + Attn (U+I)"
]

datasets = ["MovieLens-1M", "Amazon Music", "CiteULike-a"]

# HR@10 values
hr_data = {
    "MovieLens-1M": [0.6760, 0.6808, 0.6740, 0.6965, 0.6997],
    "Amazon Music": [0.4465, 0.4476, 0.4240, 0.4336, 0.4944],
    "CiteULike-a": [0.6893, 0.6772, 0.6671, 0.6812, 0.6911]
}

# NDCG@10 values
ndcg_data = {
    "MovieLens-1M": [0.3958, 0.3993, 0.3980, 0.4182, 0.4173],
    "Amazon Music": [0.2626, 0.2471, 0.2475, 0.2510, 0.2731],
    "CiteULike-a": [0.4365, 0.4306, 0.4223, 0.4337, 0.4470]
}

# ============================================================
# FIGURE 1: HR@10 BAR CHART
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(datasets))   # dataset positions
width = 0.15                   # width of each bar

# Plot 5 bars per dataset cluster
for i in range(len(models)):

    values = [
        hr_data["MovieLens-1M"][i],
        hr_data["Amazon Music"][i],
        hr_data["CiteULike-a"][i]
    ]

    ax.bar(
        x + (i - 2) * width,
        values,
        width,
        label=models[i]
    )

# Labels and formatting
ax.set_xticks(x)
ax.set_xticklabels(datasets)

ax.set_ylabel("HR@10")
ax.set_title("HR@10 Performance Across Datasets")

ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("hr10_bar_chart.png", dpi=300)

# ============================================================
# FIGURE 2: NDCG@10 BAR CHART
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))

# Plot 5 bars per dataset cluster
for i in range(len(models)):

    values = [
        ndcg_data["MovieLens-1M"][i],
        ndcg_data["Amazon Music"][i],
        ndcg_data["CiteULike-a"][i]
    ]

    ax.bar(
        x + (i - 2) * width,
        values,
        width,
        label=models[i]
    )

# Labels and formatting
ax.set_xticks(x)
ax.set_xticklabels(datasets)

ax.set_ylabel("NDCG@10")
ax.set_title("NDCG@10 Performance Across Datasets")

ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("ndcg10_bar_chart.png", dpi=300)

# ============================================================
# SHOW PLOTS
# ============================================================

plt.show()

print("Generated:")
print("- hr10_bar_chart.png")
print("- ndcg10_bar_chart.png")