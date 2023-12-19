import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import utils

font = {"size": 10, "family": "Arial"}
plt.rc("font", **font)

# Load in full nested cross-validation results
full_df = pd.read_csv("anon_results.csv", index_col=[0, 1, 2])

# Remove post-operation visits from test results
full_df = full_df[full_df["visit_type"] != "post-operation"]

# Take mean prediction if multiple recordings in one patient visit
full_df = full_df[~pd.isna(full_df["functional_grade"])]
full_df = full_df.groupby(level=[0, 1, 2]).mean()

# Assemble prediction-target pairs to plot ROC curves
pairs = []
for _, _df in full_df.groupby(["loop", "fold"]):
    pairs.append((_df["logits_2"], _df["functional_grade"] >= 2))  # Target is moderate or greater

# Plot averaged ROC
fig, axes = plt.subplots(figsize=(4.6, 4.6), dpi=300)
utils.prepare_roc_plot(axes)
_, mean_fpr, mean_tpr, _, _ = utils.plot_averaged_roc(
    axes, pairs, std_dev=1, op_point=False, op_decimals=0, alpha_fill=0.1
)

# Plot expert stertor annotation point for reference (see initial_results.py)
axes.plot(
    100 - 86.9,
    90.6,
    c="k",
    ls="none",
    marker="d",
    label="Expert stertor (Se = 91%, Sp = 87%)",
)

# Plot an example high sensitivity operating point
best_se_idx = np.argmax(2 * mean_tpr - mean_fpr)
best_se = 100 * mean_tpr[best_se_idx]
best_sp = 100 * (1 - mean_fpr[best_se_idx])
print("High sensitivity:", round(best_se, 2), round(best_sp, 2))
axes.plot(
    100 - best_sp,
    best_se,
    c="k",
    ls="none",
    marker="o",
    label=f"Sensitive OP (Se = {round(best_se):.0f}%, Sp = {round(best_sp):.0f}%)",
)

# Plot an example high specificity operating point
best_sp_idx = np.argmax(mean_tpr - mean_fpr)
best_se = 100 * mean_tpr[best_sp_idx]
best_sp = 100 * (1 - mean_fpr[best_sp_idx])
print("High specificity:", round(best_se, 2), round(best_sp, 2))
axes.plot(
    100 - best_sp,
    best_se,
    c="k",
    ls="none",
    marker="x",
    mew=3,
    label=f"Specific OP (Se = {round(best_se):.0f}%, Sp = {round(best_sp):.0f}%)",
)


axes.legend()
plt.savefig("figures/roc_functional_grade.png", bbox_inches="tight")
