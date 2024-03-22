import matplotlib.pyplot as plt
import pandas as pd

import utils

font = {"size": 10, "family": "Arial"}
plt.rc("font", **font)

full_df = pd.read_csv("anon_results.csv", index_col=[0, 1, 2])

# Remove post-operation visits from test results
full_df = full_df[full_df["visit_type"] != "post-operation"]

## First let's do prediction of stertor (per-recording)
stertor_df = full_df[~pd.isna(full_df["stertor"])]

pairs = []
for _, _df in stertor_df.groupby(["loop", "fold"]):
    pairs.append((_df["logits_2"], _df["stertor"] >= 2))  # Target is moderate stertor or greater

fig, axes = plt.subplots(figsize=(4.6, 4.6), dpi=500)
utils.prepare_roc_plot(axes)
utils.plot_averaged_roc(axes, pairs, std_dev=1, op_point=True, op_decimals=0, alpha_fill=0.1)
axes.legend()

plt.savefig("figures/roc_stertor.tif", bbox_inches="tight")
