import numpy as np
import pandas as pd
import sklearn.metrics


def decide_stertor_pred(row, threshs):
    """Pick class from logits using NNRank method"""
    row_classif = row > threshs
    for i in range(4):
        if not row_classif[i]:
            return int(max(i - 1, 0))
    return 3


full_df = pd.read_csv("anon_results.csv", index_col=[0, 1, 2])

# Remove post-operation visits from test results
full_df = full_df[full_df["visit_type"] != "post-operation"]

## Prediction of stertor (per-recording)
stertor_df = full_df[~pd.isna(full_df["stertor"])]
stertor_df["stertor"] = stertor_df["stertor"].astype(int)

dfs = []
for _, _df in stertor_df.groupby(["loop", "fold"]):
    # Pick best threshold by maximising sensitivity and specificity equally
    fpr, tpr, thr = sklearn.metrics.roc_curve(_df["stertor"] >= 2, _df["logits_2"])
    best_thr = thr[np.argmax(tpr - fpr)]

    # Decide predicted class for each patient using this threshold
    pred = _df[["logits_0", "logits_1", "logits_2", "logits_3"]].apply(
        decide_stertor_pred, threshs=best_thr, axis=1
    )

    # Record standard Re/Pr/F1 metrics for this class
    conf = sklearn.metrics.classification_report(
        _df["stertor"], pred, zero_division=0, output_dict=True
    )
    dfs.append(pd.DataFrame([conf[k] for k in ["0", "1", "2", "3"]]))

# Average per-class metrics across each of the 50 runs
mean_df = sum(dfs) / len(dfs)
std_df = np.sqrt(sum((d - mean_df) ** 2 for d in dfs) / len(dfs))
mean_df["support"] /= 100
std_df["support"] /= 100
print((mean_df * 100).round(1).astype(str) + " ± " + (std_df * 100).round(1).astype(str))
