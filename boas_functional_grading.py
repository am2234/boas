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

full_df = full_df[~pd.isna(full_df["functional_grade"])]
full_df = full_df.groupby(level=[0, 1, 2]).mean()
full_df["functional_grade"] = full_df["functional_grade"].astype(int)


dfs = []
for _, _df in full_df.groupby(["loop", "fold"]):
    # Pick best threshold by maximising sensitivity and specificity equally
    fpr, tpr, thr = sklearn.metrics.roc_curve(_df["functional_grade"] >= 2, _df["logits_2"])
    best_thr = thr[np.argmax(tpr - fpr)]

    # Decide predicted class for each patient using this threshold
    pred = _df[["logits_0", "logits_1", "logits_2", "logits_3"]].apply(
        decide_stertor_pred, threshs=best_thr, axis=1
    )

    # Record standard Re/Pr/F1 metrics for this class
    conf = sklearn.metrics.classification_report(
        _df["functional_grade"], pred, zero_division=0, output_dict=True
    )
    df = pd.DataFrame(
        [conf[k] for k in ["0", "1", "2", "3"]],
        index=["0 (None)", "1 (Mild BOAS)", "2 (Moderate BOAS)", "3 (Severe BOAS)"],
    )
    df.index = df.index.rename("Class")
    df.columns = [c.title() for c in df.columns]
    dfs.append(df)

# Average per-class metrics across each of the 50 runs
mean_df = sum(dfs) / len(dfs)
std_df = np.sqrt(sum((d - mean_df) ** 2 for d in dfs) / len(dfs))
mean_df["Support"] /= 100
std_df["Support"] /= 100
print("Mean PR table")

final_df = (mean_df * 100).round(1).astype(str) + " ± " + (std_df * 100).round(1).astype(str)
print(final_df)
final_df.to_csv("figures/functional_grade_class_results.csv")
