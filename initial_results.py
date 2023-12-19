import pandas as pd
import utils

full_df = pd.read_csv("anon_results.csv", index_col=[0, 1, 2])

# Remove post-operation visits from test results
full_df = full_df[full_df["visit_type"] != "post-operation"]

## Accuracy of BOAS positive/negative classification using ground-truth stertor grade
stertor_df = full_df[~pd.isna(full_df["stertor"])]
stertor_df = stertor_df.groupby(["anon_id", "anon_visit"]).max()

stertor_se, stertor_sp, stertor_ppv = utils.calc_binary_metrics(
    stertor_df["stertor"] > 1,  # moderate or greater
    stertor_df["functional_grade"] > 1,  # moderate or greater
)
print("Stertor expert predicting BOAS:")
print(f"   Sensitivity = {100*stertor_se:.1f}%, Specificity = {100*stertor_sp:.1f}%\n")


