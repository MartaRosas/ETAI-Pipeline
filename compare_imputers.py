import pandas as pd

from src.evaluate import corrected_paired_test

runs = pd.read_csv("results/runs.csv", dtype={"run": str}).set_index("run")
folds = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]

median_scores = runs.loc["20261120_193112", folds].to_numpy(dtype=float)

#for imputer, run_id in {
#    "mean": "20261120_193152",
#    "knn": "20261120_193237",
#    "iterative": "20261120_193348",
#    "random_forest": "20261120_193436",
#}.items():
#    scores = runs.loc[run_id, folds].to_numpy(dtype=float)
#    result = corrected_paired_test(scores, median_scores, test_train_ratio=1/4)

#    print(f"\n{imputer} vs median")
#    for key, value in result.items():
#        print(f"  {key}: {value}")



baseline = runs.loc["20261125_114703", folds].to_numpy(dtype=float)

for method, run_id in {
    "isolation_forest": "20261125_114811",
    "lof": "20261125_114905",
}.items():
    scores = runs.loc[run_id, folds].to_numpy(dtype=float)
    result = corrected_paired_test(scores, baseline, test_train_ratio=1/4)
    print(f"{method} vs none: {result}")