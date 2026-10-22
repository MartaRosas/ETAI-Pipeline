"""
Evaluation -- stratified k-fold cross-validation on the development set (week 4).


The final test set is not used anywhere in this file -- it stays locked.
See Practical/W4/notebooks/03_cross_validation.ipynb for the full walkthrough.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, get_scorer, make_scorer
from sklearn.model_selection import cross_validate

def build_scorers(metrics: list) -> dict:
    """
    {name: scorer} for scikit-learn metric names (`accuracy`, `roc_auc`, ...), plus `fbeta_<beta>`
    (e.g. `fbeta_2`: recall weighted twice as much as precision).
    """
    scorers = {}
    for name in metrics:
        if name.startswith("fbeta_"):
            scorers[name] = make_scorer(fbeta_score, beta=float(name.split("_", 1)[1]), zero_division=0)
        else:
            scorers[name] = get_scorer(name)
    return scorers

#def cross_validate_pipeline(pipeline, X, y, cv, scoring: str = "accuracy", n_jobs: int = 1):
    """
    Fits a fresh copy of `pipeline` on each fold's training part and scores it on that
    fold's validation part. Because `pipeline` contains the preprocessing too, imputation
    medians, encoder statistics and scaler means are re-learned inside every fold -- the
    validation fold never influences its own preprocessing.

    Returns (fold_scores, y_oof):
      - fold_scores: one row per fold -- train score, validation score, and the gap between
        them (a large, consistent gap = overfitting).
      - y_oof: out-of-fold predictions. Every row gets a prediction from the one fold model
        that did NOT train on it, so all of them are "unseen" -- what the classification
        report and fairness check are computed on (more rows, and the locked test set stays
        untouched). Taken from the same fits as the scores, so nothing is fitted twice.
    """
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True,
                            return_estimator=True, return_indices=True, n_jobs=n_jobs)
    fold_scores = pd.DataFrame({
        "fold": range(1, len(scores["test_score"]) + 1),
        "train": scores["train_score"],
        "validation": scores["test_score"],
    })
    fold_scores["gap"] = fold_scores["train"] - fold_scores["validation"]

    y_oof = np.empty(len(X), dtype=np.asarray(y).dtype)
    for model, val_idx in zip(scores["estimator"], scores["indices"]["test"]):
        y_oof[val_idx] = model.predict(X.iloc[val_idx])
    return fold_scores, y_oof


#def cv_report(fold_scores: pd.DataFrame, scoring: str = "accuracy") -> str:
    """Per-fold table + mean +/- std, as text (printed, and saved to results/)."""
    lines = [
        f"Cross-validation ({len(fold_scores)} stratified folds, metric: {scoring})",
        "",
        fold_scores.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
    ]
    for col in ["train", "validation", "gap"]:
        sign = "+" if col == "gap" else ""
        lines.append(f"{col.capitalize():<11s} mean = {fold_scores[col].mean():{sign}.3f}   "
                     f"std = {fold_scores[col].std(ddof=1):.3f}")
    text = "\n".join(lines)
    print(text)
    return text

def cross_validate_pipeline(pipeline, X, y, cv, scorers: dict, selection_metric: str, n_jobs: int = 1):
    """
    Fits a fresh copy of `pipeline` (preprocessing included) on each fold's training part and
    scores it on that fold's validation part, on every metric in `scorers`.

    Returns (fold_scores, oof_proba):
      - fold_scores: one row per fold with the validation score of every metric, plus the
        train score and train-validation gap of `selection_metric`;
      - oof_proba: out-of-fold probability of the positive class for every row, from the
        fold model that did not train on it. `cv` must be a partition (each row validated
        exactly once).
    """
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scorers, return_train_score=True,
                            return_estimator=True, return_indices=True, n_jobs=n_jobs)
    fold_scores = pd.DataFrame({"fold": range(1, cv.get_n_splits() + 1)})
    fold_scores["train"] = scores[f"train_{selection_metric}"]
    fold_scores["validation"] = scores[f"test_{selection_metric}"]
    fold_scores["gap"] = fold_scores["train"] - fold_scores["validation"]
    for name in scorers:
        fold_scores[name] = scores[f"test_{name}"]

    oof_proba = np.empty(len(X))
    for model, val_idx in zip(scores["estimator"], scores["indices"]["test"]):
        oof_proba[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    return fold_scores, oof_proba


def cv_report(fold_scores: pd.DataFrame, metrics: list, selection_metric: str, title: str) -> str:
    """Per-fold table and the mean and std of every metric, as text."""
    lines = [
        f"{title} ({len(fold_scores)} stratified folds, selection metric: {selection_metric})",
        "",
        fold_scores.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
        fold_scores[metrics].agg(["mean", "std"]).T.to_string(float_format=lambda v: f"{v:.3f}"),
        f"\n{selection_metric}: train-validation gap mean = {fold_scores['gap'].mean():+.3f}",
    ]
    return "\n".join(lines)


#def oof_classification_report(y_true, y_pred) -> str:
    """Classification report on the out-of-fold predictions."""
    text = "Classification report (out-of-fold predictions, development set):\n" + \
        classification_report(y_true, y_pred, zero_division=0)
    print(text)
    return text


#def fairness_report(y_true, y_pred, extras: pd.DataFrame, sensitive_attr: str = "race") -> str:
    """
    Deliberately simple fairness check -- not a substitute for a real audit, just enough
    to show that "accurate" and "fair" are not the same thing.

    For each race group, the false positive rate (share of people who did NOT reoffend
    but were predicted to) for:
        - our own model (out-of-fold predictions on the development set)
        - COMPAS's own risk score (score_text != "Low" counts as a "high risk"
          prediction), on the same rows, for comparison
    """
    df = extras.copy()
    df["y_true"] = pd.Series(y_true).values
    df["y_pred_model"] = y_pred
    df["y_pred_compas"] = (df["score_text"] != "Low").astype(int)

    lines = [
        "False positive rate by race (development set, out-of-fold)",
        "(share of people who did NOT reoffend, but were predicted to)",
        "",
    ]

    for label, col in [("Our model", "y_pred_model"), ("COMPAS's own score", "y_pred_compas")]:
        lines.append(f"  {label}:")
        for group, g in df.groupby(sensitive_attr):
            negatives = g[g["y_true"] == 0]
            if len(negatives) == 0:
                continue
            fpr = (negatives[col] == 1).mean()
            lines.append(f"    {group:<20s} FPR = {fpr:.2f}  (n={len(negatives)})")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    return text

def oof_classification_report(y_true, oof_proba, threshold: float = 0.5) -> str:
    """Confusion matrix and classification report of the out-of-fold predictions, as text."""
    y_pred = (oof_proba >= threshold).astype(int)
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=["true 0", "true 1"],
                      columns=["predicted 0", "predicted 1"])
    return (f"Out-of-fold predictions, development set (threshold {threshold}):\n\n"
            f"{cm.to_string()}\n\n{classification_report(y_true, y_pred, zero_division=0)}")


def fairness_report(y_true, oof_proba, extras: pd.DataFrame, sensitive_attr: str = "race",
                    threshold: float = 0.5) -> str:
    """
    False positive rate (share of people who did not reoffend but were predicted to) per
    group of `sensitive_attr`, for the model's out-of-fold predictions and for COMPAS's own
    score (`score_text` other than "Low" counts as predicted to reoffend), on the same rows.
    Rows with an unknown group are reported as "unknown". A simple check, not a full audit.
    """
    df = extras.copy()
    df[sensitive_attr] = df[sensitive_attr].astype(object).fillna("unknown")
    df["y_true"] = pd.Series(y_true).values
    df["y_pred_model"] = (oof_proba >= threshold).astype(int)
    df["y_pred_compas"] = (df["score_text"] != "Low").astype(int)

    lines = [
        f"False positive rate by {sensitive_attr} (development set, out-of-fold)",
        "(share of people who did NOT reoffend, but were predicted to)",
        "",
    ]
    for label, col in [("Our model", "y_pred_model"), ("COMPAS's own score", "y_pred_compas")]:
        lines.append(f"  {label}:")
        for group, g in df.groupby(sensitive_attr):
            negatives = g[g["y_true"] == 0]
            if len(negatives) == 0:
                continue
            fpr = (negatives[col] == 1).mean()
            lines.append(f"    {group:<20s} FPR = {fpr:.2f}  (n={len(negatives)})")
        lines.append("")
    return "\n".join(lines)