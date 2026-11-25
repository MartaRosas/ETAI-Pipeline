from math import ceil

import numpy as np

from main import load_config
from src.data import load_data
from src.preprocessing import (
    AnomalyScore,
    build_preprocessor,
    clean_dataset,
    drop_duplicate_rows,
    split_dev_test,
    split_features_target,
)

config = load_config()
diagnostics = config["diagnostics"]
preprocessing = config["preprocessing"]

# Reutiliza a mesma limpeza e divisão de dados que o main.py.
df = load_data(config["data"]["path"])
df = clean_dataset(df, diagnostics)
df = drop_duplicate_rows(df, diagnostics.get("id_column"))

X, y, extras = split_features_target(
    df, config["data"], preprocessing.get("mnar_indicator_sources", [])
)
X_dev, _, y_dev, _, _, _ = split_dev_test(
    X,
    y,
    extras,
    test_size=config["test_set"]["size"],
    random_state=config["test_set"]["random_state"],
)

# Ajusta o preprocessor e os detetores apenas no development set.
Z = build_preprocessor(preprocessing).fit_transform(X_dev, y_dev)
top_n = ceil(0.05 * len(y_dev))
anomaly = config.get("anomaly") or {}

for method in ("isolation_forest", "lof"):
    detector = AnomalyScore(
        method=method,
        n_neighbors=anomaly.get("n_neighbors", 20),
        random_state=anomaly.get("random_state", 42),
    )
    scores = detector.fit_transform(Z)[:, -1]
    order = np.argsort(-scores, kind="stable")

    top = y_dev.iloc[order[:top_n]]
    rest = y_dev.iloc[order[top_n:]]

    print(f"\n{method}")
    print(f"  Top 5%: {top.mean():.2%} ({top.sum()}/{len(top)})")
    print(f"  Remaining: {rest.mean():.2%} ({rest.sum()}/{len(rest)})")
    print(f"  Difference: {(top.mean() - rest.mean()) * 100:+.2f} pp")