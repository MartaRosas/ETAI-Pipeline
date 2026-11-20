import argparse

import joblib
import pandas as pd
import yaml

from src.preprocessing import clean_dataset, split_features_target


def predict(df, pipeline, config):
    clean = clean_dataset(df, config["diagnostics"])
    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])
    X, _, _ = split_features_target(clean, config["data"], mnar_sources)
    probability = pipeline.predict_proba(X)[:, 1]

    return pd.DataFrame({
        "id": df[config["diagnostics"]["id_column"]].to_numpy(),
        "probability": probability,
        "prediction": (probability >= 0.5).astype(int),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output", nargs="?", default="predictions.csv")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_path = config.get("output", {}).get("model_path", "models/model.joblib")
    pipeline = joblib.load(model_path)
    predictions = predict(pd.read_csv(args.input), pipeline, config)
    predictions.to_csv(args.output, index=False)
    print(f"{len(predictions)} rows predicted -> {args.output}")