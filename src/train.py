"""
Train a match-outcome classifier (Home win / Draw / Away win) on engineered
features and evaluate it with a chronological (time-based) train/test split,
which is the correct split for match data — a random split would leak future
form/Elo information into training.

Usage:
    python src/train.py [--data data/matches.csv] [--test-fraction 0.2]
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS, build_features, load_matches


def time_split(df: pd.DataFrame, test_fraction: float):
    cutoff_idx = int(len(df) * (1 - test_fraction))
    cutoff_date = df["Date"].iloc[cutoff_idx]
    train = df[df["Date"] < cutoff_date]
    test = df[df["Date"] >= cutoff_date]
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/matches.csv")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--model-out", default="models/model.joblib")
    args = parser.parse_args()

    matches = load_matches(args.data)
    features = build_features(matches).dropna(subset=["FTR"])

    train_df, test_df = time_split(features, args.test_fraction)
    print(f"Train: {len(train_df)} matches ({train_df['Date'].min().date()} - {train_df['Date'].max().date()})")
    print(f"Test:  {len(test_df)} matches ({test_df['Date'].min().date()} - {test_df['Date'].max().date()})")

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["FTR"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["FTR"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, pred)
    ll = log_loss(y_test, proba, labels=model.classes_)
    baseline_acc = y_test.value_counts(normalize=True).max()  # always-predict-most-common-class

    print(f"\nAccuracy: {acc:.3f}  (baseline / always predict majority class: {baseline_acc:.3f})")
    print(f"Log loss: {ll:.3f}")
    print("\n" + classification_report(y_test, pred, zero_division=0))

    joblib.dump({"model": model, "feature_columns": FEATURE_COLUMNS}, args.model_out)
    print(f"Saved model -> {args.model_out}")

    metrics = {"accuracy": acc, "log_loss": ll, "baseline_accuracy": baseline_acc, "n_test": len(test_df)}
    metrics_path = os.path.join(os.path.dirname(args.model_out) or ".", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
