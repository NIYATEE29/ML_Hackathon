"""Group-aware cross-validation evaluation for the XGBoost road candidate model."""

import time

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss, accuracy_score

from archive.legacy_config import RESULTS_DIR
from archive.nikhil_xgboost.xgboost_model import XGBoostCandidateModel
from archive.nikhil_xgboost.xgboost_dataset import build_training_dataset


def evaluate_xgboost():
    """Evaluate XGBoost using 5-fold CV grouped by trip."""

    print("Building training dataset...")

    # Dataset builder now returns four values.
    X, y, groups, point_ids = build_training_dataset()

    print("\nDataset overview")
    print(f"Candidate rows: {len(X):,}")
    print(f"Trips: {groups.nunique()}")
    print(f"Positive candidates: {int(y.sum()):,}")
    print(f"Negative candidates: {int((y == 0).sum()):,}")

    cv = GroupKFold(n_splits=5)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y, groups=groups), start=1
    ):
        print(f"\n========== Fold {fold}/5 ==========")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        val_groups = groups.iloc[val_idx].reset_index(drop=True)
        val_points = point_ids.iloc[val_idx].reset_index(drop=True)

        model = XGBoostCandidateModel()

        # Training time
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - start

        # Prediction time
        start = time.perf_counter()
        probabilities = model.predict_proba(X_val)[:, 1]
        inference_seconds = time.perf_counter() - start

        predictions = (probabilities >= 0.5).astype(int)

        macro_f1 = f1_score(
            y_val, predictions, average="macro", zero_division=0
        )

        candidate_accuracy = accuracy_score(y_val, predictions)

        fold_log_loss = log_loss(
            y_val, probabilities, labels=[0, 1]
        )

        # Correct Top-1 calculation:
        # Group by TRIP and GPS point because point indices repeat between trips.
        top1_table = pd.DataFrame({
            "trip_id": val_groups,
            "point_idx": val_points,
            "label": y_val.reset_index(drop=True),
            "probability": probabilities,
        })

        best_indices = top1_table.groupby(
            ["trip_id", "point_idx"]
        )["probability"].idxmax()

        best_candidates = top1_table.loc[best_indices]
        top1_accuracy = best_candidates["label"].mean()

        unique_points = top1_table.groupby(
            ["trip_id", "point_idx"]
        ).ngroups

        inference_ms_per_point = (
            inference_seconds / max(unique_points, 1) * 1000
        )

        result = {
            "fold": fold,
            "macro_f1": macro_f1,
            "candidate_accuracy": candidate_accuracy,
            "top1_accuracy": top1_accuracy,
            "log_loss": fold_log_loss,
            "fit_seconds": fit_seconds,
            "inference_ms_per_point": inference_ms_per_point,
            "validation_trips": val_groups.nunique(),
            "validation_points": unique_points,
            "validation_candidates": len(val_idx),
        }

        fold_results.append(result)

        print(f"Macro F1:           {macro_f1:.4f}")
        print(f"Candidate accuracy:  {candidate_accuracy:.4f}")
        print(f"Top-1 accuracy:     {top1_accuracy:.4f}")
        print(f"Log loss:           {fold_log_loss:.4f}")
        print(f"Fit time:           {fit_seconds:.2f} seconds")
        print(f"Inference:           {inference_ms_per_point:.4f} ms/point")

    fold_df = pd.DataFrame(fold_results)

    # Calculate mean and standard deviation of the fold metrics.
    summary = {
        "model": "XGBoost",
        "cv_macro_f1_mean": fold_df["macro_f1"].mean(),
        "cv_macro_f1_std": fold_df["macro_f1"].std(),
        "cv_candidate_accuracy_mean": fold_df["candidate_accuracy"].mean(),
        "cv_top1_accuracy_mean": fold_df["top1_accuracy"].mean(),
        "cv_log_loss_mean": fold_df["log_loss"].mean(),
        "cv_fit_seconds_mean": fold_df["fit_seconds"].mean(),
        "cv_inference_ms_per_point_mean": (
            fold_df["inference_ms_per_point"].mean()
        ),
    }

    print("\n========== XGBOOST CV SUMMARY ==========")

    for name, value in summary.items():
        if isinstance(value, (float, np.floating)):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")

    # Save results.
    log_dir = RESULTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    fold_df.to_csv(log_dir / "xgboost_cv_folds.csv", index=False)

    pd.DataFrame([summary]).to_csv(
        log_dir / "xgboost_cv_summary.csv", index=False
    )

    print(f"\nResults saved in: {log_dir.resolve()}")

    return fold_df, summary


if __name__ == "__main__":
    evaluate_xgboost()