import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss
from xgboost import XGBClassifier

cache_path = Path("results/logs/xgboost_dataset.pkl")
with open(cache_path, "rb") as f:
    X, y, groups, point_ids = pickle.load(f)

print("Loaded cached dataset:", X.shape)

# Unique GPS point identity = trip ID + point index
point_keys = groups.astype(str) + "_" + point_ids.astype(str)

cv = GroupKFold(n_splits=5)
fold_results = []
total_fit_time = 0

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), 1):
    print(f"\n========== FOLD {fold}/5 ==========")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        tree_method="hist",
        n_jobs=-1
    )

    start = time.time()
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    fit_time = time.time() - start
    total_fit_time += fit_time

    probabilities = model.predict_proba(X.iloc[val_idx])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    y_val = y.iloc[val_idx]
    candidate_acc = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(y_val, predictions, average="macro", zero_division=0)
    macro_precision = precision_score(y_val, predictions, average="macro", zero_division=0)
    macro_recall = recall_score(y_val, predictions, average="macro", zero_division=0)
    loss = log_loss(y_val, probabilities, labels=[0, 1])

    val_point_keys = point_keys.iloc[val_idx].reset_index(drop=True)
    val_labels = y_val.reset_index(drop=True)
    val_probs = pd.Series(probabilities)

    point_df = pd.DataFrame({
        "point_key": val_point_keys,
        "label": val_labels,
        "probability": val_probs
    })

    chosen = point_df.loc[point_df.groupby("point_key")["probability"].idxmax()]
    top1 = chosen["label"].mean()

    fold_results.append({
        "fold": fold,
        "candidate_accuracy": candidate_acc,
        "top1_accuracy": top1,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "log_loss": loss,
        "fit_time_sec": fit_time
    })

    print(f"Candidate Accuracy: {candidate_acc:.4f}")
    print(f"Point Top-1 Accuracy: {top1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Log Loss: {loss:.4f}")
    print(f"Fit time: {fit_time:.2f} sec")

results = pd.DataFrame(fold_results)
results.to_csv("results/logs/xgboost_cv_folds_cached.csv", index=False)

summary = results.drop(columns="fold").agg(["mean", "std"])
summary.to_csv("results/logs/xgboost_cv_summary_cached.csv")

print("\n========== FINAL 5-FOLD SUMMARY ==========")
print(results.drop(columns="fold").mean().round(4))
print("\nStandard deviation:")
print(results.drop(columns="fold").std().round(4))
print(f"\nTOTAL TRAINING TIME: {total_fit_time:.2f} sec")
print("\nResults saved under results/logs/")
