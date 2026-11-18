import json
import time
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, LOGS_DIR,
    SEED, N_FOLDS
)
from src.data.folds import load_folds
from src.models.mlp import PathSnapMLP, FEATURE_COLUMNS
from src.models.evaluate import evaluate_model_performance

def plot_mlp_epoch_curves(history: Dict, save_path: Path):
    """
    Plots training vs. validation loss curves across epochs.
    Mandatory artifact for Deliverable 2 (Technical Report) demonstrating
    underfitting and overfitting analysis for neural networks.
    """
    epochs = history["epoch"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    val_top1 = history["val_top1"]
    val_f1 = history["val_macro_f1"]

    plt.figure(figsize=(12, 5))

    # Subplot 1: Cross-Entropy Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Training Loss", color="#1f77b4", linewidth=2)
    plt.plot(epochs, val_loss, label="Validation Loss", color="#d62728", linewidth=2, linestyle="--")
    best_ep = epochs[int(np.argmin(val_loss))]
    best_loss = min(val_loss)
    plt.scatter([best_ep], [best_loss], color="green", zorder=5, s=60, label=f"Min Val Loss (Ep {best_ep})")
    plt.title("MLP Training & Validation Loss Curve", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Binary Cross-Entropy Log Loss", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)

    # Subplot 2: Observation-Level Metrics
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [a * 100 for a in val_top1], label="Val Top-1 Accuracy (%)", color="#2ca02c", linewidth=2)
    plt.plot(epochs, [f * 100 for f in val_f1], label="Val Macro F1 (%)", color="#9467bd", linewidth=2, linestyle="--")
    plt.title("MLP Validation Accuracy & F1 Across Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Score (%)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved epoch loss curves to {save_path}")

def run_mlp_cv_experiments() -> Dict:
    """
    Executes 5-fold GroupKFold cross-validation for the MLP model
    on the shared feature table using the shared fold assignments.
    """
    train_parquet = PROCESSED_DATA_DIR / "features_train.parquet"
    if not train_parquet.exists():
        raise FileNotFoundError(f"Feature table not found at {train_parquet}. Run extraction first.")

    print(f"Loading shared training features from {train_parquet}...")
    df_train_all = pd.read_parquet(train_parquet)
    print(f"Loaded {len(df_train_all)} rows across {df_train_all['trip_id'].nunique()} trips.")

    folds_data = load_folds()
    folds = folds_data["folds"]

    fold_results = []
    representative_history = None

    print(f"\n--- Starting 5-Fold GroupKFold Cross-Validation for MLP ---")
    for f in folds:
        fold_idx = f["fold"]
        val_trips = set(f["val_trips"])

        # Split strictly by trip (zero observation-level leakage)
        train_mask = ~df_train_all["trip_id"].isin(val_trips)
        val_mask = df_train_all["trip_id"].isin(val_trips)

        df_tr = df_train_all[train_mask]
        df_va = df_train_all[val_mask]

        X_tr = df_tr[FEATURE_COLUMNS].values
        y_tr = df_tr["label"].values
        X_va = df_va[FEATURE_COLUMNS].values
        y_va = df_va["label"].values

        mlp = PathSnapMLP(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_epochs=35,
            random_state=SEED + fold_idx
        )

        t_start = time.time()
        history = mlp.fit_with_history(X_tr, y_tr, X_va, y_va, df_va)
        train_time = time.time() - t_start

        t_infer_start = time.time()
        val_probs = mlp.predict_proba(X_va)[:, 1]
        infer_time = time.time() - t_infer_start

        metrics = evaluate_model_performance(
            y_true=y_va,
            y_prob=val_probs,
            df_eval=df_va,
            training_time_s=train_time,
            inference_time_s=infer_time
        )
        fold_results.append(metrics)

        print(f"Fold {fold_idx}: Val Top-1 = {metrics['top1_accuracy']*100:.2f}%, "
              f"Macro F1 = {metrics['macro_f1']:.4f}, Log Loss = {metrics['log_loss']:.4f}, "
              f"Train Time = {train_time:.1f}s")

        if fold_idx == 0:
            representative_history = history

    # Compute CV averages
    cv_top1 = float(np.mean([m["top1_accuracy"] for m in fold_results]))
    cv_f1 = float(np.mean([m["macro_f1"] for m in fold_results]))
    cv_loss = float(np.mean([m["log_loss"] for m in fold_results]))
    cv_train_time = float(np.mean([m["training_time_s"] for m in fold_results]))
    cv_infer_time = float(np.mean([m["inference_time_s"] for m in fold_results]))

    print(f"\n==========================================")
    print(f"MLP 5-Fold Cross-Validation Summary:")
    print(f"  Mean Observation Top-1 Accuracy: {cv_top1 * 100:.2f}%")
    print(f"  Mean Macro F1-Score:             {cv_f1:.4f}")
    print(f"  Mean Log Loss:                   {cv_loss:.4f}")
    print(f"  Mean Training Time:              {cv_train_time:.1f}s")
    print(f"==========================================")

    # Save epoch loss curves
    curve_path = FIGURES_DIR / "mlp_loss_curves.png"
    plot_mlp_epoch_curves(representative_history, curve_path)

    hist_path = LOGS_DIR / "mlp_epoch_history.json"
    with open(hist_path, "w") as f:
        json.dump(representative_history, f, indent=2)

    # Train final model on entire training dataset and evaluate on Student Test
    test_parquet = PROCESSED_DATA_DIR / "features_student_test.parquet"
    student_test_metrics = None

    if test_parquet.exists():
        print(f"\nEvaluating final MLP model on Student Testing partition...")
        df_test = pd.read_parquet(test_parquet)
        X_train_full = df_train_all[FEATURE_COLUMNS].values
        y_train_full = df_train_all["label"].values

        final_mlp = PathSnapMLP(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_epochs=35,
            random_state=SEED
        )
        t_fit = time.time()
        final_mlp.fit_with_history(
            X_train_full, y_train_full,
            df_test[FEATURE_COLUMNS].values, df_test["label"].values, df_test
        )
        final_train_time = time.time() - t_fit

        t_inf = time.time()
        test_probs = final_mlp.predict_proba(df_test[FEATURE_COLUMNS].values)[:, 1]
        final_infer_time = time.time() - t_inf

        student_test_metrics = evaluate_model_performance(
            y_true=df_test["label"].values,
            y_prob=test_probs,
            df_eval=df_test,
            training_time_s=final_train_time,
            inference_time_s=final_infer_time
        )
        print(f"Student-Test Results:")
        print(f"  Top-1 Accuracy: {student_test_metrics['top1_accuracy']*100:.2f}%")
        print(f"  Macro F1:       {student_test_metrics['macro_f1']:.4f}")
        print(f"  Log Loss:       {student_test_metrics['log_loss']:.4f}")

        # Save model artifact
        model_out = MODELS_DIR / "mlp_model.pkl"
        with open(model_out, "wb") as f:
            pickle.dump(final_mlp, f)
        print(f"Saved trained MLP model to {model_out}")

    # Output consolidated summary matching report table requirements
    summary = {
        "model_name": "Multi-Layer Perceptron (MLP)",
        "algorithm_family": "Neural Networks",
        "key_configuration": "Input(9) -> Hidden(64, ReLU) -> Hidden(32, ReLU) -> Output(1, Sigmoid), Adam, L2=1e-4",
        "cv_top1_accuracy": cv_top1,
        "cv_macro_f1": cv_f1,
        "cv_log_loss": cv_loss,
        "cv_training_time_s": cv_train_time,
        "cv_inference_time_s": cv_infer_time,
        "fold_metrics": fold_results,
        "student_test_metrics": student_test_metrics
    }

    summary_path = LOGS_DIR / "mlp_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved full experiment summary to {summary_path}")

    return summary

if __name__ == "__main__":
    run_mlp_cv_experiments()
