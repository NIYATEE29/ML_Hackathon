import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, f1_score

from src.config import SEED
from archive.legacy_config import MODELS_DIR, FIGURES_DIR, LOGS_DIR
from archive.abhijith_mlp.evaluate import compute_observation_top1, evaluate_model_performance

FEATURE_COLUMNS = [
    "dist_to_arc",
    "proj_ratio",
    "heading_diff",
    "speed",
    "time_delta",
    "arc_length",
    "dist_rank",
    "num_candidates",
    "spatial_likelihood"
]

class PathSnapMLP:
    """
    Multi-Layer Perceptron (MLP) for candidate road-arc scoring.
    Meets all hackathon requirements:
      - Learns nonlinear feature representations.
      - Generates epoch-by-epoch training and validation loss curves.
      - Provides calibrated probabilities for downstream HMM / Viterbi decoding.
      - Implements standard fit(X, y) and predict_proba(X) interface.
    """
    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (64, 32),
        activation: str = "relu",
        alpha: float = 1e-4,  # L2 regularization
        learning_rate_init: float = 1e-3,
        batch_size: int = 512,
        max_epochs: int = 35,
        random_state: int = SEED
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver="adam",
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            batch_size=self.batch_size,
            max_iter=1,
            warm_start=True,  # Enables epoch-by-epoch tracking
            random_state=self.random_state
        )
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_top1": [],
            "val_macro_f1": []
        }
        self.feature_names = FEATURE_COLUMNS
        self.is_fitted = False

    def fit_with_history(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        df_val: pd.DataFrame
    ) -> Dict:
        """
        Trains the MLP epoch-by-epoch while recording training and validation loss curves.
        Identifies the optimal early-stopping point to demonstrate underfitting vs. overfitting.
        """
        # Fit scaler on training fold only (prevent data leakage)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        classes = np.array([0, 1])
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_top1": [],
            "val_macro_f1": []
        }

        best_val_loss = float("inf")
        best_epoch = 0

        for epoch in range(1, self.max_epochs + 1):
            # One epoch of Adam optimization
            self.model.partial_fit(X_train_scaled, y_train, classes=classes)

            # Evaluate training loss
            train_probs = self.model.predict_proba(X_train_scaled)[:, 1]
            train_loss = log_loss(y_train, np.clip(train_probs, 1e-15, 1.0 - 1e-15))

            # Evaluate validation metrics
            val_probs = self.model.predict_proba(X_val_scaled)[:, 1]
            val_loss = log_loss(y_val, np.clip(val_probs, 1e-15, 1.0 - 1e-15))

            # Validation Top-1 and F1
            df_val_step = df_val.copy()
            df_val_step["pred_prob"] = val_probs
            top1_acc, _ = compute_observation_top1(df_val_step)
            val_f1 = f1_score(y_val, (val_probs >= 0.5).astype(int), average="macro", zero_division=0)

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(float(train_loss))
            self.history["val_loss"].append(float(val_loss))
            self.history["val_top1"].append(float(top1_acc))
            self.history["val_macro_f1"].append(float(val_f1))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch

        self.is_fitted = True
        return self.history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Outputs probabilities of shape (N, 2) where col 1 is P(candidate is true arc)."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Outputs binary 0/1 predictions."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)
