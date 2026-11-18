import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss, confusion_matrix

def compute_observation_top1(df_eval: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    """
    Computes observation-level Top-1 Accuracy:
    For each GPS observation (point_idx), the model selects the candidate arc with
    the highest predicted probability. Top-1 is correct if that candidate is the true traversed arc.
    Returns: (top1_accuracy, df_best_predictions)
    """
    # Sort so top candidate per point comes first
    df_sorted = df_eval.sort_values(by=["trip_id", "point_idx", "pred_prob"], ascending=[True, True, False])
    # Pick top-1 candidate for each (trip_id, point_idx)
    top1_df = df_sorted.drop_duplicates(subset=["trip_id", "point_idx"], keep="first").copy()
    
    # Correct if the top candidate has label == 1
    top1_correct = (top1_df["label"] == 1).sum()
    total_observations = len(top1_df)
    acc = float(top1_correct / max(1, total_observations))
    return acc, top1_df

def compute_route_continuity(
    predicted_arcs: List[int],
    arcs_table: np.ndarray
) -> float:
    """
    Computes route continuity score:
    Fraction of consecutive transitions where arc[t].to_node == arc[t+1].from_node.
    """
    if len(predicted_arcs) <= 1:
        return 1.0

    connected = 0
    total = len(predicted_arcs) - 1

    for i in range(total):
        curr_arc = predicted_arcs[i]
        next_arc = predicted_arcs[i + 1]
        if curr_arc == next_arc:
            connected += 1
        elif arcs_table[curr_arc, 1] == arcs_table[next_arc, 0]:
            connected += 1

    return float(connected / max(1, total))

def evaluate_model_performance(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    df_eval: pd.DataFrame,
    training_time_s: float = 0.0,
    inference_time_s: float = 0.0,
    candidate_miss_rate: Optional[float] = None,
    arcs_table: Optional[np.ndarray] = None
) -> Dict[str, Union[float, Dict]]:
    """
    Shared team evaluation function as required by Section 5 of AGENTS.md.
    Returns:
        - macro_f1
        - precision
        - recall
        - top1_accuracy
        - log_loss
        - candidate_miss_rate
        - route_continuity (if arcs_table provided)
        - training_time_s
        - inference_time_s
    """
    y_pred = (y_prob >= 0.5).astype(int)

    # Candidate-level classification metrics
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))

    # Log loss (clamped to prevent numerical infinity)
    eps = 1e-15
    y_prob_clamped = np.clip(y_prob, eps, 1.0 - eps)
    try:
        loss = float(log_loss(y_true, y_prob_clamped))
    except Exception:
        loss = float("nan")

    # Observation-level Top-1 Accuracy
    df_eval_copy = df_eval.copy()
    df_eval_copy["pred_prob"] = y_prob
    top1_acc, top1_df = compute_observation_top1(df_eval_copy)

    # Route continuity if road network topology is available
    continuity = None
    if arcs_table is not None:
        predicted_route = top1_df["cand_arc_id"].tolist()
        continuity = compute_route_continuity(predicted_route, arcs_table)

    metrics = {
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall,
        "top1_accuracy": top1_acc,
        "log_loss": loss,
        "candidate_miss_rate": candidate_miss_rate,
        "route_continuity": continuity,
        "training_time_s": training_time_s,
        "inference_time_s": inference_time_s,
        "num_candidate_rows": len(y_true),
        "num_observations": len(top1_df)
    }
    return metrics
