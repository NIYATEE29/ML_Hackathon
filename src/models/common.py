"""Shared data access, fixed CV folds and evaluation, used by EVERY model.

Do not re-implement metrics in model scripts; call `evaluate()`.

Definitions (observation = one GPS point with a reference label):
  * Top-1 accuracy   : the highest-scoring candidate is the reference arc. A point whose
                       reference arc is not in the candidate set counts as wrong.
  * Macro F1 (primary): per-point arg-max decoding -> every candidate row becomes
                       predicted 1 (the top-1 candidate) or 0, compared with the row label,
                       macro-averaged over the two classes. (`macro_f1_thr` is the same
                       with a fixed 0.5 probability threshold, for reference.)
  * Log loss         : binary log loss over candidate rows of labelled points.
  * Candidate miss   : share of labelled points whose reference arc is not a candidate.
  * Route continuity : among consecutive points' top-1 arcs, share that are the same arc or
                       connected (to_node of previous == from_node of next).
  * Route continuity (transitions): the same check restricted to consecutive points whose
                       arc actually changes. The plain figure above counts "still on the
                       same arc" as continuous, and with ~8 GPS points per 17 m arc about
                       75-81% of consecutive pairs are same-arc, so it starts near 0.75
                       before the model does anything and compresses every model into the
                       top quarter of its range. Reference point: the ground-truth labels
                       themselves score 0.9666 plain but only 0.8210 on transitions, because
                       a route legitimately skips short arcs between two 1 Hz samples.
                       Judge models against that 0.8210 ceiling, not against 1.0.
                       See src/models/route_diagnostics.py for the hop-tolerant version.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score

from src.config import PROCESSED_DIR, RESULTS_DIR, SEED
from src.features.candidates import FEATURES

# Train trips whose reference route does not match the GPS track (< 50% of points lie
# within 30 m of the route): see results/tables/trip_quality.csv. Excluded from fitting/CV.
MIN_LABEL_FRAC = 0.5
FOLDS_PATH = RESULTS_DIR / "folds.json"
N_FOLDS = 5


def excluded_trips() -> list[str]:
    q = pd.read_csv(RESULTS_DIR / "tables" / "trip_quality.csv", dtype={"trip_id": str})
    q = q[q.partition == "train"]
    return sorted(q.loc[q.label_frac < MIN_LABEL_FRAC, "trip_id"])


def load_table(partition: str, drop_excluded: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / f"candidates_{partition}.parquet")
    if partition == "train" and drop_excluded:
        df = df[~df.trip_id.isin(excluded_trips())]
    return df.reset_index(drop=True)


def labelled(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of points that have a reference label (used for fitting and metrics)."""
    return df[df.pt_has_label].reset_index(drop=True)


def make_folds(df: pd.DataFrame, k: int = N_FOLDS, seed: int = SEED) -> dict[str, int]:
    """Trip-level folds balanced by row count. Deterministic; saved to results/folds.json."""
    sizes = df.groupby("trip_id").size()
    order = sizes.sample(frac=1.0, random_state=seed).sort_values(ascending=False, kind="stable")
    load, fold_of = np.zeros(k), {}
    for trip, n in order.items():
        f = int(np.argmin(load)); fold_of[trip] = f; load[f] += n
    return dict(sorted(fold_of.items()))


def get_folds(train_df: pd.DataFrame) -> dict[str, int]:
    if FOLDS_PATH.exists():
        return json.loads(FOLDS_PATH.read_text())
    folds = make_folds(train_df)
    FOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FOLDS_PATH.write_text(json.dumps(folds, indent=1))
    return folds


def evaluate(df: pd.DataFrame, proba: np.ndarray) -> dict:
    """df: candidate rows (from load_table); proba: P(label=1) per row, same order."""
    d = df[["trip_id", "pt", "cand_arc", "from_node", "to_node", "pt_has_label", "label"]].copy()
    d["p"] = np.asarray(proba, dtype=float)
    d["top"] = d.groupby(["trip_id", "pt"])["p"].rank(method="first", ascending=False) == 1

    # --- route continuity: all points with candidates, original order ---
    top = d[d.top].sort_values(["trip_id", "pt"])
    same_trip = top.trip_id.to_numpy()[1:] == top.trip_id.to_numpy()[:-1]
    same_arc = top.cand_arc.to_numpy()[1:] == top.cand_arc.to_numpy()[:-1]
    linked = top.to_node.to_numpy()[:-1] == top.from_node.to_numpy()[1:]
    continuity = float(((same_arc | linked)[same_trip]).mean()) if same_trip.any() else np.nan
    # Same check over real arc changes only, so staying put cannot inflate it.
    moved = same_trip & ~same_arc
    continuity_trans = float(linked[moved].mean()) if moved.any() else np.nan

    # --- observation-level metrics on labelled points ---
    L = d[d.pt_has_label]
    y, p, pred = L.label.to_numpy(), L.p.to_numpy(), L.top.to_numpy().astype(int)
    pt_hit = L[L.top].groupby(["trip_id", "pt"]).label.max()          # top-1 correct per point
    pt_has_cand_true = L.groupby(["trip_id", "pt"]).label.max()       # reference arc in candidates
    per_trip = pt_hit.groupby(level=0).mean()
    return dict(
        n_points=int(len(pt_has_cand_true)), n_rows=int(len(L)),
        top1_acc=float(pt_hit.mean()), top1_acc_per_trip_mean=float(per_trip.mean()),
        macro_f1=float(f1_score(y, pred, average="macro")),
        macro_precision=float(precision_score(y, pred, average="macro", zero_division=0)),
        macro_recall=float(recall_score(y, pred, average="macro", zero_division=0)),
        macro_f1_thr=float(f1_score(y, (p >= 0.5).astype(int), average="macro")),
        log_loss=float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
        candidate_miss_rate=float(1 - pt_has_cand_true.mean()),
        route_continuity=continuity,
        route_continuity_transitions=continuity_trans,
        n_arc_transitions=int(moved.sum()),
    )
