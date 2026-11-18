"""Confusion matrices for every model on student_test (diagnostic figure for the report).

A row is one (GPS point, candidate arc) pair. The "prediction" is the one our Top-1 metric uses: a candidate is
predicted 1 if it is the highest-probability candidate of its GPS point, otherwise 0. The label is 1 if the candidate
is the reference route arc. This is exactly the prediction behind the logged macro F1, and the script checks that
the macro F1 recomputed from each matrix equals the logged one.

For the last panel the HMM decides which candidate of each point is chosen (the decoded route).

Usage: python -m src.models.confusion
"""

import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import MODELS_DIR, RESULTS_DIR
from src.features.candidates import FEATURES
from src.hmm.run_hmm_eval import decode_rows
from src.hmm.viterbi import HMMParams
from src.models.common import labelled, load_table

LOGS = RESULTS_DIR / "logs"
MODELS = [
    ("logreg", "Logistic Regression"),
    ("dtree", "Decision Tree"),
    ("rf", "Random Forest"),
    ("xgb", "XGBoost"),
    ("xgb_tuned", "XGBoost (tuned)"),
    ("mlp", "MLP"),
]


def top_choice(df, p):
    """1 for the highest-probability candidate of each GPS point (same rule as evaluate())."""
    d = df[["trip_id", "pt"]].assign(p=np.asarray(p, dtype=float))
    return (d.groupby(["trip_id", "pt"])["p"].rank(method="first", ascending=False) == 1).to_numpy().astype(int)


def matrix(y, pred):
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    return np.array([[tn, fp], [fn, tp]])


def macro_f1(m):
    (tn, fp), (fn, tp) = m
    f1_pos = 2 * tp / (2 * tp + fp + fn)
    f1_neg = 2 * tn / (2 * tn + fn + fp)
    return (f1_pos + f1_neg) / 2


def logged_macro_f1(key):
    if key == "hmm":
        return json.loads((LOGS / "hmm_xgb_tuned_summary.json").read_text())["student_test"]["hmm_tuned"]["metrics"][
            "macro_f1"
        ]
    return json.loads((LOGS / f"{key}_summary.json").read_text())["student_macro_f1"]


def draw(ax, m, title):
    pct = m / m.sum(axis=1, keepdims=True) * 100
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{pct[i, j]:.1f}%\n{m[i, j]:,}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if pct[i, j] > 55 else "#0b0b0b",
            )
    ax.set_xticks([0, 1], ["not chosen", "chosen"], fontsize=8)
    ax.set_yticks([0, 1], ["wrong arc", "true arc"], fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    st = labelled(load_table("student_test")).reset_index(drop=True)
    X, y = st[FEATURES].to_numpy(np.float32), st.label.to_numpy()
    panels, rows = [], []
    for key, name in MODELS:
        model = joblib.load(MODELS_DIR / f"{key}.joblib")["model"]
        panels.append((key, name, matrix(y, top_choice(st, model.predict_proba(X)[:, 1]))))
    params = HMMParams(**json.loads((MODELS_DIR / "hmm_params_xgb_tuned.json").read_text()))
    p_clf = joblib.load(MODELS_DIR / "xgb_tuned.joblib")["model"].predict_proba(X)[:, 1]
    hmm, _ = decode_rows(st, p_clf, "student_test", params, {})
    panels.append(("hmm", "XGBoost (tuned) + HMM", matrix(y, top_choice(st, hmm))))

    for key, name, m in panels:
        (tn, fp), (fn, tp) = m
        got, logged = macro_f1(m), logged_macro_f1(key)
        assert abs(got - logged) < 2e-3, f"{name}: matrix gives macro F1 {got:.4f}, log says {logged:.4f}"
        rows.append(
            dict(
                model=name,
                tn=tn,
                fp=fp,
                fn=fn,
                tp=tp,
                true_arc_recall=tp / (tp + fn),
                true_arc_precision=tp / (tp + fp),
                macro_f1_from_matrix=got,
                macro_f1_logged=logged,
            )
        )
    out = pd.DataFrame(rows)
    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / "tables" / "confusion_matrices.csv", index=False)

    fig, axs = plt.subplots(2, 4, figsize=(13, 6.4))
    for ax, (_, name, m) in zip(axs.ravel(), panels):
        draw(ax, m, name)
    axs.ravel()[-1].axis("off")
    fig.suptitle(
        "Confusion matrices on student_test: is the chosen candidate the true arc? (rows add to 100%)", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "confusion_matrices.png", dpi=150)
    plt.close(fig)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
