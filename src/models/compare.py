"""Consolidated model comparison: table + figures, built only from committed logs.

Reads every results/logs/<model>_summary.json (the schema written by run_baselines.py /
run_decision_tree.py; HMM summaries are ignored) and, when present, the per-fold CV file
results/logs/<model>_cv_folds.csv. Nothing is re-run and nothing is typed in by hand, so every
number in the outputs is traceable to a log. A new model appears automatically once its
summary JSON follows the same schema.

Usage: python -m src.models.compare
Writes: results/tables/model_comparison.csv, results/tables/model_comparison.md
        results/figures/cmp_cv_macro_f1.png, cmp_student_top1_f1.png, cmp_log_loss.png,
        cmp_f1_vs_inference.png
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR

LOGS = RESULTS_DIR / "logs"

# Display names and a FIXED colour per model (colour follows the entity, never its rank).
NAMES = {"logreg": "Logistic Regression", "rf": "Random Forest", "dtree": "Decision Tree",
         "lgbm": "LightGBM", "xgb": "XGBoost", "mlp": "MLP", "knn": "KNN"}
ORDER = ["logreg", "knn", "dtree", "rf", "lgbm", "xgb", "mlp"]          # stable plotting order
COLORS = {"logreg": "#2a78d6", "rf": "#eb6834", "dtree": "#1baf7a",
          "lgbm": "#eda100", "mlp": "#e87ba4", "knn": "#008300", "xgb": "#6250d6"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def load_summaries(logs: Path = LOGS) -> dict[str, dict]:
    out = {}
    for p in sorted(logs.glob("*_summary.json")):
        key = p.name[: -len("_summary.json")]
        if key.startswith("hmm_"):
            continue
        s = json.loads(p.read_text())
        if "cv_macro_f1_mean" in s:
            out[key] = s
    return dict(sorted(out.items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else 99))


def load_folds(key: str, summary: dict, logs: Path = LOGS) -> np.ndarray | None:
    p = logs / f"{key}_cv_folds.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "max_depth" in df.columns and "max_depth" in summary.get("params", {}):   # depth-sweep file
        df = df[df.max_depth.astype(str) == str(summary["params"]["max_depth"])]
    return df["test_macro_f1"].to_numpy() if "test_macro_f1" in df else None


def key_config(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in params.items())


def build_table(summaries: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for key, s in summaries.items():
        g = s.get
        rows.append({
            "Model": NAMES.get(key, key), "key": key, "Algorithm Family": g("family"),
            "Key Configuration": key_config(g("params", {})),
            "CV Macro F1 (mean)": g("cv_macro_f1_mean"), "CV Macro F1 (std)": g("cv_macro_f1_std"),
            "CV Top-1": g("cv_top1_acc_mean"), "CV Log Loss": g("cv_log_loss_mean"),
            "Student Top-1": g("student_top1_acc"), "Route Continuity": g("student_route_continuity"),
            "Route Continuity (transitions only)": g("student_route_continuity_transitions"),
            "Student Macro F1": g("student_macro_f1"), "Student Log Loss": g("student_log_loss"),
            "Train Time (s)": g("final_fit_seconds"), "Inference (ms/point)": g("student_inference_ms_per_point"),
        })
    t = pd.DataFrame(rows)
    num = [c for c in t.columns if c not in ("Model", "key", "Algorithm Family", "Key Configuration")]
    t[num] = t[num].apply(pd.to_numeric, errors="coerce")   # missing metric -> NaN (rendered N/A)
    return t


def _fmt(v, nd=4):
    return "N/A" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def to_markdown(t: pd.DataFrame) -> str:
    """The Deliverables 3.1 layout. Missing metrics are written N/A (state why in the report)."""
    head = ["Model", "Algorithm Family", "Key Configuration", "CV Macro F1", "Top-1 Accuracy (student-test)",
            "Route Continuity", "Student-Test Macro F1", "Log Loss (student-test)", "Inference Time (ms/point)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in t.iterrows():
        cv = "N/A" if pd.isna(r["CV Macro F1 (mean)"]) else f"{r['CV Macro F1 (mean)']:.4f} ± {_fmt(r['CV Macro F1 (std)'])}"
        lines.append("| " + " | ".join([r["Model"], str(r["Algorithm Family"]), str(r["Key Configuration"]), cv,
                                         _fmt(r["Student Top-1"]), _fmt(r["Route Continuity"]), _fmt(r["Student Macro F1"]),
                                         _fmt(r["Student Log Loss"]), _fmt(r["Inference (ms/point)"], 4)]) + " |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- figures
def _style():
    import matplotlib as mpl
    mpl.use("Agg")
    mpl.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                         "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})
    import matplotlib.pyplot as plt
    return plt


def _color(key):
    return COLORS.get(key, "#6250d6")


def fig_cv_macro_f1(t, folds, path):
    plt = _style(); fig, ax = plt.subplots(figsize=(7.2, 4))
    for i, r in t.reset_index(drop=True).iterrows():
        c = _color(r.key); f = folds.get(r.key)
        if f is not None:
            ax.scatter(np.full(len(f), i) + np.linspace(-0.12, 0.12, len(f)), f, s=16, color=c, alpha=0.45, zorder=2, linewidths=0)
        ax.errorbar(i, r["CV Macro F1 (mean)"], yerr=r["CV Macro F1 (std)"], fmt="o", ms=8, color=c, ecolor=c, elinewidth=1.6,
                    capsize=4, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.annotate(f"{r['CV Macro F1 (mean)']:.4f}", (i, r["CV Macro F1 (mean)"] + r["CV Macro F1 (std)"]), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(t))); ax.set_xticklabels(t.Model, rotation=0)
    ax.set_ylabel("CV macro F1 (5 trip-grouped folds)"); ax.margins(x=0.12, y=0.18)
    ax.set_title("Cross-validation macro F1: mean ± std, dots are the individual folds")
    ax.text(0, -0.2, "Y-axis does not start at zero; differences are small relative to fold spread.", transform=ax.transAxes, fontsize=8, color=INK2)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_student(t, path):
    plt = _style(); fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    for ax, col, title in ((axes[0], "Student Top-1", "Top-1 accuracy"), (axes[1], "Student Macro F1", "Macro F1")):
        for i, r in t.reset_index(drop=True).iterrows():
            ax.scatter(i, r[col], s=70, color=_color(r.key), edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.annotate(f"{r[col]:.4f}", (i, r[col]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
        ax.set_xticks(range(len(t))); ax.set_xticklabels(t.Model, rotation=20, ha="right"); ax.margins(x=0.12, y=0.18)
        ax.set_title(f"Student-test {title}")
    axes[0].set_ylabel("score (axis does not start at zero)")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_log_loss(t, path):
    plt = _style(); fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharey=True)
    for ax, col, title in ((axes[0], "CV Log Loss", "Cross-validation"), (axes[1], "Student Log Loss", "Student-test")):
        for i, r in t.reset_index(drop=True).iterrows():
            v = r[col]
            if pd.isna(v):
                ax.text(i, 0.002, "N/A", ha="center", fontsize=9, color=INK2); continue
            ax.bar(i, v, width=0.55, color=_color(r.key), edgecolor=SURFACE, linewidth=1.5)
            ax.annotate(f"{v:.3f}", (i, v), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
        ax.set_xticks(range(len(t))); ax.set_xticklabels(t.Model, rotation=20, ha="right"); ax.set_title(f"{title} log loss (lower is better)")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("log loss"); axes[0].set_ylim(0, None)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_tradeoff(t, path):
    plt = _style(); fig, ax = plt.subplots(figsize=(7.2, 4.2))
    d = t.dropna(subset=["Inference (ms/point)", "Student Macro F1"])
    for _, r in d.iterrows():
        ax.scatter(r["Inference (ms/point)"], r["Student Macro F1"], s=90, color=_color(r.key), edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(r.Model, (r["Inference (ms/point)"], r["Student Macro F1"]), xytext=(8, 6), textcoords="offset points", fontsize=9)
    ax.set_xscale("log"); ax.margins(x=0.25, y=0.2)
    ax.set_xlabel("inference time, ms per GPS point (log scale)"); ax.set_ylabel("student-test macro F1")
    ax.set_title("Accuracy versus inference cost")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main():
    summaries = load_summaries()
    if not summaries:
        raise SystemExit(f"no *_summary.json found in {LOGS}")
    t = build_table(summaries)
    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True); (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    t.drop(columns="key").to_csv(RESULTS_DIR / "tables" / "model_comparison.csv", index=False)
    (RESULTS_DIR / "tables" / "model_comparison.md").write_text(to_markdown(t))
    folds = {k: load_folds(k, s) for k, s in summaries.items()}
    F = RESULTS_DIR / "figures"
    fig_cv_macro_f1(t, folds, F / "cmp_cv_macro_f1.png"); fig_student(t, F / "cmp_student_top1_f1.png")
    fig_log_loss(t, F / "cmp_log_loss.png"); fig_tradeoff(t, F / "cmp_f1_vs_inference.png")
    print(f"{len(t)} models: {', '.join(t.Model)}\n"); print(to_markdown(t))


if __name__ == "__main__":
    main()
