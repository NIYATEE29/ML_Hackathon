"""Decision Tree candidate scorer (tree family).

Listed in AGENTS.md as an optional extra rather than one of the five required
models. It earns its place by showing the overfitting behaviour of this feature
set more clearly than anything else in the comparison, and it is cheap enough
to sweep properly rather than guess at.

Protocol is identical to src/models/run_baselines.py so the numbers are
comparable: the same shared table, the same fixed folds from results/folds.json,
the same excluded trips and the same `evaluate()` from src/models/common.py.
The only addition is a max_depth sweep, which is the point of this model.

Usage: python -m src.models.run_decision_tree
Writes: results/logs/dtree_depth_sweep.csv
        results/logs/dtree_cv_folds.csv
        results/logs/dtree_summary.json
        results/tables/dtree_feature_importance.csv
        results/figures/dtree_depth.png
        models/dtree.joblib
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from src.config import MODELS_DIR, RESULTS_DIR, SEED
from src.features.candidates import FEATURES
from src.models.common import evaluate, get_folds, labelled, load_table

FAMILY = "Decision tree (single, non-ensemble)"
DEPTHS = [4, 6, 8, 10, 14, None]
MIN_SAMPLES_LEAF = 20  # matches the RF baseline's leaf size, so depth is the only variable


def build(max_depth):
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=SEED,
    )


def fit_eval(max_depth, tr, te):
    """Same shape as run_baselines.fit_eval, kept local so that file is untouched."""
    model = build(max_depth)
    X_tr = tr[FEATURES].to_numpy(np.float32)
    X_te = te[FEATURES].to_numpy(np.float32)

    t0 = time.time()
    model.fit(X_tr, tr.label)
    fit_s = time.time() - t0

    t0 = time.time()
    p_te = model.predict_proba(X_te)[:, 1]
    inf_s = time.time() - t0
    p_tr = model.predict_proba(X_tr)[:, 1]

    out = {f"test_{k}": v for k, v in evaluate(te, p_te).items()}
    out.update({
        f"train_{k}": v
        for k, v in evaluate(tr, p_tr).items()
        if k in ("top1_acc", "macro_f1", "log_loss")
    })
    n_points = max(1, len(te[["trip_id", "pt"]].drop_duplicates()))
    out.update(
        fit_seconds=fit_s,
        predict_seconds=inf_s,
        inference_ms_per_point=1000 * inf_s / n_points,
        n_leaves=int(model.get_n_leaves()),
        actual_depth=int(model.get_depth()),
    )
    return model, out


def cv_for_depth(train, max_depth):
    rows = []
    for f in sorted(train.fold.unique()):
        tr = train[train.fold != f].reset_index(drop=True)
        va = train[train.fold == f].reset_index(drop=True)
        _, m = fit_eval(max_depth, tr, va)
        m.update(max_depth="none" if max_depth is None else max_depth, fold=int(f))
        rows.append(m)
    return pd.DataFrame(rows)


def main():
    train = labelled(load_table("train"))
    stest = labelled(load_table("student_test"))
    folds = get_folds(train)
    train["fold"] = train.trip_id.map(folds)

    print(
        f"train rows {len(train):,} trips {train.trip_id.nunique()} | "
        f"student_test rows {len(stest):,} trips {stest.trip_id.nunique()} | "
        f"{len(FEATURES)} features",
        flush=True,
    )
    for d in (RESULTS_DIR / "logs", RESULTS_DIR / "tables", RESULTS_DIR / "figures", MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # --- depth sweep: the overfitting evidence ------------------------------
    sweep, all_folds = [], []
    for depth in DEPTHS:
        t0 = time.time()
        cv = cv_for_depth(train, depth)
        all_folds.append(cv)
        row = dict(
            max_depth="none" if depth is None else depth,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            cv_macro_f1_mean=cv.test_macro_f1.mean(),
            cv_macro_f1_std=cv.test_macro_f1.std(),
            cv_top1_acc_mean=cv.test_top1_acc.mean(),
            cv_log_loss_mean=cv.test_log_loss.mean(),
            cv_route_continuity_mean=cv.test_route_continuity.mean(),
            train_macro_f1_mean=cv.train_macro_f1.mean(),
            train_top1_acc_mean=cv.train_top1_acc.mean(),
            overfit_gap_macro_f1=cv.train_macro_f1.mean() - cv.test_macro_f1.mean(),
            mean_leaves=cv.n_leaves.mean(),
            mean_actual_depth=cv.actual_depth.mean(),
            cv_fit_seconds_mean=cv.fit_seconds.mean(),
            sweep_seconds=time.time() - t0,
        )
        sweep.append(row)
        print(
            f"depth={str(row['max_depth']):>4}  CV f1={row['cv_macro_f1_mean']:.4f}"
            f"±{row['cv_macro_f1_std']:.4f}  train f1={row['train_macro_f1_mean']:.4f}"
            f"  gap={row['overfit_gap_macro_f1']:+.4f}  top1={row['cv_top1_acc_mean']:.4f}"
            f"  ll={row['cv_log_loss_mean']:.4f}  cont={row['cv_route_continuity_mean']:.4f}"
            f"  leaves={row['mean_leaves']:,.0f}  [{row['sweep_seconds']:.0f}s]",
            flush=True,
        )

    sweep_df = pd.DataFrame(sweep)
    sweep_df.to_csv(RESULTS_DIR / "logs" / "dtree_depth_sweep.csv", index=False)
    pd.concat(all_folds, ignore_index=True).to_csv(
        RESULTS_DIR / "logs" / "dtree_cv_folds.csv", index=False
    )

    # --- pick on CV macro F1 (the agreed primary metric) --------------------
    best = sweep_df.loc[sweep_df.cv_macro_f1_mean.idxmax()]
    best_depth = None if best.max_depth == "none" else int(best.max_depth)
    print(f"\nBest by CV macro F1: max_depth={best.max_depth}", flush=True)

    model, m = fit_eval(best_depth, train, stest)
    joblib.dump(
        dict(model=model, features=FEATURES,
             params=dict(max_depth=best_depth, min_samples_leaf=MIN_SAMPLES_LEAF)),
        MODELS_DIR / "dtree.joblib", compress=3,
    )

    imp = (
        pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    imp.to_csv(RESULTS_DIR / "tables" / "dtree_feature_importance.csv", index=False)

    # Summary in the same schema as the logreg / rf summaries.
    summ = dict(
        model="dtree", family=FAMILY,
        params=dict(max_depth=best_depth, min_samples_leaf=MIN_SAMPLES_LEAF),
        seed=SEED, n_features=len(FEATURES),
        cv_macro_f1_mean=best.cv_macro_f1_mean, cv_macro_f1_std=best.cv_macro_f1_std,
        cv_top1_acc_mean=best.cv_top1_acc_mean, cv_log_loss_mean=best.cv_log_loss_mean,
        cv_route_continuity_mean=best.cv_route_continuity_mean,
        cv_fit_seconds_mean=best.cv_fit_seconds_mean,
        **{f"student_{k.replace('test_', '')}": v for k, v in m.items() if k.startswith("test_")},
        student_train_top1_acc=m["train_top1_acc"],
        student_train_macro_f1=m["train_macro_f1"],
        final_fit_seconds=m["fit_seconds"],
        student_predict_seconds=m["predict_seconds"],
        student_inference_ms_per_point=m["inference_ms_per_point"],
        n_leaves=m["n_leaves"], actual_depth=m["actual_depth"],
    )
    (RESULTS_DIR / "logs" / "dtree_summary.json").write_text(json.dumps(summ, indent=1, default=float))

    print(
        f"[dtree] STUDENT-TEST top1={m['test_top1_acc']:.4f} f1={m['test_macro_f1']:.4f} "
        f"ll={m['test_log_loss']:.4f} cont={m['test_route_continuity']:.4f} | "
        f"CV f1={best.cv_macro_f1_mean:.4f}±{best.cv_macro_f1_std:.4f}",
        flush=True,
    )
    print("\nTop features:")
    print(imp.head(10).to_string(index=False))

    # --- depth plot ---------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [str(v) for v in sweep_df.max_depth]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, sweep_df.train_macro_f1_mean, "o-", label="train macro F1")
    ax.errorbar(x, sweep_df.cv_macro_f1_mean, yerr=sweep_df.cv_macro_f1_std,
                fmt="s-", capsize=3, label="CV macro F1 (5 trip-grouped folds)")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("macro F1")
    ax.set_title(f"Decision Tree: train vs CV macro F1 (min_samples_leaf={MIN_SAMPLES_LEAF})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "dtree_depth.png", dpi=150)


if __name__ == "__main__":
    main()
