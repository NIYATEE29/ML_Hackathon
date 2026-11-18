"""Logistic Regression and Random Forest baselines.

Protocol (shared with every other model):
  1. 5-fold group CV by trip on the 80 train trips (minus the 2 excluded), folds from results/folds.json.
  2. Refit on all kept train trips, evaluate on the 10 student_test trips.
  3. Training-set metrics are logged too, as overfitting evidence.
Scalers live inside the sklearn Pipeline, so they are fitted on training folds only.

Usage: python -m src.models.run_baselines [logreg] [rf] [xgb]
"""

import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, RESULTS_DIR, SEED
from src.features.candidates import FEATURES
from src.models.common import evaluate, get_folds, labelled, load_table

MODELS = {
    "logreg": dict(
        family="Linear / probabilistic",
        params=dict(C=1.0, max_iter=1000, solver="lbfgs"),  # L2 penalty (default)
        build=lambda p: make_pipeline(StandardScaler(), LogisticRegression(random_state=SEED, **p)),
    ),
    "rf": dict(
        family="Bagging ensemble (trees)",
        params=dict(n_estimators=100, max_depth=None, min_samples_leaf=20, max_features="sqrt", max_samples=0.25),
        build=lambda p: RandomForestClassifier(n_jobs=2, random_state=SEED, **p),
    ),
    # Boosting. Hyper-parameters are the ones our XGBoost teammate chose (src/models/xgboost_model.py);
    # here they run on the shared 27-feature table, shared folds and shared metrics.
    "xgb": dict(
        family="Gradient boosting (trees)",
        params=dict(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            reg_lambda=1.0,
            tree_method="hist",
        ),
        build=lambda p: _xgb(p),
    ),
}


def _load_tuned():
    """Best parameters found by src/models/tune_xgb.py (absent until the search has been run)."""
    f = RESULTS_DIR / "logs" / "xgb_best_params.json"
    return json.loads(f.read_text())["params"] if f.exists() else None


def _xgb(p):
    from xgboost import XGBClassifier  # imported lazily so the other models do not need xgboost installed

    return XGBClassifier(objective="binary:logistic", eval_metric="logloss", n_jobs=2, random_state=SEED, **p)


if _load_tuned():
    MODELS["xgb_tuned"] = dict(
        family="Gradient boosting (trees), Bayesian-tuned", params=_load_tuned(), build=lambda p: _xgb(p)
    )


def fit_eval(name, tr, te):
    spec = MODELS[name]
    model = spec["build"](spec["params"])
    t0 = time.time()
    model.fit(tr[FEATURES].to_numpy(np.float32), tr.label)
    fit_s = time.time() - t0
    t0 = time.time()
    p_te = model.predict_proba(te[FEATURES].to_numpy(np.float32))[:, 1]
    inf_s = time.time() - t0
    p_tr = model.predict_proba(tr[FEATURES].to_numpy(np.float32))[:, 1]
    out = {f"test_{k}": v for k, v in evaluate(te, p_te).items()}
    out.update({f"train_{k}": v for k, v in evaluate(tr, p_tr).items() if k in ("top1_acc", "macro_f1", "log_loss")})
    out.update(
        fit_seconds=fit_s,
        predict_seconds=inf_s,
        inference_ms_per_point=1000 * inf_s / max(1, len(te[["trip_id", "pt"]].drop_duplicates())),
    )
    return model, out


def main(names):
    train = labelled(load_table("train"))
    stest = labelled(load_table("student_test"))
    folds = get_folds(train)
    train["fold"] = train.trip_id.map(folds)
    print(f"train rows {len(train):,} trips {train.trip_id.nunique()} | student_test rows {len(stest):,}", flush=True)
    (RESULTS_DIR / "logs").mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    for name in names:
        cv = []
        for f in sorted(train.fold.unique()):
            tr, va = train[train.fold != f].reset_index(drop=True), train[train.fold == f].reset_index(drop=True)
            _, m = fit_eval(name, tr, va)
            m.update(model=name, fold=int(f), n_val_trips=int(va.trip_id.nunique()))
            cv.append(m)
            print(
                f"[{name}] fold {f}: top1={m['test_top1_acc']:.4f} f1={m['test_macro_f1']:.4f} "
                f"ll={m['test_log_loss']:.4f} fit={m['fit_seconds']:.0f}s",
                flush=True,
            )
        pd.DataFrame(cv).to_csv(RESULTS_DIR / "logs" / f"{name}_cv_folds.csv", index=False)
        model, m = fit_eval(name, train, stest)
        joblib.dump(
            dict(model=model, features=FEATURES, params=MODELS[name]["params"]),
            MODELS_DIR / f"{name}.joblib",
            compress=3,
        )
        cvd = pd.DataFrame(cv)
        summ = dict(
            model=name,
            family=MODELS[name]["family"],
            params=MODELS[name]["params"],
            seed=SEED,
            n_features=len(FEATURES),
            cv_macro_f1_mean=cvd.test_macro_f1.mean(),
            cv_macro_f1_std=cvd.test_macro_f1.std(),
            cv_top1_acc_mean=cvd.test_top1_acc.mean(),
            cv_log_loss_mean=cvd.test_log_loss.mean(),
            cv_route_continuity_mean=cvd.test_route_continuity.mean(),
            cv_fit_seconds_mean=cvd.fit_seconds.mean(),
            **{f"student_{k.replace('test_', '')}": v for k, v in m.items() if k.startswith("test_")},
            student_train_top1_acc=m["train_top1_acc"],
            student_train_macro_f1=m["train_macro_f1"],
            final_fit_seconds=m["fit_seconds"],
            student_predict_seconds=m["predict_seconds"],
            student_inference_ms_per_point=m["inference_ms_per_point"],
        )
        (RESULTS_DIR / "logs" / f"{name}_summary.json").write_text(json.dumps(summ, indent=1, default=float))
        print(
            f"[{name}] STUDENT-TEST top1={m['test_top1_acc']:.4f} f1={m['test_macro_f1']:.4f} ll={m['test_log_loss']:.4f} "
            f"cont={m['test_route_continuity']:.4f} | CV f1={cvd.test_macro_f1.mean():.4f}±{cvd.test_macro_f1.std():.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main(sys.argv[1:] or ["logreg", "rf"])
