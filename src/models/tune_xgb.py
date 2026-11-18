"""Bayesian hyper-parameter optimisation of XGBoost (Optuna, TPE sampler).

Protocol (what to say in the viva):
  * Objective = mean macro F1 (per-point top-1 decoding) over the SAME 5 trip-grouped folds
    (results/folds.json) used for every other model. Only the training trips are used.
    student_test is never touched during the search; it is evaluated once, at the end, in `finalize`.
  * Trial 0 is the untuned default, so the study itself shows how much tuning gained.
  * A MedianPruner stops a trial after a fold if it is clearly worse than earlier trials at the same fold.
  * The study lives in a SQLite file, so the search can be stopped and resumed (python -m ... search again).

Usage:
  python -m src.models.tune_xgb search --minutes 8     # run (or continue) the search
  python -m src.models.tune_xgb finalize               # refit best params, CV + student_test, write logs/model/figures
"""
import argparse
import json
import time

import numpy as np
import optuna
import pandas as pd

from src.config import RESULTS_DIR, SEED
from src.features.candidates import FEATURES
from src.models import run_baselines
from src.models.common import evaluate, get_folds, labelled, load_table

LOGS = RESULTS_DIR / "logs"
DB = f"sqlite:///{LOGS / 'xgb_optuna.db'}"
STUDY = "xgb_cv_macro_f1"
DEFAULT = run_baselines.MODELS["xgb"]["params"]


def suggest(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        min_child_weight=trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 0.01, 30.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 0.001, 10.0, log=True),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
        tree_method="hist",
    )


def load_train():
    train = labelled(load_table("train"))
    train["fold"] = train.trip_id.map(get_folds(train))
    return train


def objective_factory(train):
    folds = sorted(train.fold.unique())
    parts = {f: (train[train.fold != f].reset_index(drop=True), train[train.fold == f].reset_index(drop=True)) for f in folds}

    def objective(trial):
        params = suggest(trial)
        scores, t0 = [], time.time()
        for i, f in enumerate(folds):
            tr, va = parts[f]
            model = run_baselines._xgb(params)
            model.fit(tr[FEATURES].to_numpy(np.float32), tr.label)
            p = model.predict_proba(va[FEATURES].to_numpy(np.float32))[:, 1]
            m = evaluate(va, p)
            scores.append(m["macro_f1"])
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                trial.set_user_attr("pruned_after_folds", i + 1)
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_macro_f1", [float(s) for s in scores])
        trial.set_user_attr("seconds", time.time() - t0)
        return float(np.mean(scores))

    return objective


def get_study():
    LOGS.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(
        study_name=STUDY, storage=DB, load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))


def search(minutes):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    train = load_train()
    study = get_study()
    if len(study.trials) == 0:                       # trial 0 = the untuned default, for a like-for-like baseline
        study.enqueue_trial({k: v for k, v in DEFAULT.items() if k != "tree_method"} | dict(reg_alpha=0.001, gamma=0.0))

    def log(study, trial):
        v = f"{trial.value:.4f}" if trial.value is not None else "pruned"
        print(f"trial {trial.number:>3} {v}  best={study.best_value:.4f}  {int(trial.duration.total_seconds())}s  "
              f"depth={trial.params.get('max_depth')} lr={trial.params.get('learning_rate', 0):.3f} "
              f"n={trial.params.get('n_estimators')}", flush=True)

    study.optimize(objective_factory(train), timeout=minutes * 60, callbacks=[log])
    done = [t for t in study.trials if t.state.name == "COMPLETE"]
    print(f"\n{len(study.trials)} trials ({len(done)} complete). best so far {study.best_value:.4f} (trial {study.best_trial.number})")


def finalize():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = get_study()
    best = {**study.best_params, "tree_method": "hist"}
    trials = study.trials_dataframe()
    trials.to_csv(LOGS / "xgb_tuning_trials.csv", index=False)
    (LOGS / "xgb_best_params.json").write_text(json.dumps(
        dict(params=best, cv_macro_f1_mean=study.best_value, trial=study.best_trial.number,
             n_trials=len(study.trials), n_complete=int((trials.state == "COMPLETE").sum()), seed=SEED,
             default_trial0_cv_macro_f1=study.trials[0].value), indent=1))
    # Re-use the shared runner: CV folds + final fit + single student_test evaluation + joblib + summary json.
    run_baselines.MODELS["xgb_tuned"] = dict(
        family="Gradient boosting (trees), Bayesian-tuned", params=best, build=lambda p: run_baselines._xgb(p))
    run_baselines.main(["xgb_tuned"])
    make_figures(study)


def make_figures(study):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = RESULTS_DIR / "figures"
    done = [t for t in study.trials if t.state.name == "COMPLETE"]
    x = [t.number for t in done]; y = [t.value for t in done]
    best_so_far = np.maximum.accumulate(y)
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.scatter(x, y, s=22, color="#6250d6", alpha=0.7, label="completed trial")
    ax.step(x, best_so_far, where="post", color="#0b0b0b", lw=2, label="best so far")
    ax.axhline(study.trials[0].value, color="#52514e", ls="--", lw=1, label="untuned default (trial 0)")
    ax.set(xlabel="trial", ylabel="CV macro F1 (5 folds, train trips)", title="XGBoost Bayesian optimisation (Optuna TPE)")
    ax.legend(frameon=False); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(fig_dir / "xgb_tuning_history.png", dpi=160); plt.close(fig)
    try:
        imp = optuna.importance.get_param_importances(study)
        fig, ax = plt.subplots(figsize=(7.2, 4))
        ax.barh(list(imp)[::-1], list(imp.values())[::-1], color="#6250d6")
        ax.set(xlabel="importance (fANOVA)", title="Which hyper-parameters mattered")
        fig.tight_layout(); fig.savefig(fig_dir / "xgb_tuning_importance.png", dpi=160); plt.close(fig)
    except Exception as e:                           # importance needs enough completed trials
        print("importance plot skipped:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["search", "finalize"])
    ap.add_argument("--minutes", type=float, default=8)
    a = ap.parse_args()
    search(a.minutes) if a.cmd == "search" else finalize()
