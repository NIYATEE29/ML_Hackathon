"""Classifier-only vs classifier+HMM, and decoder-parameter tuning.

Rules followed (Models and Training Guidelines, section 9): decoder parameters are tuned on
TRAIN data only (out-of-fold classifier probabilities, so the emissions are honest) and then
reported once on student_test. Evaluator data is never used.

Usage: python -m src.hmm.run_hmm_eval --model rf --trials 30
"""

import argparse
import json
import time

import joblib
import numpy as np
import optuna
import pandas as pd

from src.config import MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, SEED
from src.data.loader import load_trip
from src.features.candidates import FEATURES
from src.hmm.viterbi import ArcGraph, HMMParams, decode_trip
from src.models.common import evaluate, get_folds, labelled, load_table

optuna.logging.set_verbosity(optuna.logging.WARNING)


def route_f1(pred_arcs, ref_arcs):
    p, r = set(map(int, pred_arcs)), set(map(int, ref_arcs))
    tp = len(p & r)
    prec, rec = tp / max(1, len(p)), tp / max(1, len(r))
    return 2 * prec * rec / max(1e-9, prec + rec)


def decode_rows(df, p, part, params, trips_cache):
    """Decode every trip of df. Returns (hmm_score array aligned to df rows, per-trip decoded frames)."""
    d = df.assign(p=p)
    hmm = np.zeros(len(d))
    decoded = {}
    for tid, g in d.groupby("trip_id", sort=False):
        if tid not in trips_cache:
            tr = load_trip(part_of(tid, part), tid)
            trips_cache[tid] = (ArcGraph(tr.arcs.to_numpy().astype(int), len(tr.nodes)), tr.route)
        graph, _ = trips_cache[tid]
        dec = decode_trip(g, graph, params)
        key = dec.assign(hit=1)
        merged = (
            g[["pt", "cand_arc"]].reset_index().merge(key[["pt", "cand_arc", "hit"]], on=["pt", "cand_arc"], how="left")
        )
        hmm[merged["index"].to_numpy()] = merged["hit"].fillna(0).to_numpy()
        decoded[tid] = dec
    return hmm, decoded


def part_of(tid, default):
    return default


def compare(df, p_clf, p_hmm, decoded, trips_cache):
    """Shared metrics for classifier-only vs HMM, plus route-level F1 (arc-set overlap)."""
    out = {}
    for name, p in (("classifier", p_clf), ("hmm", p_hmm)):
        m = evaluate(df, p)
        if name == "hmm":
            m.pop("log_loss")  # decoded 0/1 scores have no probability meaning
        out[name] = m
    rows = []
    for tid, g in df.assign(pc=p_clf).groupby("trip_id", sort=False):
        ref = trips_cache[tid][1]
        clf_top = g.loc[g.groupby("pt").pc.idxmax(), "cand_arc"]
        rows.append(
            dict(
                trip_id=tid,
                route_f1_classifier=route_f1(clf_top, ref),
                route_f1_hmm=route_f1(decoded[tid].cand_arc, ref),
            )
        )
    rf = pd.DataFrame(rows)
    out["classifier"]["route_f1_mean"], out["hmm"]["route_f1_mean"] = (
        rf.route_f1_classifier.mean(),
        rf.route_f1_hmm.mean(),
    )
    return out, rf


def per_trip(df, p_clf, p_hmm):
    rows = []
    for tid in df.trip_id.unique():
        m = (df.trip_id == tid).to_numpy()
        a, b = evaluate(df[m], p_clf[m]), evaluate(df[m], p_hmm[m])
        rows.append(
            dict(
                trip_id=tid,
                n_points=a["n_points"],
                top1_classifier=a["top1_acc"],
                top1_hmm=b["top1_acc"],
                continuity_classifier=a["route_continuity"],
                continuity_hmm=b["route_continuity"],
            )
        )
    return pd.DataFrame(rows)


def make_oof(model_name):
    """Out-of-fold probabilities for the kept train trips (fixed folds), cached on disk."""
    path = PROCESSED_DIR / f"oof_{model_name}.parquet"
    full = load_table("train")
    if path.exists():
        return full.assign(p=pd.read_parquet(path)["p"].to_numpy())
    from src.models.run_baselines import MODELS

    spec = MODELS[model_name]
    folds = get_folds(labelled(full))
    full["fold"] = full.trip_id.map(folds)
    p = np.zeros(len(full))
    for f in sorted(full.fold.unique()):
        tr = full[(full.fold != f) & full.pt_has_label]
        m = spec["build"](spec["params"]).fit(tr[FEATURES].to_numpy(np.float32), tr.label)
        te = (full.fold == f).to_numpy()
        p[te] = m.predict_proba(full.loc[te, FEATURES].to_numpy(np.float32))[:, 1]
        print(f"OOF fold {f} done", flush=True)
    pd.DataFrame({"p": p}).to_parquet(path)
    return full.assign(p=p)


def tune(oof, trials, n_trips=20, max_pts=4000):
    """Optuna TPE on train OOF probabilities. Objective: Top-1 accuracy of the decoded route."""
    pts = oof.groupby("trip_id").pt.nunique()
    ok = pts[(pts >= 300) & (pts <= max_pts)].sample(frac=1.0, random_state=SEED).index[:n_trips]
    sub = oof[oof.trip_id.isin(ok)].reset_index(drop=True)
    cache = {}
    print(f"tuning on {len(ok)} train trips, {sub.pt_has_label.sum():,} label rows", flush=True)

    def objective(trial):
        P = HMMParams(
            emission_weight=trial.suggest_float("emission_weight", 0.3, 3.0, log=True),
            switch_penalty=trial.suggest_float("switch_penalty", 0.0, 3.0),
            hop_penalty=trial.suggest_float("hop_penalty", 0.0, 2.0),
            reverse_penalty=trial.suggest_float("reverse_penalty", 0.0, 8.0),
            disconnected_penalty=trial.suggest_float("disconnected_penalty", 2.0, 15.0),
            max_hops=trial.suggest_int("max_hops", 0, 5),
            hops_per_sec=trial.suggest_float("hops_per_sec", 0.0, 1.5),
        )
        hmm, _ = decode_rows(sub, sub.p.to_numpy(), "train", P, cache)
        m = evaluate(sub, hmm)
        trial.set_user_attr("continuity", m["route_continuity"])
        return m["top1_acc"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    base = HMMParams()
    study.enqueue_trial(
        {
            k: getattr(base, k)
            for k in (
                "emission_weight",
                "switch_penalty",
                "hop_penalty",
                "reverse_penalty",
                "disconnected_penalty",
                "max_hops",
                "hops_per_sec",
            )
        }
    )
    t0 = time.time()
    study.optimize(objective, n_trials=trials)
    print(f"tuning done in {time.time() - t0:.0f}s, best top1 {study.best_value:.4f}", flush=True)
    return study, sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rf")
    ap.add_argument("--trials", type=int, default=30)
    a = ap.parse_args()
    (RESULTS_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)

    # 1) tune decoder on train OOF
    oof = make_oof(a.model)
    study, sub = tune(oof, a.trials)
    best = HMMParams(**{**HMMParams().as_dict(), **study.best_params})
    study.trials_dataframe().to_csv(RESULTS_DIR / "logs" / f"hmm_{a.model}_decoder_trials.csv", index=False)
    (MODELS_DIR / f"hmm_params_{a.model}.json").write_text(json.dumps(best.as_dict(), indent=1))

    # 2) tuned-vs-default on the tuning trips (train OOF), for the record
    cache = {}
    hmm_def, _ = decode_rows(sub, sub.p.to_numpy(), "train", HMMParams(), cache)
    hmm_best, _ = decode_rows(sub, sub.p.to_numpy(), "train", best, cache)
    tune_rec = {
        n: evaluate(sub, x)
        for n, x in (("classifier_only", sub.p.to_numpy()), ("hmm_default", hmm_def), ("hmm_tuned", hmm_best))
    }

    # 3) report once on student_test
    st = load_table("student_test")
    model = joblib.load(MODELS_DIR / f"{a.model}.joblib")["model"]
    p_clf = model.predict_proba(st[FEATURES].to_numpy(np.float32))[:, 1]
    cache = {}
    res = {}
    for name, P in (("hmm_default", HMMParams()), ("hmm_tuned", best)):
        t0 = time.time()
        hmm, dec = decode_rows(st, p_clf, "student_test", P, cache)
        out, rf = compare(st, p_clf, hmm, dec, cache)
        res[name] = dict(metrics=out["hmm"], decode_seconds=time.time() - t0)
        res["classifier_only"] = dict(metrics=out["classifier"])
        if name == "hmm_tuned":
            pt = per_trip(st, p_clf, hmm).merge(rf, on="trip_id")
            pt.to_csv(RESULTS_DIR / "tables" / f"hmm_vs_classifier_{a.model}_per_trip.csv", index=False)
    summary = dict(
        model=a.model,
        decoder_params=best.as_dict(),
        tuning=dict(
            trials=a.trials,
            n_train_trips=int(sub.trip_id.nunique()),
            objective="decoded Top-1 on train OOF",
            best_value=study.best_value,
            results_on_tuning_trips=tune_rec,
        ),
        student_test=res,
    )
    (RESULTS_DIR / "logs" / f"hmm_{a.model}_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    c, d, t = (res[k]["metrics"] for k in ("classifier_only", "hmm_default", "hmm_tuned"))
    print("\nSTUDENT-TEST      top1    macroF1  continuity  routeF1")
    for n, m in (("classifier-only", c), ("HMM default", d), ("HMM tuned", t)):
        print(
            f"{n:16s} {m['top1_acc']:.4f}  {m['macro_f1']:.4f}   {m['route_continuity']:.4f}     {m['route_f1_mean']:.4f}"
        )


if __name__ == "__main__":
    main()
