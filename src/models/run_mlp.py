"""MLP on the SHARED pipeline (same table, same 27 features, same folds, same evaluate()).

Architecture and optimiser follow Abhijith's PathSnapMLP (src/models/mlp.py): hidden layers (64, 32),
ReLU, Adam, L2 alpha=1e-4, lr 1e-3. That file trained on its own 9-feature table and its own folds, so
its numbers are not comparable with logreg / rf / dtree; this script makes them comparable.

Epoch curves: every epoch we record log loss on a fixed 100k-row sample of the training folds and on
the validation fold. The number of epochs is fixed in advance (EPOCHS); the validation curve is only a
diagnostic for over/underfitting and is NOT used to stop or select anything. These are the only real
training histories in our comparison, so they are the only curves we plot.

Usage: python -m src.models.run_mlp
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, RESULTS_DIR, SEED
from src.features.candidates import FEATURES
from src.models.common import evaluate, get_folds, labelled, load_table

PARAMS = dict(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=1e-4,
              learning_rate_init=1e-3, batch_size=1024)
EPOCHS = 20
FAMILY = "Neural network (MLP)"


def fit_with_history(tr, va):
    """Train EPOCHS epochs (fixed). Returns (pipeline, per-epoch history rows)."""
    scaler = StandardScaler().fit(tr[FEATURES].to_numpy(np.float32))          # fitted on training rows only
    X = scaler.transform(tr[FEATURES].to_numpy(np.float32)); y = tr.label.to_numpy()
    sub = np.random.RandomState(SEED).choice(len(X), size=min(100_000, len(X)), replace=False)
    Xv = scaler.transform(va[FEATURES].to_numpy(np.float32)); yv = va.label.to_numpy()
    mlp = MLPClassifier(random_state=SEED, **PARAMS)
    hist = []
    for e in range(1, EPOCHS + 1):
        mlp.partial_fit(X, y, classes=[0, 1])                                  # one pass over the data
        hist.append(dict(epoch=e,
                         train_loss=log_loss(y[sub], np.clip(mlp.predict_proba(X[sub])[:, 1], 1e-7, 1 - 1e-7)),
                         val_loss=log_loss(yv, np.clip(mlp.predict_proba(Xv)[:, 1], 1e-7, 1 - 1e-7))))
    return Pipeline([("scale", scaler), ("mlp", mlp)]), hist


def main():
    train = labelled(load_table("train")); stest = labelled(load_table("student_test"))
    folds = get_folds(train); train["fold"] = train.trip_id.map(folds)
    print(f"train rows {len(train):,} trips {train.trip_id.nunique()} | student_test rows {len(stest):,} | {len(FEATURES)} features", flush=True)
    for d in (RESULTS_DIR / "logs", MODELS_DIR, RESULTS_DIR / "figures"):
        d.mkdir(parents=True, exist_ok=True)

    cv, history = [], []
    for f in sorted(train.fold.unique()):
        tr, va = train[train.fold != f].reset_index(drop=True), train[train.fold == f].reset_index(drop=True)
        t0 = time.time(); model, h = fit_with_history(tr, va); fit_s = time.time() - t0
        t0 = time.time(); p = model.predict_proba(va[FEATURES].to_numpy(np.float32))[:, 1]; inf_s = time.time() - t0
        m = {f"test_{k}": v for k, v in evaluate(va, p).items()}
        p_tr = model.predict_proba(tr[FEATURES].to_numpy(np.float32))[:, 1]
        m.update({f"train_{k}": v for k, v in evaluate(tr, p_tr).items() if k in ("top1_acc", "macro_f1", "log_loss")})
        m.update(model="mlp", fold=int(f), fit_seconds=fit_s, predict_seconds=inf_s)
        cv.append(m); history += [dict(fold=int(f), **r) for r in h]
        print(f"[mlp] fold {f}: top1={m['test_top1_acc']:.4f} f1={m['test_macro_f1']:.4f} ll={m['test_log_loss']:.4f} "
              f"fit={fit_s:.0f}s | final epoch train/val loss {h[-1]['train_loss']:.4f}/{h[-1]['val_loss']:.4f}", flush=True)
    cvd = pd.DataFrame(cv); cvd.to_csv(RESULTS_DIR / "logs" / "mlp_cv_folds.csv", index=False)
    hist = pd.DataFrame(history); hist.to_csv(RESULTS_DIR / "logs" / "mlp_epoch_history_shared.csv", index=False)

    # final model on all kept train trips, scored once on student_test
    t0 = time.time(); final, _ = fit_with_history(train, train.iloc[:1000].reset_index(drop=True)); final_fit_s = time.time() - t0   # curve of this fit is ignored
    t0 = time.time(); p_te = final.predict_proba(stest[FEATURES].to_numpy(np.float32))[:, 1]; inf_s = time.time() - t0
    p_tr = final.predict_proba(train[FEATURES].to_numpy(np.float32))[:, 1]; ev_tr = evaluate(train, p_tr)
    m = {f"test_{k}": v for k, v in evaluate(stest, p_te).items()}
    n_pts = len(stest[["trip_id", "pt"]].drop_duplicates())
    joblib.dump(dict(model=final, features=FEATURES, params=dict(PARAMS, epochs=EPOCHS)), MODELS_DIR / "mlp.joblib", compress=3)
    summ = dict(model="mlp", family=FAMILY, params=dict(PARAMS, epochs=EPOCHS), seed=SEED, n_features=len(FEATURES),
                cv_macro_f1_mean=cvd.test_macro_f1.mean(), cv_macro_f1_std=cvd.test_macro_f1.std(),
                cv_top1_acc_mean=cvd.test_top1_acc.mean(), cv_log_loss_mean=cvd.test_log_loss.mean(),
                cv_route_continuity_mean=cvd.test_route_continuity.mean(), cv_fit_seconds_mean=cvd.fit_seconds.mean(),
                **{f"student_{k.replace('test_', '')}": v for k, v in m.items()},
                student_train_top1_acc=ev_tr["top1_acc"], student_train_macro_f1=ev_tr["macro_f1"],
                final_fit_seconds=final_fit_s, student_predict_seconds=inf_s,
                student_inference_ms_per_point=1000 * inf_s / max(1, n_pts))
    (RESULTS_DIR / "logs" / "mlp_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(f"[mlp] STUDENT-TEST top1={m['test_top1_acc']:.4f} f1={m['test_macro_f1']:.4f} ll={m['test_log_loss']:.4f} "
          f"cont={m['test_route_continuity']:.4f} | CV f1={cvd.test_macro_f1.mean():.4f}±{cvd.test_macro_f1.std():.4f}", flush=True)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    g = hist.groupby("epoch")[["train_loss", "val_loss"]].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for col, c, lab in (("train_loss", "#2a78d6", "training loss (100k-row sample)"), ("val_loss", "#eb6834", "validation-fold loss")):
        ax.plot(g.index, g[(col, "mean")], color=c, lw=2, label=lab)
        ax.fill_between(g.index, g[(col, "mean")] - g[(col, "std")], g[(col, "mean")] + g[(col, "std")], color=c, alpha=0.15, lw=0)
    ax.set_xlabel("epoch"); ax.set_ylabel("log loss"); ax.set_title("MLP training vs validation loss (mean ± std over 5 trip-grouped folds)", loc="left", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(); fig.tight_layout(); fig.savefig(RESULTS_DIR / "figures" / "mlp_loss_curves_shared.png", dpi=150)


if __name__ == "__main__":
    main()
