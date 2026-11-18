"""How legal is a predicted route, really?

`route_continuity` in src/models/common.py asks whether consecutive top-1 arcs
are directly linked. That under-credits every model, because the reference
route itself skips arcs: arcs have a median length of about 17 m and the GPS
samples at 1 Hz, so at 20 m/s a vehicle can cross a whole short arc without a
single point landing on it. The reference labels only score 0.821 on direct
links for exactly that reason, not because the reference route is broken.

This script asks the fairer question: is the next arc *reachable* from the
current one within a few hops of the road graph? A prediction that skips one
genuine arc is fine; a prediction that teleports to an unconnected street is
not. The ground-truth row is the ceiling to judge models against.

It runs on saved models, so it retrains nothing and changes no committed
metric. Needs data/raw present for the road networks.

Usage:  python -m src.models.route_diagnostics [logreg rf dtree ...]
Writes: results/tables/route_continuity_diagnostic.csv
"""

import sys
from collections import defaultdict, deque

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, RESULTS_DIR
from src.data.loader import load_trip
from src.features.candidates import FEATURES
from src.models.common import labelled, load_table

HOP_LIMIT = 6
REPORT_HOPS = (1, 2, 3, 6)
PARTITION = "student_test"


def _adjacency(partition: str, trip_id: str):
    """(arcs, from_node -> arc ids leaving it) for one trip."""
    arcs = load_trip(partition, trip_id).arcs.to_numpy().astype(int)
    by_from = defaultdict(list)
    for i, (u, _v) in enumerate(arcs):
        by_from[u].append(i)
    return arcs, by_from


def hops_between(adj, src: int, dst: int, limit: int = HOP_LIMIT) -> int:
    """Fewest arc hops from src to dst, or -1 if not reachable within `limit`."""
    arcs, by_from = adj
    if src == dst:
        return 0
    seen, queue = {src}, deque([(src, 0)])
    while queue:
        cur, d = queue.popleft()
        if d >= limit:
            continue
        for nxt in by_from.get(arcs[cur][1], ()):
            if nxt == dst:
                return d + 1
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, d + 1))
    return -1


def top1_sequence(df: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    d = df[["trip_id", "pt", "cand_arc"]].copy()
    d["p"] = np.asarray(proba, float)
    d["top"] = d.groupby(["trip_id", "pt"])["p"].rank(method="first", ascending=False) == 1
    return d[d.top].sort_values(["trip_id", "pt"])


def diagnose(df: pd.DataFrame, proba: np.ndarray, label: str, adj_cache: dict) -> dict:
    top = top1_sequence(df, proba)
    tid, arc = top.trip_id.to_numpy(), top.cand_arc.to_numpy()
    changed = (arc[1:] != arc[:-1]) & (tid[1:] == tid[:-1])
    idx = np.flatnonzero(changed)

    h = (
        np.array([hops_between(adj_cache[tid[i]], int(arc[i]), int(arc[i + 1])) for i in idx])
        if idx.size
        else np.empty(0, int)
    )

    row = {"model": label, "n_transitions": int(h.size)}
    for k in REPORT_HOPS:
        row[f"reachable_within_{k}_hops"] = float(((h >= 0) & (h <= k)).mean()) if h.size else np.nan
    row["unreachable"] = float((h < 0).mean()) if h.size else np.nan
    row["median_hops"] = float(np.median(h[h >= 0])) if (h >= 0).any() else np.nan
    return row


def main(names):
    df = labelled(load_table(PARTITION))
    X = df[FEATURES].to_numpy(np.float32)
    adj_cache = {t: _adjacency(PARTITION, t) for t in sorted(df.trip_id.unique())}

    rows = [diagnose(df, df.label.to_numpy(float), "ground truth (ceiling)", adj_cache)]
    for name in names:
        path = MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            print(f"skipping {name}: {path} not found")
            continue
        model = joblib.load(path)["model"]
        rows.append(diagnose(df, model.predict_proba(X)[:, 1], name, adj_cache))

    out = pd.DataFrame(rows)
    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / "tables" / "route_continuity_diagnostic.csv", index=False)
    print(f"Arc-to-arc transitions on {PARTITION}, reachability in the road graph\n")
    print(out.to_string(index=False, float_format=lambda v: f"{v:7.4f}"))
    print(f"\nWrote {RESULTS_DIR / 'tables' / 'route_continuity_diagnostic.csv'}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["logreg", "rf", "dtree"])
