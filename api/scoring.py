"""Server-side scoring against the reference route.

This is the ONLY place in the API that reads a .route file. Its output is a handful of numbers;
the reference route and the per-point labels are never put into a response.
"""

from pathlib import Path

import pandas as pd

from src.data.loader import load_trip
from src.features.candidates import build_trip_table
from src.models.common import evaluate


def _route_f1(pred_arcs, ref_arcs) -> float:
    p, r = set(map(int, pred_arcs)), set(map(int, ref_arcs))
    tp = len(p & r)
    prec, rec = tp / max(1, len(p)), tp / max(1, len(r))
    return 2 * prec * rec / max(1e-9, prec + rec)


def score_against_reference(trips_dir: Path, trip_id: str, cands: pd.DataFrame, decoded: pd.DataFrame) -> dict | None:
    """Metrics for the final (HMM) route and for the classifier alone. None if no reference exists."""
    trips_dir = Path(trips_dir)
    if not (trips_dir / trip_id / f"{trip_id}.route").exists():
        return None
    trip = load_trip(trips_dir.name, trip_id, trips_dir.parent)  # reads the route (server side)
    lab, _ = build_trip_table(trip, with_labels=True)
    assert (
        len(lab) == len(cands)
        and (lab.pt.to_numpy() == cands.pt.to_numpy()).all()
        and (lab.cand_arc.to_numpy() == cands.cand_arc.to_numpy()).all()
    ), "label/inference rows are misaligned"
    p_clf = cands["p"].to_numpy()
    key = decoded.assign(hit=1.0)[["pt", "cand_arc", "hit"]]
    p_hmm = lab[["pt", "cand_arc"]].merge(key, on=["pt", "cand_arc"], how="left")["hit"].fillna(0.0).to_numpy()

    top_clf = cands.loc[cands.groupby("pt")["p"].idxmax(), "cand_arc"]
    out = {}
    for name, p, arcs in (("final", p_hmm, decoded.cand_arc), ("classifier_only", p_clf, top_clf)):
        m = evaluate(lab, p)
        out[name] = {
            "top1_accuracy": m["top1_acc"],
            "macro_f1": m["macro_f1"],
            "route_continuity": m["route_continuity"],
            "route_f1": _route_f1(arcs, trip.route),
            "n_scored_points": m["n_points"],
            "candidate_miss_rate": m["candidate_miss_rate"],
        }
    return {
        k: {kk: round(float(vv), 4) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in out.items()
    }
