"""PathSnap prediction service (pure Python, no web code).

`PathSnap.predict()` runs the full pipeline on one trip: features -> classifier -> HMM/Viterbi.
It NEVER opens the .route file or any label. Scoring against the reference route lives in
api/scoring.py and is called separately by the web layer, server side only.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, ROOT
from src.hmm.viterbi import ArcGraph, HMMParams, decode_trip
from src.inference.predict import load_model, load_trip_for_inference, score_trip


def arcs_to_paths(arc_ids, arcs: np.ndarray, nodes: pd.DataFrame) -> list[list[list[float]]]:
    """Decoded arc sequence -> list of [lat, lon] paths. A new path starts wherever two
    consecutive arcs are not connected, so the map never draws a fake line across a break."""
    lon, lat = nodes["lon"].to_numpy(), nodes["lat"].to_numpy()
    paths, cur, last_to = [], [], None
    prev = None
    for a in arc_ids:
        if prev is not None and a == prev:
            continue                                   # same arc repeated for many GPS points
        f, t = int(arcs[a, 0]), int(arcs[a, 1])
        if last_to is not None and f != last_to:
            paths.append(cur); cur = []
        if not cur:
            cur.append([float(lat[f]), float(lon[f])])
        cur.append([float(lat[t]), float(lon[t])])
        last_to, prev = t, a
    if cur:
        paths.append(cur)
    return paths


class PathSnap:
    def __init__(self, trips_dir: Path, model_name: str = "xgb_tuned", models_dir: Path = MODELS_DIR):
        self.trips_dir = Path(trips_dir)
        self.partition, self.raw_dir = self.trips_dir.name, self.trips_dir.parent
        self.model_name = model_name
        self.model = load_model(model_name, models_dir)
        self.params = HMMParams(**json.loads((models_dir / f"hmm_params_{model_name}.json").read_text()))
        self._trip_ids = sorted(p.name for p in self.trips_dir.iterdir() if p.is_dir() and (p / f"{p.name}.track").exists())

    def trip_ids(self) -> list[str]:
        return list(self._trip_ids)

    def list_trips(self) -> list[dict]:
        out = []
        for t in self._trip_ids:
            with open(self.trips_dir / t / f"{t}.track") as f:
                out.append({"trip_id": t, "n_points": sum(1 for _ in f)})
        return out

    def predict(self, trip_id: str) -> dict:
        """Full pipeline on one trip. Returns the prediction payload plus private arrays
        (keys starting with '_') for the scorer; the web layer strips those before replying."""
        if trip_id not in self._trip_ids:
            raise KeyError(trip_id)
        t0 = time.perf_counter()
        trip = load_trip_for_inference(self.partition, trip_id, self.raw_dir)      # no route is read
        t1 = time.perf_counter()
        cands = score_trip(trip, self.model)                                         # candidates + features + classifier
        t2 = time.perf_counter()
        arcs = trip.arcs.to_numpy().astype(int)
        dec = decode_trip(cands, ArcGraph(arcs, len(trip.nodes)), self.params)      # HMM / Viterbi
        t3 = time.perf_counter()
        clf_top = cands.loc[cands.groupby("pt")["p"].idxmax()].sort_values("pt")
        gps = trip.track
        return {
            "trip_id": trip_id, "model": self.model_name, "decoder": "hmm_viterbi",
            "n_points": int(len(gps)),
            "gps": np.c_[gps["lat"].to_numpy(), gps["lon"].to_numpy()].round(6).tolist(),
            "predicted_route": {"arcs": _dedupe(dec.cand_arc.tolist()),
                                "paths": arcs_to_paths(dec.cand_arc.tolist(), arcs, trip.nodes)},
            "classifier_route": {"arcs": _dedupe(clf_top.cand_arc.tolist()),
                                 "paths": arcs_to_paths(clf_top.cand_arc.tolist(), arcs, trip.nodes)},
            "timing_ms": {"load": round(1000 * (t1 - t0)), "features_and_classifier": round(1000 * (t2 - t1)),
                          "hmm": round(1000 * (t3 - t2)), "total": round(1000 * (t3 - t0))},
            "_cands": cands, "_decoded": dec,
        }


def _dedupe(seq):
    out = []
    for a in seq:
        if not out or out[-1] != a:
            out.append(int(a))
    return out
