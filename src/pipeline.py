import time
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import (
    MODELS_DIR, PROCESSED_DATA_DIR, CANDIDATE_SEARCH_RADIUS_M, CANDIDATE_TOP_K, FEATURE_COLUMNS
)
from src.data.loader import load_trip_data
from src.features.spatial import TripSpatialIndex
from src.features.extract import extract_features_for_trip
from src.hmm.viterbi import PathSnapHMMDecoder
from src.models.evaluate import compute_route_continuity, compute_observation_top1

class PathSnapPipeline:
    """
    End-to-end production inference pipeline for PathSnap.
    Coordinates:
      1. Dynamic spatial candidate generation and feature extraction.
      2. Multi-Layer Perceptron (MLP) probability estimation.
      3. Hidden Markov Model (HMM) + Viterbi global route decoding.
      4. Ground-truth server-side scoring (when ground truth is available).
    """
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        sigma_z: float = 12.0,
        beta: float = 8.0,
        eps_floor: float = 1e-6
    ):
        if model_path is None:
            model_path = MODELS_DIR / "mlp_model.pkl"
        self.model_path = Path(model_path)
        self.sigma_z = sigma_z
        self.beta = beta
        self.eps_floor = eps_floor

        self.model = None
        if self.model_path.exists():
            self.load_model(self.model_path)

    def load_model(self, model_path: Union[str, Path]):
        """Loads a pre-trained scikit-learn / PathSnap model wrapper."""
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        print(f"Loaded PathSnap model from {model_path}")

    def run_trip(
        self,
        trip_id: str,
        split: str = "student_test",
        use_cached_features: bool = True
    ) -> Dict:
        """
        Runs the full candidate generation -> MLP -> HMM pipeline on a single trip.
        Returns detailed predictions and performance comparisons.
        """
        t0 = time.time()
        trip_data = load_trip_data(trip_id, split)
        nodes = trip_data["nodes"]
        arcs = trip_data["arcs"]
        track = trip_data["track"]
        route = trip_data["route"]
        has_ground_truth = route is not None and len(route) > 0

        T = len(track)
        t_cand_start = time.time()

        # Step 1: Feature Extraction / Candidate Generation
        df_candidates = None
        if use_cached_features:
            cache_file = PROCESSED_DATA_DIR / f"features_{split}.parquet"
            if cache_file.exists():
                try:
                    df_all = pd.read_parquet(cache_file, filters=[("trip_id", "==", trip_id)])
                    if len(df_all) > 0:
                        df_candidates = df_all.sort_values(["point_idx", "dist_rank"]).reset_index(drop=True)
                except Exception:
                    df_candidates = None

        spatial_index = None
        if df_candidates is None:
            spatial_index = TripSpatialIndex(nodes, arcs)
            df_candidates, _ = extract_features_for_trip(
                trip_id=trip_id,
                split=split,
                spatial_index=spatial_index
            )

        cand_time_ms = (time.time() - t_cand_start) * 1000.0

        # Step 2: MLP Inference
        t_infer_start = time.time()
        if self.model is None:
            raise RuntimeError("Pipeline model is not loaded. Train or specify a valid model_path.")

        X = df_candidates[FEATURE_COLUMNS].values
        probs = self.model.predict_proba(X)[:, 1]
        df_candidates["prob"] = probs
        infer_time_ms = (time.time() - t_infer_start) * 1000.0

        # Classifier-only Top-1 prediction per point
        clf_top1_df = df_candidates.loc[df_candidates.groupby("point_idx")["prob"].idxmax()].sort_values("point_idx")
        clf_predicted_route = clf_top1_df["cand_arc_id"].tolist()

        # Step 3: HMM + Viterbi Decoding
        t_hmm_start = time.time()
        grouped = df_candidates.groupby("point_idx")
        candidate_lists = []
        candidate_probs = []
        candidate_dists = []

        for p_idx in range(T):
            if p_idx in grouped.groups:
                g = grouped.get_group(p_idx)
                candidate_lists.append(g["cand_arc_id"].tolist())
                candidate_probs.append(g["prob"].tolist())
                candidate_dists.append(g["dist_to_arc"].tolist())
            else:
                candidate_lists.append([0])
                candidate_probs.append([1.0])
                candidate_dists.append([50.0])

        if spatial_index is None:
            spatial_index = TripSpatialIndex(nodes, arcs)
        gps_xy = spatial_index.project_gps(track[:, 0], track[:, 1])

        time_deltas = np.zeros(T, dtype=np.float64)
        time_deltas[1:] = np.maximum(1e-3, track[1:, 2] - track[:-1, 2])

        decoder = PathSnapHMMDecoder(
            nodes=nodes,
            arcs=arcs,
            sigma_z=self.sigma_z,
            beta=self.beta,
            eps_floor=self.eps_floor
        )
        hmm_predicted_route = decoder.decode(
            candidate_lists=candidate_lists,
            candidate_probs=candidate_probs,
            candidate_dists=candidate_dists,
            gps_xy=gps_xy,
            time_deltas=time_deltas
        )
        hmm_time_ms = (time.time() - t_hmm_start) * 1000.0
        total_time_ms = (time.time() - t0) * 1000.0

        # Metrics & Route Continuity
        clf_continuity = compute_route_continuity(clf_predicted_route, arcs)
        hmm_continuity = compute_route_continuity(hmm_predicted_route, arcs)

        result = {
            "trip_id": trip_id,
            "split": split,
            "num_points": T,
            "num_candidates": len(df_candidates),
            "classifier_predicted_route": clf_predicted_route,
            "hmm_predicted_route": hmm_predicted_route,
            "classifier_continuity": clf_continuity,
            "hmm_continuity": hmm_continuity,
            "timings": {
                "candidate_gen_ms": round(cand_time_ms, 2),
                "mlp_inference_ms": round(infer_time_ms, 2),
                "hmm_viterbi_ms": round(hmm_time_ms, 2),
                "total_pipeline_ms": round(total_time_ms, 2)
            }
        }

        # Server-side Ground Truth Evaluation (if route available)
        if has_ground_truth:
            # Point-level Top-1 accuracy against derived ground truth labels
            clf_acc, _ = compute_observation_top1(df_candidates.rename(columns={"prob": "pred_prob"}))
            result["classifier_top1_accuracy"] = round(clf_acc, 4)

            # Evaluate HMM Top-1 on derived labels
            true_point_labels = df_candidates[df_candidates["label"] == 1].set_index("point_idx")["cand_arc_id"].to_dict()
            correct_hmm = 0
            evaluated_pts = 0
            for pt_idx, true_arc in true_point_labels.items():
                if pt_idx < len(hmm_predicted_route):
                    if hmm_predicted_route[pt_idx] == true_arc:
                        correct_hmm += 1
                    evaluated_pts += 1
            hmm_acc = (correct_hmm / evaluated_pts) if evaluated_pts > 0 else 0.0
            result["hmm_top1_accuracy"] = round(hmm_acc, 4)

            # Route arc set overlap (Jaccard similarity)
            pred_set = set(hmm_predicted_route)
            true_set = set(route)
            jaccard = len(pred_set & true_set) / max(1, len(pred_set | true_set))
            result["route_jaccard_similarity"] = round(jaccard, 4)

        return result
