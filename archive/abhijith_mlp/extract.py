import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import SEED
from archive.legacy_config import TRAIN_DIR, STUDENT_TEST_DIR, PROCESSED_DATA_DIR, CANDIDATE_SEARCH_RADIUS_M, CANDIDATE_TOP_K, ARC_DENSIFY_STEP_M
from src.data.loader import list_trips
from archive.legacy_loader import load_trip_data
from archive.abhijith_mlp.spatial import TripSpatialIndex, point_to_segment_dist, angle_diff, to_local_xy

def assign_ground_truth_arcs(
    gps_xy: np.ndarray,
    route_arcs: np.ndarray,
    arcs: np.ndarray,
    node_xy: np.ndarray
) -> np.ndarray:
    """
    Chronological, order-aware projection of GPS points onto the ground-truth route arcs.
    Returns: array of shape (T,) where element t is the true arc_id for GPS point t.
    """
    T = len(gps_xy)
    R = len(route_arcs)
    true_arcs = np.full(T, -1, dtype=np.int64)

    # Pre-extract route segment coordinates
    from_nodes = arcs[route_arcs, 0]
    to_nodes = arcs[route_arcs, 1]
    seg_starts = node_xy[from_nodes]
    seg_ends = node_xy[to_nodes]

    # Sliding window pointer along the reference route
    route_idx = 0
    window_ahead = 20
    window_behind = 5

    for t in range(T):
        pt = gps_xy[t]
        start_w = max(0, route_idx - window_behind)
        end_w = min(R, route_idx + window_ahead)

        best_dist = float("inf")
        best_r_idx = route_idx

        for r_i in range(start_w, end_w):
            d, _ = point_to_segment_dist(pt, seg_starts[r_i], seg_ends[r_i])
            if d < best_dist:
                best_dist = d
                best_r_idx = r_i

        true_arcs[t] = route_arcs[best_r_idx]
        route_idx = best_r_idx  # Advance window monotonically

    return true_arcs

def extract_features_for_trip(
    trip_id: str,
    split: str = "train",
    spatial_index: Optional[TripSpatialIndex] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Extracts candidate features and ground-truth labels for a single trip.
    Returns:
        (df_candidates, trip_stats)
    """
    data = load_trip_data(trip_id, split)
    nodes = data["nodes"]
    arcs = data["arcs"]
    track = data["track"]
    route = data["route"]

    T = len(track)
    if spatial_index is None:
        spatial_index = TripSpatialIndex(nodes, arcs, densify_step_m=ARC_DENSIFY_STEP_M)

    gps_xy = spatial_index.project_gps(track[:, 0], track[:, 1])
    timestamps = track[:, 2]

    # Kinematics: Speeds, time deltas, and trajectory headings
    time_deltas = np.zeros(T, dtype=np.float64)
    speeds = np.zeros(T, dtype=np.float64)
    headings = np.zeros(T, dtype=np.float64)

    time_deltas[1:] = np.maximum(1e-3, timestamps[1:] - timestamps[:-1])
    disp = gps_xy[1:] - gps_xy[:-1]
    step_dists = np.sqrt(np.sum(disp ** 2, axis=1))
    speeds[1:] = step_dists / time_deltas[1:]
    speeds[0] = speeds[1] if T > 1 else 0.0

    # Heading angles (from central differencing where possible)
    if T >= 3:
        central_disp = gps_xy[2:] - gps_xy[:-2]
        headings[1:-1] = np.arctan2(central_disp[:, 1], central_disp[:, 0])
        headings[0] = np.arctan2(disp[0, 1], disp[0, 0])
        headings[-1] = np.arctan2(disp[-1, 1], disp[-1, 0])
    elif T == 2:
        h = np.arctan2(disp[0, 1], disp[0, 0])
        headings[0] = h
        headings[1] = h

    # Match ground-truth route if available
    has_labels = route is not None and len(route) > 0
    true_arc_per_point = None
    if has_labels:
        true_arc_per_point = assign_ground_truth_arcs(gps_xy, route, arcs, spatial_index.node_xy)

    rows = []
    missed_count = 0
    total_candidates_sum = 0

    for t in range(T):
        pt = gps_xy[t]
        v_speed = speeds[t]
        v_heading = headings[t]
        dt = time_deltas[t]

        # Spatial candidate retrieval
        candidates = spatial_index.find_candidates(
            pt,
            radius_m=CANDIDATE_SEARCH_RADIUS_M,
            top_k=CANDIDATE_TOP_K
        )
        total_candidates_sum += len(candidates)

        true_arc = true_arc_per_point[t] if has_labels else -1
        if has_labels and true_arc not in candidates:
            missed_count += 1

        # Compute point-to-arc metrics for each candidate
        cand_metrics = []
        for c_arc in candidates:
            u, v = arcs[c_arc]
            a_start = spatial_index.node_xy[u]
            a_end = spatial_index.node_xy[v]
            dist, proj_ratio = point_to_segment_dist(pt, a_start, a_end)
            bearing = spatial_index.arc_bearings[c_arc]
            ang_diff = angle_diff(v_heading, bearing)
            cand_metrics.append((c_arc, dist, proj_ratio, ang_diff))

        # Sort by distance to compute rank
        cand_metrics.sort(key=lambda x: x[1])
        num_c = len(cand_metrics)

        # Distance likelihood score (Gaussian decay sigma = 15m)
        dist_sq = np.array([m[1]**2 for m in cand_metrics])
        exp_scores = np.exp(-dist_sq / (2.0 * (15.0 ** 2)))
        prob_dist = exp_scores / (np.sum(exp_scores) + 1e-12)

        for rank, (c_arc, dist, proj_ratio, ang_diff) in enumerate(cand_metrics):
            label = 1 if (has_labels and c_arc == true_arc) else 0
            rows.append({
                "trip_id": trip_id,
                "point_idx": t,
                "cand_arc_id": c_arc,
                "dist_to_arc": dist,
                "proj_ratio": proj_ratio,
                "heading_diff": ang_diff,
                "speed": v_speed,
                "time_delta": dt,
                "arc_length": spatial_index.arc_lengths[c_arc],
                "dist_rank": (rank + 1) / max(1, num_c),
                "num_candidates": num_c,
                "spatial_likelihood": float(prob_dist[rank]),
                "label": label
            })

    df = pd.DataFrame(rows)
    stats = {
        "trip_id": trip_id,
        "num_gps_points": T,
        "total_rows": len(df),
        "avg_candidates_per_point": total_candidates_sum / max(1, T),
        "candidate_miss_rate": (missed_count / max(1, T)) if has_labels else None
    }
    return df, stats

def extract_all_trips(split: str = "train", max_trips: Optional[int] = None) -> pd.DataFrame:
    """Extracts features for all trips in the partition and saves as Parquet & CSV."""
    trips = list_trips(split)
    if max_trips:
        trips = trips[:max_trips]

    print(f"Extracting features for {len(trips)} trips from {split} partition...")
    all_dfs = []
    stats_list = []

    for trip_id in tqdm(trips, desc=f"Processing {split}"):
        df_trip, trip_stats = extract_features_for_trip(trip_id, split)
        all_dfs.append(df_trip)
        stats_list.append(trip_stats)

    full_df = pd.concat(all_dfs, ignore_index=True)
    out_parquet = PROCESSED_DATA_DIR / f"features_{split}.parquet"
    full_df.to_parquet(out_parquet, index=False)

    stats_df = pd.DataFrame(stats_list)
    stats_csv = PROCESSED_DATA_DIR / f"candidate_stats_{split}.csv"
    stats_df.to_csv(stats_csv, index=False)

    print(f"Done! Saved {len(full_df)} rows to {out_parquet}")
    if "candidate_miss_rate" in stats_df.columns:
        print(f"Overall Mean Candidate Miss Rate: {stats_df['candidate_miss_rate'].mean() * 100:.2f}%")
        print(f"Overall Mean Candidates/Point: {stats_df['avg_candidates_per_point'].mean():.2f}")

    return full_df

if __name__ == "__main__":
    extract_all_trips("train")
