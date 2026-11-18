
"""Generate candidate labels from the ground-truth route."""

import numpy as np
import pandas as pd

from archive.nikhil_xgboost.candidate_generation import _point_to_segment_distance


def generate_labels(trip, candidates, max_label_distance_m=30.0):
    """
    Assign binary labels to candidates using the ground-truth route.

    A GPS point is assigned to its nearest ground-truth route arc.
    Points farther than max_label_distance_m are considered unreliable
    and are excluded from training.

    Returns
    -------
    DataFrame
        Candidate rows with a binary label and a point-level validity flag.
    """
    if candidates.empty:
        result = candidates.copy()
        result["label"] = pd.Series(dtype=int)
        result["valid_label"] = pd.Series(dtype=bool)
        return result

    nodes = trip.nodes
    arcs = trip.arcs
    track = trip.track

    # Get the unique arcs that appear in the ground-truth route.
    route_arc_ids = np.unique(trip.route)

    route_segments = []

    for arc_id in route_arc_ids:
        arc = arcs.iloc[int(arc_id)]

        start = nodes.iloc[int(arc["from_node"])]
        end = nodes.iloc[int(arc["to_node"])]

        route_segments.append(
            (
                int(arc_id),
                float(start["x"]),
            float(start["y"]),
                float(end["x"]),
            float(end["y"]),
            )
        )

    labels_by_point = {}

    for point_idx, point in track.iterrows():
        px, py = float(point["x"]), float(point["y"])

        best_arc_id = None
        best_distance = float("inf")

        # Find the closest ground-truth route arc.
        for arc_id, x1, y1, x2, y2 in route_segments:
            distance, _ = _point_to_segment_distance(
                px, py, x1, y1, x2, y2
            )

            if distance < best_distance:
                best_distance = distance
                best_arc_id = arc_id

        # Mark the point unreliable if it is too far from the reference route.
        valid = best_distance <= max_label_distance_m

        labels_by_point[int(point_idx)] = {
            "true_arc_id": best_arc_id if valid else None,
            "label_distance": best_distance,
            "valid_label": valid,
        }

    result = candidates.copy()

    result["true_arc_id"] = result["point_idx"].map(
        lambda idx: labels_by_point[int(idx)]["true_arc_id"]
    )

    result["label_distance"] = result["point_idx"].map(
        lambda idx: labels_by_point[int(idx)]["label_distance"]
    )

    result["valid_label"] = result["point_idx"].map(
        lambda idx: labels_by_point[int(idx)]["valid_label"]
    )

    # Only reliable points receive labels.
    result["label"] = (
        result["arc_id"] == result["true_arc_id"]
    ).astype(int)

    return result
