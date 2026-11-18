
"""Generate nearby road-arc candidates for each GPS observation."""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def _sample_arcs(nodes, arcs, spacing_m=10.0):
    """Sample points along every road arc and retain its arc ID."""
    sampled_points = []
    sampled_arc_ids = []

    for arc_id, arc in arcs.iterrows():
        start = nodes.iloc[int(arc["from_node"])]
        end = nodes.iloc[int(arc["to_node"])]

        x1, y1 = float(start["x"]), float(start["y"])
        x2, y2 = float(end["x"]), float(end["y"])

        length = np.hypot(x2 - x1, y2 - y1)

        # Include both endpoints and intermediate points.
        n_samples = max(2, int(np.ceil(length / spacing_m)) + 1)

        fractions = np.linspace(0.0, 1.0, n_samples)

        xs = x1 + fractions * (x2 - x1)
        ys = y1 + fractions * (y2 - y1)

        sampled_points.extend(zip(xs, ys))
        sampled_arc_ids.extend([int(arc_id)] * n_samples)

    return np.asarray(sampled_points), np.asarray(sampled_arc_ids)


def _point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Return distance from a point to a segment and its projection fraction."""
    dx = x2 - x1
    dy = y2 - y1

    length_sq = dx * dx + dy * dy

    if length_sq == 0:
        fraction = 0.0
    else:
        fraction = ((px - x1) * dx + (py - y1) * dy) / length_sq
        fraction = np.clip(fraction, 0.0, 1.0)

    closest_x = x1 + fraction * dx
    closest_y = y1 + fraction * dy

    distance = np.hypot(px - closest_x, py - closest_y)

    return float(distance), float(fraction)


def generate_candidates(
    trip,
    radius_m=40.0,
    sample_spacing_m=10.0,
):
    """
    Generate candidate arcs for every GPS point in a trip.

    Returns one row per (GPS point, candidate arc) pair.
    """
    nodes = trip.nodes
    arcs = trip.arcs
    track = trip.track

    if len(nodes) == 0 or len(arcs) == 0 or len(track) == 0:
        return pd.DataFrame(
            columns=[
                "point_idx",
                "arc_id",
                "distance",
                "projection_fraction",
            ]
        )

    sampled_points, sampled_arc_ids = _sample_arcs(
        nodes, arcs, spacing_m=sample_spacing_m
    )

    tree = cKDTree(sampled_points)

    candidate_rows = []

    for point_idx, point in track.iterrows():
        px, py = float(point["x"]), float(point["y"])

        nearby_sample_indices = tree.query_ball_point(
            [px, py], r=radius_m + sample_spacing_m
        )

        # Convert nearby sampled points to unique candidate arc IDs.
        candidate_arc_ids = np.unique(
            sampled_arc_ids[nearby_sample_indices]
        )

        for arc_id in candidate_arc_ids:
            arc = arcs.iloc[int(arc_id)]

            start = nodes.iloc[int(arc["from_node"])]
            end = nodes.iloc[int(arc["to_node"])]

            distance, fraction = _point_to_segment_distance(
                px,
                py,
                float(start["x"]),
                float(start["y"]),
                float(end["x"]),
                float(end["y"]),
            )

            # Exact segment distance is checked after spatial-index lookup.
            if distance <= radius_m:
                candidate_rows.append(
                    {
                        "point_idx": int(point_idx),
                        "arc_id": int(arc_id),
                        "distance": distance,
                        "projection_fraction": fraction,
                    }
                )

    candidates = pd.DataFrame(candidate_rows)

    if not candidates.empty:
        candidates = candidates.sort_values(
            ["point_idx", "distance"]
        ).reset_index(drop=True)

    return candidates
