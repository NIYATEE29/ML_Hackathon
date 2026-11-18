
"""Extract features for GPS-point / candidate-road pairs."""

import numpy as np
import pandas as pd


def _angle_difference(a, b):
    """Smallest absolute difference between two angles in degrees."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _bearing(x1, y1, x2, y2):
    """Calculate bearing clockwise from north."""
    dx = x2 - x1
    dy = y2 - y1
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def extract_features(trip, candidates):
    """
    Build one feature row per GPS point and candidate road.

    Parameters
    ----------
    trip : Trip
        Loaded trip from the shared loader.
    candidates : DataFrame
        Output from candidate generation.

    Returns
    -------
    DataFrame
        Candidate rows with model features.
    """
    if candidates.empty:
        return candidates.copy()

    track = trip.track
    nodes = trip.nodes
    arcs = trip.arcs

    # Calculate movement information for each GPS observation.
    gps_info = {}

    for i in range(len(track)):
        current = track.iloc[i]
        x, y = float(current["x"]), float(current["y"])
        t = float(current["t"])

        if i < len(track) - 1:
            next_point = track.iloc[i + 1]
            dx = float(next_point["x"]) - x
            dy = float(next_point["y"]) - y
            dt = float(next_point["t"]) - t
        elif i > 0:
            previous = track.iloc[i - 1]
            dx = x - float(previous["x"])
            dy = y - float(previous["y"])
            dt = t - float(previous["t"])
        else:
            dx, dy, dt = 0.0, 0.0, 0.0

        distance_m = np.hypot(dx, dy)

        if dt > 0:
            speed = distance_m / dt
        else:
            speed = 0.0

        if distance_m > 0:
            heading = _bearing(0.0, 0.0, dx, dy)
        else:
            heading = 0.0

        gps_info[i] = {
            "gps_heading": heading,
            "speed": speed,
            "time_gap": max(dt, 0.0),
        }

    feature_rows = []

    for _, candidate in candidates.iterrows():
        point_idx = int(candidate["point_idx"])
        arc_id = int(candidate["arc_id"])

        arc = arcs.iloc[arc_id]

        start = nodes.iloc[int(arc["from_node"])]
        end = nodes.iloc[int(arc["to_node"])]

        x1, y1 = float(start["x"]), float(start["y"])
        x2, y2 = float(end["x"]), float(end["y"])

        road_length = np.hypot(x2 - x1, y2 - y1)
        road_bearing = _bearing(x1, y1, x2, y2)

        info = gps_info[point_idx]

        heading_difference = _angle_difference(
            info["gps_heading"],
            road_bearing,
        )

        feature_rows.append({
            "point_idx": point_idx,
            "arc_id": arc_id,
            "distance": float(candidate["distance"]),
            "projection_fraction": float(candidate["projection_fraction"]),
            "road_length": float(road_length),
            "road_bearing": float(road_bearing),
            "gps_heading": float(info["gps_heading"]),
            "heading_difference": float(heading_difference),
            "speed": float(info["speed"]),
            "time_gap": float(info["time_gap"]),
        })

    features = pd.DataFrame(feature_rows)

    # Rank candidates from nearest to farthest for each GPS observation.
    features["candidate_rank"] = (
        features.groupby("point_idx")["distance"]
        .rank(method="first")
        .astype(int)
    )

    return features
