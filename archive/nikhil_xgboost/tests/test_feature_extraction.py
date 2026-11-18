
import pandas as pd

from archive.nikhil_xgboost.candidate_generation import generate_candidates
from archive.nikhil_xgboost.feature_extraction import extract_features


class FakeTrip:
    pass


def test_feature_extraction():
    trip = FakeTrip()

    trip.nodes = pd.DataFrame({
        "x": [0.0, 100.0, 0.0, 100.0],
        "y": [0.0, 0.0, 20.0, 20.0],
    })

    trip.arcs = pd.DataFrame({
        "from_node": [0, 2],
        "to_node": [1, 3],
    })

    trip.track = pd.DataFrame({
        "x": [10.0, 20.0, 30.0],
        "y": [2.0, 2.0, 2.0],
        "t": [0.0, 1.0, 2.0],
    })

    candidates = generate_candidates(
        trip,
        radius_m=40.0,
        sample_spacing_m=10.0,
    )

    features = extract_features(trip, candidates)

    # Required features must exist.
    expected_features = {
        "distance",
        "projection_fraction",
        "road_length",
        "road_bearing",
        "gps_heading",
        "heading_difference",
        "speed",
        "time_gap",
        "candidate_rank",
    }

    assert expected_features.issubset(features.columns)

    # All feature values should be finite.
    assert features[list(expected_features)].notna().all().all()

    # The road lengths should be 100 metres.
    assert (features["road_length"] == 100.0).all()

    # GPS movement is 10 metres per second.
    assert (features["speed"] == 10.0).all()

    # Candidate ranks should start at 1 for each GPS point.
    assert (features.groupby("point_idx")["candidate_rank"].min() == 1).all()
