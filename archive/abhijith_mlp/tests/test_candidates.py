import pytest
import numpy as np
from archive.legacy_loader import load_trip_data
from archive.abhijith_mlp.spatial import TripSpatialIndex, point_to_segment_dist, angle_diff

def test_point_to_segment_dist():
    p = np.array([0.0, 10.0])
    a = np.array([0.0, 0.0])
    b = np.array([10.0, 0.0])
    d, proj = point_to_segment_dist(p, a, b)
    assert abs(d - 10.0) < 1e-5
    assert abs(proj - 0.0) < 1e-5

def test_angle_diff():
    # Identical angle
    assert abs(angle_diff(0.5, 0.5)) < 1e-5
    # Opposite angle (pi)
    assert abs(angle_diff(0.0, np.pi) - np.pi) < 1e-5

from src.config import RAW_DIR


@pytest.mark.skipif(not (RAW_DIR / "train").exists(), reason="raw data not present")
def test_spatial_index_candidates():
    data = load_trip_data("00000001", "train")
    spatial = TripSpatialIndex(data["nodes"], data["arcs"])
    gps_xy = spatial.project_gps(data["track"][:5, 0], data["track"][:5, 1])
    candidates = spatial.find_candidates(gps_xy[0], radius_m=40.0, top_k=10)
    assert len(candidates) > 0
    assert len(candidates) <= 10
