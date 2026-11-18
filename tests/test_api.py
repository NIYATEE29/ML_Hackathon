import json
from pathlib import Path

import pytest

from src.config import RAW_DIR

pytestmark = pytest.mark.skipif(not (RAW_DIR / "student_test").exists(), reason="raw data not present")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


def test_health_and_trips(client):
    assert client.get("/health").json()["status"] == "ok"
    trips = client.get("/trips").json()["trips"]
    assert len(trips) == 10 and {"trip_id", "n_points"} <= set(trips[0])


def test_predict_shape_and_no_ground_truth_leak(client):
    r = client.post("/predict", json={"trip_id": "00000052"})
    assert r.status_code == 200
    b = r.json()
    assert {"gps", "predicted_route", "classifier_route", "metrics", "timing_ms"} <= set(b)
    assert len(b["gps"]) == b["n_points"] and all(len(p) == 2 for p in b["gps"][:5])
    assert b["metrics"]["final"]["top1_accuracy"] >= b["metrics"]["classifier_only"]["top1_accuracy"]
    text = json.dumps(b)
    for forbidden in ("true_arc", "reference", "label", "ground"):
        assert forbidden not in text                      # no ground truth in the response


def test_unknown_trip_is_404_and_path_traversal_is_blocked(client):
    assert client.post("/predict", json={"trip_id": "99999999"}).status_code == 404
    assert client.post("/predict", json={"trip_id": "../train/00000001"}).status_code == 404
