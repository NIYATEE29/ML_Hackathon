import pytest
from fastapi.testclient import TestClient
from archive.abhijith_mlp.api_main import app

from src.config import RAW_DIR

client = TestClient(app)

def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "endpoints" in data

@pytest.mark.skipif(not (RAW_DIR / "student_test").exists(), reason="raw data not present")
def test_api_trips():
    response = client.get("/trips?split=student_test")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 10
    assert len(data["trips"]) == 10
    assert data["trips"][0]["split"] == "student_test"
