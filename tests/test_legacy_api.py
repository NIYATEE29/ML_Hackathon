import pytest
from fastapi.testclient import TestClient
from api.legacy_main import app

client = TestClient(app)

def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "endpoints" in data

def test_api_trips():
    response = client.get("/trips?split=student_test")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 10
    assert len(data["trips"]) == 10
    assert data["trips"][0]["split"] == "student_test"
