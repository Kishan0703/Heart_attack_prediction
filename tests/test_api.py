import pytest
from fastapi.testclient import TestClient
import os
import sys

# Add the project root to sys.path to import backend.main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set dummy API_KEY for tests
os.environ["API_KEY"] = "test_key"

from backend.main import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_health(client):
    """Test the /health endpoint returns 200 and correct status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "artifacts" in data

def test_predict_valid(client):
    """Test /predict with a valid payload."""
    payload = {
        "I_620": 585.7,
        "I_540": 735.3,
        "R620": 1.485,
        "R540": 3.025,
        "cTnI_ng_mL": 14.35,
        "Myoglobin_ng_mL": 15.025,
        "Raw_Fluorescence_au": 2149.01,
        "DeltaF_au": 949.01,
        "Calculated_Troponin_ng_mL": 1.1863,
        "Peak_Current_uA": 280.739,
        "BNP_pg_mL": 623.42
    }
    response = client.post(
        "/predict",
        json=payload,
        headers={"X-API-Key": "test_key"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "risk_category" in data
    assert data["risk_category"] in ["Low", "Moderate", "High"]
    assert "prediction_id" in data
    assert "probability" in data

def test_predict_out_of_range(client):
    """Test /predict with out-of-range values returns 422."""
    payload = {
        "I_620": 585.7,
        "I_540": 735.3,
        "R620": 1.485,
        "R540": 3.025,
        "cTnI_ng_mL": 999.0, # Max is 500
        "Myoglobin_ng_mL": 15.025,
        "Raw_Fluorescence_au": 2149.01,
        "DeltaF_au": 949.01,
        "Calculated_Troponin_ng_mL": 1.1863,
        "Peak_Current_uA": 280.739,
        "BNP_pg_mL": 623.42
    }
    response = client.post(
        "/predict",
        json=payload,
        headers={"X-API-Key": "test_key"}
    )
    assert response.status_code == 422

def test_predict_missing_fields(client):
    """Test /predict with missing fields returns 422."""
    payload = {
        "cTnI_ng_mL": 14.35,
        "BNP_pg_mL": 623.42
    }
    response = client.post(
        "/predict",
        json=payload,
        headers={"X-API-Key": "test_key"}
    )
    assert response.status_code == 422

def test_predict_invalid_key(client):
    """Test /predict with invalid API key returns 403."""
    payload = {
        "I_620": 585.7,
        "I_540": 735.3,
        "R620": 1.485,
        "R540": 3.025,
        "cTnI_ng_mL": 14.35,
        "Myoglobin_ng_mL": 15.025,
        "Raw_Fluorescence_au": 2149.01,
        "DeltaF_au": 949.01,
        "Calculated_Troponin_ng_mL": 1.1863,
        "Peak_Current_uA": 280.739,
        "BNP_pg_mL": 623.42
    }
    response = client.post(
        "/predict",
        json=payload,
        headers={"X-API-Key": "wrong_key"}
    )
    assert response.status_code == 403
