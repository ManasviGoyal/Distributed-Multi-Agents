import sys
from unittest import mock

# Patch early to avoid import errors if database_manager isn't available
sys.modules["database_manager"] = mock.MagicMock()

import pytest
import asyncio
import base64
from datetime import datetime
from fastapi.testclient import TestClient

# Correct import for robust monkeypatching
from src import server as server_module
from src.server import app, active_jobs, MODELS, process_query_background

client = TestClient(app)

@pytest.fixture
def sample_job():
    return {
        "status": "completed",
        "progress": 100,
        "result": {
            "responses": {
                "qwen": "Response A",
                "llama3": "Summary B"
            },
            "analysis": {
                "consensus_score": 0.82,
                "heatmap": "data:image/png;base64," + base64.b64encode(b"fakeimage").decode(),
                "emotion_chart": "data:image/png;base64," + base64.b64encode(b"fakeimage").decode(),
                "polarity_chart": "data:image/png;base64," + base64.b64encode(b"fakeimage").decode(),
                "radar_chart": "data:image/png;base64," + base64.b64encode(b"fakeimage").decode(),
            },
            "consensus_score": 82,
            "query": "test query",
            "question_type": "Open-ended",
            "domain": "Science/Technology",
            "aggregator_id": "llama3"
        }
    }

def test_history_endpoint_mocked(monkeypatch):
    mock_db = mock.MagicMock()
    monkeypatch.setattr(server_module, "db_manager", mock_db)
    mock_db.get_user_history.return_value = [{"job_id": "job1", "query": "Q", "timestamp": 1234, "responses": {}, "roles": ""}]
    res = client.get("/history", params={"username": "testuser"})
    assert res.status_code == 200
    assert "history" in res.json()

def test_delete_history_success(monkeypatch):
    monkeypatch.setattr(server_module.db_manager, "delete_interaction", lambda jid, u: True)
    res = client.delete("/history/jobX", params={"username": "someone"})
    assert res.status_code == 200
    assert res.json() == {"success": True}

def test_delete_history_failure(monkeypatch):
    monkeypatch.setattr(server_module.db_manager, "delete_interaction", lambda jid, u: False)
    res = client.delete("/history/jobX", params={"username": "failuser"})
    assert res.status_code == 500

def test_job_result_fallback_from_db(monkeypatch):
    job_id = "abc-123"
    if job_id in active_jobs:
        del active_jobs[job_id]

    fake_result = {
        "job_id": job_id,
        "query": "Q",
        "timestamp": 123,
        "domain": "Science/Technology",
        "question_type": "Open-ended",
        "responses": {
            "llama3": {"response": "Aggregator response", "is_aggregator": True},
            "qwen": {"response": "Agent response", "is_aggregator": False}
        },
        "consensus_score": 0.76,
        "analysis": {"emotion_chart": "data:image/png;base64," + base64.b64encode(b"img").decode()}
    }

    monkeypatch.setattr(server_module.db_manager, "get_interaction", lambda jid: fake_result)
    res = client.get(f"/job_result/{job_id}")
    assert res.status_code == 200
    assert res.json()["aggregator_id"] == "llama3"
    assert "responses" in res.json()

def test_image_route_success(sample_job):
    job_id = "image-job"
    active_jobs[job_id] = sample_job
    res = client.get(f"/image/{job_id}/heatmap")
    assert res.status_code == 200

def test_image_route_invalid_type(sample_job):
    job_id = "image-job2"
    active_jobs[job_id] = sample_job
    res = client.get(f"/image/{job_id}/invalid_image_type")
    assert res.status_code == 400

def test_image_route_not_ready():
    job_id = "notready"
    active_jobs[job_id] = {"status": "processing"}
    res = client.get(f"/image/{job_id}/heatmap")
    assert res.status_code == 400

def test_image_route_no_analysis():
    job_id = "noanalysis"
    active_jobs[job_id] = {"status": "completed", "result": {}}
    res = client.get(f"/image/{job_id}/heatmap")
    assert res.status_code == 400

@pytest.mark.asyncio
async def test_process_query_background_db_failure(monkeypatch):
    """Simulate background query processing where DB write fails."""

    monkeypatch.setattr(server_module.db_manager, "save_interaction", lambda *a, **kw: False)
    monkeypatch.setattr(server_module.db_manager, "save_responses", lambda *a, **kw: True)
    monkeypatch.setattr(server_module.db_manager, "save_analysis", lambda *a, **kw: True)

    mock_aggregator = mock.MagicMock()
    mock_aggregator.process_query = mock.AsyncMock(return_value={
        "responses": {mid: "test response" for mid in MODELS},
        "query": "Q?",
        "analysis": {"consensus_score": 0.8},
        "aggregator_id": "llama3"
    })
    monkeypatch.setattr(server_module, "response_aggregator", mock_aggregator)

    await process_query_background(
        job_id="failjob",
        query="Q?",
        question_type="Open-ended",
        domain="Healthcare",
        ethical_views=["Utilitarian"],
        username="test_user"
    )

    assert active_jobs["failjob"]["status"] == "completed"
    assert active_jobs["failjob"]["result"]["responses"]["llama3"].startswith("test")
