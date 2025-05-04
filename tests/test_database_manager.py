import os
import tempfile
import pytest
from src.database_manager import DatabaseManager

@pytest.fixture
def temp_db():
    """Creates a temporary database for testing."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        db_path = tmp.name
    db = DatabaseManager(db_path=db_path)
    yield db
    os.remove(db_path)

def test_create_user_and_verify(temp_db):
    assert temp_db.create_user("alice", "Password123") is True
    assert temp_db.verify_user("alice", "Password123") is True
    assert temp_db.verify_user("alice", "WrongPassword") is False
    assert temp_db.create_user("alice", "Password123") == "duplicate"

def test_delete_user(temp_db):
    temp_db.create_user("bob", "Secret123")
    assert temp_db.delete_user("bob", "Secret123") is True
    assert temp_db.verify_user("bob", "Secret123") is False

def test_save_and_get_interaction(temp_db):
    job_id = "job-1"
    temp_db.save_interaction(job_id, "What's AI?", "Science", "Open-ended", "carol")
    result = temp_db.get_interaction(job_id)
    assert result is not None
    assert result["job_id"] == job_id
    assert result["query"] == "What's AI?"

def test_save_and_get_responses(temp_db):
    job_id = "job-2"
    temp_db.save_interaction(job_id, "Test?", "Test", "None", "dan")
    responses = {"agent1": "Response A", "agent2": "Response B"}
    assert temp_db.save_responses(job_id, responses, aggregator_id="agent2")
    result = temp_db.get_interaction(job_id)
    assert result["responses"]["agent1"]["response"] == "Response A"
    assert result["responses"]["agent2"]["is_aggregator"] is True

def test_save_and_get_analysis(temp_db):
    job_id = "job-3"
    temp_db.save_interaction(job_id, "Analyze?", "Custom", "None", "eve")
    analysis_data = {"polarity": 0.5}
    assert temp_db.save_analysis(job_id, 0.75, analysis_data)
    result = temp_db.get_interaction(job_id)
    assert result["consensus_score"] == 0.75
    assert result["analysis"]["polarity"] == 0.5

def test_get_user_history(temp_db):
    for i in range(3):
        temp_db.save_interaction(f"job-{i}", f"query {i}", "General", "None", "frank")
    history = temp_db.get_user_history("frank")
    assert len(history) == 3
    assert all("job_id" in h for h in history)

def test_get_interaction_history(temp_db):
    for i in range(2):
        temp_db.save_interaction(f"jobx-{i}", f"query {i}", "General", "None", "gwen")
    history = temp_db.get_interaction_history()
    assert isinstance(history, list)
    assert len(history) >= 2

def test_delete_interaction_authorized(temp_db):
    job_id = "job-del"
    temp_db.save_interaction(job_id, "Delete me", "Policy", "None", "hank")
    temp_db.save_responses(job_id, {"a1": "Resp"})
    temp_db.save_analysis(job_id, 0.5, {"foo": "bar"})
    assert temp_db.delete_interaction(job_id, "hank") is True
    assert temp_db.get_interaction(job_id) is None

def test_delete_interaction_unauthorized(temp_db):
    job_id = "job-protected"
    temp_db.save_interaction(job_id, "Protected", "Policy", "None", "irene")
    assert temp_db.delete_interaction(job_id, "not_irene") is False
