import pytest
from unittest import mock
import sys
import os
import importlib.util

# Import src.client in a way that works across test runs
module_name = "src.client"
file_path = os.path.join(os.path.dirname(__file__), "../src/client.py")
spec = importlib.util.spec_from_file_location(module_name, file_path)
client_module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = client_module
sys.modules['database_manager'] = mock.MagicMock()

spec.loader.exec_module(client_module)

@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "http://mocked-backend")
    monkeypatch.setenv("OPENROUTER_API_KEY", "mocked-api-key")

def test_init_client_fails(monkeypatch):
    # Simulate backend request failures
    monkeypatch.setattr(client_module.requests, "get", mock.Mock(side_effect=Exception("fail")))
    result = client_module.init_client(retries=2, delay=0.1)
    assert result is False

def test_update_aggregator_success(monkeypatch):
    mock_post = mock.Mock()
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"aggregator_id": "llama3"}
    monkeypatch.setattr(client_module.requests, "post", mock_post)

    result = client_module.update_aggregator("llama3")
    assert result == "llama3"

def test_update_aggregator_failure(monkeypatch):
    mock_post = mock.Mock()
    mock_post.return_value.status_code = 500
    monkeypatch.setattr(client_module.requests, "post", mock_post)

    result = client_module.update_aggregator("llama3")
    assert result is None

def test_get_example_choices_none():
    update = client_module.get_example_choices("")
    assert update["choices"] == []

def test_get_example_choices_success(monkeypatch):
    mock_resp = mock.Mock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"examples": ["a", "b"]}
    monkeypatch.setattr(client_module.requests, "post", mock.Mock(return_value=mock_resp))

    update = client_module.get_example_choices("Education")
    assert "a" in update["choices"]

def test_fill_query_and_type_success(monkeypatch):
    monkeypatch.setattr(client_module.requests, "post", mock.Mock(return_value=mock.Mock(
        status_code=200,
        json=lambda: {"query": "What is AI?", "question_type": "Open-ended"}
    )))
    q, qtype = client_module.fill_query_and_type("What is AI?", "Science")
    assert q == "What is AI?"
    assert qtype == "Open-ended"

def test_fill_query_and_type_failure(monkeypatch):
    monkeypatch.setattr(client_module.requests, "post", mock.Mock(side_effect=Exception("fail")))
    q, qtype = client_module.fill_query_and_type("What is AI?", "Science")
    assert q == ""
    assert qtype == "None"
