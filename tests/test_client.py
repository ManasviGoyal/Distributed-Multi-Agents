import pytest
import sys
import os
import importlib.util
from unittest import mock

# Import src.client in isolation-safe way
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

def test_init_client_success(monkeypatch):
    mock_get = mock.Mock()
    mock_get.side_effect = [
        mock.Mock(status_code=200, json=lambda: {"qwen": {"name": "qwen/..."} }),
        mock.Mock(status_code=200, json=lambda: {"domains": {"Education": "desc"}}),
        mock.Mock(status_code=200, json=lambda: {"question_types": ["Yes/No"]}),
        mock.Mock(status_code=200, json=lambda: {"examples": {"Education": [["ex", "type"]]}})
    ]
    monkeypatch.setattr(client_module.requests, "get", mock_get)

    success = client_module.init_client()
    assert success
    assert "qwen" in client_module.model_info
    assert "Education" in client_module.domains
    assert "Yes/No" in client_module.question_types
    assert "Education" in client_module.examples_by_domain

def test_get_example_choices_valid(monkeypatch):
    mock_post = mock.Mock(return_value=mock.Mock(
        status_code=200,
        json=lambda: {"examples": ["example1", "example2"]}
    ))
    monkeypatch.setattr(client_module.requests, "post", mock_post)

    update = client_module.get_example_choices("Education")
    assert update["choices"] == ["example1", "example2"]

def test_get_example_choices_empty():
    update = client_module.get_example_choices("")
    assert update["choices"] == []

def test_fill_query_and_type(monkeypatch):
    mock_post = mock.Mock(return_value=mock.Mock(
        status_code=200,
        json=lambda: {
            "query": "What are the ethical implications of using AI tools to assist students in writing essays?",
            "question_type": "Open-ended"
        }
    ))
    monkeypatch.setattr(client_module.requests, "post", mock_post)

    example = "What are the ethical implications of using AI tools to assist students in writing essays?"
    domain = "Education"

    query, question_type = client_module.fill_query_and_type(example, domain)
    assert query == example
    assert question_type == "Open-ended"

def test_update_aggregator(monkeypatch):
    mock_post = mock.Mock(return_value=mock.Mock(
        status_code=200,
        json=lambda: {"aggregator_id": "mistral"}
    ))
    monkeypatch.setattr(client_module.requests, "post", mock_post)

    result = client_module.update_aggregator("mistral")
    assert result == "mistral"
