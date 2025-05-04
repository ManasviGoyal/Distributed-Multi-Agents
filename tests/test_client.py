import pytest
import sys
import os
import importlib.util
from unittest import mock

# Ensure client.py is imported as src.client
module_name = "src.client"
file_path = os.path.join(os.path.dirname(__file__), "../src/client.py")
spec = importlib.util.spec_from_file_location(module_name, file_path)
client_module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = client_module

sys.modules['database_manager'] = mock.MagicMock()

spec.loader.exec_module(client_module)

# Patch environment and requests globally for all tests
@pytest.fixture(autouse=True)
def mock_env_and_requests(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "http://mocked-backend")
    monkeypatch.setenv("OPENROUTER_API_KEY", "mocked-api-key")

    with mock.patch("src.client.requests") as mock_requests:
        yield mock_requests

def test_init_client_success(mock_env_and_requests):
    # Setup mock responses
    mock_env_and_requests.get.side_effect = [
        mock.Mock(status_code=200, json=lambda: {"qwen": {"name": "qwen/..."} }),
        mock.Mock(status_code=200, json=lambda: {"domains": {"Education": "desc"}}),
        mock.Mock(status_code=200, json=lambda: {"question_types": ["Yes/No"]}),
        mock.Mock(status_code=200, json=lambda: {"examples": {"Education": [["ex", "type"]]}})
    ]

    success = client_module.init_client()
    assert success
    assert "qwen" in client_module.model_info
    assert "Education" in client_module.domains
    assert "Yes/No" in client_module.question_types
    assert "Education" in client_module.examples_by_domain

# def test_update_aggregator_success(mock_env_and_requests):
#     mock_env_and_requests.post.return_value = mock.Mock(
#         status_code=200,
#         json=lambda: {"aggregator_id": "llama3"}
#     )
#     result = client_module.update_aggregator("llama3")
#     assert result == "llama3"

def test_get_example_choices_valid(mock_env_and_requests):
    mock_env_and_requests.post.return_value = mock.Mock(
        status_code=200,
        json=lambda: {"examples": ["example1", "example2"]}
    )
    update = client_module.get_example_choices("Education")
    assert update["choices"] == ["example1", "example2"]

def test_get_example_choices_empty(mock_env_and_requests):
    update = client_module.get_example_choices("")
    assert update["choices"] == []

def test_fill_query_and_type(mock_env_and_requests):
    mock_env_and_requests.post.return_value = mock.Mock(
        status_code=200,
        json=lambda: {
            "query": "What are the ethical implications of using AI tools to assist students in writing essays?",
            "question_type": "Open-ended"
        }
    )

    example = "What are the ethical implications of using AI tools to assist students in writing essays?"
    domain = "Education"

    query, question_type = client_module.fill_query_and_type(example, domain)
    assert query == example
    assert question_type == "Open-ended"


def test_update_aggregator(mock_env_and_requests):
    mock_env_and_requests.post.return_value = mock.Mock(
        status_code=200,
        json=lambda: {"aggregator_id": "mistral"}
    )
    result = client_module.update_aggregator("mistral")
    assert result == "mistral"
