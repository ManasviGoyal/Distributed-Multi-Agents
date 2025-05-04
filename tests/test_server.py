import pytest
import json
import asyncio
import unittest.mock as mock
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from fastapi import BackgroundTasks
from pytest_mock import MockFixture
import importlib.util
import os

# Mock the environment modules to avoid loading actual environment variables during tests
import sys

sys.modules['database_manager'] = mock.MagicMock()

# Ensure src/server.py is imported as src.server
module_name = "src.server"
file_path = os.path.join(os.path.dirname(__file__), "../src/server.py")
spec = importlib.util.spec_from_file_location(module_name, file_path)
server = importlib.util.module_from_spec(spec)
sys.modules[module_name] = server
spec.loader.exec_module(server)
from src.server import app, OpenRouterClient, ResponseAggregator, agent_health, MODELS, active_jobs, update_agent_heartbeat, check_agent_health, reset_failed_agent
from src.server import api_process_query, QueryRequest

sys.modules['dotenv'] = mock.MagicMock()
sys.modules['dotenv.load_dotenv'] = mock.MagicMock()
sys.modules['nltk'] = mock.MagicMock()
sys.modules['nltk.data'] = mock.MagicMock()
sys.modules['nltk.download'] = mock.MagicMock()
sys.modules['nltk.sentiment'] = mock.MagicMock()
sys.modules['nltk.sentiment.SentimentIntensityAnalyzer'] = mock.MagicMock()
sys.modules['sentence_transformers'] = mock.MagicMock()
sys.modules['textblob'] = mock.MagicMock()
sys.modules['PIL'] = mock.MagicMock()
sys.modules['matplotlib'] = mock.MagicMock()
sys.modules['matplotlib.pyplot'] = mock.MagicMock()
sys.modules['seaborn'] = mock.MagicMock()
# sys.modules['io'] = mock.MagicMock()

# Mock the database_manager
# with mock.patch.dict(sys.modules, {'database_manager': mock.MagicMock()}):
# from src.server import app, OpenRouterClient, ResponseAggregator, agent_health, MODELS, active_jobs, update_agent_heartbeat, check_agent_health, reset_failed_agent

# Create a test client
client = TestClient(app)

# Sample data for testing
test_job_id = "test-job-123"
test_query = "What is the most ethical way to handle AI development?"
test_domain = "Science/Technology"
test_question_type = "Open-ended"
test_username = "test_user"
test_api_key = "test_api_key"

@pytest.fixture
def mock_openrouter_client():
    """Fixture to create a mocked OpenRouterClient instance."""
    with mock.patch('src.server.OpenRouterClient') as mock_client:
        client_instance = mock.MagicMock()
        mock_client.return_value = client_instance
        
        # Mock async generate_response method
        async def mock_generate_response(model_name, prompt, temperature=0.7):
            return f"Response from {model_name} about {prompt[:20]}..."
        
        client_instance.generate_response = mock_generate_response
        client_instance.analyze_sentiment.return_value = {
            'polarity': 0.5,
            'compound': 0.6,
            'subjectivity': 0.4,
            'emotional_tones': {
                'Empathetic': 0.4,
                'Analytical': 0.3,
                'Curious': 0.3,
                'Judgmental': 0.0,
                'Ambivalent': 0.0,
                'Defensive': 0.0,
            },
            'tone_context': 'Empathetic and Supportive'
        }
        
        client_instance.get_embeddings.return_value = [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]
        
        yield client_instance

@pytest.fixture
def mock_response_aggregator(mock_openrouter_client):
    """Fixture to create a mocked ResponseAggregator instance."""
    with mock.patch('src.server.ResponseAggregator') as mock_aggregator:
        aggregator_instance = mock.MagicMock()
        mock_aggregator.return_value = aggregator_instance
        
        # Mock async process_query method
        async def mock_process_query(query, question_type="none", ethical_views=["None"]):
            return {
                "query": query,
                "formatted_query": f"Formatted: {query}",
                "responses": {
                    "qwen": "Response from qwen",
                    "llama3": "SUMMARY\n\nConsensus summary from llama3",
                    "mistral": "Response from mistral"
                },
                "analysis": {
                    "consensus_score": 0.8,
                    "heatmap": "data:image/png;base64,heatmap",
                    "emotion_chart": "data:image/png;base64,emotion",
                    "polarity_chart": "data:image/png;base64,polarity",
                    "radar_chart": "data:image/png;base64,radar"
                },
                "consensus_summary": "SUMMARY\n\nConsensus summary from llama3",
                "aggregator_id": "llama3"
            }
        
        aggregator_instance.process_query = mock_process_query
        
        yield aggregator_instance

# @pytest.fixture
# def mock_database_manager():
#     """Fixture to create a mocked DatabaseManager instance."""
#     with mock.patch('src.server.DatabaseManager') as mock_db:
#         db_instance = mock.MagicMock()
#         mock_db.return_value = db_instance
        
#         # Mock methods
#         db_instance.save_interaction.return_value = True
#         db_instance.save_responses.return_value = True
#         db_instance.save_analysis.return_value = True
#         db_instance.get_interaction.return_value = {
#             "job_id": test_job_id,
#             "query": test_query,
#             "domain": test_domain,
#             "question_type": test_question_type,
#             "username": test_username,
#             "responses": {
#                 "qwen": {"response": "Response from qwen", "is_aggregator": False},
#                 "llama3": {"response": "Response from llama3", "is_aggregator": True},
#                 "mistral": {"response": "Response from mistral", "is_aggregator": False}
#             },
#             "consensus_score": 0.8,
#             "analysis": {
#                 "heatmap": "data:image/png;base64,heatmap"
#             }
#         }
        
#         yield db_instance

@pytest.fixture
def mock_database_manager():
    db_instance = mock.MagicMock()
    sys.modules['database_manager'].DatabaseManager.return_value = db_instance
    yield db_instance

@pytest.fixture
def mock_background_tasks():
    """Fixture to create a mocked BackgroundTasks instance."""
    background_tasks = mock.MagicMock(spec=BackgroundTasks)
    return background_tasks

# Define tests for main API endpoints

def test_root_endpoint():
    """Test the root endpoint returns the expected message."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Multi-Agent LLM Backend is running"}

def test_get_models():
    """Test the /models endpoint returns the correct model information."""
    response = client.get("/models")
    assert response.status_code == 200
    
    # Verify response contains model info
    data = response.json()
    assert isinstance(data, dict)
    assert "qwen" in data
    assert "llama3" in data
    assert "aggregator" in data["llama3"]
    assert data["llama3"]["aggregator"] == True

def test_get_domains():
    """Test the /domains endpoint returns domain information."""
    response = client.get("/domains")
    assert response.status_code == 200
    
    # Verify domains include expected values
    data = response.json()
    assert "domains" in data
    assert "Education" in data["domains"]
    assert "Custom" in data["domains"]

def test_get_question_types():
    """Test the /question_types endpoint returns question type information."""
    response = client.get("/question_types")
    assert response.status_code == 200
    
    # Verify question types include expected values
    data = response.json()
    assert "question_types" in data
    assert "Open-ended" in data["question_types"]
    assert "Yes/No" in data["question_types"]
    assert "None" in data["question_types"]

def test_get_examples():
    """Test the /examples endpoint returns example information."""
    response = client.get("/examples")
    assert response.status_code == 200
    
    # Verify examples include expected domains
    data = response.json()
    assert "examples" in data
    assert "Education" in data["examples"]
    assert "Custom" in data["examples"]

def test_get_agent_health():
    """Test the /agent_health endpoint returns health status."""
    response = client.get("/agent_health")
    assert response.status_code == 200
    
    # Verify agent health data structure
    data = response.json()
    assert "qwen" in data
    assert "llama3" in data
    assert "status" in data["qwen"]
    assert "last_heartbeat" in data["qwen"]

def test_get_example_choices():
    """Test the /get_example_choices endpoint returns examples for a domain."""
    response = client.post("/get_example_choices", json={"domain": "Education"})
    assert response.status_code == 200
    
    data = response.json()
    assert "examples" in data

def test_fill_query_and_type():
    """Test the /fill_query_and_type endpoint with example and domain."""
    response = client.post(
        "/fill_query_and_type", 
        json={
            "selected_example": "What are the ethical implications of using AI tools to assist students in writing essays?",
            "domain": "Education"
        }
    )
    assert response.status_code == 200
    
    data = response.json()
    assert "query" in data
    assert "question_type" in data

def test_update_aggregator():
    """Test the /update_aggregator endpoint sets an aggregator model."""
    response = client.post("/update_aggregator", json={"aggregator_id": "mistral"})
    assert response.status_code == 200
    
    data = response.json()
    assert data["aggregator_id"] == "mistral"
    assert data["success"] == True

def test_reset_agent():
    """Test the /reset_agent endpoint resets a failed agent."""
    # Setup - mark an agent as failed
    agent_health["qwen"]["status"] = "failed"
    agent_health["qwen"]["retries"] = 3
    
    response = client.post("/reset_agent/qwen")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "success"
    assert agent_health["qwen"]["status"] == "healthy"
    assert agent_health["qwen"]["retries"] == 0

def test_reset_agent_not_found():
    """Test the /reset_agent endpoint with non-existent agent."""
    response = client.post("/reset_agent/nonexistent_agent")
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_update_agent_heartbeat():
    """Test updating agent heartbeat."""
    # Reset the agent health to a known state
    agent_health["qwen"] = {
        "status": "unhealthy",
        "last_heartbeat": datetime.now() - timedelta(minutes=5),
        "failures": 2,
        "retries": 1,
        "has_processed_request": False
    }
    
    # Update the heartbeat
    await update_agent_heartbeat("qwen")
    
    # Verify the agent state has been updated correctly
    assert agent_health["qwen"]["status"] == "healthy"
    assert (datetime.now() - agent_health["qwen"]["last_heartbeat"]).total_seconds() < 1
    assert agent_health["qwen"]["has_processed_request"] == True

@pytest.mark.asyncio
async def test_check_agent_health():
    """Test the agent health check logic."""
    # Set up initial agent states
    agent_health["qwen"] = {
        "status": "healthy",
        "last_heartbeat": datetime.now() - timedelta(seconds=130),  # Simulate > 2x HEALTH_TIMEOUT
        "failures": 0,
        "retries": 0,
        "has_processed_request": True
    }

    agent_health["llama3"] = {
        "status": "healthy",
        "last_heartbeat": datetime.now() - timedelta(seconds=10),  # Fresh
        "failures": 0,
        "retries": 0,
        "has_processed_request": True
    }

    with mock.patch("src.server.update_agent_heartbeat", side_effect=lambda mid: None):
        await check_agent_health()

    assert agent_health["llama3"]["status"] == "healthy"
    assert agent_health["qwen"]["status"] == "unhealthy"


@pytest.mark.asyncio
async def test_reset_failed_agent():
    """Test resetting a failed agent."""
    # Set up a failed agent
    agent_health["mistral"] = {
        "status": "failed",
        "last_heartbeat": datetime.now() - timedelta(minutes=10),
        "failures": 5,
        "retries": 3,
        "has_processed_request": True
    }
    
    # Reset the agent
    await reset_failed_agent("mistral")
    
    # Verify the agent has been reset correctly
    assert agent_health["mistral"]["status"] == "healthy"
    assert agent_health["mistral"]["failures"] == 0
    assert agent_health["mistral"]["retries"] == 0
    assert (datetime.now() - agent_health["mistral"]["last_heartbeat"]).total_seconds() < 1

# @pytest.mark.asyncio
# async def test_process_query_background(mock_database_manager, mock_response_aggregator):
#     """Test the background process_query function."""
#     from src.server import process_query_background
    
#     # Call the background function
#     await process_query_background(
#         job_id=test_job_id,
#         query=test_query,
#         question_type=test_question_type,
#         domain=test_domain,
#         ethical_views=["Utilitarian"],
#         username=test_username
#     )
    
#     # Verify job was processed and added to active_jobs
#     assert test_job_id in active_jobs
#     assert active_jobs[test_job_id]["status"] == "completed"
#     assert active_jobs[test_job_id]["progress"] == 100
    
#     # Verify database interactions
#     mock_database_manager.return_value.save_responses.assert_called_once()
#     mock_database_manager.return_value.save_interaction.assert_called_once()
#     mock_database_manager.return_value.save_analysis.assert_called_once()
    
#     # Clean up
#     if test_job_id in active_jobs:
#         del active_jobs[test_job_id]

# def test_process_query_api(mock_background_tasks, monkeypatch):
#     """Test the /process_query endpoint."""
#     # Mock the uuid module to return a consistent job ID
#     monkeypatch.setattr('uuid.uuid4', lambda: test_job_id)
    
#     response = client.post(
#         "/process_query",
#         json={
#             "query": test_query,
#             "api_key": test_api_key,
#             "question_type": test_question_type,
#             "domain": test_domain,
#             "aggregator_id": "llama3",
#             "username": test_username,
#             "ethical_views": ["Utilitarian"]
#         }
#     )
    
#     assert response.status_code == 200
#     data = response.json()
#     assert data["job_id"] == test_job_id
#     assert data["status"] == "processing"
    
#     # Verify background task was added
#     mock_background_tasks.add_task.assert_called_once()
    
#     # Clean up
#     if test_job_id in active_jobs:
#         del active_jobs[test_job_id]

@pytest.mark.asyncio
async def test_process_query_api(mock_background_tasks, monkeypatch):
    monkeypatch.setattr("uuid.uuid4", lambda: test_job_id)

    req = QueryRequest(
        query=test_query,
        api_key=test_api_key,
        question_type=test_question_type,
        domain=test_domain,
        aggregator_id="llama3",
        username=test_username,
        ethical_views=["Utilitarian"]
    )

    result = await api_process_query(req, mock_background_tasks)

    assert result["job_id"] == test_job_id
    assert result["status"] == "processing"
    mock_background_tasks.add_task.assert_called_once()

def test_job_status_not_found():
    """Test job_status endpoint with non-existent job ID."""
    response = client.get(f"/job_status/nonexistent-job")
    assert response.status_code == 404

def test_job_result_not_found():
    """Test job_result endpoint with non-existent job ID."""
    job_id = "nonexistent-job"

    # Ensure the job isn't in memory
    if job_id in active_jobs:
        del active_jobs[job_id]

    # Patch the actual instance used in the app
    from src.server import db_manager
    db_manager.get_interaction = mock.MagicMock(return_value=None)

    response = client.get(f"/job_result/{job_id}")
    assert response.status_code == 404

def test_job_status_and_result(monkeypatch):
    """Test the job_status and job_result endpoints."""
    # Add a mock job to active_jobs
    active_jobs[test_job_id] = {
        "status": "completed",
        "progress": 100,
        "result": {
            "responses": {
                "qwen": "Response from qwen",
                "llama3": "Consensus from llama3",
                "mistral": "Response from mistral"
            },
            "analysis": {
                "consensus_score": 0.8,
                "heatmap": "data:image/png;base64,heatmap"
            },
            "consensus_score": 80,
            "query": test_query,
            "question_type": test_question_type,
            "domain": test_domain,
            "aggregator_id": "llama3"
        }
    }
    
    # Test job_status
    status_response = client.get(f"/job_status/{test_job_id}")
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert status_data["status"] == "completed"
    assert status_data["progress"] == 100
    
    # Test job_result
    result_response = client.get(f"/job_result/{test_job_id}")
    assert result_response.status_code == 200
    result_data = result_response.json()
    assert "responses" in result_data
    assert "llama3" in result_data["responses"]
    assert result_data["aggregator_id"] == "llama3"
    
    # Clean up
    if test_job_id in active_jobs:
        del active_jobs[test_job_id]

@pytest.mark.asyncio
async def test_openrouter_client_generate_response(monkeypatch):
    with mock.patch('src.server.requests.post') as mock_post:
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "This is a test response"}}]
        }
        mock_post.return_value = mock_response

        # Use real model name, not model ID
        client = OpenRouterClient("test_api_key", "http://test.site", "Test Site")
        response = await client.generate_response("qwen/qwen-2.5-7b-instruct:free", "Test prompt", 0.5)

        assert response == "This is a test response"

@pytest.mark.asyncio
async def test_openrouter_client_generate_response_error(monkeypatch):
    with mock.patch('src.server.requests.post') as mock_post:
        mock_response = mock.MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
        mock_post.return_value = mock_response

        client = OpenRouterClient("test_api_key", "http://test.site", "Test Site")
        response = await client.generate_response("qwen/qwen-2.5-7b-instruct:free", "Test prompt", 0.5)

        assert "Error: API returned status code 400" in response

@pytest.mark.asyncio
async def test_response_aggregator_process_query(monkeypatch, mock_openrouter_client):
    """Test ResponseAggregator's process_query method."""
    # Create a test instance with our mocked OpenRouterClient
    aggregator = ResponseAggregator(mock_openrouter_client)
    
    # Mock the generate_consensus_summary method
    original_method = aggregator.generate_consensus_summary
    
    async def mock_consensus(*args, **kwargs):
        return "This is a test consensus summary"
    
    aggregator.generate_consensus_summary = mock_consensus
    
    # Also mock the analyze_responses method
    original_analyze = aggregator.analyze_responses
    
    async def mock_analyze(*args, **kwargs):
        return {
            "similarity_matrix": {"model1": {"model2": 0.8}},
            "response_lengths": {"model1": 100, "model2": 120},
            "sentiment_analysis": {"model1": {"polarity": 0.5}},
            "consensus_score": 0.8
        }
    
    aggregator.analyze_responses = mock_analyze
    
    # Mock agent health to have some healthy agents
    agent_health.clear()
    for model_id in ["qwen", "llama3", "mistral"]:
        agent_health[model_id] = {
            "status": "healthy",
            "last_heartbeat": datetime.now(),
            "failures": 0,
            "retries": 0,
            "has_processed_request": True
        }
    
    # Test the method
    result = await aggregator.process_query("Test query", "open-ended", ["Utilitarian"])
    
    # Verify the result structure
    assert "query" in result
    assert "responses" in result
    assert "analysis" in result
    assert "consensus_summary" in result
    assert result["consensus_summary"] == "This is a test consensus summary"
    
    # Restore original methods
    aggregator.generate_consensus_summary = original_method
    aggregator.analyze_responses = original_analyze

# Test the image generation endpoints

def test_get_image_not_found():
    """Test getting an image for a non-existent job."""
    response = client.get("/image/nonexistent-job/heatmap")
    assert response.status_code == 404

def test_get_image_invalid_type():
    """Test getting an image with an invalid image type."""
    # First add a mock job
    active_jobs[test_job_id] = {
        "status": "completed",
        "result": {
            "analysis": {
                "heatmap": "data:image/png;base64,heatmap"
            }
        }
    }
    
    response = client.get(f"/image/{test_job_id}/invalid_type")
    assert response.status_code == 400
    
    # Clean up
    if test_job_id in active_jobs:
        del active_jobs[test_job_id]

def test_get_image_job_not_completed():
    """Test getting an image for a job that's not yet completed."""
    # Add a processing job
    active_jobs[test_job_id] = {
        "status": "processing",
        "progress": 50
    }
    
    response = client.get(f"/image/{test_job_id}/heatmap")
    assert response.status_code == 400
    
    # Clean up
    if test_job_id in active_jobs:
        del active_jobs[test_job_id]

def test_get_image_missing_analysis():
    """Test getting an image for a job without analysis data."""
    # Add a job without analysis
    active_jobs[test_job_id] = {
        "status": "completed",
        "result": {
            "responses": {}
        }
    }
    
    response = client.get(f"/image/{test_job_id}/heatmap")
    assert response.status_code == 400
    
    # Clean up
    if test_job_id in active_jobs:
        del active_jobs[test_job_id]

def test_get_image_success(monkeypatch):
    """Test successfully getting an image."""
    import base64
    
    # Mock the StreamingResponse
    with mock.patch('src.server.StreamingResponse') as mock_streaming:
        mock_streaming.return_value = "mocked_response"
        
        # Add a job with image data
        active_jobs[test_job_id] = {
            "status": "completed",
            "result": {
                "analysis": {
                    "heatmap": "data:image/png;base64,aGVsbG8="  # base64 for "hello"
                }
            }
        }
        
        response = client.get(f"/image/{test_job_id}/heatmap")
        
        # The TestClient doesn't process the StreamingResponse correctly
        # but we can verify the mock was called with the right params
        mock_streaming.assert_called_once()
        
        # Clean up
        if test_job_id in active_jobs:
            del active_jobs[test_job_id]