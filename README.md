# Distributed Multi-Agent LLM System

This project implements a fault-tolerant distributed application framework that leverages multiple LLM agents to better understand model behavior, bias, consensus, and reliability. The system distributes queries across multiple language models, aggregates their responses using a consensus mechanism, and supports automatic failover and retry strategies for fault tolerance.

## Table of Contents

- [Main Features](#main-features)
- [Use Cases](#use-cases)
- [Running the Application](#running-the-application)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [User Interface](#user-interface)
- [API Reference](#api-reference)
- [Testing and Documentation](#testing-and-documentation)

## Main Features

- **Multi-Agent Architecture**: Distributes queries to multiple LLM models in parallel.
- **Consensus Aggregation**: Synthesizes responses from different models to generate a more comprehensive answer.
- **Agent Health Monitoring**: Continuously tracks agent performance and availability.
- **Response Analysis**: We get insights into model agreement, sentiment analysis, and emotional tone.
- **Visualizations**: Includes heatmaps, emotion charts, polarity analysis, and radar comparisons.
- **User Management**: Authentication and history tracking for multiple users.
- **Domain Expertise**: Configurable domain prefixes to provide specialized context for queries.
- **Ethical Role Assignment**: Supports assigning ethical reasoning roles (Utilitarian, Libertarian, etc.) to models dynamically.

## Use Cases

The Distributed Multi-Agent LLM System is ideal for:

- **Enhanced Decision-Making**: Combining multiple model perspectives to achieve more robust and informed conclusions.
- **Bias**: Detecting training biases by contrasting how different models reason and respond.
- **Policy and Ethics Analysis**: Examining complex policy or ethical dilemmas from multiple philosophical standpoints.
- **Model Benchmarking**: Comparing model behavior side-by-side to analyze performance consistency and response diversity.

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenRouter API key (for accessing LLM models)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ManasviGoyal/Distributed-Multi-Agents.git
cd Distributed-Multi-Agents
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file (or use the auto-generated one on first run):
```
OPENROUTER_API_KEY=your_openrouter_api_key
SITE_URL=http://localhost:7860
SITE_NAME=Multi-Agent LLM System
```

## Running the Application

### Single Server

1. Start the backend server:
```bash
python src/server.py --host 0.0.0.0 --port 8000
```

2. Start the client interface:
```bash
python src/client.py --backend-url http://localhost:8000 --host 0.0.0.0 --port 7860
```

3. Access the web interface at:
```
http://localhost:7860
```

### Multiple Servers with Load Balancer

1. Start multiple backend servers (on different ports):
```bash
python src/server.py --host 0.0.0.0 --port 8001
python src/server.py --host 0.0.0.0 --port 8002
python src/server.py --host 0.0.0.0 --port 8003
```

2. Start the load balancer using Uvicorn:
```
uvicorn src.load_balancer:app --host 0.0.0.0 --port 8000
```
This starts the load balancer FastAPI server at `http://localhost:8000`, which routes traffic across the backends.

3. Start the client interface pointing to the load balancer:
```bash
python src/client.py --backend-url http://localhost:8000 --host 0.0.0.0 --port 7860
```

4. Access the web interface at:
```
http://localhost:7860
```

## Architecture

The system consists of four main components:

### Backend Server (`server.py`)

- Handles query processing and agent communication
- Manages agent health monitoring
- Performs response aggregation and analysis
- Provides REST API endpoints for the client

### Client Application (`client.py`)

- Spins up a user-friendly Gradio interface
- Displays model responses and analysis visualizations
- Manages user sessions and interaction history
- Communicates with the backend server via HTTP

### Load Balancer (`load_balancer.py`)

- Routes incoming client requests intelligently
- Provides sticky routing for job-related queries (e.g., fetching images or results)
- Uses round-robin dispatch with retries for new processing requests
- Tolerates up to two backend server failures transparentl

### Database Management (`database_manager.py`)

- Stores user information and interaction history
- Manages response and analysis data persistence
- Provides query and retrieval functionality

## Configuration

### Agent Models

You can configure the LLM models used by editing the `MODELS` dictionary in `server.py` (refer to following example):

```python
MODELS = {
    "qwen": {
        "name": "qwen/qwen-2.5-7b-instruct:free",
        "temperature": 0.3,
    },
    "llama3": {
        "name": "meta-llama/llama-3.1-8b-instruct:free",
        "temperature": 0.3,
        "aggregator": True,  # Marks llama3 as the default aggregator model
    },
    # Add more models as needed
}
```

### Domain Expertise

Customize the domains and their prefixes in `server.py` (refer to following example):

```python
DOMAINS = {
    "Custom": "",
    "Education": "As an education policy expert, ",
    "Healthcare": "As a health policy expert, ",
    # Add more domains as needed
}
```

## User Interface

The client provides a Gradio interface with multiple tabs:

### Query Input
- Select a domain expertise area
- Choose a question type
- Input your query or select from examples
- Select the aggregator model
- Optionally assign ethical perspectives to agent models

### Model Responses
- View the aggregator's consensus summary
- Compare individual model responses

### Analysis Visualizations
- Response similarity heatmap
- Emotional tone analysis
- Sentiment polarity comparison
- Radar chart comparing response features

### Interaction History
- Browse previous queries and responses
- Load or delete past interactions

## API Reference

The backend server exposes the following key endpoints:

- `POST /process_query`: Submit a new query for processing
- `GET /job_status/{job_id}`: Check the status of a processing job
- `GET /job_result/{job_id}`: Retrieve the results of a completed job
- `GET /image/{job_id}/{image_type}`: Get analysis visualizations
- `GET /models`: List available LLM models
- `GET /domains`: List available domains
- `GET /history`: Retrieve user interaction history
- `DELETE /history/{job_id}`: Delete a specific interaction
- `POST /update_aggregator`: Update the aggregator model dynamically

## Testing and Documentation
This project is thoroughly tested and documented. 

We use `pytest` for unit testing across the server, client, database manager, and load balancer components. Run the tests from project root:

```
PYTHONPATH=src pytest tests/ --cov=src --cov-config=.coveragerc
```

All classes and methods are documented with Google-style docstrings for consistency and clarity. The complete developer and API documentation is available via Sphinx and rendered using the Read the Docs theme. 

The HTML documentation can be rebuilt locally as:

```bash
cd docs
make html   # or on Windows: .\make.bat html
```

The generated docs can be found at the deployed at [GitHub Actions](https://manasvigoyal.github.io/Distributed-Multi-Agents/).