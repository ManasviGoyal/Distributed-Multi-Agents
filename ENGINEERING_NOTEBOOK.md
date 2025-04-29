# Engineering Notebook: Distributed Multi-Agent LLM System

This engineering notebook documents the design, development, and iterative progress of our Distributed Multi-Agent LLM System. The project explores how multiple lightweight language models can be orchestrated in parallel to improve response reliability, ethical coverage, and system resilience. By combining a modular architecture with fault-tolerant mechanisms, we aimed to overcome the limitations of single-model systems—such as inconsistency, fragility, and narrow ethical reasoning.

The system we built consists of multiple independently deployed agents, a central aggregator, a smart load balancer, and a persistent database. These components communicate through well-defined APIs and are coordinated to deliver high-availability, low-latency query handling. Our design also emphasizes transparency and interpretability by including response visualizations and configurable ethical roles. The following notebook sections document the motivation, technical architecture, implementation choices, and development timeline for this project.

## Table of Contents

1. [Need and Motivation](#need-and-motivation)
2. [Components](#components)
3. [Key Features](#key-features)
4. [Architecture](#architecture)
5. [Design Decisions](#design-decisions)
6. [How a Query Flows Through the System](#how-a-query-flows-through-the-system)
7. [Day to Day Progress](#day-to-day-progress)

## Need and Motivation

Large Language Models (LLMs) have demonstrated impressive capabilities across a wide range of reasoning and generation tasks, but relying on a single LLM often leads to limitations in reliability, consistency, ethical reasoning, and overall response quality. Single-model systems can produce inconsistent or biased outputs, and they typically lack fault tolerance. If the model fails, the entire system fails. Additionally, frequent external API usage incurs high costs and latency, especially when re-querying similar prompts. Moreover, individual models tend to represent narrow ethical perspectives, limiting the diversity of viewpoints in sensitive or morally complex domains.

To address these challenges, we designed and built a distributed multi-agent system in which multiple LLMs reason independently on the same prompt, and an aggregator model synthesizes a consensus response. This setup improves reliability by introducing redundancy, ensures fault tolerance through dynamic failover mechanisms, and enhances ethical diversity by allowing each model to adopt different normative perspectives. The system also significantly reduces API costs and latency through local caching and persistence.

Crucially, we chose a distributed architecture to enable parallelism, modular scalability, and resilience. Separating the client, load balancer, backend servers, and model agents into independent components allowed us to isolate failures, deploy updates without downtime, and dynamically balance load across agents. This distributed design was essential to meeting our goals of high availability, low latency, and ethical interpretability, while maintaining a system that is easy to maintain, extend, and evolve.

## Components

The system follows a client-server architecture:

1. **Backend Server** (`server.py`): Manages LLM agents, handles query processing, performs response aggregation and analysis, and provides REST API endpoints.

2. **Client Application** (`client.py`): Offers a Gradio-based user interface for submitting queries, viewing responses, and analyzing results. Talks to backend via REST APIs.

3. **Database Manager** (`database_manager.py`): SQLite database for handling data persistence, user authentication, and interaction history.

4. **Load Balancer** (`load_balancer.py`): Smart proxy using round-robin and sticky job-based routing. Provides 2-fault tolerance (tries up to 3 servers if a backend fails).

## Key Features

- **Multi-Model Querying**: Distributes queries to multiple LLM models
- **Consensus Aggregation**: Synthesizes responses to produce a comprehensive answer
- **Agent Health Monitoring**: Tracks model availability and performance
- **Response Analysis**: Provides visualization and metrics for model agreement
- **User Management**: Authentication and history tracking
- **Domain Expertise**: Adds domain-specific context to queries
- **Ethical Role Assignment**: Supports assigning ethical reasoning roles (Utilitarian, Libertarian, etc.) to models dynamically.

## Architecture
The system architecture is built to support robust, scalable, and fault-tolerant multi-agent reasoning across multiple LLMs. It organizes the workflow into distinct, specialized layers: a Gradio-based client interface for user interaction, a smart load balancer for intelligent request routing, distributed backend servers for query processing and model aggregation, and a structured database for persistent storage and retrieval. This separation of concerns ensures seamless user experiences, resilient query handling, and comprehensive response analysis, even under partial system failures. The following sections describe each major component of the architecture in detail.

### Database Schema and Persistence

The structure of the system’s database is depicted in **Figure 1**, which shows the relationships between users, queries, model responses, and analysis results. The `USERS` table stores authentication credentials, with each user identified by a unique ID, a username, and a hashed password. Each user interaction with the system is recorded in the `INTERACTIONS` table, which links back to the `USERS` table via a foreign key on the username. Every interaction is assigned a unique `job_id` and captures the query text, domain, question type, and a timestamp. Associated with each interaction are multiple entries in the `RESPONSES` table, where each response corresponds to a specific model (identified by `agent_id`), includes the model's output, a flag indicating whether it was generated by the aggregator, and a timestamp. Additionally, for every interaction, a detailed analysis is stored in the `ANALYSIS` table, linked via the `job_id`. This table records metrics such as the consensus score and serialized analysis data, including sentiment scores and similarity matrices.

Persistence plays a crucial role in ensuring that all user queries, model-generated responses, and consensus summaries are reliably stored within the system. By maintaining a complete historical record, the platform allows users to revisit past queries, reload previous results instantly, and continue sessions seamlessly without re-querying external APIs. This approach not only enhances user experience by providing fast and reliable access to historical data but also improves system robustness. Furthermore, by caching both raw model outputs and aggregated analyses locally, the system significantly reduces redundant API calls, lowering computational costs, minimizing external API usage, and reducing latency. Overall, persistence strengthens the system’s efficiency, scalability, and resilience, ensuring long-term sustainability and operational cost-effectiveness.

### System Component Overview

The overall system design is illustrated in **Figure 2**, showing the interaction between major components. Users interact with the system through the **Client Application**, which is built using Gradio and provides interfaces for query submission, user authentication, history management, and visualizations. The Client communicates with a **Load Balancer** that intelligently distributes incoming requests to one of several available **Backend Servers**. These backend servers are responsible for processing queries by interacting with various **LLM models** such as Qwen, Llama3, Mistral, and DeepHermes. Each server can independently communicate with the models, execute query processing tasks, perform response aggregation, and generate analytical insights. Once responses and analysis are produced, the backend servers update the central **SQLite Database**, ensuring that all user interactions, model outputs, and generated visualizations are persistently stored. This modular design, with clearly separated responsibilities across clients, servers, and models, enables the system to scale efficiently, maintain robustness, and allow seamless expansion or replacement of individual components without disrupting the overall workflow.

### Fault Tolerance and Agent Failover Mechanism

The system's resilience and self-healing capabilities are depicted in **Figure 3**, which illustrates the fault tolerance and agent failover mechanism. Each agent model periodically sends heartbeat signals—every 10 seconds—to a centralized Heartbeat Tracker. This component monitors agent health in real-time by updating a shared Health Status Store based on heartbeat timeouts and API error logs. If an agent fails to send a heartbeat or exhibits repeated API errors, it is marked as unhealthy or failed. The system then attempts recovery using exponential backoff, and if necessary, triggers an **Agent Reset** procedure. This reset mechanism allows the system to retain degraded agents and reintroduce them when they recover.

In cases where the failed agent is an aggregator—responsible for synthesizing responses across models—the system enters an **Aggregator Failover** state. It detects failure through invalid response validation and immediately selects a healthy backup model to assume the aggregator role. This ensures that the core functionality of consensus generation is not disrupted. For standard agents (e.g., Qwen, Mistral, DeepHermes), the system uses the same health signals to determine when to replace a model or assign a new one dynamically. The failover design guarantees that even if multiple agents crash or become unresponsive, the system continues to process user queries without interruption. By maintaining health checks and allowing automated role reassignment, this mechanism provides robust fault tolerance with minimal human intervention. 

## Design Decisions

Throughout the development of this system, we made a series of deliberate design decisions to ensure that the architecture would be robust, scalable, fault-tolerant, and user-friendly. Our choices reflect a strong emphasis on modularity, efficiency, and resilience. The key decisions and the reasoning behind them are outlined below:

### 1. Layered Modular Architecture

We chose to separate the system into distinct layers — Client Application, Load Balancer, Backend Servers, LLM Models, and Database, to make the system modular and maintainable. This allowed us to independently scale or upgrade each component without affecting the others, enabling future flexibility and smoother system evolution.

### 2. Gradio-Based Client Application

For the client interface, we intentionally built the application using Gradio because it offered a lightweight, web-based solution that was quick to develop yet highly interactive. Gradio allowed us to implement login functionality, query submission, history management, and visualization features seamlessly in a single consistent interface.

### 3. Smart Load Balancer with Fault Tolerance

To balance requests efficiently across backend servers, we designed a custom load balancer with both round-robin distribution and sticky routing based on job IDs. Recognizing the need for high availability, we also built in 2-fault tolerance so that if a server failed, the load balancer would automatically retry on alternate servers without interrupting the user experience.

### 4. Distributed Backend Servers

We architected the backend to consist of multiple independent FastAPI servers. Each server was designed to handle model interactions, response aggregation, and analysis generation. This distributed approach ensured that the system could scale horizontally as user load increased, and that failures in one server would not affect others.

### 5. Multi-Agent Query Processing

Rather than relying on a single model, we intentionally designed the system to collect responses from multiple LLM agents. By involving models like Llama, Qwen, Mistral, and DeepHermes for every query, we ensured diversity in reasoning. We believed this would lead to richer consensus building and greater overall robustness in the final outputs.

### 6. Dynamic Aggregator Failover

Understanding that the aggregator model (Llama3) is critical for consensus generation, we implemented a dynamic failover system. If the aggregator became unhealthy, we automatically selected a backup model. This decision was crucial to ensuring that system reliability did not hinge on a single point of failure.

### 7. Heartbeat-Based Health Monitoring

We introduced a heartbeat tracker where each agent sends a signal every 10 seconds. By monitoring heartbeats and API errors, we could detect unhealthy agents early and either reset them or select backups. This decision allowed the system to maintain operational integrity without requiring manual supervision.

### 8. Persistence and Caching Strategy

From early on, we prioritized persistence. We designed the system to cache queries, model responses, and analysis results locally in an SQLite database. This allowed users to revisit prior queries without needing to re-call external APIs, which greatly reduced latency, saved API costs, and enhanced user experience.

### 9. Visualization of Analysis Results

To make the analysis transparent and actionable, we decided to generate visualizations, including heatmaps, sentiment polarity charts, emotional tone distributions, and radar plots. These were integrated into the client to allow users not only to read responses but also to interpret underlying model behavior visually.

### 10. Emphasis on Fault Tolerance and Resilience

Our design focused on resilience. We consciously added retry mechanisms in the load balancer, heartbeat-based health monitoring, aggregator failover logic, and database caching. Each of these elements ensured that even under partial system failures, users would experience uninterrupted service.

### 11. API-Driven Stateless Communication

We structured communication between the client, load balancer, and backend servers using clean REST APIs. By keeping each transaction stateless, we made the system easier to scale horizontally and deploy across distributed infrastructures in the future if needed.

## How a Query Flows Through the System

1. **User** logs in and submits a **query** using the Gradio client.

2. **Client** sends the request to the **Load Balancer**.

3. **Load Balancer** forwards the request to an available **Backend Server**.

4. **Backend Server**:
   - Distributes the query to **multiple agent models** (Qwen, Mistral, DeepHermes, etc.).
   - Collects individual **agent responses** asynchronously.
   - Synthesizes a **consensus summary** using the selected **aggregator model** (e.g., Llama3).
   - Performs **response analysis**:
     - Embedding similarity matrix (heatmap)
     - Sentiment polarity (bar chart)
     - Emotional tone distribution (stacked chart)
     - Response feature comparison (radar plot)
   - Saves the interaction, responses, and analysis to the **SQLite database**.

5. **Client** fetches the processed results:
   - **Consensus summary** (aggregator output)
   - **Individual agent responses**
   - **Analysis visualizations** (heatmap, sentiment, emotion, radar)

6. **User** views the outputs interactively in the Gradio GUI.

7. **User** can:
   - **View** previous interaction history.
   - **Reload** any past job.
   - **Delete** specific jobs from their history.

## Day to Day Progress

#### March 29, 2025

We initiated development of the multi-agent consensus system for analyzing responses from various LLMs. The primary focus was on setting up the repository structure, defining the project’s scope, and installing any relevant dependencies.

##### Work Completed 

- Repository Setup
    - Created the initial GitHub repository and put source code we were going to use as base for our project (the source code was generated using Claude AI, which we would then update and edit to better fit our project objectives)

- Development Environment
    - Installed key dependencies and libraries.
    - Started to establish the framework for leader-follower interaction with the LLM models and model aggregation.

- At this point in our project, rather than evaluating consensus and bias in LLMs, we were planning on trying to create a similar application but for aggregating responses for solving math problems. The idea was that since lightweight LLM models are bad at solving math problems, if we aggregate and put together multiple model responses, we may be more likely to get to the right answer with a consensus mechanism. What we ended up finding was that the math models were still performing pretty poorly and so we needed to re-evaluate our project approach.

#### April 7, 2025

We pivoted to evaluating more generally consensus amongst language transformer models, since there are more models to choose from and it would be easier to test conceptually. We implemented the starter codebase, defining the modular structure that would host the LLM interaction logic, visualization, and UI components. We also focused on testing lightweight models again that we could access, use for our project, and had decent latency.

##### Work Completed

- Code Structure & Modules
    - Added configuration files and base Python modules.
    - Separated responsibilities between model handling, UI, and response processing.

- Environment Setup
    - Created reproducible environments using requirements.txt and conda setup.
    - Tested basic script execution to ensure compatibility.

#### April 14, 2025

We implemented the foundational architecture for the multi-agent consensus system. Key components like LLM interaction logic, client-server structure, and basic plotting utilities were added.

##### Work Completed

- Base Architecture
    - Implemented core logic for querying multiple LLMs and retrieving responses.
    - Created initial Gradio-based UI for client interaction.

- Visualization Features
    - Added early plotting utilities for visualizing outputs from multiple agents.

- Server Endpoints
    - Built server endpoints to handle incoming user queries and return model responses.

#### April 15, 2025

We focused on enhancing visualization and enabling users to select different aggregator models for consensus generation. Radar plots and emotional analysis charts were added.

##### Work Completed

- Advanced Visualizations
    - Implemented radar charts for feature comparison.
    - Added polarity and emotion analysis charts for response sentiment.

- Aggregator Logic
    - Enabled model selection as the aggregator responsible for consensus.
    - Improved UI readability with clearer formatting.


#### April 16, 2025

We added domain-specific expertise by allowing users to query different domains like Education, Policy, and Healthcare. We also improved visual rendering and query examples.

The following are some of the plots that we generated with out updated analysis code for the following query

<p align="center">
  <img src="imgs/prompt.png">
</p>

<p align="center">
  <img src="imgs/response_similarity.png">
</p>

<p align="center">
  <img src="imgs/emotional_tone.png">
</p>

<p align="center">
  <img src="imgs/sentiment_polarity.png">
</p>

<p align="center">
  <img src="imgs/response_feature.png">
</p>

##### Work Completed

- Domain Expert Mode
    - Integrated example queries and visual theming for domains: Education, Healthcare, Policy, Science, and Environmental.

- Formatting & Rendering
    - Enhanced response chart rendering.
    - Fixed image handling issues to display consistent visual feedback.


#### April 18, 2025

Persistence was implemented using SQLite to store user profiles, query history, and model responses. Challenges arose around concurrency and session management.

##### Work Completed

- SQLite Integration
    - Persisted user authentication data and interaction logs.
    - Implemented basic query history retrieval and storage.

- Challenges Tackled
    - Addressed database connection sharing across components.
    - Handled simultaneous query access and session stability.

#### April 19, 2025

LLM fault tolerance was added, with health checks, heartbeat pings, and automatic agent fallback mechanisms. This ensures robust behavior during agent downtimes.

##### Work Completed

- Heartbeat Monitoring
    - Health checks every 30 seconds for each agent.
    - Agent states categorized as healthy, unhealthy, or failed.

- Recovery System
    - Automatic reassignment of aggregator models if a primary model fails.
    - Retry mechanisms with exponential backoff.


#### April 21, 2025

We added secure user authentication and finalized documentation. Passwords are hashed securely, and sessions are managed to preserve query history.

This is what our Gradio interface looks like after incorporating the new account-related features:

<p align="center">
  <img src="imgs/gui_top.png">
</p>

<p align="center">
  <img src="imgs/gui_bottom.png">
</p>

##### Work Completed

- Authentication System
    - Implemented login/signup with SHA-256 password hashing.
    - Managed secure session tokens and logout functionality.

- Comprehensive Documentation
    - Added Google-style docstrings for all major functions.
    - Updated README.md with installation steps and usage instructions.

- Authentication Flow
    - Designed login/signup interface with history association.
    - Ensured secure logout and session reset procedures.