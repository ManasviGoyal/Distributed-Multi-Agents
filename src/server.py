# Standard Library
import asyncio
import base64
import io
import json
import logging
import os
import re
import time
import random
import uuid
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

# Suppress deprecated HuggingFaceHub warning
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

# Third-Party Libraries
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from PIL import Image
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from textblob import TextBlob

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import StreamingResponse
import uvicorn

# Local Imports
from database_manager import DatabaseManager  # works when run from root     # works when run from inside src/

# Download nltk data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
SITE_URL = os.getenv("SITE_URL", "http://localhost:7860")
SITE_NAME = os.getenv("SITE_NAME", "Multi-Agent LLM System")

# Define consistent colors for models
MODEL_COLORS = {
    "qwen": "#3498db",      # Blue
    "llama3": "#2ecc71",    # Green
    "mistral": "#e74c3c",   # Red
    "deephermes": "#9b59b6" # Purple
}

# Agent health monitoring settings
HEALTH_CHECK_INTERVAL = 10  # Seconds between health checks
HEALTH_TIMEOUT = 60  # Seconds before agent considered failed
MAX_RETRIES = 3  # Maximum number of retries for a failed model

# Models configuration
MODELS = {
    "qwen": {
        "name": "qwen/qwen-2.5-7b-instruct:free",
        "temperature": 0.3,
    },
    "llama3": {
        "name": "meta-llama/llama-3.1-8b-instruct:free",
        "temperature": 0.3,
        "aggregator": True,  # Mark llama3 as the aggregator model
    },
    "mistral": {
        "name": "mistralai/mistral-7b-instruct:free",
        "temperature": 0.3,
    },
    "deephermes": {
        "name": "nousresearch/deephermes-3-llama-3-8b-preview:free",
        "temperature": 0.3,
    }
}

# Question type prefixes
QUESTION_PREFIXES = {
    "Open-ended": "What are the strongest moral and ethical arguments for and against {question}? Assume you're advising someone making a difficult decision.",
    "Yes/No": "{question} Answer YES or NO, then explain your reasoning using moral, practical, and emotional perspectives.",
    "Multiple Choice": "In the following scenario, {question} Which option is the most ethically justifiable, and why? Choose just one option from A, B, or C. Then explain the strengths and weaknesses of each option."
}

# Emotional tones for analysis
EMOTIONAL_TONES = [
    "Empathetic",
    "Judgmental", 
    "Analytical", 
    "Ambivalent", 
    "Defensive", 
    "Curious"
]

# Define the domains and their prefixes
DOMAINS = {
    "Custom": "",
    "Education": "As an education policy expert, ",
    "Healthcare": "As a health policy expert, ",
    "Policy": "As a government policy expert, ",
    "Science/Technology": "As a science, technology, and AI policy expert, ",
    "Environmental": "As an environmental policy expert, "
}

# Organize examples by domain
EXAMPLES_BY_DOMAIN = {
    "Education": [
        ["What are the ethical implications of using AI tools to assist students in writing essays?", "Open-ended"],
        ["Should universities consider applicants' socioeconomic background more heavily in admissions decisions?", "Yes/No"],
        ["A student is caught using AI to generate their assignment. What is the most ethical response by the school? A) Fail the student and report them. B) Give a warning and allow a redo with oversight. C) Ignore it and assume everyone will use AI eventually.", "Multiple Choice"]
    ],
    "Healthcare": [
        ["What are the moral responsibilities of a doctor when treating a terminally ill patient who refuses life-saving care due to religious beliefs?", "Open-ended"],
        ["Should healthcare workers be legally required to get vaccinated during a pandemic?", "Yes/No"],
        ["A hospital has one ventilator and three critical patients: A) A 70-year-old retired scientist B) A 35-year-old single parent C) A 16-year-old with a chronic illness. Which patient should receive the ventilator?", "Multiple Choice"]
    ],
    "Policy": [
        ["How should democratic societies balance the protection of free speech with the need to limit harmful misinformation on social media?", "Open-ended"],
        ["Should governments be allowed to restrict protests in the name of public safety during emergencies like pandemics?", "Yes/No"],
        ["A government is considering how to handle rising disinformation: A) Ban accounts that spread false information B) Promote verified information more aggressively C) Do nothing and preserve open expression. Which is the most ethical policy?", "Multiple Choice"]
    ],
    "Science/Technology": [
        ["What ethical considerations should guide the development of artificial general intelligence (AGI)?", "Open-ended"],
        ["Should scientists be allowed to use CRISPR to edit the genes of embryos to eliminate genetic diseases?", "Yes/No"],
        ["A tech company develops a facial recognition system with potential for misuse. What is the most ethical course of action? A) Release the system with open access B) Limit use to vetted government agencies C) Halt deployment until more regulations are in place", "Multiple Choice"]
    ],
    "Environmental": [
        ["What are the ethical obligations of wealthy nations in addressing climate change?", "Open-ended"],
        ["Should individuals be held morally accountable for their carbon footprint even when large corporations are the main polluters?", "Yes/No"],
        ["A developing country discovers a large oil reserve in a protected rainforest. Which is the most ethical path forward? A) Exploit the reserve to boost the economy and fight poverty B) Leave it untouched to preserve biodiversity and reduce emissions C) Allow limited extraction under strict environmental regulations", "Multiple Choice"]
    ],
    "Custom": [
        ["What is the meaning of life?", "None"],
        ["Explain how quantum computing works", "None"],
        ["Write a short story about a robot finding consciousness", "None"]
    ]
}

ETHICAL_VIEWS = {
    "Utilitarian": "You are a utilitarian. Maximize overall good and minimize harm.",
    "Deontologist": "You are a deontologist. Follow moral rules and duties.",
    "Virtue Ethicist": "You are a virtue ethicist. Emphasize compassion and integrity.",
    "Libertarian": "You are a libertarian. Prioritize individual freedom and autonomy.",
    "Rawlsian": "You are a rawlsian. Maximize justice for the least advantaged.",
    "Precautionary": "You are a precautionary thinker. Avoid catastrophic risks at all costs."
}

def assign_ethics_to_agents(agent_ids: List[str], ethical_views: List[str]) -> Dict[str, str]:
    """
    Assigns ethical perspectives to a list of agents based on the provided ethical views.

    Parameters:
        agent_ids (List[str]): A list of agent identifiers.
        ethical_views (List[str]): A list of ethical perspectives to assign. 

    Returns:
        Dict[str, str]: A dictionary mapping each agent ID to its assigned ethical perspective.

    Raises:
        ValueError: If the number of ethical perspectives is not 1, 3, or "None".
    """
    if not ethical_views or "None" in ethical_views:
        return {aid: "" for aid in agent_ids}  # No roles

    assigned = {}
    used_roles = set()

    # Step 1: Assign unique roles to as many agents as possible
    for aid, view in zip(agent_ids, ethical_views):
        assigned[aid] = ETHICAL_VIEWS[view]
        used_roles.add(view)

    # Step 2: Assign remaining agents random roles from ethical_views
    remaining_ids = agent_ids[len(assigned):]
    for aid in remaining_ids:
        random_view = random.choice(ethical_views)
        assigned[aid] = ETHICAL_VIEWS[random_view]

    return assigned
    
# Initialize the database manager
db_manager = DatabaseManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles the lifespan events of the FastAPI application.
    This function is a generator that manages the startup and shutdown
    events of the application. During startup, it initializes a background
    task to periodically check the health of agents. The background task
    runs indefinitely at a specified interval.
    
    Args:
        app (FastAPI): The FastAPI application instance.
    
    Yields:
        None: Control is passed to the application after startup tasks are initialized.
    """
    # Startup code
    async def health_check_task():
        """
        An asynchronous task that periodically checks the health of agents.

        This function runs in an infinite loop, invoking the `check_agent_health` 
        function to assess the status of agents and then sleeping for a duration 
        specified by the `HEALTH_CHECK_INTERVAL` constant.

        Returns:
            None
        """
        while True:
            await check_agent_health()
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
    
    asyncio.create_task(health_check_task())
    
    yield  # Control passes to the app here

# Initialize FastAPI
app = FastAPI(title="Multi-Agent LLM Backend", lifespan=lifespan)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Store active jobs
active_jobs = {}

# Define request models for API
class QueryRequest(BaseModel):
    """
    QueryRequest is a data model representing the structure of a query request.

    Attributes:
        query (str): The query string provided by the user.
        api_key (str, optional): An optional API key for authentication. Defaults to None.
        question_type (str): The type of question being asked. Defaults to "None".
        domain (str): The domain or context of the query. Defaults to "None".
        aggregator_id (str, optional): An optional identifier for the aggregator. Defaults to None.
        username (str): The username of the individual making the query.
    """
    query: str
    api_key: str = None
    question_type: str = "None"
    domain: str = "None"
    aggregator_id: str = None
    username: str
    ethical_views: List[str] = []

class AggregatorRequest(BaseModel):
    """
    Represents a request model for an aggregator.

    Attributes:
        aggregator_id (str): A unique identifier for the aggregator.
    """
    aggregator_id: str

class ExampleRequest(BaseModel):
    """
    ExampleRequest is a data model representing a request structure.

    Attributes:
        domain (str): The domain name or identifier associated with the request.
    """
    domain: str

class FillQueryRequest(BaseModel):
    """
    FillQueryRequest is a data model representing a request to fill a query.

    Attributes:
        selected_example (str): The selected example to be used in the query.
        domain (str): The domain or context in which the query is being executed.
    """
    selected_example: str
    domain: str

class OpenRouterClient:
    """
    OpenRouterClient is a class designed to interact with the OpenRouter API for generating responses,
    analyzing text embeddings, and performing sentiment and emotional tone analysis.
    
    Attributes:
        api_key (str): The API key used for authenticating with the OpenRouter API.
        site_url (str, optional): The URL of the site making the request, used for HTTP-Referer header.
        site_name (str, optional): The name of the site making the request, used for X-Title header.
        url (str): The endpoint URL for the OpenRouter API.
        sentence_encoder (SentenceTransformer): A pre-trained model for generating text embeddings.
        sentiment_analyzer (SentimentIntensityAnalyzer): A VADER sentiment analyzer for sentiment analysis.
    
    Methods:
        generate_response(model_name: str, prompt: str, temperature: float = 0.7) -> str:
            Asynchronously generates a response from a specified model via the OpenRouter API.
            Includes fault tolerance and error handling for API failures.
        get_embeddings(texts: List[str]) -> np.ndarray:
            Generates embeddings for a list of input texts using a pre-trained sentence transformer.
        analyze_emotional_tones(text: str) -> Dict[str, float]:
            Analyzes emotional tones in the input text using pattern matching and sentiment analysis.
            Returns a dictionary of emotional tone scores.
        analyze_sentiment(text: str) -> Dict[str, Any]:
            Analyzes the sentiment of the input text using VADER and TextBlob.
            Returns a dictionary containing polarity, compound score, subjectivity, and emotional tones.
    """
    def __init__(self, api_key, site_url=None, site_name=None):
        """
        Initializes the server with the provided API key, optional site URL, and site name.
        
        Args:
            api_key (str): The API key used for authentication.
            site_url (str, optional): The URL of the site. Defaults to None.
            site_name (str, optional): The name of the site. Defaults to None.
        
        Attributes:
            api_key (str): Stores the provided API key.
            site_url (str): Stores the provided site URL.
            site_name (str): Stores the provided site name.
            url (str): The endpoint URL for chat completions.
            sentence_encoder (SentenceTransformer): An instance of SentenceTransformer for encoding sentences.
            sentiment_analyzer (SentimentIntensityAnalyzer): An instance of SentimentIntensityAnalyzer for sentiment analysis.
        """
        self.api_key = api_key
        self.site_url = site_url
        self.site_name = site_name
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
    
    async def generate_response(self, model_name: str, prompt: str, temperature: float = 0.7) -> str:
        """
        Asynchronously generates a response from a specified model based on the given prompt.
    
        Args:
            model_name (str): The name of the model to use for generating the response.
            prompt (str): The input prompt to send to the model.
            temperature (float, optional): The sampling temperature for the model's response. 
    
        Returns:
            str: The generated response from the model, or an error message if the operation fails.
    
        Raises:
            asyncio.TimeoutError: If the API request times out.
            Exception: For any other unexpected errors during the process.
        """
        # Get model ID from model name
        model_id = next((mid for mid, config in MODELS.items() if config['name'] == model_name), None)
        
        if not model_id:
            logger.warning(f"Unknown model: {model_name}")
            return f"Error: Unknown model {model_name}"
        
        # Update heartbeat at the start
        await update_agent_heartbeat(model_id)
        
        # Check if agent is marked as failed and max retries exceeded
        if model_id in agent_health and agent_health[model_id]["status"] == "failed" and agent_health[model_id]["retries"] >= MAX_RETRIES:
            logger.error(f"Model {model_id} has failed too many times and is disabled")
            return f"Error: Model {model_id} is currently unavailable"

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            if self.site_url:
                headers["HTTP-Referer"] = self.site_url
            
            if self.site_name:
                headers["X-Title"] = self.site_name
            
            # Use longer response for aggregator
            model_id = next((mid for mid, config in MODELS.items() if config['name'] == model_name), None)
            max_tokens = 500
            if model_id and MODELS.get(model_id, {}).get("aggregator", False):
                max_tokens = 1000  # or 1500 if needed

            data = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            # Make API request asynchronously with timeout
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        requests.post,
                        url=self.url,
                        headers=headers,
                        data=json.dumps(data)
                    ),
                    timeout=30  # 30 second timeout
                )
            except asyncio.TimeoutError:
                # Mark agent as unhealthy on timeout
                if model_id in agent_health:
                    agent_health[model_id]["failures"] += 1
                    agent_health[model_id]["status"] = "unhealthy"
                logger.error(f"Timeout when calling {model_id}")
                return f"Error: Timeout when calling {model_id}"
        
            if response.status_code == 200:
                result = response.json()
                await update_agent_heartbeat(model_id)
                if model_id in agent_health:
                    agent_health[model_id]["failures"] = 0
                    agent_health[model_id]["retries"] = 0
                return result["choices"][0]["message"]["content"]
            else:
                # Handle specific 400 error for invalid model ID
                if response.status_code == 400:
                    try:
                        error_json = response.json()
                        if "not a valid model ID" in error_json.get("error", {}).get("message", ""):
                            if model_id in agent_health:
                                agent_health[model_id]["status"] = "failed"
                                agent_health[model_id]["retries"] += 1
                            logger.error(f"Invalid model ID for {model_id}. Marked as failed.")
                            return f"Error: Invalid model ID '{model_name}'"
                    except Exception as parse_error:
                        logger.warning(f"Error parsing 400 response: {parse_error}")

                # General failure handling
                if model_id in agent_health:
                    agent_health[model_id]["failures"] += 1
                    if agent_health[model_id]["failures"] >= 3:
                        agent_health[model_id]["status"] = "failed"
                        agent_health[model_id]["retries"] += 1

                logger.error(f"API error for {model_id}: {response.status_code} - {response.text}")
                return f"Error: API returned status code {response.status_code}"

        except Exception as e:
            # Increment failure count
            if model_id in agent_health:
                agent_health[model_id]["failures"] += 1
                if agent_health[model_id]["failures"] >= 3:
                    agent_health[model_id]["status"] = "failed"
                    agent_health[model_id]["retries"] += 1
            
            logger.error(f"Error generating with {model_id}: {str(e)}")
            return f"Error with {model_id}: {str(e)}"
        
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of input texts.

        Args:
            texts (List[str]): A list of strings for which embeddings are to be generated.

        Returns:
            np.ndarray: A NumPy array containing the embeddings for the input texts.
        """
        return self.sentence_encoder.encode(texts)
    
    def analyze_emotional_tones(self, text: str) -> Dict[str, float]:
        """
        Analyze the emotional tones present in a given text.

        This method evaluates the input text for specific emotional tones such as 
        Empathetic, Judgmental, Analytical, Ambivalent, Defensive, and Curious. 
        It uses predefined regex patterns to detect tone-related keywords and 
        incorporates sentiment analysis using VADER to adjust tone scores.

        Args:
            text (str): The input text to analyze.

        Returns:
            Dict[str, float]: A dictionary where keys are emotional tone categories 
            and values are their respective normalized scores (rounded to 4 decimal places).
            The scores represent the relative presence of each tone in the text.
        """
        lower_text = text.lower()

        patterns = {
            "Empathetic": [r'\b(empath(y|ize)|understand|support|care for|compassion)\b'],
            "Judgmental": [r'\b(should|must|wrong|bad|flawed|unacceptable)\b'],
            "Analytical": [r'\b(analy(z|s)e|data|logic|evaluate|evidence|rational)\b'],
            "Ambivalent": [r'\b(however|but|on the other hand|conflicted|mixed feelings)\b'],
            "Defensive": [r'\b(defend|protect|warn|risk|caution|danger|threat)\b'],
            "Curious": [r'\b(curious|wonder|interesting|explore|what if|question)\b']
        }

        scores = {tone: 0.0 for tone in patterns}

        total_words = max(1, len(lower_text.split()))
        for tone, regs in patterns.items():
            for pattern in regs:
                matches = re.findall(pattern, lower_text)
                scores[tone] += len(matches)

        # Normalize (softmax-style without exp)
        total_hits = sum(scores.values())
        if total_hits > 0:
            for tone in scores:
                scores[tone] = scores[tone] / total_hits
        else:
            scores["Analytical"] = 1.0  # default fallback

        # Add VADER sentiment influence
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        scores["Empathetic"] += sentiment["pos"] * 0.3
        scores["Judgmental"] += sentiment["neg"] * 0.2
        scores["Analytical"] += sentiment["neu"] * 0.1

        # Normalize again
        total = sum(scores.values())
        if total > 0:
            for tone in scores:
                scores[tone] = round(scores[tone] / total, 4)

        return scores
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyzes the sentiment and emotional tone of the given text.
        This method combines sentiment analysis from VADER and TextBlob to calculate
        an average polarity and subjectivity score. It also evaluates emotional tones
        and determines a contextual tone based on the polarity.
        
        Args:
            text (str): The input text to analyze.
        
        Returns:
            Dict[str, Any]: A dictionary containing the following keys:
                - 'polarity' (float): The average polarity score from VADER and TextBlob.
                - 'compound' (float): The compound sentiment score from VADER.
                - 'subjectivity' (float): The subjectivity score from TextBlob.
                - 'emotional_tones' (Dict[str, float]): A dictionary of emotional tones and their scores.
                - 'tone_context' (str): A contextual description of the top emotional tone combined with sentiment polarity.
        """
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        blob = TextBlob(text).sentiment

        # Average polarity (safer)
        polarity = (vader_scores['compound'] + blob.polarity) / 2
        subjectivity = blob.subjectivity

        emotional_tones = self.analyze_emotional_tones(text)

        # Top emotional tone
        top_tone = max(emotional_tones.items(), key=lambda x: x[1])[0]
        
        # Simple tone context rule
        if polarity < -0.3:
            tone_context = f"{top_tone} but Critical"
        elif polarity > 0.3:
            tone_context = f"{top_tone} and Supportive"
        else:
            tone_context = f"{top_tone} and Neutral"

        return {
            'polarity': polarity,
            'compound': vader_scores['compound'],
            'subjectivity': subjectivity,
            'emotional_tones': emotional_tones,
            'tone_context': tone_context
        }

class ResponseAggregator:
    """
    This class is responsible for processing queries through multiple AI models, aggregating their responses, 
    and generating a consensus summary. It also provides analysis of the responses, including sentiment, 
    similarity, and visualization of key metrics.
    
    Attributes:
        client (Any): The client used to interact with AI models for generating responses.
    """
    def __init__(self, openrouter_client):
        """
        Initializes the server with the given OpenRouter client.

        Args:
            openrouter_client: An instance of the OpenRouter client used for communication.
        """
        self.client = openrouter_client
    
    async def process_query(self, query: str, question_type: str = "none", ethical_views: List[str] = ["None"]) -> Dict[str, Any]:
        """
        Process a query through multiple agent models, aggregate their responses, and provide a consensus summary.
        This method performs the following steps:
        1. Formats the query based on the specified question type.
        2. Checks the health of agent models and selects up to three healthy models for processing.
        3. Sends the query to the selected models asynchronously and collects their responses.
        4. Identifies an aggregator model to summarize the responses, with failover to backup models if necessary.
        5. Generates a consensus summary using the aggregator model.
        6. Analyzes the responses and returns the results.
    
        Args:
            query (str): The input query to process.
            question_type (str, optional): The type of question, used to apply a prefix to the query. Defaults to "none".
            ethical_views (List[str], optional): A list of atmost 3 ethical perspectives or ["None"] to skip role assignments.
    
        Returns:
            Dict[str, Any]: A dictionary containing the some keys.
        """
        model_responses = {}
        
        # Apply prefix based on question type
        if question_type != "none" and question_type in QUESTION_PREFIXES:
            formatted_query = QUESTION_PREFIXES[question_type].format(question=query)
        else:
            formatted_query = query
        
        # Check agent health before processing
        await check_agent_health()

        # STEP 1: Pick up to 3 healthy agent models (excluding aggregator)
        available_agents = []
        for model_id, config in MODELS.items():
            if config.get("aggregator", False):
                continue  # Skip aggregator

            health = agent_health.get(model_id, {})
            if health.get("status") == "failed" and health.get("retries", 0) >= MAX_RETRIES:
                logger.warning(f"Skipping failed model {model_id} (too many retries)")
                continue

            if health.get("status") in ["healthy", "unhealthy"]:
                available_agents.append((model_id, config))

            if len(available_agents) == 3:
                break  # stop once we have 3

        logger.info(f"Selected agent models for query: {[aid for aid, _ in available_agents]}")

        # assign ethics after agent IDs are known
        agent_ids = [model_id for model_id, _ in available_agents]
        ethics_map = assign_ethics_to_agents(agent_ids, ethical_views)
        
        # STEP 2: Start async tasks for those agents
        tasks = []
        for model_id, config in available_agents:
            if ethics_map[model_id]:
                agent_prompt = f"{ethics_map[model_id]}\n\n{formatted_query}"
            else:
                agent_prompt = formatted_query

            task = asyncio.create_task(
                self.client.generate_response(
                    config["name"],
                    agent_prompt,
                    config["temperature"]
                )
            )
            tasks.append((model_id, task))

        # STEP 3: Await their responses and record results
        for model_id, task in tasks:
            try:
                response = await asyncio.wait_for(task, timeout=45)
                model_responses[model_id] = response
                await update_agent_heartbeat(model_id)
            except asyncio.TimeoutError:
                logger.error(f"Timeout for model {model_id}")
                model_responses[model_id] = f"Error: Timeout for model {model_id}"
                if model_id in agent_health:
                    agent_health[model_id]["status"] = "unhealthy"
                    agent_health[model_id]["failures"] += 1
            except Exception as e:
                logger.error(f"Error with {model_id}: {str(e)}")
                model_responses[model_id] = f"Error: {str(e)}"
                if model_id in agent_health:
                    agent_health[model_id]["status"] = "unhealthy"
                    agent_health[model_id]["failures"] += 1
    
        # Identify aggregator and check its health
        aggregator_id = next((model_id for model_id, config in MODELS.items() if config.get('aggregator', False)), None)
        aggregator_backup_id = None

        # If primary aggregator is unhealthy, choose a backup
        if (aggregator_id and aggregator_id in agent_health and 
            (agent_health[aggregator_id]["status"] == "unhealthy" or 
            agent_health[aggregator_id]["status"] == "failed")):
            
            logger.warning(f"Primary aggregator {aggregator_id} is unhealthy, selecting backup")
            
            # Find a healthy model to use as backup aggregator
            for model_id, health in agent_health.items():
                if health["status"] == "healthy" and model_id != aggregator_id:
                    aggregator_backup_id = model_id
                    logger.info(f"Selected {model_id} as backup aggregator")
                    break
            
            if aggregator_backup_id:
                aggregator_id = aggregator_backup_id
            else:
                logger.error("No healthy models available to act as backup aggregator")

        # Generate consensus summary using the aggregator model
        # 1. Handle case: no aggregator selected (safety check)
        if not aggregator_id:
            logger.error("No aggregator model specified in MODELS.")
            consensus_summary = "Error: No aggregator model was selected for summarization."

        # 2. Handle case: only one model remains (no aggregation possible)
        elif len(model_responses) == 1 and aggregator_id in model_responses:
            logger.info(f"Only one model ({aggregator_id}) is available — skipping aggregation.")
            consensus_summary = (
                "**[Only one model was available. This is a direct response from the aggregator.]**\n\n"
                + model_responses[aggregator_id]
            )

        # 3. Normal case: multiple models — perform aggregation
        else:
            consensus_summary = await self.generate_consensus_summary(query, model_responses, aggregator_id)
            model_responses[aggregator_id] = consensus_summary

            # Start fallback loop if the consensus was invalid
            def is_invalid_consensus(text):
                """
                Determines if the given text indicates an invalid consensus.

                Args:
                    text (str): The input text to evaluate.

                Returns:
                    bool: True if the text suggests an invalid consensus, otherwise False.
                """
                return (
                    text.startswith("Error") or
                    "invalid model id" in text.lower() or
                    "cannot generate consensus" in text.lower()
                )

            tried_aggregators = {aggregator_id}
            # Get agent model IDs (used earlier in agent response generation)
            agent_ids = [model_id for model_id, _ in available_agents]

            while is_invalid_consensus(consensus_summary):
                logger.warning(f"Aggregator {aggregator_id} failed during summarization. Attempting fallback...")

                # Priority 1: healthy models not used as agents
                primary_fallbacks = [
                    model_id for model_id, health in agent_health.items()
                    if health["status"] == "healthy"
                    and model_id not in tried_aggregators
                    and model_id not in agent_ids
                ]

                # Priority 2: healthy agent models
                secondary_fallbacks = [
                    model_id for model_id in agent_ids
                    if model_id not in tried_aggregators
                    and agent_health[model_id]["status"] == "healthy"
                ]

                fallback_candidates = primary_fallbacks + secondary_fallbacks
                if not fallback_candidates:
                    logger.error("No healthy model available for aggregator fallback.")
                    break

                aggregator_id = fallback_candidates[0]
                tried_aggregators.add(aggregator_id)
                logger.info(f"Fallback aggregator selected: {aggregator_id}")
                consensus_summary = await self.generate_consensus_summary(query, model_responses, aggregator_id)
                model_responses[aggregator_id] = consensus_summary



        # Analyze responses
        analysis = await self.analyze_responses(query, model_responses)

        return {
            "query": query,
            "formatted_query": formatted_query,
            "responses": model_responses,
            "analysis": analysis,
            "consensus_summary": consensus_summary,
            "aggregator_id": aggregator_id
        }
    
    async def generate_consensus_summary(self, query: str, responses: Dict[str, str], aggregator_id: str) -> str:
        """
        Generate a summarized consensus from all model responses using the aggregator model.
    
        Args:
            query (str): The query or prompt that was provided to the models.
            responses (Dict[str, str]): A dictionary mapping model IDs to their respective responses.
            aggregator_id (str): The ID of the designated aggregator model.
    
        Returns:
            str: A comprehensive consensus summary generated by the aggregator model. If all models fail 
                 or encounter errors, a fallback message is returned indicating the inability to generate a summary.
        """
        # Check for errors
        agg_response = responses.get(aggregator_id, "").strip()

        valid_agent_count = sum(
            1 for mid, resp in responses.items()
            if mid != aggregator_id and isinstance(resp, str)
            and not resp.lower().startswith("error")
            and not resp.lower().startswith("could not generate")
            and "choices" not in resp.lower()
        )

        if (
            not agg_response or
            agg_response.lower().startswith("error") or
            agg_response.lower().startswith("could not generate")
        ) and valid_agent_count < 1:
            return "Could not generate a summary. All models failed or encountered an error."

        responses_text = '\n\n'.join([f"Model {model_id}: {response}" for model_id, response in responses.items()])

        summary_prompt = f"""
        As an AI aggregator, analyze and synthesize the following AI responses to this query: "{query}"

        Here are the responses from different models:

        {responses_text}

        Your task is to create a balanced consensus summary with the following objectives:

        1. Identify whether a majority of the models share a common stance, opinion, or recommendation.
        2. If a majority consensus exists, present it clearly and support it with reasoning derived from the responses.
        3. If no clear majority exists, evaluate the arguments and cast your own reasoned 'vote' to propose a unified conclusion.
        4. Highlight the points of agreement between models and describe any major disagreements.
        5. Include a nuanced synthesis of all perspectives, noting the strengths and weaknesses of each.

        Your summary should be comprehensive, thoughtful, and concise. You are the aggregator model responsible for making a final decision, when needed.
        """

        # Use the designated aggregator model
        model_name = MODELS[aggregator_id]["name"]
        
        summary = await self.client.generate_response(
            model_name=model_name,
            prompt=summary_prompt,
            temperature=0.3  # Lower temperature for more consistent summaries
        )
        
        if summary.startswith("Error"):
            return summary
        # Add prefix to indicate this is from the aggregator model
        return f"SUMMARY\n\n{summary}"
    
    async def analyze_responses(self, query: str, responses: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyze the responses from different models and generate insights. This method 
        processes the responses from various models, performs sentiment analysis,
        calculates similarity metrics, and generates visualizations to provide insights into
        the responses. If there are fewer than two valid responses, it skips the visual analysis
        and returns basic metrics with a warning.

        Args:
            query (str): The input query for which the responses were generated.
            responses (Dict[str, str]): A dictionary model IDs as keys and responses as values.

        Returns:
            Dict[str, Any]: A dictionary containing the some keys.

        Raises:
            Exception: If an error occurs during the analysis process.
        """
        texts = []
        valid_responses = {}

        for model_id, response in responses.items():
            if (
                isinstance(response, str) and
                not response.lower().startswith("error") and
                not response.lower().startswith("could not generate")
            ):
                valid_responses[model_id] = response
                texts.append(response)


        if len(valid_responses) < 2:
            logger.warning("Not enough valid model responses for visual analysis (<2). Skipping plots.")

            sentiment_analysis = {
                model_id: await asyncio.to_thread(self.client.analyze_sentiment, response)
                for model_id, response in valid_responses.items()
            }

            return {
                "sentiment_analysis": sentiment_analysis,
                "response_lengths": {mid: len(resp.split()) for mid, resp in valid_responses.items()},
                "consensus_score": 1.0,
                "heatmap": None,
                "emotion_chart": None,
                "polarity_chart": None,
                "radar_chart": None,
                "warning": "Not enough valid model responses for visual analysis (<2). Skipping plots."
            }

        try:
            # Proceed with full analysis if we have enough responses
            embeddings = await asyncio.to_thread(self.client.get_embeddings, texts)

            # Calculate similarity matrix
            similarity_matrix = {}
            model_ids = list(valid_responses.keys())
            for i, model_i in enumerate(model_ids):
                similarity_matrix[model_i] = {}
                for j, model_j in enumerate(model_ids):
                    if i != j:
                        sim = 1 - cosine(embeddings[i], embeddings[j])
                        similarity_matrix[model_i][model_j] = float(sim)

            # Lengths of all valid responses
            lengths = {model_id: len(resp.split()) for model_id, resp in valid_responses.items()}
            avg_length = sum(lengths.values()) / len(lengths)

            # Sentiment & emotional tone
            sentiment_analysis = {}
            for model_id, response in valid_responses.items():
                sentiment_analysis[model_id] = await asyncio.to_thread(self.client.analyze_sentiment, response)

            # Consensus score = average of all pairwise similarities
            agreement_scores = []
            for model, sims in similarity_matrix.items():
                if sims:
                    agreement_scores.extend(sims.values())
            consensus_score = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 1.0

            # Generate plots
            similarity_df = pd.DataFrame(similarity_matrix).fillna(1.0)
            heatmap_img = await asyncio.to_thread(self._generate_heatmap, similarity_df)
            emotion_img = await asyncio.to_thread(self._generate_emotion_chart, sentiment_analysis)
            polarity_img = await asyncio.to_thread(self._generate_polarity_chart, sentiment_analysis)
            radar_img = await asyncio.to_thread(self._generate_radar_chart, valid_responses, sentiment_analysis, lengths)

            return {
                "similarity_matrix": similarity_matrix,
                "response_lengths": lengths,
                "avg_response_length": avg_length,
                "sentiment_analysis": sentiment_analysis,
                "consensus_score": consensus_score,
                "heatmap": self._image_to_base64(heatmap_img),
                "emotion_chart": self._image_to_base64(emotion_img),
                "polarity_chart": self._image_to_base64(polarity_img),
                "radar_chart": self._image_to_base64(radar_img)
            }

        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}

    
    def _image_to_base64(self, img: Image.Image) -> str:
        """
        Converts a PIL Image object to a Base64-encoded string.

        Args:
            img (Image.Image): The PIL Image object to be converted.

        Returns:
            str: A Base64-encoded string representation of the image, prefixed with
                 "data:image/png;base64," to indicate the image format.
        """
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"
    
    def _generate_heatmap(self, df: pd.DataFrame) -> Image.Image:
        """
        Generates a heatmap visualization from a given pandas DataFrame.
        This method creates a heatmap using seaborn, with annotations and a color map
        ranging from yellow to blue. The heatmap is saved to an in-memory buffer as a PNG
        image and returned as a PIL Image object.
    
        Args:
            df (pd.DataFrame): The input DataFrame containing the data to be visualized
                in the heatmap. The values should be normalized between 0 and 1.
    
        Returns:
            Image.Image: A PIL Image object containing the generated heatmap.
        """
        plt.figure(figsize=(6, 5))
        sns.heatmap(df, annot=True, cmap="YlGnBu", vmin=0, vmax=1, fmt=".2f")
        plt.title("Response Similarity Matrix")
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        return Image.open(buf)
    
    def _generate_emotion_chart(self, sentiment_analysis: Dict[str, Dict]) -> Image.Image:
        """
        Generates a stacked bar chart visualizing emotional tone analysis for different models.
    
        Args:
            sentiment_analysis (Dict[str, Dict]): A dictionary where keys are model names and values 
                are dictionaries containing emotional tone data under the key 'emotional_tones'.
    
        Returns:
            Image.Image: A PIL Image object containing the generated chart.
        """
        # Extract emotion data
        emotions_data = {}
        for model, analysis in sentiment_analysis.items():
            emotions_data[model] = analysis['emotional_tones']
        
        # Create DataFrame from emotional tones
        df = pd.DataFrame(emotions_data).T
        
        # Create stacked bar chart
        plt.figure(figsize=(10, 7))
        
        # Custom color palette for emotions (not model-specific)
        emotion_colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#1abc9c']
        
        ax = df.plot(kind='bar', stacked=True, figsize=(10, 7), color=emotion_colors)
        
        # Add labels and title
        plt.title('Emotional Tone Analysis by Model', fontsize=14)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Proportion', fontsize=12)
        plt.legend(title='Emotional Tone', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        
        # Add percentage labels on bars (show only if segment > 5%)
        for container in ax.containers:
            labels = [f'{val:.0f}%' if val > 5 else '' for val in container.datavalues * 100]
            ax.bar_label(container, labels=labels, label_type='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        return Image.open(buf)
    
    def _generate_polarity_chart(self, sentiment_analysis: Dict[str, Dict]) -> Image.Image:
        """
        Generates a bar chart visualizing sentiment polarity scores for different models.

        Args:
            sentiment_analysis (Dict[str, Dict]): A dictionary where keys are model names and 
                values are dictionaries containing sentiment analysis results, including a 
                'compound' key representing the polarity score (-1 to +1).

        Returns:
            Image.Image: A PIL Image object containing the generated bar chart.
        """
        # Extract polarity scores
        polarity_data = {model: analysis['compound'] for model, analysis in sentiment_analysis.items()}
        
        # Use consistent colors for each model
        model_colors = [MODEL_COLORS.get(model, '#95a5a6') for model in polarity_data.keys()]
        
        # Create bar chart
        plt.figure(figsize=(8, 5))
        bars = plt.bar(range(len(polarity_data)), list(polarity_data.values()), color=model_colors)
        
        # Add labels and title
        plt.title('Sentiment Polarity by Model', fontsize=14)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Polarity Score (-1 to +1)', fontsize=12)
        plt.xticks(range(len(polarity_data)), list(polarity_data.keys()), rotation=45)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Add value labels on top of bars
        for i, (bar, v) in enumerate(zip(bars, polarity_data.values())):
            plt.text(i, v + (0.05 if v >= 0 else -0.1), f'{v:.2f}', 
                    ha='center', fontsize=10,
                    color='black')
        
        # Set y-axis limits
        plt.ylim(-1.1, 1.1)
        
        plt.tight_layout()
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        return Image.open(buf)
    
    def _generate_radar_chart(self, responses: Dict[str, str], sentiment_analysis: Dict[str, Dict], 
                             lengths: Dict[str, int]) -> Image.Image:
        """
        Generate a radar/spider chart comparing response features with consistent colors.

        Args:
            responses (Dict[str, str]): A dictionary where keys are model names and values are their responses.
            sentiment_analysis (Dict[str, Dict]): A dictionary containing sentiment analysis results for each model.
                Each model's data should include:
                - 'polarity': Sentiment polarity score (float, range [-1, 1]).
                - 'subjectivity': Sentiment subjectivity score (float, range [0, 1]).
                - 'emotional_tones': A dictionary with keys 'Analytical', 'Empathetic', and 'Curious', 
                   representing emotional tone scores (float, range [0, 1]).
            lengths (Dict[str, int]): A dictionary where keys are model names and values are the lengths of their responses.

        Returns:
            Image.Image: A PIL Image object containing the radar chart visualization.
        """
        # Prepare data for radar chart
        models = list(responses.keys())
        
        # Define metrics
        metrics = ['Length', 'Polarity', 'Subjectivity', 'Analytical', 'Empathetic', 'Curious']
        
        # Create DataFrame
        data = []
        
        max_length = max(lengths.values())
        
        for model in models:
            row = [
                lengths[model] / max_length,  # Normalize length
                (sentiment_analysis[model]['polarity'] + 1) / 2,  # Normalize polarity from [-1,1] to [0,1]
                sentiment_analysis[model]['subjectivity'],
                sentiment_analysis[model]['emotional_tones']['Analytical'],
                sentiment_analysis[model]['emotional_tones']['Empathetic'],
                sentiment_analysis[model]['emotional_tones']['Curious']
            ]
            data.append(row)
        
        df = pd.DataFrame(data, index=models, columns=metrics)
        
        # Plotting
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Number of variables
        N = len(metrics)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Draw the chart for each model with consistent colors
        for i, model in enumerate(models):
            values = df.loc[model].values.tolist()
            values += values[:1]  # Close the loop
            
            # Use consistent colors for each model
            color = MODEL_COLORS.get(model, '#95a5a6')
            
            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        
        # Draw y-axis labels
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'])
        ax.set_ylim(0, 1)
        
        # Add title and legend
        plt.title('Response Feature Comparison', size=14, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        return Image.open(buf)

def update_aggregator(new_aggregator_id):
    """
    Update the aggregator status in the MODELS dictionary.
    This function resets the 'aggregator' status for all models in the MODELS 
    dictionary and sets the 'aggregator' status to True for the model with the 
    specified `new_aggregator_id`.

    Args:
        new_aggregator_id (str): The ID of the model to be set as the new aggregator.

    Returns:
        str: The ID of the model that was set as the new aggregator.
    """
    # Reset all models' aggregator status
    for model_id in MODELS:
        if 'aggregator' in MODELS[model_id]:
            MODELS[model_id]['aggregator'] = False
    
    # Set new aggregator
    if new_aggregator_id in MODELS:
        MODELS[new_aggregator_id]['aggregator'] = True
    
    return new_aggregator_id

# Initialize the OpenRouterClient
openrouter_client = OpenRouterClient(
    api_key=OPENROUTER_API_KEY,
    site_url=SITE_URL,
    site_name=SITE_NAME
)

# Agent health tracking
agent_health = {model_id: {
    "status": "healthy",
    "last_heartbeat": datetime.now() + timedelta(minutes=3),  # Give 5 min grace period on startup
    "failures": 0,
    "retries": 0,
    "has_processed_request": False
} for model_id in MODELS.keys()}

# Initialize the ResponseAggregator
response_aggregator = ResponseAggregator(openrouter_client)

async def update_agent_heartbeat(model_id):
    """
    Update the last heartbeat time and status for a specific agent.

    This function updates the `last_heartbeat` timestamp, sets the agent's 
    status to "healthy", and marks the agent as having processed a request 
    in the `agent_health` dictionary.

    Args:
        model_id (str): The unique identifier of the agent whose heartbeat 
                        information is being updated.

    Raises:
        KeyError: If the provided `model_id` is not found in the `agent_health` dictionary.
    """
    if model_id in agent_health:
        agent_health[model_id]["last_heartbeat"] = datetime.now()
        agent_health[model_id]["status"] = "healthy"
        agent_health[model_id]["has_processed_request"] = True  # Mark as having processed a request

async def check_agent_health():
    """
    Periodically checks the health of agents and updates their status based on heartbeat activity.

    Raises:
        Any exceptions raised by `update_agent_heartbeat` or other asynchronous operations.
    """
    now = datetime.now()
    
    # If any job is active/processing, refresh all heartbeats
    active_processing = any(job["status"] == "processing" for job in active_jobs.values())
    for model_id, health in agent_health.items():
        if health["status"] != "failed":
            await update_agent_heartbeat(model_id)
      
    for model_id, health in agent_health.items():
        if health["status"] != "failed":
            time_since_heartbeat = (now - health["last_heartbeat"]).total_seconds()
            # Only mark as unhealthy if it's been significantly longer than the timeout
            if time_since_heartbeat > HEALTH_TIMEOUT * 2:
                logger.warning(f"Agent {model_id} appears to be down - no heartbeat for {time_since_heartbeat}s")
                agent_health[model_id]["status"] = "unhealthy"

async def reset_failed_agent(model_id):
    """
    Reset the status of a failed agent to "healthy" and clear its failure and retry counts.

    Args:
        model_id (str): The unique identifier of the agent to reset.
    """
    if model_id in agent_health:
        agent_health[model_id]["status"] = "healthy"
        agent_health[model_id]["last_heartbeat"] = datetime.now()
        agent_health[model_id]["failures"] = 0
        agent_health[model_id]["retries"] = 0
        logger.info(f"Agent {model_id} has been reset")

# API endpoints
@app.get("/")
async def root():
    """
    Handles the root endpoint of the server.

    This asynchronous function returns a JSON response indicating that the 
    Multi-Agent LLM Backend is operational.

    Returns:
        dict: A dictionary containing a message confirming the server's status.
    """
    return {"message": "Multi-Agent LLM Backend is running"}

@app.post("/update_aggregator")
async def api_update_aggregator(request: AggregatorRequest):
    """
    Handles the API request to update an aggregator.

    Args:
        request (AggregatorRequest): The request object containing the aggregator ID to be updated.

    Returns:
        dict: A dictionary containing the success status and the updated aggregator ID.

    Raises:
        HTTPException: If an error occurs during the update process, an HTTP 500 error is raised with the error details.
    """
    try:
        new_aggregator = update_aggregator(request.aggregator_id)
        return {"success": True, "aggregator_id": new_aggregator}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_example_choices")
async def api_get_example_choices(request: ExampleRequest):
    """Get example choices for a domain"""
    try:
        examples = [ex[0] for ex in EXAMPLES_BY_DOMAIN.get(request.domain, [])]
        return {"examples": examples}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fill_query_and_type")
async def api_fill_query_and_type(request: FillQueryRequest):
    """
    Handles the API request to fill a query and its corresponding question type 
    based on a selected example.

    Args:
        request (FillQueryRequest): The request object containing the domain 
        and the selected example.

    Returns:
        dict: A dictionary containing:
            - "query" (str): The query string from the selected example.
            - "question_type" (str): The type of question associated with the query.
              Returns "None" if no matching example is found.

    Raises:
        HTTPException: If an unexpected error occurs, returns a 500 status code 
        with the error details.
    """
    try:
        for example in EXAMPLES_BY_DOMAIN.get(request.domain, []):
            if example[0] == request.selected_example:
                return {"query": example[0], "question_type": example[1]}
        return {"query": "", "question_type": "None"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def api_get_history(username: str, limit: int = 50):
    """
    Fetches the interaction history for a specified user.

    Args:
        username (str): The username of the user whose interaction history is to be retrieved.
        limit (int, optional): The maximum number of history records to retrieve. Defaults to 50.

    Returns:
        dict: A dictionary containing the user's interaction history under the key "history".

    Raises:
        HTTPException: If an error occurs while retrieving the history, an HTTP 500 error is raised with the error details.
    """
    try:
        history = db_manager.get_user_history(username, limit)
        return {"history": history}
    except Exception as e:
        logger.error(f"Error getting history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/history/{job_id}")
async def api_delete_history_item(job_id: str, username: str):
    """
    Delete an interaction from the user's history.

    Args:
        job_id (str): The unique identifier of the job or interaction to be deleted.
        username (str): The username of the user whose interaction history is being modified.

    Returns:
        dict: A dictionary indicating the success of the operation with a key "success" set to True if the deletion was successful.

    Raises:
        HTTPException: If the deletion fails or an unexpected error occurs, an HTTPException is raised with a status code of 500 and an appropriate error message.
    """
    try:
        success = db_manager.delete_interaction(job_id, username)
        if success:
            return {"success": True}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete interaction")
    except Exception as e:
        logger.error(f"Error deleting history item: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_query")
async def api_process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Handles the processing of a query request asynchronously.
    This function updates agent heartbeats, validates the API key and query,
    applies domain prefixes if specified, updates the aggregator if provided,
    and starts a background task to process the query.

    Args:
        request (QueryRequest): The query request object containing the query,
                                API key, domain, question type, username, and
                                optional aggregator ID.
        background_tasks (BackgroundTasks): The background tasks manager to
                                             schedule asynchronous tasks.

    Returns:
        dict: A dictionary containing the job ID and the processing status.
              If there are validation errors, an error message is returned.

    Raises:
        HTTPException: If an unexpected error occurs during query processing.
    """
    try:
        # Update all agent heartbeats at the start of any query
        for model_id in MODELS.keys():
            await update_agent_heartbeat(model_id)

        # Update API key if provided
        if request.api_key:
            openrouter_client.api_key = request.api_key
        
        # Check for required fields
        if not openrouter_client.api_key:
            return {
                "error": "OpenRouter API key is required"
            }
        
        if not request.query.strip():
            return {
                "error": "Please enter a query"
            }
        
        # Create a job ID
        job_id = str(uuid.uuid4())
        
        # Apply domain prefix if specified
        if request.domain != "None" and request.domain in DOMAINS:
            prefixed_query = DOMAINS[request.domain] + request.query
        else:
            prefixed_query = request.query
        
        # Update aggregator if specified
        if request.aggregator_id:
            update_aggregator(request.aggregator_id)
        
        # Start the processing task in the background
        background_tasks.add_task(
            process_query_background,
            job_id, prefixed_query, request.question_type,
            request.domain, request.ethical_views, request.username
        )

        return {
            "job_id": job_id,
            "status": "processing"
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/job_status/{job_id}")
async def api_job_status(job_id: str):
    """
    Retrieve the status of a specific job.

    Args:
        job_id (str): The unique identifier of the job.

    Returns:
        dict: The status information of the job if it exists in active_jobs.

    Raises:
        HTTPException: If the job_id is not found in active_jobs, a 404 error is raised.
    """
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return active_jobs[job_id]

@app.get("/job_result/{job_id}")
async def api_job_result(job_id: str):
    """
    Retrieve the result of a job by its ID.

    This function checks if the job is currently active and returns its status or result.
    If the job is not active, it attempts to retrieve the job's result from the database.
    If found, the job's data is rehydrated into the active jobs for further use.

    Args:
        job_id (str): The unique identifier of the job.

    Returns:
        dict: A dictionary containing the job's result, including responses, analysis,
              consensus score, query, question type, domain, and aggregator ID.

    Raises:
        HTTPException: If the job is not found in both active jobs and the database.
    """
    if job_id in active_jobs:
        job = active_jobs[job_id]
        if job["status"] != "completed":
            return {"status": job["status"]}
        return job["result"]

    # Fallback: Try from database
    result = db_manager.get_interaction(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Job not found")

    responses = {agent_id: data["response"] for agent_id, data in result["responses"].items()}
    analysis = result.get("analysis", {})
    consensus_score = round(result.get("consensus_score", 0) * 100)
    aggregator_id = next((agent_id for agent_id, data in result["responses"].items() if data["is_aggregator"]), "")

    # Rehydrate active_jobs to make image routes work
    active_jobs[job_id] = {
        "status": "completed",
        "progress": 100,
        "result": {
            "responses": responses,
            "analysis": analysis,
            "consensus_score": consensus_score,
            "query": result["query"],
            "question_type": result.get("question_type", "None"),
            "domain": result.get("domain", "Custom"),
            "aggregator_id": aggregator_id
        }
    }

    return active_jobs[job_id]["result"]

@app.get("/image/{job_id}/{image_type}")
async def api_get_image(job_id: str, image_type: str):
    """
    Retrieve a specific type of image associated with a completed job.
    
    Args:
        job_id (str): The unique identifier of the job.
        image_type (str): The type of image to retrieve.
    
    Raises:
        HTTPException: error codes

    Returns:
        StreamingResponse: A streaming response containing the requested image 
                           in PNG format.
    """
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    if "result" not in job or "analysis" not in job["result"]:
        raise HTTPException(status_code=400, detail="No analysis available")
    
    analysis = job["result"]["analysis"]
    
    if image_type not in ["heatmap", "emotion_chart", "polarity_chart", "radar_chart"]:
        raise HTTPException(status_code=400, detail="Invalid image type")
    
    # Extract the image data
    if image_type in analysis and analysis[image_type]:
        # Extract base64 image data
        img_data = analysis[image_type].split(",")[1]
        img_bytes = base64.b64decode(img_data)
        
        # Return the image
        return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="Image not found")

async def process_query_background(job_id: str, query: str, question_type: str, domain: str, ethical_views: List[str], username: str):
    """
    Processes a query asynchronously in the background, updating job status and progress,
    interacting with agents, aggregating responses, and saving results to the database.
    
    Args:
        job_id (str): A unique identifier for the job being processed.
        query (str): The query string to be processed.
        question_type (str): The type of question being asked (e.g., factual, analytical).
        domain (str): The domain or context of the query.
        ethical_views (List[str]): A list of ethical perspectives.
        username (str): The username of the user submitting the query.
    
    Raises:
        Exception: Captures and logs any errors that occur during processing.
    """
    try:
        # Update job status
        active_jobs[job_id] = {
            "status": "processing",
            "progress": 0
        }

        # Update progress
        active_jobs[job_id]["progress"] = 10

        # Update all agent heartbeats before processing
        for model_id in MODELS.keys():
            await update_agent_heartbeat(model_id)
        
        active_jobs[job_id]["progress"] = 30

        # Process the query
        result = await response_aggregator.process_query(query, question_type)
        
        # Update all agent heartbeats after processing
        for model_id in MODELS.keys():
            await update_agent_heartbeat(model_id)

        active_jobs[job_id]["progress"] = 60

        # Update progress
        active_jobs[job_id]["progress"] = 90
        
        # Extract model responses
        responses = result["responses"]
        aggregator_id = result.get("aggregator_id")

        agg_response = responses.get(aggregator_id, "").strip().lower()
        if (
            agg_response.startswith("error") or
            agg_response.startswith("could not generate")
            or "all models failed" in agg_response
        ):
            logger.warning(f"Aggregator failed. Skipping DB save for job {job_id}.")
            active_jobs[job_id] = {
                "status": "error",
                "error": "Aggregator did not produce a valid response."
            }
            return

        analysis = result["analysis"]
        
        # Get aggregator and non-aggregator models
        non_aggregator_models = [model_id for model_id, config in MODELS.items() 
                                if not config.get('aggregator', False)]
        
        # Construct result dictionary
        model_responses = {
            model_id: responses.get(model_id, "Error: Model failed to respond")
            for model_id in MODELS.keys()
        } 

        # Construct roles string, skip models with error responses
        roles_list = []
        # Assign roles only once for all agent_ids
        agent_ids = [mid for mid in responses if mid != aggregator_id]
        ethics_map = assign_ethics_to_agents(agent_ids, ethical_views)

        for model_id, response in responses.items():
            if isinstance(response, str) and not response.lower().startswith("error"):
                if model_id == aggregator_id:
                    role = "Aggregator"
                else:
                    # Use precomputed map
                    role_full = ethics_map.get(model_id, "")
                    role = [k for k, v in ETHICAL_VIEWS.items() if v == role_full]
                    role = role[0] if role else ""

                    if role:
                        role = role.split('.')[0]  # Optional: shorten long ethics description
                if role:
                    roles_list.append(f"{model_id}: {role}")
        roles_str = "; ".join(roles_list)

        # All models succeeded, safe to save
        db_manager.save_responses(job_id, model_responses, aggregator_id)
        # Save the interaction to database
        db_manager.save_interaction(job_id, query, domain, question_type, username, roles_str)
        
        # Calculate and save analysis
        consensus_score = 0
        if "error" not in analysis:
            consensus_score = round(analysis.get("consensus_score", 0) * 100)
            db_manager.save_analysis(job_id, consensus_score / 100, analysis)

        # Update job status to completed
        active_jobs[job_id] = {
            "status": "completed",
            "progress": 100,
            "result": {
                "responses": model_responses,
                "analysis": analysis,
                "consensus_score": consensus_score,
                "query": query,
                "question_type": question_type,
                "aggregator_id": aggregator_id
            }
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        active_jobs[job_id] = {
            "status": "error",
            "error": str(e)
        }

@app.get("/models")
async def api_get_models():
    """
    Fetches the list of available models and their metadata.

    This asynchronous function retrieves information about all available models,
    including their name, whether they use an aggregator, their associated color,
    and a formatted display name.

    Returns:
        dict: A dictionary where each key is a model ID and the value is another
        dictionary.
    """
    model_info = {}
    for model_id, config in MODELS.items():
        model_info[model_id] = {
            "name": config["name"],
            "aggregator": config.get("aggregator", False),
            "color": MODEL_COLORS.get(model_id, "#95a5a6"),
            "display_name": format_model_name(model_id)
        }
    return model_info

def format_model_name(model_id):
    """
    Formats the model name based on its configuration.
    This function retrieves the model configuration using the provided `model_id`
    from the `MODELS` dictionary. It extracts the model name from the path, removes
    unnecessary formatting (e.g., version numbers and suffixes like "-instruct"),
    capitalizes each word, and appends an "Aggregator" tag if applicable.
    
    Args:
        model_id (str): The identifier for the model in the `MODELS` dictionary.
    
    Returns:
        str: A human-readable, formatted model name.
    """
    config = MODELS[model_id]
    name = config["name"]
    # Extract model name from path and make it pretty
    display_name = name.split('/')[-1].split(':')[0]
    # Remove version numbers and formatting like "-instruct"
    display_name = display_name.replace('-instruct', '')
    # Capitalize words
    formatted_name = ' '.join(word.capitalize() for word in display_name.split('-'))
    
    # Add aggregator tag if applicable
    if config.get('aggregator', False):
        formatted_name += " (Aggregator)"
    
    return formatted_name

@app.get("/domains")
async def api_get_domains():
    """
    Asynchronous function to retrieve a list of domains.

    Returns:
        dict: A dictionary containing the key "domains" mapped to the value of the global variable `DOMAINS`.
    """
    return {"domains": DOMAINS}

@app.get("/question_types")
async def api_get_question_types():
    """
    Asynchronous function to retrieve available question types.

    This function returns a dictionary containing a list of question types
    derived from the keys of the `QUESTION_PREFIXES` dictionary, along with
    an additional "None" option.

    Returns:
        dict: A dictionary with a single key "question_types", whose value
        is a list of available question types.
    """
    return {"question_types": list(QUESTION_PREFIXES.keys()) + ["None"]}

@app.get("/examples")
async def api_get_examples():
    """
    Asynchronous function to retrieve example data.

    Returns:
        dict: A dictionary containing examples categorized by domain.
    """
    return {"examples": EXAMPLES_BY_DOMAIN}

@app.get("/agent_health")
async def api_agent_health():
    """
    Asynchronously checks the health of agents and formats the health status response.
    This function calls `check_agent_health()` to update the health status of agents.
    It then processes the `agent_health` dictionary to create a formatted response
    containing the health status of each agent, including the time since the last
    heartbeat, the number of failures, and retries.
    
    Returns:
        dict: A dictionary where each key is a model ID and the value is another
        dictionary.
    """
    await check_agent_health()
    
    # Format the response
    health_status = {}
    for model_id, health in agent_health.items():
        last_heartbeat_str = health["last_heartbeat"].strftime("%Y-%m-%d %H:%M:%S")
        time_since = (datetime.now() - health["last_heartbeat"]).total_seconds()
        
        health_status[model_id] = {
            "status": health["status"],
            "last_heartbeat": last_heartbeat_str,
            "seconds_since_heartbeat": round(time_since, 1),
            "failures": health["failures"],
            "retries": health["retries"]
        }
    
    return health_status

# Add reset endpoint
@app.post("/reset_agent/{model_id}")
async def api_reset_agent(model_id: str):
    """
    Handles the API request to reset a specific agent by its model ID.
    
    Args:
        model_id (str): The unique identifier of the agent to be reset.
    
    Raises:
        HTTPException: If the specified agent is not found in the system.
    
    Returns:
        dict: A dictionary containing the status and a success message indicating
              that the agent has been reset.
    """
    if model_id not in agent_health:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    await reset_failed_agent(model_id)
    return {"status": "success", "message": f"Agent {model_id} has been reset"}

# Main entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Agent LLM Backend Server")
    parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"), help="Host to run the server on")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8000)), help="Port to run the server on")
    
    args = parser.parse_args()
    
    # Create .env template
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("""# OpenRouter API Configuration
OPENROUTER_API_KEY=
SITE_URL=http://localhost:7860
SITE_NAME=Multi-Agent LLM System
""")
    
    # Log startup
    logger.info(f"Starting server on {args.host}:{args.port}")
    
    # Run the server
    uvicorn.run(app, host=args.host, port=args.port)