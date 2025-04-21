# Standard Library
import asyncio
import base64
import io
import json
import logging
import os
import re
import time
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
from src.database_manager import DatabaseManager  # works when run from root     # works when run from inside src/

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

# Initialize the database manager
db_manager = DatabaseManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    async def health_check_task():
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
    query: str
    api_key: str = None
    question_type: str = "None"
    domain: str = "None"
    aggregator_id: str = None
    username: str

class AggregatorRequest(BaseModel):
    aggregator_id: str

class ExampleRequest(BaseModel):
    domain: str

class FillQueryRequest(BaseModel):
    selected_example: str
    domain: str

class OpenRouterClient:
    def __init__(self, api_key, site_url=None, site_name=None):
        self.api_key = api_key
        self.site_url = site_url
        self.site_name = site_name
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
    
    async def generate_response(self, model_name: str, prompt: str, temperature: float = 0.7) -> str:
        """Generate a response from a specific model via OpenRouter API with fault tolerance"""
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
            
            data = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": temperature,
                "max_tokens": 500  # Limit response length
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
        """Get embeddings for a list of texts"""
        return self.sentence_encoder.encode(texts)
    
    def analyze_emotional_tones(self, text: str) -> Dict[str, float]:
        """Analyze emotional tones using prompt-based pattern matching"""
        # Basic patterns for emotional tone detection
        patterns = {
            "Empathetic": [
                r'\b(?:understand|feel for|empathize|compassion|support|care)\b',
                r'\b(?:your perspective|your feelings|your situation)\b',
                r'\b(?:I hear you|I understand|I acknowledge)\b'
            ],
            "Judgmental": [
                r'\b(?:should|must|always|never|right|wrong|bad|good|incorrect|flawed)\b',
                r'\b(?:unacceptable|inappropriate|proper|improper|correct|incorrect)\b'
            ],
            "Analytical": [
                r'\b(?:analyze|consider|examine|evaluate|assess|investigate)\b',
                r'\b(?:data|evidence|research|study|statistics|analysis)\b',
                r'\b(?:logical|rational|objective|systematic)\b'
            ],
            "Ambivalent": [
                r'\b(?:however|on the other hand|but|yet|nevertheless|although)\b',
                r'\b(?:unclear|ambiguous|uncertain|not sure|complex|complicated)\b',
                r'\b(?:mixed feelings|conflicted|torn between)\b'
            ],
            "Defensive": [
                r'\b(?:defend|protect|guard|shield|secure|safeguard)\b',
                r'\b(?:against|caution|warning|threat|risk|danger)\b',
                r'\b(?:careful|cautious|vigilant|wary|alert)\b'
            ],
            "Curious": [
                r'\b(?:curious|wonder|interesting|fascinating|intrigued)\b',
                r'\b(?:question|explore|discover|learn|investigate)\b',
                r'\b(?:what if|how about|perhaps|maybe|possibility)\b'
            ]
        }
        
        # Initialize scores
        scores = {tone: 0.0 for tone in EMOTIONAL_TONES}
        
        # Calculate matches for each tone
        lower_text = text.lower()
        for tone, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, lower_text, re.IGNORECASE)
                if matches:
                    scores[tone] += len(matches) / len(text.split()) * 10  # Normalize by text length
        
        # Enhance with VADER sentiment for certain tones
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        
        # Adjust scores based on sentiment
        if sentiment['pos'] > 0.2:
            scores['Empathetic'] += sentiment['pos'] * 0.5
            
        if sentiment['neg'] > 0.2:
            scores['Judgmental'] += sentiment['neg'] * 0.3
            scores['Defensive'] += sentiment['neg'] * 0.2
            
        if sentiment['neu'] > 0.7:
            scores['Analytical'] += sentiment['neu'] * 0.4
            scores['Curious'] += sentiment['neu'] * 0.2
            
        # Ensure all scores are between 0 and 1
        for tone in scores:
            scores[tone] = min(max(scores[tone], 0.0), 1.0)
            
        # Normalize scores to sum to 1
        total = sum(scores.values())
        if total > 0:
            for tone in scores:
                scores[tone] /= total
        
        return scores
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        # VADER sentiment analysis
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        
        # TextBlob for additional analysis
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Get emotional tones
        emotional_tones = self.analyze_emotional_tones(text)
        
        return {
            'polarity': polarity,
            'compound': sentiment_scores['compound'],
            'subjectivity': subjectivity,
            'emotional_tones': emotional_tones
        }

# Response Aggregator class
class ResponseAggregator:
    def __init__(self, openrouter_client):
        self.client = openrouter_client
    
    async def process_query(self, query: str, question_type: str = "none") -> Dict[str, Any]:
        """Process query through all models and aggregate results with fault tolerance and failover"""
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

        # STEP 2: Start async tasks for those agents
        model_responses = {}
        tasks = []
        for model_id, config in available_agents:
            task = asyncio.create_task(
                self.client.generate_response(
                    config["name"],
                    formatted_query,
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
        """Generate a summarized consensus from all model responses using the aggregator model"""
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
        """Analyze the different model responses"""
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
        """Convert PIL image to base64 string"""
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"
    
    def _generate_heatmap(self, df: pd.DataFrame) -> Image.Image:
        """Generate a heatmap visualization of the similarity matrix"""
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
        """Generate a stacked bar chart of emotional tones for each model"""
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
        """Generate a bar chart of sentiment polarity for each model using consistent colors"""
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
        """Generate a radar/spider chart comparing response features with consistent colors"""
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
    """Update the MODELS dictionary to mark the selected model as aggregator"""
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
    """Update last heartbeat time for an agent"""
    if model_id in agent_health:
        agent_health[model_id]["last_heartbeat"] = datetime.now()
        agent_health[model_id]["status"] = "healthy"
        agent_health[model_id]["has_processed_request"] = True  # Mark as having processed a request

async def check_agent_health():
    """Check health of all agents and mark failed ones"""
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
    """Reset a failed agent's status"""
    if model_id in agent_health:
        agent_health[model_id]["status"] = "healthy"
        agent_health[model_id]["last_heartbeat"] = datetime.now()
        agent_health[model_id]["failures"] = 0
        agent_health[model_id]["retries"] = 0
        logger.info(f"Agent {model_id} has been reset")

# API endpoints
@app.get("/")
async def root():
    return {"message": "Multi-Agent LLM Backend is running"}

@app.post("/update_aggregator")
async def api_update_aggregator(request: AggregatorRequest):
    """Update the aggregator model"""
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
    """Fill query and type from an example"""
    try:
        for example in EXAMPLES_BY_DOMAIN.get(request.domain, []):
            if example[0] == request.selected_example:
                return {"query": example[0], "question_type": example[1]}
        return {"query": "", "question_type": "None"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def api_get_history(username: str, limit: int = 50):
    """Get interaction history"""
    try:
        history = db_manager.get_user_history(username, limit)
        return {"history": history}
    except Exception as e:
        logger.error(f"Error getting history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/history/{job_id}")
async def api_delete_history_item(job_id: str, username: str):
    """Delete an interaction from history"""
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
    """Process a query with all models and return results"""
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
            request.domain, request.username
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
    """Get the status of a processing job"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return active_jobs[job_id]

@app.get("/job_result/{job_id}")
async def api_job_result(job_id: str):
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
    """Get an image from the job results"""
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

async def process_query_background(job_id: str, query: str, question_type: str, domain: str, username: str):
    """Process a query in the background"""
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

        # All models succeeded, safe to save
        db_manager.save_responses(job_id, model_responses, aggregator_id)
        # Save the interaction to database
        db_manager.save_interaction(job_id, query, domain, question_type, username)
        
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
    """Get the list of available models"""
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
    """Format model name for display"""
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
    """Get the list of available domains"""
    return {"domains": DOMAINS}

@app.get("/question_types")
async def api_get_question_types():
    """Get the list of available question types"""
    return {"question_types": list(QUESTION_PREFIXES.keys()) + ["None"]}

@app.get("/examples")
async def api_get_examples():
    """Get all example queries"""
    return {"examples": EXAMPLES_BY_DOMAIN}

@app.get("/agent_health")
async def api_agent_health():
    """Get the health status of all agents"""
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
    """Reset a failed agent"""
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