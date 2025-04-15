# main.py - Distributed multi-agent LLM system using OpenRouter API

import asyncio
import logging
import numpy as np
import gradio as gr
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import re
from PIL import Image
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from typing import Dict, List, Any, Tuple
import os
from dotenv import load_dotenv
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

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
    "Multiple choice": "In the following scenario, {question} Which option is the most ethically justifiable, and why? Choose just one option from A, B, or C. Then explain the strengths and weaknesses of each option."
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

class OpenRouterClient:
    def __init__(self, api_key, site_url=None, site_name=None):
        self.api_key = api_key
        self.site_url = site_url
        self.site_name = site_name
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
    
    async def generate_response(self, model_name: str, prompt: str, temperature: float = 0.7) -> str:
        """Generate a response from a specific model via OpenRouter API"""
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
            
            # Make API request asynchronously
            response = await asyncio.to_thread(
                requests.post,
                url=self.url,
                headers=headers,
                data=json.dumps(data)
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return f"Error: API returned status code {response.status_code}"
        
        except Exception as e:
            logger.error(f"Error generating with {model_name}: {str(e)}")
            return f"Error with {model_name}: {str(e)}"
    
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
        """Process query through all models and aggregate results"""
        model_responses = {}
        
        # Apply prefix based on question type
        if question_type != "none" and question_type in QUESTION_PREFIXES:
            formatted_query = QUESTION_PREFIXES[question_type].format(question=query)
        else:
            formatted_query = query
        
        # Create tasks for non-aggregator models
        tasks = []
        for model_id, config in MODELS.items():
            if not config.get('aggregator', False):
                task = asyncio.create_task(
                    self.client.generate_response(
                        config['name'], 
                        formatted_query,
                        config['temperature']
                    )
                )
                tasks.append((model_id, task))
        
        # Gather responses from non-aggregator models
        for model_id, task in tasks:
            try:
                response = await task
                model_responses[model_id] = response
            except Exception as e:
                logger.error(f"Error with {model_id}: {str(e)}")
                model_responses[model_id] = f"Error: {str(e)}"
        
        # Analyze responses
        analysis = await self.analyze_responses(query, model_responses)
        
        # Generate consensus summary using the aggregator model
        aggregator_id = next((model_id for model_id, config in MODELS.items() if config.get('aggregator', False)), None)
        
        if aggregator_id:
            consensus_summary = await self.generate_consensus_summary(query, model_responses, aggregator_id)
            model_responses[aggregator_id] = consensus_summary  # Add aggregator response
        else:
            # Fallback if no aggregator is specified
            consensus_summary = "No aggregator model specified"
        
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
        if any(response.startswith("Error:") for response in responses.values()):
            return "Cannot generate consensus due to model errors."
        
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
        
        # Add prefix to indicate this is from the aggregator model
        return f"SUMMARY\n\n{summary}"
    
    async def analyze_responses(self, query: str, responses: Dict[str, str]) -> Dict[str, Any]:
        """Analyze the different model responses"""
        # Get response texts
        texts = list(responses.values())
        
        # Skip analysis if there are errors
        if any(text.startswith("Error:") for text in texts):
            return {"error": "Cannot analyze responses due to model errors"}
        
        try:
            # Get embeddings
            embeddings = await asyncio.to_thread(self.client.get_embeddings, texts)
            
            # Calculate similarity matrix
            similarity_matrix = {}
            for i, model_i in enumerate(responses.keys()):
                similarity_matrix[model_i] = {}
                for j, model_j in enumerate(responses.keys()):
                    if i != j:
                        sim = 1 - cosine(embeddings[i], embeddings[j])
                        similarity_matrix[model_i][model_j] = float(sim)
            
            # Calculate response length statistics
            lengths = {model_id: len(response.split()) for model_id, response in responses.items()}
            avg_length = sum(lengths.values()) / len(lengths)
            
            # Calculate sentiment and emotional tones
            sentiment_analysis = {}
            for model_id, response in responses.items():
                sentiment_analysis[model_id] = await asyncio.to_thread(self.client.analyze_sentiment, response)
            
            # Calculate overall agreement score (average pairwise similarity)
            agreement_scores = []
            for model, sims in similarity_matrix.items():
                if sims:  # Check if the model has similarity scores
                    agreement_scores.extend(sims.values())
            
            consensus_score = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0
            
            # Generate similarity heatmap
            similarity_df = pd.DataFrame(similarity_matrix).fillna(1.0)
            heatmap_img = self._generate_heatmap(similarity_df)
            
            # Generate emotion comparison chart
            emotion_img = self._generate_emotion_chart(sentiment_analysis)
            
            # Generate polarity comparison chart
            polarity_img = self._generate_polarity_chart(sentiment_analysis)
            
            # Generate radar chart
            radar_img = self._generate_radar_chart(responses, sentiment_analysis, lengths)
            
            return {
                "similarity_matrix": similarity_matrix,
                "response_lengths": lengths,
                "avg_response_length": avg_length,
                "sentiment_analysis": sentiment_analysis,
                "consensus_score": consensus_score,
                "heatmap": heatmap_img,
                "emotion_chart": emotion_img,
                "polarity_chart": polarity_img,
                "radar_chart": radar_img
            }
            
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
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

# Gradio Interface
def create_gradio_interface():
    # Initialize components
    openrouter_client = OpenRouterClient(
        api_key=OPENROUTER_API_KEY,
        site_url=SITE_URL,
        site_name=SITE_NAME
    )
    aggregator = ResponseAggregator(openrouter_client)
    
    # Format model name for display
    def format_model_name(model_id):
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
    
    def update_output_labels(aggregator_id):
        # Get non-aggregator model IDs
        non_aggregator_models = [model_id for model_id, config in MODELS.items() 
                                if not config.get('aggregator', False)]
        
        # Update model labels
        return {
            output_aggregator: gr.update(label=format_model_name(aggregator_id)),
            output_model1: gr.update(label=format_model_name(non_aggregator_models[0])),
            output_model2: gr.update(label=format_model_name(non_aggregator_models[1])),
            output_model3: gr.update(label=format_model_name(non_aggregator_models[2])),
        }

    # Define processing function
    async def process_query(query: str, api_key: str, question_type: str, aggregator_id: str, progress=gr.Progress()):
        # Update API key if provided
        if api_key and api_key != openrouter_client.api_key:
            openrouter_client.api_key = api_key
        
        if not openrouter_client.api_key:
            return {
                output_aggregator: "Error: OpenRouter API key is required",
                output_model1: "Error: OpenRouter API key is required",
                output_model2: "Error: OpenRouter API key is required",
                output_model3: "Error: OpenRouter API key is required",
                consensus_score: 0,
                output_heatmap: None,
                output_emotion: None,
                output_polarity: None,
                output_radar: None,
            }
        
        if not query.strip():
            return {
                output_aggregator: "Please enter a query",
                output_model1: "",
                output_model2: "",
                output_model3: "",
                consensus_score: 0,
                output_heatmap: None,
                output_emotion: None,
                output_polarity: None,
                output_radar: None,
            }
        
        progress(0.05, desc="Initializing...")
        await asyncio.sleep(0.2)

        progress(0.3, desc="Generating responses from models...")
        result = await aggregator.process_query(query, question_type)

        progress(0.6, desc="Analyzing sentiment and similarity...")
        await asyncio.sleep(0.4)

        progress(0.9, desc="Finalizing output...")
        await asyncio.sleep(0.2)
                
        # Extract responses
        responses = result["responses"]
        analysis = result["analysis"]
        consensus_summary = result["consensus_summary"]
        aggregator_id = result.get("aggregator_id")
        
        # Get model IDs (except aggregator)
        non_aggregator_models = [model_id for model_id, config in MODELS.items() 
                                if not config.get('aggregator', False)]
        
        # Check for errors in analysis
        if "error" in analysis:
            consensus = 0
            heatmap_img = None
            emotion_img = None
            polarity_img = None
            radar_img = None
        else:
            consensus = analysis["consensus_score"]
            heatmap_img = analysis["heatmap"]
            emotion_img = analysis["emotion_chart"]
            polarity_img = analysis["polarity_chart"]
            radar_img = analysis["radar_chart"]
        
        progress(1.0, desc="Complete!")
        
        # Return results
        return {
            output_aggregator: responses.get(aggregator_id, "Error: Aggregator model failed to respond"),
            output_model1: responses.get(non_aggregator_models[0], "Error: Model failed to respond"),
            output_model2: responses.get(non_aggregator_models[1], "Error: Model failed to respond"),
            output_model3: responses.get(non_aggregator_models[2], "Error: Model failed to respond"),
            consensus_score: round(consensus * 100),
            output_heatmap: heatmap_img,
            output_emotion: emotion_img,
            output_polarity: polarity_img,
            output_radar: radar_img,
        }
    
    # Define the interface
    with gr.Blocks(title="Multi-Agent LLM System") as app:
        gr.Markdown("# Distributed Multi-Agent LLM System")
        
        with gr.Row():
            with gr.Column():
                api_key_input = gr.Textbox(
                    label="OpenRouter API Key",
                    placeholder="Enter your OpenRouter API key...",
                    value=OPENROUTER_API_KEY,
                    type="password"
                )
                
                # Add aggregator selection radio button
                aggregator_radio = gr.Radio(
                    choices=list(MODELS.keys()),
                    label="Select Aggregator Model",
                    value=next((m for m, c in MODELS.items() if c.get('aggregator', False)), list(MODELS.keys())[0])
                )
                
                question_type = gr.Radio(
                    ["None", "Open-ended", "Yes/No", "Multiple Choice"],
                    label="Question Type",
                    value="None",
                    info="Select prompt format to improve results"
                )
                input_query = gr.Textbox(
                    label="Your Query",
                    placeholder="Enter your question or prompt...",
                    lines=3
                )
                submit_btn = gr.Button("Submit", variant="primary")
        
        with gr.Tabs():
            with gr.TabItem("Model Responses"):
                # Get current aggregator model ID
                current_aggregator_id = next((model_id for model_id, config in MODELS.items() 
                                      if config.get('aggregator', False)), None)
                
                # Get non-aggregator model IDs
                non_aggregator_models = [model_id for model_id, config in MODELS.items() 
                                        if not config.get('aggregator', False)]
                
                # Full-width aggregator response
                with gr.Row():
                    output_aggregator = gr.Textbox(
                        label=format_model_name(current_aggregator_id) if current_aggregator_id else "Consensus Summary",
                        lines=10,
                        max_lines=10
                    )
                
                # Three equal columns for agent models
                with gr.Row():
                    output_model1 = gr.Textbox(
                        label=format_model_name(non_aggregator_models[0]), 
                        lines=8,
                        max_lines=8
                    )
                    output_model2 = gr.Textbox(
                        label=format_model_name(non_aggregator_models[1]),
                        lines=8,
                        max_lines=8
                    )
                    output_model3 = gr.Textbox(
                        label=format_model_name(non_aggregator_models[2]),
                        lines=8,
                        max_lines=8
                    )
            
            with gr.TabItem("Analysis Visualizations"):
                with gr.Row():
                    with gr.Column():
                        consensus_score = gr.Slider(
                            label="Consensus Score", 
                            minimum=0, 
                            maximum=100, 
                            value=0,
                            interactive=False
                        )
                
                with gr.Row():
                    with gr.Column():
                        output_heatmap = gr.Image(label="Response Similarity Matrix")
                    with gr.Column():
                        output_emotion = gr.Image(label="Emotional Tone Analysis")
                
                with gr.Row():
                    with gr.Column():
                        output_polarity = gr.Image(label="Sentiment Polarity")
                    with gr.Column():
                        output_radar = gr.Image(label="Response Feature Comparison")
        
        # Add examples
        gr.Examples(
            examples=[
                ["What is the meaning of life?", "none"],
                ["Explain how quantum computing works", "none"],
                ["Write a short story about a robot finding consciousness", "None"],
                ["What are the ethical implications of artificial intelligence?", "Open-ended"],
                ["Should businesses prioritize profit over environmental concerns?", "Yes/No"],
                ["In a medical emergency with limited resources, should we treat: A) The youngest patients first, B) The most severely injured, or C) Those with highest survival chance", "Multiple Choice"]
            ],
            inputs=[input_query, question_type]
        )
        
        # Connect the radio button to update the aggregator
        aggregator_radio.change(
            fn=lambda x: update_aggregator(x),
            inputs=aggregator_radio,
            outputs=aggregator_radio
        ).then(
            fn=update_output_labels,
            inputs=aggregator_radio,
            outputs=[output_aggregator, output_model1, output_model2, output_model3]
        )

        # Connect the button to the process function
        submit_btn.click(
            fn=process_query,
            inputs=[input_query, api_key_input, question_type, aggregator_radio],
            outputs=[
                output_aggregator,
                output_model1, 
                output_model2, 
                output_model3,
                consensus_score,
                output_heatmap,
                output_emotion,
                output_polarity,
                output_radar
            ]
        )
    
    return app

# Create .env file template
def create_env_template():
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("""# OpenRouter API Configuration
OPENROUTER_API_KEY=
SITE_URL=http://localhost:7860
SITE_NAME=Multi-Agent LLM System
""")

# Main function to run the system
def main():
    # Create .env template
    create_env_template()
    
    # Check for API key
    # if not OPENROUTER_API_KEY:
    #     print("⚠️ OpenRouter API key not found. Please add it to the .env file or provide it in the UI.")
    
    # Create and launch Gradio app
    app = create_gradio_interface()
    app.launch()

if __name__ == "__main__":
    main()