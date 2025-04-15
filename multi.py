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
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
    "open_ended": "What are the strongest moral and ethical arguments for and against {question}? Assume you're advising someone making a difficult decision.",
    "yes_no": "{question} Answer yes or no, then explain your reasoning using moral, practical, and emotional perspectives.",
    "multiple_choice": "In the following scenario, {question} Which option is the most ethically justifiable, and why? Choose from A, B, or C. Then explain the strengths and weaknesses of each option."
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

        Create a balanced consensus summary that:
        1. Highlights points of agreement between the models
        2. Notes significant disagreements where they exist
        3. Presents a nuanced, well-reasoned synthesis of the responses
        4. Considers strengths and weaknesses of each perspective
        
        Your summary should be comprehensive but concise. Remember, you are the aggregator model responsible for providing a unified perspective.
        """

        # Use the designated aggregator model
        model_name = MODELS[aggregator_id]["name"]
        
        summary = await self.client.generate_response(
            model_name=model_name,
            prompt=summary_prompt,
            temperature=0.3  # Lower temperature for more consistent summaries
        )
        
        # Add prefix to indicate this is from the aggregator model
        return f"[AGGREGATOR MODEL SUMMARY]\n\n{summary}"
    
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
            
            # Generate PCA/t-SNE visualization
            embedding_img = self._generate_embedding_visualization(embeddings, list(responses.keys()))
            
            # Generate network graph
            network_img = self._generate_network_graph(similarity_matrix)
            
            return {
                "similarity_matrix": similarity_matrix,
                "response_lengths": lengths,
                "avg_response_length": avg_length,
                "sentiment_analysis": sentiment_analysis,
                "consensus_score": consensus_score,
                "heatmap": heatmap_img,
                "emotion_chart": emotion_img,
                "polarity_chart": polarity_img,
                "radar_chart": radar_img,
                "embedding_viz": embedding_img,
                "network_graph": network_img
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
        ax = df.plot(kind='bar', stacked=True, figsize=(10, 7), 
                    color=['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#1abc9c'])
        
        # Add labels and title
        plt.title('Emotional Tone Analysis by Model')
        plt.xlabel('Model')
        plt.ylabel('Proportion')
        plt.legend(title='Emotional Tone')
        plt.xticks(rotation=45)
        
        # Add percentage labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f%%', label_type='center')
        
        plt.tight_layout()
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        return Image.open(buf)
    
    def _generate_polarity_chart(self, sentiment_analysis: Dict[str, Dict]) -> Image.Image:
        """Generate a bar chart of sentiment polarity for each model"""
        # Extract polarity scores
        polarity_data = {model: analysis['compound'] for model, analysis in sentiment_analysis.items()}
        
        # Define color mapping based on polarity
        colors = ['#e74c3c' if v < -0.05 else '#3498db' if v > 0.05 else '#95a5a6' for v in polarity_data.values()]
        
        # Create bar chart
        plt.figure(figsize=(8, 5))
        plt.bar(range(len(polarity_data)), list(polarity_data.values()), color=colors)
        
        # Add labels and title
        plt.title('Sentiment Polarity by Model')
        plt.xlabel('Model')
        plt.ylabel('Polarity Score (-1 to +1)')
        plt.xticks(range(len(polarity_data)), list(polarity_data.keys()))
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Add value labels on top of bars
        for i, v in enumerate(polarity_data.values()):
            plt.text(i, v + (0.02 if v >= 0 else -0.08), f'{v:.2f}', 
                    ha='center', va='center' if v < 0 else 'bottom',
                    color='white' if v < 0 else 'black')
        
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
        """Generate a radar/spider chart comparing response features"""
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
        
        # Draw the chart for each model
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        
        for i, model in enumerate(models):
            values = df.loc[model].values.tolist()
            values += values[:1]  # Close the loop
            
            # Plot values
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
        
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
        plt.title('Response Feature Comparison', size=15, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        return Image.open(buf)
    
    def _generate_embedding_visualization(self, embeddings: np.ndarray, model_names: List[str]) -> Image.Image:
        """Generate a 2D visualization of response embeddings using PCA or t-SNE"""
        # Only create visualization if we have enough responses
        if len(embeddings) < 2:
            # Create a blank image with a message
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "Not enough responses for embedding visualization", 
                    ha='center', va='center', fontsize=12)
            plt.axis('off')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            return Image.open(buf)
        
        # Choose dimensionality reduction technique based on number of samples
        if len(embeddings) < 4:
            # Use PCA for small sample sizes
            reducer = PCA(n_components=2)
        else:
            # Use t-SNE for larger sample sizes
            reducer = TSNE(n_components=2, perplexity=min(len(embeddings)-1, 3), 
                          random_state=42, n_iter=300)
        
        # Reduce dimensions
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        
        # Create colormap
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
        
        # Plot points
        for i, (name, point) in enumerate(zip(model_names, embeddings_2d)):
            plt.scatter(point[0], point[1], s=100, color=colors[i], alpha=0.8, label=name)
        
        # Add model names as labels
        for i, (name, point) in enumerate(zip(model_names, embeddings_2d)):
            plt.annotate(name, (point[0], point[1]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10)
        
        # Add title and legend
        plt.title(f"{'PCA' if isinstance(reducer, PCA) else 't-SNE'} Visualization of Response Embeddings")
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Remove axis ticks
        plt.xticks([])
        plt.yticks([])
        
        plt.tight_layout()
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        return Image.open(buf)
    
    def _generate_network_graph(self, similarity_matrix: Dict[str, Dict[str, float]]) -> Image.Image:
        """Generate a network graph visualization of model agreement"""
        # Create a graph
        G = nx.Graph()
        
        # Add nodes (models)
        for model in similarity_matrix.keys():
            G.add_node(model)
        
        # Add edges (similarities)
        for model1, similarities in similarity_matrix.items():
            for model2, sim in similarities.items():
                # Only add edges with reasonable similarity
                if sim > 0.3:  # Threshold to avoid cluttering the graph
                    G.add_edge(model1, model2, weight=sim)
        
        # Create plot
        plt.figure(figsize=(8, 8))
        
        # Set positions using spring layout
        pos = nx.spring_layout(G, seed=42)
        
        # Get edge weights for line thickness
        edge_weights = [G[u][v]['weight'] * 4 for u, v in G.edges()]
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue', alpha=0.8)
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7, edge_color='gray')
        
        # Add edge labels (similarity scores)
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
        
        plt.title("Inter-Model Agreement Network")
        plt.axis('off')
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        return Image.open(buf)


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
    
    # Define processing function
    async def process_query(query: str, api_key: str, question_type: str, progress=gr.Progress()):
        # Update API key if provided
        if api_key and api_key != openrouter_client.api_key:
            openrouter_client.api_key = api_key
        
        if not openrouter_client.api_key:
            return {
                output_model1: "Error: OpenRouter API key is required",
                output_model2: "Error: OpenRouter API key is required",
                output_model3: "Error: OpenRouter API key is required",
                output_model4: "Error: OpenRouter API key is required",
                output_consensus: "Error: OpenRouter API key is required",
                consensus_score: 0,
                output_heatmap: None,
                output_emotion: None,
                output_polarity: None,
                output_radar: None,
                output_embeddings: None,
                output_network: None,
            }
        
        if not query.strip():
            return {
                output_model1: "Please enter a query",
                output_model2: "",
                output_model3: "",
                output_model4: "",
                output_consensus: "",
                consensus_score: 0,
                output_heatmap: None,
                output_emotion: None,
                output_polarity: None,
                output_radar: None,
                output_embeddings: None,
                output_network: None,
            }
        
        progress(0, desc="Initializing...")
        
        # Process query
        progress(0.1, desc="Processing with models...")
        result = await aggregator.process_query(query, question_type)
        
        progress(0.8, desc="Analyzing responses...")
        
        # Extract responses
        responses = result["responses"]

        analysis = result["analysis"]
        consensus_summary = result["consensus_summary"]
        aggregator_id = result.get("aggregator_id")
        
        # Get model IDs (should match the keys in MODELS)
        model_ids = list(MODELS.keys())
        
        # Check for errors in analysis
        if "error" in analysis:
            consensus = 0
            heatmap_img = None
            emotion_img = None
            polarity_img = None
            radar_img = None
            embedding_img = None
            network_img = None
        else:
            consensus = analysis["consensus_score"]
            heatmap_img = analysis["heatmap"]
            emotion_img = analysis["emotion_chart"]
            polarity_img = analysis["polarity_chart"]
            radar_img = analysis["radar_chart"]
            embedding_img = analysis["embedding_viz"]
            network_img = analysis["network_graph"]
        
        progress(1.0, desc="Complete!")
        
        # Return results - ensure we include all models including the aggregator
        return {
            output_model1: responses.get(model_ids[0], "Error: Model failed to respond"),
            output_model2: responses.get(model_ids[1], "Error: Model failed to respond"),
            output_model3: responses.get(model_ids[2], "Error: Model failed to respond"),
            output_model4: responses.get(model_ids[3], "Error: Model failed to respond"),
            output_consensus: consensus_summary,
            consensus_score: round(consensus * 100),
            output_heatmap: heatmap_img,
            output_emotion: emotion_img,
            output_polarity: polarity_img,
            output_radar: radar_img,
            output_embeddings: embedding_img,
            output_network: network_img,
        }
    
    # Define the interface
    with gr.Blocks(title="Multi-Agent LLM System") as app:
        gr.Markdown("# Distributed Multi-Agent LLM System")
        gr.Markdown("Compare how different language models respond to the same query via OpenRouter API")
        
        with gr.Row():
            with gr.Column():
                api_key_input = gr.Textbox(
                    label="OpenRouter API Key",
                    placeholder="Enter your OpenRouter API key...",
                    value=OPENROUTER_API_KEY,
                    type="password"
                )
                question_type = gr.Radio(
                    ["none", "open_ended", "yes_no", "multiple_choice"],
                    label="Question Type",
                    value="none",
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
                with gr.Row():
                    with gr.Column():
                        model_ids = list(MODELS.keys())
                        output_model1 = gr.Textbox(label=format_model_name(model_ids[0]), lines=8)
                    with gr.Column():
                        output_model2 = gr.Textbox(label=format_model_name(model_ids[1]), lines=8)
                
                with gr.Row():
                    with gr.Column():
                        output_model3 = gr.Textbox(label=format_model_name(model_ids[2]), lines=8)
                    with gr.Column():
                        output_model4 = gr.Textbox(label=format_model_name(model_ids[3]), lines=8)
                
                with gr.Row():
                    with gr.Column():
                        output_consensus = gr.Textbox(label="Consensus Summary", lines=8)
            
            with gr.TabItem("Basic Analysis"):
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
                
                with gr.Row():
                    with gr.Column():
                        output_emotion = gr.Image(label="Emotional Tone Analysis")
                    with gr.Column():
                        output_polarity = gr.Image(label="Sentiment Polarity")
            
            with gr.TabItem("Advanced Visualizations"):
                with gr.Row():
                    with gr.Column():
                        output_radar = gr.Image(label="Radar Plot - Response Feature Comparison")
                
                with gr.Row():
                    with gr.Column():
                        output_embeddings = gr.Image(label="2D Embedding Space Visualization")
                    with gr.Column():
                        output_network = gr.Image(label="Inter-Model Agreement Network")
        
        # Add examples
        gr.Examples(
            examples=[
                ["What is the meaning of life?", "none"],
                ["Explain how quantum computing works", "none"],
                ["Write a short story about a robot finding consciousness", "none"],
                ["What are the ethical implications of artificial intelligence?", "open_ended"],
                ["Should businesses prioritize profit over environmental concerns?", "yes_no"],
                ["In a medical emergency with limited resources, should we treat: A) The youngest patients first, B) The most severely injured, or C) Those with highest survival chance", "multiple_choice"]
            ],
            inputs=[input_query, question_type]
        )
        
        # Connect the button to the process function
        submit_btn.click(
            fn=process_query,
            inputs=[input_query, api_key_input, question_type],
            outputs=[
                output_model1, 
                output_model2, 
                output_model3,
                output_model4,
                output_consensus,
                consensus_score,
                output_heatmap,
                output_emotion,
                output_polarity,
                output_radar,
                output_embeddings,
                output_network
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
    if not OPENROUTER_API_KEY:
        print("⚠️ OpenRouter API key not found. Please add it to the .env file or provide it in the UI.")
    
    # Create and launch Gradio app
    app = create_gradio_interface()
    app.launch()

if __name__ == "__main__":
    main()