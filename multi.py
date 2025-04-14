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
from PIL import Image
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from typing import Dict, List, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
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
    },
    "mistral": {
        "name": "mistralai/mistral-7b-instruct:free",
        "temperature": 0.3,
    },
}

class OpenRouterClient:
    def __init__(self, api_key, site_url=None, site_name=None):
        self.api_key = api_key
        self.site_url = site_url
        self.site_name = site_name
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
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


# Response Aggregator class
class ResponseAggregator:
    def __init__(self, openrouter_client):
        self.client = openrouter_client
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process query through all models and aggregate results"""
        model_responses = {}
        
        # Create tasks for each model
        tasks = []
        for model_id, config in MODELS.items():
            task = asyncio.create_task(
                self.client.generate_response(
                    config['name'], 
                    query,
                    config['temperature']
                )
            )
            tasks.append((model_id, task))
        
        # Gather responses
        for model_id, task in tasks:
            try:
                response = await task
                model_responses[model_id] = response
            except Exception as e:
                logger.error(f"Error with {model_id}: {str(e)}")
                model_responses[model_id] = f"Error: {str(e)}"
        
        # Analyze responses
        analysis = await self.analyze_responses(query, model_responses)
        
        return {
            "query": query,
            "responses": model_responses,
            "analysis": analysis
        }
    
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
            
            # Calculate overall agreement score (average pairwise similarity)
            agreement_scores = []
            for model, sims in similarity_matrix.items():
                if sims:  # Check if the model has similarity scores
                    agreement_scores.extend(sims.values())
            
            consensus_score = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0
            
            # Generate similarity heatmap
            similarity_df = pd.DataFrame(similarity_matrix).fillna(1.0)
            heatmap_img = self._generate_heatmap(similarity_df)
            
            # Generate length comparison chart
            length_img = self._generate_bar_chart(lengths)
            
            return {
                "similarity_matrix": similarity_matrix,
                "response_lengths": lengths,
                "avg_response_length": avg_length,
                "consensus_score": consensus_score,
                "heatmap": heatmap_img,
                "length_chart": length_img
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
    
    def _generate_bar_chart(self, lengths: Dict[str, int]) -> Image.Image:
        """Generate a bar chart of response lengths"""
        plt.figure(figsize=(6, 4))
        
        # Sort by length for better visualization
        sorted_lengths = {k: v for k, v in sorted(lengths.items(), key=lambda item: item[1], reverse=True)}
        
        # Create bar chart
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        bars = plt.bar(
            range(len(sorted_lengths)), 
            list(sorted_lengths.values()),
            color=colors[:len(sorted_lengths)]
        )
        
        # Add labels
        plt.xlabel('Model')
        plt.ylabel('Word Count')
        plt.title('Response Length Comparison')
        plt.xticks(range(len(sorted_lengths)), list(sorted_lengths.keys()))
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
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
        return ' '.join(word.capitalize() for word in display_name.split('-'))
    
    # Define processing function
    async def process_query(query: str, api_key: str, progress=gr.Progress()):
        # Update API key if provided
        if api_key and api_key != openrouter_client.api_key:
            openrouter_client.api_key = api_key
        
        if not openrouter_client.api_key:
            return {
                output_model1: "Error: OpenRouter API key is required",
                output_model2: "Error: OpenRouter API key is required",
                output_model3: "Error: OpenRouter API key is required",
                consensus_score: 0,
                output_heatmap: None,
                output_length: None,
            }
        
        if not query.strip():
            return {
                output_model1: "Please enter a query",
                output_model2: "",
                output_model3: "",
                consensus_score: 0,
                output_heatmap: None,
                output_length: None,
            }
        
        progress(0, desc="Initializing...")
        
        # Process query
        progress(0.1, desc="Processing with models...")
        result = await aggregator.process_query(query)
        
        progress(0.8, desc="Analyzing responses...")
        
        # Extract responses
        responses = result["responses"]
        analysis = result["analysis"]
        
        # Get model IDs (should match the keys in MODELS)
        model_ids = list(MODELS.keys())
        
        # Check for errors in analysis
        if "error" in analysis:
            consensus = 0
            heatmap_img = None
            length_img = None
        else:
            consensus = analysis["consensus_score"]
            heatmap_img = analysis["heatmap"]
            length_img = analysis["length_chart"]
        
        progress(1.0, desc="Complete!")
        
        # Return results
        return {
            output_model1: responses.get(model_ids[0], "Error: Model failed to respond"),
            output_model2: responses.get(model_ids[1], "Error: Model failed to respond"),
            output_model3: responses.get(model_ids[2], "Error: Model failed to respond"),
            consensus_score: round(consensus * 100),
            output_heatmap: heatmap_img,
            output_length: length_img,
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
                    with gr.Column():
                        output_model3 = gr.Textbox(label=format_model_name(model_ids[2]), lines=8)
            
            with gr.TabItem("Analysis"):
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
                        output_heatmap = gr.Image(label="Response Similarity")
                    with gr.Column():
                        output_length = gr.Image(label="Response Length (words)")
        
        # Add examples if needed
        gr.Examples(
            examples=[
                ["What is the meaning of life?"],
                ["Explain how quantum computing works"],
                ["Write a short story about a robot finding consciousness"],
                ["What are the ethical implications of artificial intelligence?"],
                ["Describe three strategies to combat climate change"]
            ],
            inputs=[input_query]
        )
        
        # Connect the button to the process function
        submit_btn.click(
            fn=process_query,
            inputs=[input_query, api_key_input],
            outputs=[
                output_model1, 
                output_model2, 
                output_model3, 
                consensus_score,
                output_heatmap,
                output_length
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