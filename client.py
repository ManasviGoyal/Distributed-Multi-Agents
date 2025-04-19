# client.py - Frontend for Distributed multi-agent LLM system
import gradio as gr
import requests
import json
import time
import argparse
import os
from datetime import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv

from PIL import Image
from io import BytesIO

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("OPENROUTER_API_KEY", "")
BACKEND_URL = os.getenv("BACKEND_URL") or f"http://{os.getenv('HOST', 'localhost')}:{os.getenv('PORT', '8000')}"

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define client-side state
current_job_id = None
model_info = None
domains = None
question_types = None
examples_by_domain = None

def init_client(retries: int = 3, delay: float = 1.5) -> bool:
    """Initialize the client by fetching data from the backend with retries"""
    global model_info, domains, question_types, examples_by_domain
    attempt = 0

    while attempt < retries:
        try:
            logger.info(f"Attempting to connect to backend ({attempt + 1}/{retries}) at {BACKEND_URL}")

            # Try one representative endpoint first to check connectivity
            response = requests.get(f"{BACKEND_URL}/models", timeout=5)
            if response.status_code != 200:
                raise Exception(f"Status code {response.status_code} on /models")

            # If models are reachable, fetch everything
            model_info = response.json()

            response = requests.get(f"{BACKEND_URL}/domains")
            domains = response.json().get("domains", {}) if response.status_code == 200 else {}

            response = requests.get(f"{BACKEND_URL}/question_types")
            question_types = response.json().get("question_types", []) if response.status_code == 200 else []

            response = requests.get(f"{BACKEND_URL}/examples")
            examples_by_domain = response.json().get("examples", {}) if response.status_code == 200 else {}

            return True

        except Exception as e:
            logger.warning(f"Failed to connect to backend: {e}")
            attempt += 1
            time.sleep(delay)

    logger.error("Failed to connect after multiple attempts.")
    return False

def update_aggregator(aggregator_id):
    """Update the aggregator model on the backend"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/update_aggregator",
            json={"aggregator_id": aggregator_id}
        )
        if response.status_code == 200:
            return response.json().get("aggregator_id")
        else:
            logger.error(f"Failed to update aggregator: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error updating aggregator: {str(e)}")
        return None

def get_example_choices(domain):
    """Get example choices for a domain"""
    try:
        # Check if domain is null or empty
        if not domain:
            return gr.update(choices=[], value=None)
            
        response = requests.post(
            f"{BACKEND_URL}/get_example_choices",
            json={"domain": domain}
        )
        if response.status_code == 200:
            examples = response.json().get("examples", [])
            return gr.update(choices=examples, value=None)
        else:
            logger.error(f"Failed to get example choices: {response.status_code}")
            return gr.update(choices=[], value=None)
    except Exception as e:
        logger.error(f"Error getting example choices: {str(e)}")
        return gr.update(choices=[], value=None)

def fill_query_and_type(selected_example, domain):
    """Fill query and type from an example"""
    if not selected_example:
        return "", "None"  # Prevent 422 errors

    try:
        response = requests.post(
            f"{BACKEND_URL}/fill_query_and_type",
            json={"selected_example": selected_example, "domain": domain}
        )
        if response.status_code == 200:
            result = response.json()
            return result.get("query", ""), result.get("question_type", "None")
        else:
            logger.error(f"Failed to fill query and type: {response.status_code}")
            return "", "None"
    except Exception as e:
        logger.error(f"Error filling query and type: {str(e)}")
        return "", "None"

def update_output_labels(aggregator_id):
    """Update the output labels based on the actual aggregator used in the last query"""
    global model_info

    try:
        response = requests.get(f"{BACKEND_URL}/models")
        if response.status_code == 200:
            model_info = response.json()
        else:
            logger.error(f"Failed to refresh models: {response.status_code}")
            return {}
    except Exception as e:
        logger.error(f"Error refreshing models: {str(e)}")
        return {}

    # Always build fresh label based on actual aggregator used
    base_name = model_info.get(aggregator_id, {}).get("name", aggregator_id)
    short_name = base_name.split("/")[-1].split(":")[0].replace("-instruct", "").replace("-", " ").title()

    # Label aggregator correctly
    if model_info.get(aggregator_id, {}).get("aggregator", False):
        agg_label = f"{short_name} (Aggregator)"
    else:
        agg_label = f"{short_name} (Fallback Aggregator)"

    # List other agent models, excluding the actual aggregator
    non_aggregator_models = [mid for mid in model_info if mid != aggregator_id]

    agent_labels = []
    for i in range(3):
        if i < len(non_aggregator_models):
            model_id = non_aggregator_models[i]
            base_name = model_info[model_id]["name"]
            short_name = base_name.split("/")[-1].split(":")[0].replace("-instruct", "").replace("-", " ").title()
            agent_labels.append(short_name)
        else:
            agent_labels.append("Agent Not Available")

    return {
        output_aggregator: gr.update(label=agg_label),
        output_model1: gr.update(label=agent_labels[0]),
        output_model2: gr.update(label=agent_labels[1]),
        output_model3: gr.update(label=agent_labels[2]),
    }

def process_query(query, api_key, question_type, domain, aggregator_id, progress=gr.Progress()):
    """Process a query and wait for results"""
    global current_job_id
    consensus_score = 0  # Ensure it's always defined

    # Check for required inputs
    if not api_key:
        return [
            "Error: OpenRouter API key is required", "", "", "", 0,
            None, None, None, None,
            gr.update(value="", visible=False),
            gr.update(label="Aggregator"),
            gr.update(label="Model 1"),
            gr.update(label="Model 2"),
            gr.update(label="Model 3")
        ]


    if not query.strip():
        return [
            "Please enter a query", "", "", "", 0,
            None, None, None, None,
            gr.update(value="", visible=False),
            gr.update(label="Aggregator"),
            gr.update(label="Model 1"),
            gr.update(label="Model 2"),
            gr.update(label="Model 3")
        ]

    try:
        progress(0.05, desc="Sending request to backend...")

        # Submit query to backend
        response = requests.post(
            f"{BACKEND_URL}/process_query",
            json={
                "query": query,
                "api_key": api_key,
                "question_type": question_type,
                "domain": domain,
                "aggregator_id": aggregator_id
            }
        )

        if response.status_code != 200:
            logger.error(f"Failed to process query: {response.status_code}")
            return [
                f"Error: Backend returned status code {response.status_code}", "", "", "", 0, None, None, None, None
            ]

        # Get job ID
        job_data = response.json()
        current_job_id = job_data.get("job_id")

        if not current_job_id:
            logger.error("No job ID returned from backend")
            return [
                "Error: No job ID returned from backend", "", "", "", 0, None, None, None, None
            ]

        # Poll for job status
        status = "processing"
        poll_count = 0
        max_polls = 120

        progress(0.1, desc="Models generating responses...")

        while status == "processing" and poll_count < max_polls:
            time.sleep(1)
            poll_count += 1

            status_response = requests.get(f"{BACKEND_URL}/job_status/{current_job_id}")
            if status_response.status_code != 200:
                logger.error(f"Failed to get job status: {status_response.status_code}")
                continue

            status_data = status_response.json()
            status = status_data.get("status", "processing")
            current_progress = status_data.get("progress", 0)
            progress(0.1 + (current_progress / 100) * 0.8, desc=f"Processing... {current_progress}%")

        if status != "completed":
            return [
                "Error: Backend processing timed out or failed", "", "", "", 0,
                None, None, None, None,
                gr.update(value="", visible=False),  # plot warning
                gr.update(label="Aggregator"),
                gr.update(label="Model 1"),
                gr.update(label="Model 2"),
                gr.update(label="Model 3")
            ]

        # Get job result
        progress(0.95, desc="Retrieving results...")
        result_response = requests.get(f"{BACKEND_URL}/job_result/{current_job_id}")

        if result_response.status_code != 200:
            logger.error(f"Failed to get job result: {result_response.status_code}")
            return [
                f"Error: Failed to retrieve results (status code {result_response.status_code})", "", "", "", 0, None, None, None, None
            ]

        # Parse result
        result = result_response.json()
        responses = result.get("responses", {})
        analysis = result.get("analysis", {})
        warning_msg = analysis.get("warning", "")
        consensus_score = result.get("consensus_score", 0)
        aggregator_id = result.get("aggregator_id")
        label_updates = update_output_labels(aggregator_id)

        # Get non-aggregator agents from the response
        agent_ids = [mid for mid in responses.keys() if mid != aggregator_id]
        agent_boxes = []

        # Prepare 3 output boxes: either real response or fallback text
        for i in range(3):
            if i < len(agent_ids):
                agent_id = agent_ids[i]
                response_text = responses.get(agent_id, f"No response from {agent_id}")
                label = model_info.get(agent_id, {}).get("display_name", agent_id)
            else:
                response_text = "No agent available"
                label = "Agent Not Available"
            agent_boxes.append((label, response_text))

        # Get non-aggregator models
        non_aggregator_models = [model_id for model_id, info in model_info.items()
                                 if not info.get("aggregator", False)]

        def fetch_image(url):
            try:
                res = requests.get(url)
                if res.status_code == 200:
                    return Image.open(BytesIO(res.content))
            except Exception as e:
                logger.warning(f"Failed to fetch image from {url}: {e}")
            return None

        # Get images
        heatmap_url = f"{BACKEND_URL}/image/{current_job_id}/heatmap"
        emotion_url = f"{BACKEND_URL}/image/{current_job_id}/emotion_chart"
        polarity_url = f"{BACKEND_URL}/image/{current_job_id}/polarity_chart"
        radar_url = f"{BACKEND_URL}/image/{current_job_id}/radar_chart"

        heatmap_url = fetch_image(f"{BACKEND_URL}/image/{current_job_id}/heatmap")
        emotion_url = fetch_image(f"{BACKEND_URL}/image/{current_job_id}/emotion_chart")
        polarity_url = fetch_image(f"{BACKEND_URL}/image/{current_job_id}/polarity_chart")
        radar_url = fetch_image(f"{BACKEND_URL}/image/{current_job_id}/radar_chart")

        progress(1.0, desc="Complete!")

        return [
            responses.get(aggregator_id, "Error: Aggregator model failed to respond"),
            agent_boxes[0][1],
            agent_boxes[1][1],
            agent_boxes[2][1],
            consensus_score,
            heatmap_url if "error" not in analysis else None,
            emotion_url if "error" not in analysis else None,
            polarity_url if "error" not in analysis else None,
            radar_url if "error" not in analysis else None,
            gr.update(value=warning_msg, visible=bool(warning_msg)),
            label_updates[output_aggregator],
            label_updates[output_model1],
            label_updates[output_model2],
            label_updates[output_model3],
        ]


    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return [
            f"Error: {str(e)}", "", "", "", 0, None, None, None, None
        ]

# Create the Gradio interface
def create_gradio_interface():
    """Create the Gradio interface"""
    global output_aggregator, output_model1, output_model2, output_model3, consensus_score
    global output_heatmap, output_emotion, output_polarity, output_radar
    global plot_warning_box

    # Initialize client
    init_success = init_client()
    if not init_success:
        raise RuntimeError(f"Failed to connect to backend at {BACKEND_URL} after 3 attempts. Please check the server and try again.")
    
    with gr.Blocks(title="Multi-Agent LLM System") as app:
        gr.Markdown(
            "<h1 style='text-align: center;'>Distributed Multi-Agent LLM System</h1>",
            elem_id="title"
        )
        
        with gr.Row():
            with gr.Column():
                api_key_input = gr.Textbox(
                    label="OpenRouter API Key",
                    placeholder="Enter your OpenRouter API key",
                    value=API_KEY,
                    type="password"
                )
                
                # Add backend URL input
                backend_url_input = gr.Textbox(
                    label="Backend URL",
                    placeholder="Enter backend server URL",
                    value=BACKEND_URL
                )
                
                # Add connection test button
                with gr.Column():
                    connect_button = gr.Button("Test Connection")
                    connection_status = gr.Textbox(label="Connection Status", interactive=False)
                
                # Add aggregator selection radio button
                aggregator_options = list(model_info.keys()) if model_info else []
                default_aggregator = next((model_id for model_id, info in model_info.items() 
                                        if info.get("aggregator", False)), None) if model_info else None
                
                aggregator_radio = gr.Radio(
                    choices=aggregator_options,
                    label="Select Aggregator Model",
                    value=default_aggregator
                )

                domain_options = list(domains.keys()) if domains else []
                domain_radio = gr.Radio(
                    choices=domain_options, 
                    label="Select Domain Expertise", 
                    value="Custom" if "Custom" in domain_options else None
                )
                
                question_type_options = question_types if question_types else []
                question_type = gr.Radio(
                    choices=question_type_options,
                    label="Question Type",
                    value="None" if "None" in question_type_options else None
                )
                
                input_query = gr.Textbox(
                    label="Your Query",
                    placeholder="Enter your question or prompt",
                    lines=3
                )

                example_options = examples_by_domain.get("Custom", []) if examples_by_domain else []
                example_selector = gr.Dropdown(
                    choices=[ex[0] for ex in example_options] if example_options else [],
                    label="Choose an Example (Optional)"
                )
                
                submit_btn = gr.Button("Submit", variant="primary")
        
        with gr.Tabs():
            with gr.TabItem("Model Responses"):
                # Get current aggregator model ID
                current_aggregator_id = default_aggregator
                
                # Get non-aggregator model IDs
                non_aggregator_models = [model_id for model_id, info in model_info.items() 
                                       if not info.get("aggregator", False)] if model_info else []
                
                # Full-width aggregator response
                with gr.Row():
                    output_aggregator = gr.Textbox(
                        label=model_info[current_aggregator_id]["display_name"] if model_info and current_aggregator_id in model_info else "Consensus Summary",
                        lines=10,
                        max_lines=10
                    )
                
                # Three equal columns for agent models
                with gr.Row():
                    output_model1 = gr.Textbox(
                        label=model_info[non_aggregator_models[0]]["display_name"] if model_info and len(non_aggregator_models) > 0 else "Model 1", 
                        lines=8,
                        max_lines=8
                    )
                    output_model2 = gr.Textbox(
                        label=model_info[non_aggregator_models[1]]["display_name"] if model_info and len(non_aggregator_models) > 1 else "Model 2",
                        lines=8,
                        max_lines=8
                    )
                    output_model3 = gr.Textbox(
                        label=model_info[non_aggregator_models[2]]["display_name"] if model_info and len(non_aggregator_models) > 2 else "Model 3",
                        lines=8,
                        max_lines=8
                    )
            
            with gr.TabItem("Analysis Visualizations"):
                with gr.Row():
                    plot_warning_box = gr.Markdown("", visible=False)
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

            with gr.TabItem("Interaction History"):
                with gr.Row():
                    refresh_history_btn = gr.Button("🔄 Refresh History")
                
                # Use Gradio Dataframe with appropriate columns
                history_list = gr.Dataframe(
                    headers=["Time", "Query", "Domain", "Question Type", "Consensus"],
                    datatype=["str", "str", "str", "str", "number"],
                    interactive=False,
                    row_count=10
                )

                # Hidden state to store job IDs
                history_job_ids = gr.State([])
                
                def refresh_history():
                    try:
                        response = requests.get(f"{BACKEND_URL}/history?limit=100")
                        if response.status_code != 200:
                            return [], []
                        
                        history = response.json().get("history", [])
                        if not history:
                            return [], []
                        
                        rows = []
                        job_ids = []
                        
                        for item in history:
                            timestamp = datetime.fromtimestamp(item["timestamp"]).strftime("%Y-%m-%d %H:%M")
                            query = item["query"]
                            preview = query[:100] + "..." if len(query) > 100 else query
                            domain = item.get("domain", "Custom")
                            question_type = item.get("question_type", "None")
                            
                            # Get aggregator response preview - improved handling
                            responses = item.get("responses", {})
                            aggregator_id = item.get("aggregator_id", "")
                            
                            # If aggregator_id is empty or not in responses, try to find an aggregator response
                            if not aggregator_id or aggregator_id not in responses:
                                # Try to identify an aggregator from the model_info
                                for model_id, info in model_info.items():
                                    if info.get("aggregator", False) and model_id in responses:
                                        aggregator_id = model_id
                                        break
                            
                            aggregator_response = responses.get(aggregator_id, "")
                            if not aggregator_response and responses:
                                # If still no aggregator response but we have responses, use the first one
                                aggregator_response = next(iter(responses.values()), "")
                                                            
                            consensus = item.get("consensus_score", 0)
                            job_id = item["job_id"]
                            
                            rows.append([timestamp, preview, domain, question_type, consensus])
                            job_ids.append(job_id)
                        
                        return rows, job_ids
                    except Exception as e:
                        logger.error(f"Error fetching history: {str(e)}")
                        return [], []
                
                # Selection indication
                with gr.Row():
                    gr.Markdown("First, select a row by clicking on it, then use these buttons:")
                
                # Add explicit buttons with selected row indicator
                with gr.Row():
                    selected_row_info = gr.Markdown("No row selected")
                    selected_row_idx = gr.State(-1)
                
                with gr.Row():
                    load_btn = gr.Button("📥 Load Selected Entry")
                    delete_btn = gr.Button("🗑️ Delete Selected Entry")
                
                # Update selected row when user clicks
                def update_selected_row(evt: gr.SelectData, current_idx):
                    row_idx = evt.index[0]
                    if row_idx == current_idx:  # If clicking already selected row
                        return -1, "No row selected"
                    return row_idx, f"Selected row: {row_idx + 1}"

                # Connect selection event
                history_list.select(
                    fn=update_selected_row,
                    inputs=[selected_row_idx],  # Pass current index
                    outputs=[selected_row_idx, selected_row_info]
                )
                
                # Load selected row
                def load_selected_row(row_idx, job_ids):
                    if row_idx < 0 or row_idx >= len(job_ids):
                        return [None] * 14  # Include aggregator_radio in the count (now 14 total outputs)
                    
                    try:
                        job_id = job_ids[row_idx]
                        
                        # Get full job details
                        result_response = requests.get(f"{BACKEND_URL}/job_result/{job_id}")
                        
                        if result_response.status_code == 404:
                            logger.warning(f"Job {job_id} not found. It may have been deleted.")
                            return ["Job not found - refresh history", gr.update(value="Custom"), gr.update(value="None"), None, "", "", "", "", 0, None, None, None, None, gr.update(value=None)]

                        if result_response.status_code == 200:
                            result = result_response.json()
                            
                            # Get responses
                            responses = result.get("responses", {})
                            aggregator_id = result.get("aggregator_id", "")
                            
                            # Get non-aggregator model IDs
                            non_aggregator_models = [model_id for model_id, info in model_info.items() 
                                                if not info.get("aggregator", False)] if model_info else []
                            
                            consensus_score_value = result.get("consensus_score", 0)
                            
                            # Get query and metadata
                            query = result.get("query", "")
                            domain = result.get("domain", "Custom")
                            q_type = result.get("question_type", "None")

                            # Define fetch_image function locally if not already defined
                            def fetch_image(url):
                                try:
                                    res = requests.get(url)
                                    if res.status_code == 200:
                                        return Image.open(BytesIO(res.content))
                                except Exception as e:
                                    logger.warning(f"Failed to fetch image from {url}: {e}")
                                return None
                            
                            # Prepare return values
                            return [
                                gr.update(value=query), 
                                domain,  # Force domain update using gr.update()
                                q_type,  # Force question type update
                                None,  # Clear example dropdown
                                responses.get(aggregator_id, ""),
                                responses.get(non_aggregator_models[0], "") if len(non_aggregator_models) > 0 else "",
                                responses.get(non_aggregator_models[1], "") if len(non_aggregator_models) > 1 else "",
                                responses.get(non_aggregator_models[2], "") if len(non_aggregator_models) > 2 else "",
                                consensus_score_value,
                                fetch_image(f"{BACKEND_URL}/image/{job_id}/heatmap"),
                                fetch_image(f"{BACKEND_URL}/image/{job_id}/emotion_chart"),
                                fetch_image(f"{BACKEND_URL}/image/{job_id}/polarity_chart"),
                                fetch_image(f"{BACKEND_URL}/image/{job_id}/radar_chart"),  # Re-added radar chart
                                aggregator_id  # Force aggregator update
                            ]
                        elif result_response.status_code == 404:
                            logger.warning(f"Job {job_id} not found. It may have been deleted.")
                            # Refresh the history to remove stale entries
                            refresh_history()
                            return [None] * 14
                        else:
                            logger.error(f"Failed to fetch job {job_id}: {result_response.status_code}")
                            return [None] * 14
                    except Exception as e:
                        logger.error(f"Error loading job: {str(e)}")
                        return [None] * 14
                
                # Delete selected row
                def delete_selected_row(row_idx, job_ids):
                    if row_idx < 0 or row_idx >= len(job_ids):
                        return gr.update(), gr.update(), "No row selected"
                    
                    try:
                        job_id = job_ids[row_idx]
                        
                        # Delete the job
                        delete_response = requests.delete(f"{BACKEND_URL}/history/{job_id}")
                        if delete_response.status_code == 200:
                            logger.info(f"Deleted job {job_id}")
                            # Refresh the history
                            rows, job_ids = refresh_history()
                            return rows, job_ids, "Row deleted successfully"
                        else:
                            logger.error(f"Failed to delete job {job_id}: {delete_response.status_code}")
                            return gr.update(), gr.update(), f"Failed to delete: {delete_response.status_code}"
                    except Exception as e:
                        logger.error(f"Error deleting job: {str(e)}")
                        return gr.update(), gr.update(), f"Error: {str(e)}"
                
                # Connect buttons
                load_btn.click(
                    fn=load_selected_row,
                    inputs=[selected_row_idx, history_job_ids],
                    outputs=[
                        input_query, domain_radio, question_type, example_selector,
                        output_aggregator, output_model1, output_model2, output_model3,
                        consensus_score, output_heatmap, output_emotion, output_polarity,
                        output_radar, aggregator_radio
                    ]
                )
                
                delete_btn.click(
                    fn=delete_selected_row,
                    inputs=[selected_row_idx, history_job_ids],
                    outputs=[history_list, history_job_ids, selected_row_info]
                )
                
                # Connect refresh button
                refresh_history_btn.click(
                    fn=refresh_history,
                    inputs=None,
                    outputs=[history_list, history_job_ids]
                )
                
                # Load history on startup
                app.load(fn=refresh_history, inputs=None, outputs=[history_list, history_job_ids])

        # Connect the domain radio button to update example choices
        domain_radio.change(fn=get_example_choices, inputs=domain_radio, outputs=example_selector)
        
        # Connect the example selector to fill query and type
        example_selector.change(fn=fill_query_and_type, inputs=[example_selector, domain_radio], outputs=[input_query, question_type])
        
        # Connect backend URL input to update the backend URL
        def update_backend_url(url):
            global BACKEND_URL
            BACKEND_URL = url
            return url
        
        backend_url_input.change(fn=update_backend_url, inputs=backend_url_input, outputs=backend_url_input)
        
        # Connect the connect button to test the connection
        def test_connection(url):
            try:
                # Test basic connectivity
                response = requests.get(f"{url}/")
                if response.status_code != 200:
                    return f"Failed to connect to backend: Status code {response.status_code}"
                
                # Test models endpoint
                models_response = requests.get(f"{url}/models")
                if models_response.status_code != 200:
                    return "Connected to backend, but models endpoint failed."
                
                # Test domains endpoint
                domains_response = requests.get(f"{url}/domains")
                if domains_response.status_code != 200:
                    return "Connected to backend, but domains endpoint failed."
                    
                # Test history endpoint
                history_response = requests.get(f"{url}/history?limit=1")
                if history_response.status_code != 200:
                    return "Connected to backend, but history endpoint failed."
                
                # If all tests pass, refresh the client data
                global model_info, domains, question_types, examples_by_domain
                init_success = init_client()
                if init_success:
                    return "Connected successfully! All endpoints validated."
                else:
                    return "Connected to backend, but failed to update client information."
            except Exception as e:
                return f"Error connecting to backend: {str(e)}"
        
        connect_button.click(fn=test_connection, inputs=backend_url_input, outputs=connection_status)
        
        # Connect the radio button to update the aggregator
        aggregator_radio.change(
            fn=update_aggregator,
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
            inputs=[input_query, api_key_input, question_type, domain_radio, aggregator_radio],
            outputs=[
                output_aggregator,
                output_model1,
                output_model2,
                output_model3,
                consensus_score,
                output_heatmap,
                output_emotion,
                output_polarity,
                output_radar,
                plot_warning_box,
                output_aggregator,
                output_model1,
                output_model2,
                output_model3
            ]

        )
                    
    return app

# Create .env file template
def create_env_template():
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("""# OpenRouter API Configuration
OPENROUTER_API_KEY=
BACKEND_URL=http://localhost:8000
""")

# Main function to run the system
def main():
    global BACKEND_URL  # Declare this before using BACKEND_URL anywhere

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Multi-Agent LLM Client")
    parser.add_argument("--backend-url", type=str, default=BACKEND_URL, help="URL of the backend server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for Gradio client")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the Gradio interface on")
    parser.add_argument("--share", action="store_true", help="Create a shareable link")
    
    args = parser.parse_args()
    
    # Update backend URL
    BACKEND_URL = args.backend_url

    # Create .env template
    create_env_template()
    
    # Create and launch Gradio app
    app = create_gradio_interface()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)

if __name__ == "__main__":
    main()