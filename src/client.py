import gradio as gr
import requests
import time
import argparse
import os
from datetime import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO

from src.database_manager import DatabaseManager  # works when run from root     # works when run from inside src/

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
    """
    Initializes the client by attempting to connect to the backend and fetching necessary data.

    This function tries to establish a connection to the backend server and retrieve information
    such as model details, domains, question types, and examples. It retries the connection
    a specified number of times with a delay between attempts if the connection fails.

    Args:
        retries (int): The maximum number of connection attempts. Defaults to 3.
        delay (float): The delay (in seconds) between consecutive connection attempts. Defaults to 1.5.

    Returns:
        bool: True if the connection and data retrieval are successful, False otherwise.

    Raises:
        None: All exceptions are caught and logged, and the function will return False
        if the connection fails after the specified number of retries.

    Global Variables:
        model_info (dict): Stores information about models retrieved from the backend.
        domains (dict): Stores domain data retrieved from the backend.
        question_types (list): Stores question types retrieved from the backend.
        examples_by_domain (dict): Stores examples categorized by domain retrieved from the backend.

    """
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
    """
    Updates the aggregator model on the backend server.

    This function sends a POST request to the backend server to update the 
    aggregator model identified by the given `aggregator_id`. If the request 
    is successful, the updated aggregator ID is returned. Otherwise, an error 
    is logged, and `None` is returned.

    Args:
        aggregator_id (str): The unique identifier of the aggregator to update.

    Returns:
        str or None: The updated aggregator ID if the request is successful, 
        or `None` if the request fails or an exception occurs.

    Raises:
        None: Any exceptions are caught and logged.
    """
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
    """
    Fetches example choices for a given domain from a backend service.
    Args:
        domain (str): The domain for which to fetch example choices. 
                      If the domain is None or empty, an empty list of choices is returned.
    Returns:
        gr.update: An object containing the updated choices and value. 
                   - If successful, the choices are populated with examples fetched from the backend.
                   - If the request fails or an error occurs, the choices are empty.
    Raises:
        Logs errors if the backend request fails or an exception occurs during execution.
    """
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
    """
    Sends a POST request to the backend to fill in a query and determine the question type
    based on the provided example and domain.

    Args:
        selected_example (str): The selected example to be processed.
        domain (str): The domain context for the query.

    Returns:
        tuple: A tuple containing:
            - query (str): The filled query string. Returns an empty string if an error occurs.
            - question_type (str): The type of the question. Defaults to "None" if not determined.
    """
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
    """
    Updates the output labels for the aggregator and agent models based on the 
    provided aggregator ID and the model information retrieved from the backend.

    Args:
        aggregator_id (str): The ID of the aggregator whose labels need to be updated.

    Returns:
        dict: A dictionary containing updated labels for the aggregator and up to 
              three agent models. The keys correspond to the output components 
              (e.g., `output_aggregator`, `output_model1`, `output_model2`, `output_model3`), 
              and the values are `gr.update` objects with the new labels.
    """
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

def process_query(query, api_key, question_type, domain, aggregator_id, ethical_views, session, progress=gr.Progress()):
    """
    Processes a user query by sending it to a backend service, polling for the job status, 
    and retrieving the results, including responses from models, analysis, and visualizations.

    Args:
        query (str): The user query to be processed.
        api_key (str): The API key for authentication with the backend.
        question_type (str): The type of question being asked.
        domain (str): The domain or context of the query.
        aggregator_id (str): The ID of the aggregator model to be used.
        session (str): The session or username for tracking the request.
        progress (gr.Progress, optional): A progress tracker for updating the UI.

    Returns:
        list: Contains:
            - str: Aggregator model response or error message.
            - str: Responses from up to three non-aggregator models.
            - float: Consensus score.
            - Image or None: Visualizations (heatmap, emotion, polarity, radar).
            - gr.update: Updates for warnings and model labels.

    Raises:
        Exception: For errors during query processing, including network or backend issues.
    """
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
                "aggregator_id": aggregator_id,
                "username": session,
                "ethical_views": ethical_views
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
            """
            Fetches an image from the specified URL.

            This function sends a GET request to the provided URL and attempts to 
            retrieve an image. If the request is successful and the response status 
            code is 200, the image is returned as a PIL Image object. If the request 
            fails or an exception occurs, a warning is logged, and None is returned.

            Args:
                url (str): The URL of the image to fetch.

            Returns:
                PIL.Image.Image or None: The fetched image as a PIL Image object if 
                successful, otherwise None.
            """
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
    """
    Creates a Gradio interface for a Distributed Multi-Agent LLM System.
    This function initializes the Gradio application with multiple interactive components,
    including login/signup functionality, query submission, model response display, 
    analysis visualizations, and interaction history management. It also handles user 
    authentication, session management, and backend communication for processing queries 
    and retrieving historical data.

    Returns:
            gr.Blocks: The Gradio Blocks application instance.
    """
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

        # Login/Signup UI block
        session = gr.State(value=None)  # to store current user session

        with gr.Row():
            with gr.Column():
                auth_toggle = gr.Radio(choices=["Login", "Signup"], value="Login", label="Choose Action")
                username_input = gr.Textbox(label="Username")
                password_input = gr.Textbox(
                    label="Password",
                    type="password"
                )

                show_password_checkbox = gr.Checkbox(
                    label="👁 Show Password",
                    value=False
                )

                login_button = gr.Button("Login", visible=True)
                signup_button = gr.Button("Create Account", visible=False)
                with gr.Row():
                    logout_button = gr.Button("Logout", visible=False)
                    delete_account_button = gr.Button("Delete Account", visible=False)
                login_status = gr.Markdown("Not logged in")
        
                delete_password_input = gr.Textbox(
                    label="Confirm your password to delete account",
                    type="password",
                    visible=False
                )
                show_delete_password_checkbox = gr.Checkbox(
                    label="👁 Show Password",
                    value=False,
                    visible=False
                )
                confirm_delete_button = gr.Button("Confirm Deletion", visible=False)

        def toggle_auth_mode(mode, session):
            """
            Toggles the visibility of authentication UI components based on the current mode 
            ("Signup" or "Login") and session status.

            Args:
                mode (str): The current authentication mode, either "Signup" or "Login".
                session (bool): Indicates whether a user session is active (True) or not (False).

            Returns:
                tuple: A pair of `gr.update` objects that control the visibility of the 
                       authentication UI components. The first element corresponds to the 
                       "Login" visibility, and the second corresponds to the "Signup" visibility.
            """
            if session:
                # Already logged in, don't toggle anything
                return gr.update(visible=False), gr.update(visible=False)
            if mode == "Signup":
                return gr.update(visible=False), gr.update(visible=True)
            else:
                return gr.update(visible=True), gr.update(visible=False)


        auth_toggle.change(
            fn=toggle_auth_mode,
            inputs=[auth_toggle, session],
            outputs=[login_button, signup_button]
        )

        with gr.Row():
            with gr.Column():
                api_key_input = gr.Textbox(
                    label="OpenRouter API Key",
                    placeholder="Enter your OpenRouter API key",
                    value=API_KEY,
                    type="password"
                )
                
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

                ethical_view_selector = gr.CheckboxGroup(
                    choices=[
                        "None",
                        "Utilitarian",
                        "Deontologist",
                        "Virtue Ethicist",
                        "Libertarian",
                        "Rawlsian",
                        "Precautionary"
                    ],
                    value=["None"],
                    label="Choose Ethical View(s) for Agents",
                    info="Select exactly 1 or 3 perspectives, or choose 'None' to skip ethics."
                )

                ethical_warning_box = gr.Markdown("", visible=False)
                
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
                
                def refresh_history(session_user):
                    """
                    Fetches and processes the history of user queries from the backend.
                    Args:
                        session_user (str): The username of the session user whose history is to be fetched.
                    Returns:
                        tuple: A tuple containing:
                            - rows (list): A list of lists where each inner list represents a row with the following details:
                                - timestamp (str): The formatted timestamp of the query.
                                - preview (str): A preview of the query (truncated to 100 characters if necessary).
                                - domain (str): The domain of the query (default is "Custom").
                                - question_type (str): The type of question (default is "None").
                                - consensus (float): The consensus score of the query.
                            - job_ids (list): A list of job IDs corresponding to the queries.
                    """
                    try:
                        response = requests.get(f"{BACKEND_URL}/history?username={session_user}&limit=100")
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
                
                def select_history_row(evt: gr.SelectData):
                    """
                    Handles the selection of a row in a history table.

                    Args:
                        evt (gr.SelectData): The event data containing information about the selected row.

                    Returns:
                        tuple: A tuple containing the index of the selected row (int) and a string message
                               indicating the selected row number (1-based index).
                    """
                    row_idx = evt.index[0]
                    return row_idx, f"Selected row: {row_idx + 1}"

                # Connect selection event
                history_list.select(
                    fn=select_history_row,
                    outputs=[selected_row_idx, selected_row_info]
                )
                
                # Load selected row
                def load_selected_row(row_idx, job_ids):
                    """
                    Loads the details of a selected job row based on the provided row index and job IDs.
                    Args:
                        row_idx (int): The index of the selected row in the job list.
                        job_ids (list): A list of job IDs corresponding to the rows.
                    Returns:
                        list: A list of 14 elements containing the following:
                            - gr.update(value=query): The query string associated with the job.
                            - domain (str): The domain of the query (e.g., "Custom").
                            - q_type (str): The question type (e.g., "None").
                            - None: Placeholder to clear the example dropdown.
                            - str: The response from the aggregator model.
                            - str: The response from the first non-aggregator model (if available).
                            - str: The response from the second non-aggregator model (if available).
                            - str: The response from the third non-aggregator model (if available).
                            - float: The consensus score value.
                            - Image or None: The heatmap image fetched from the backend.
                            - Image or None: The emotion chart image fetched from the backend.
                            - Image or None: The polarity chart image fetched from the backend.
                            - Image or None: The radar chart image fetched from the backend.
                            - str: The aggregator ID.
                    Raises:
                        Exception: Logs any unexpected errors encountered during the process.
                    """
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
                                """
                                Fetches an image from the given URL.

                                This function sends a GET request to the specified URL and attempts to 
                                retrieve an image. If the request is successful and the response status 
                                code is 200, the image is returned as a PIL Image object. If the request 
                                fails or an exception occurs, a warning is logged, and None is returned.

                                Args:
                                    url (str): The URL of the image to fetch.

                                Returns:
                                    PIL.Image.Image or None: The fetched image as a PIL Image object if 
                                    successful, otherwise None.
                                """
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
                def delete_selected_row(row_idx, job_ids, session_user):
                    """
                    Deletes a selected row from the job history based on the provided row index.
                    Args:
                        row_idx (int): The index of the row to delete. Must be within the range of `job_ids`.
                        job_ids (list): A list of job IDs corresponding to the rows in the job history.
                        session_user (str): The username of the current session user.
                    Returns:
                        tuple: A tuple containing:
                            - Updated rows (or `gr.update()` if no update is performed).
                            - Updated job IDs (or `gr.update()` if no update is performed).
                            - A message string indicating the result of the operation.
                    Raises:
                        None: Any exceptions encountered during the operation are caught and logged.
                    """
                    if row_idx < 0 or row_idx >= len(job_ids):
                        return gr.update(), gr.update(), "No row selected"
                    
                    try:
                        job_id = job_ids[row_idx]
                        
                        # Delete the job
                        delete_response = requests.delete(f"{BACKEND_URL}/history/{job_id}?username={session_user}")
                        if delete_response.status_code == 200:
                            logger.info(f"Deleted job {job_id}")
                            # Refresh the history
                            rows, job_ids = refresh_history(session_user)
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
                ).then(fn=lambda: (-1, "No row selected"), outputs=[selected_row_idx, selected_row_info])
                
                delete_btn.click(
                    fn=delete_selected_row,
                    inputs=[selected_row_idx, history_job_ids, session],
                    outputs=[history_list, history_job_ids, selected_row_info]
                ).then(fn=lambda: (-1, "No row selected"), outputs=[selected_row_idx, selected_row_info])
                
                # Connect refresh button
                refresh_history_btn.click(
                    fn=refresh_history,
                    inputs=[session],
                    outputs=[history_list, history_job_ids]
                )

                def is_valid_password(password):
                    """
                    Validates whether a given password meets the following criteria:
                    - Contains at least 8 characters.
                    - Includes at least one uppercase letter.
                    - Includes at least one lowercase letter.
                    - Includes at least one numeric digit.

                    Args:
                        password (str): The password string to validate.

                    Returns:
                        bool: True if the password meets all the criteria, False otherwise.
                    """
                    return (
                        len(password) >= 8 and
                        any(c.isupper() for c in password) and
                        any(c.islower() for c in password) and
                        any(c.isdigit() for c in password)
                    )

                def login_user(username, password):
                    """
                    Authenticates a user based on the provided username and password.

                    Args:
                        username (str): The username of the user attempting to log in.
                        password (str): The password associated with the username.

                    Returns:
                        tuple: Contains the username (or None) and UI component updates.
                    """
                    db = DatabaseManager()
                    if db.verify_user(username, password):
                        return (
                            username,
                            gr.update(visible=False),  # login button
                            gr.update(visible=False),  # signup button
                            gr.update(visible=True),   # logout button
                            gr.update(visible=True),   # delete button
                            gr.update(visible=False),  # password field
                            gr.update(visible=False),  # confirm delete button
                            gr.update(value=f"✅ Logged in as **{username}**"),
                            gr.update(value=False),  # reset checkbox
                            gr.update(type="password")
                        )
                    else:
                        return (
                            None,
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(value="⚠️ Invalid username or password."),
                            gr.update(value=False),  # reset checkbox
                            gr.update(type="password")
                        )

                def signup_user(username, password):
                    """
                    Handles the signup process for a new user.

                    Args:
                        username (str): Desired username for the new account.
                        password (str): Password for the new account.

                    Returns:
                        tuple: Updates UI components and displays signup result.
                    """
                    db = DatabaseManager()

                    if not is_valid_password(password):
                        return (
                            None,
                            gr.update(visible=False), gr.update(visible=False),  # hide both
                            gr.update(visible=False), gr.update(visible=False),
                            gr.update(visible=False), gr.update(visible=False),
                            gr.update(value="❌ Password must be at least 8 characters, include a lowercase, uppercase letter, and a number.")
                        )

                    result = db.create_user(username, password)

                    if result is True:
                        return (
                            username,
                            gr.update(visible=False), gr.update(visible=False),
                            gr.update(visible=True), gr.update(visible=True),
                            gr.update(visible=False), gr.update(visible=False),
                            gr.update(value=f"✅ **Account created! Welcome, {username}**"),
                            gr.update(value=False),  # reset checkbox
                            gr.update(type="password")
                        )
                    elif result == "duplicate":
                        return (
                            None,
                            gr.update(visible=False), gr.update(visible=False),  # hide both
                            gr.update(visible=False), gr.update(visible=False),
                            gr.update(visible=False), gr.update(visible=False),
                            gr.update(value="⚠️ **The username already exists.**"),
                            gr.update(value=False),  # reset checkbox
                            gr.update(type="password")
                        )
                    else:
                        return (
                            None,
                            gr.update(visible=False), gr.update(visible=False),  # hide both
                            gr.update(visible=False), gr.update(visible=False),
                            gr.update(visible=False), gr.update(visible=False),
                            gr.update(value="❌ **Signup failed due to a server error.**"),
                            gr.update(value=False),  # reset checkbox
                            gr.update(type="password")
                        )

                def logout_user():
                    """
                    Logs out the user and updates the UI components to reflect the logged-out state.

                    Returns:
                        tuple: Updates for UI components to reset to the logged-out state.
                    """
                    return (
                        None,
                        gr.update(visible=False),   # login
                        gr.update(visible=False),   # signup
                        gr.update(visible=False),  # logout
                        gr.update(visible=False),  # delete
                        gr.update(visible=False),  # password field
                        gr.update(visible=False),  # confirm delete
                        gr.update(value="✅ **You have been logged out successfully.**"),
                        gr.update(value=False, visible=False),  # hide + uncheck the checkbox
                        gr.update(value=False),  # reset checkbox
                        gr.update(type="password")
                    )

                def delete_account(username, password):
                    """
                    Deletes a user account from the database if the provided credentials are valid.

                    Args:
                        username (str): The username of the account to be deleted.
                        password (str): The password associated with the account.

                    Returns:
                        tuple: A tuple containing updates for the UI components based on the success or failure of the operation.
                    """
                    db = DatabaseManager()
                    if db.delete_user(username, password):
                        return (
                            None,  # clear session
                            gr.update(visible=True),   # login
                            gr.update(visible=True),   # signup
                            gr.update(visible=False),  # logout
                            gr.update(visible=False),  # delete account
                            gr.update(value=""),       # clear password field
                            gr.update(visible=False),  # hide confirm delete
                            gr.update(value="✅ Account deleted successfully."),
                            gr.update(value=False, visible=False)  # hide + uncheck the checkbox
                        )
                    else:
                        return (
                            username,  # session still active
                            gr.update(visible=False), gr.update(visible=False),  # hide login/signup
                            gr.update(visible=True), gr.update(visible=True),    # keep logout + delete
                            gr.update(value=""),       # clear password input
                            gr.update(visible=False),  # hide confirm delete (to prevent retry loop)
                            gr.update(value="❌ Invalid password. Account not deleted.")

                        )

                def show_delete_inputs():
                    """
                    Updates the visibility of UI components related to the delete operation.

                    This function returns a tuple of `gr.update` calls that make the following
                    components visible:
                    - `delete_password_input`: Input field for entering the delete password.
                    - `confirm_delete_button`: Button to confirm the delete action.
                    - `show_delete_password_checkbox`: Checkbox to toggle the visibility of the delete password.

                    Returns:
                        tuple: A tuple containing `gr.update` objects to set the visibility of
                        the delete-related UI components to `True`.
                    """
                    return (
                        gr.update(visible=True),   # delete_password_input
                        gr.update(visible=True),   # confirm_delete_button
                        gr.update(visible=True)    # show_delete_password_checkbox
                    )
                
                def toggle_password_visibility(show):
                    """
                    Toggles the visibility of a password input field.

                    Args:
                        show (bool): A boolean indicating whether to show the password. 
                                     If True, the password will be displayed as plain text. 
                                     If False, the password will be hidden.

                    Returns:
                        dict: A dictionary containing the updated type for the input field, 
                              either "text" for visible or "password" for hidden.
                    """
                    return gr.update(type="text" if show else "password")

                def validate_ethical_selection(selected):
                    """
                    Validates the user's selection of ethical perspectives.

                    This function ensures that the selection adheres to the following rules:
                    1. The option "None" cannot be selected alongside other ethical perspectives.
                    2. If "None" is not selected, the user must select either exactly 1 or exactly 3 ethical perspectives.

                    Args:
                        selected (list): A list of strings representing the selected ethical perspectives.

                    Returns:
                        gr.update: An update object for the UI, containing:
                            - value (str): A warning message if the selection is invalid, or an empty string if valid.
                            - visible (bool): A flag indicating whether the warning message should be displayed.
                    """
                    if "None" in selected and len(selected) > 1:
                        return gr.update(value="⚠️ Cannot select 'None' with other ethical views.", visible=True)
                    elif "None" not in selected and len(selected) not in [1, 3]:
                        return gr.update(value="⚠️ Select either exactly 1 or 3 ethical perspectives.", visible=True)
                    else:
                        return gr.update(value="", visible=False)

                ethical_view_selector.change(
                    fn=validate_ethical_selection,
                    inputs=[ethical_view_selector],
                    outputs=[ethical_warning_box]
                )

                login_button.click(
                    fn=login_user,
                    inputs=[username_input, password_input],
                    outputs=[
                        session, 
                        login_button, signup_button, logout_button, 
                        delete_account_button, delete_password_input,
                        confirm_delete_button, login_status,
                        show_password_checkbox, password_input  # main login checkbox & field
                    ]
                ).then(
                    fn=toggle_auth_mode,
                    inputs=[auth_toggle, session],
                    outputs=[login_button, signup_button]
                )

                signup_button.click(
                    fn=signup_user,
                    inputs=[username_input, password_input],
                    outputs=[
                        session, 
                        login_button, signup_button, logout_button, 
                        delete_account_button, delete_password_input,
                        confirm_delete_button, login_status,
                        show_password_checkbox, password_input  # main login checkbox & field
                    ]
                ).then(
                    fn=toggle_auth_mode,
                    inputs=[auth_toggle, session],
                    outputs=[login_button, signup_button]
                )

                logout_button.click(
                    fn=logout_user,
                    outputs=[
                        session, 
                        login_button, signup_button, logout_button, 
                        delete_account_button, delete_password_input,
                        confirm_delete_button, login_status,
                        show_delete_password_checkbox,
                        show_password_checkbox, password_input  # main login checkbox & field
                    ]
                ).then(
                    fn=lambda: (-1, "No row selected"),
                    outputs=[selected_row_idx, selected_row_info]
                ).then(
                    fn=lambda: ([], []),  # This clears history table and job IDs
                    outputs=[history_list, history_job_ids]
                ).then(
                    fn=toggle_auth_mode,
                    inputs=[auth_toggle, session],
                    outputs=[login_button, signup_button]
                )

                delete_account_button.click(
                    fn=show_delete_inputs,
                    inputs=[],
                    outputs=[
                        delete_password_input,
                        confirm_delete_button,
                        show_delete_password_checkbox
                    ]
                )

                confirm_delete_button.click(
                    fn=delete_account,
                    inputs=[session, delete_password_input],
                    outputs=[
                        session,
                        login_button, signup_button,
                        logout_button, delete_account_button,
                        delete_password_input, confirm_delete_button,
                        login_status, show_delete_password_checkbox
                    ]
                ).then(
                    fn=lambda: (-1, "No row selected"),
                    outputs=[selected_row_idx, selected_row_info]
                ).then(
                    fn=lambda: ([], []),  # <-- This clears history table and job IDs
                    outputs=[history_list, history_job_ids]
                ).then(
                    fn=toggle_auth_mode,
                    inputs=[auth_toggle, session],
                    outputs=[login_button, signup_button]
                )

                show_password_checkbox.change(
                    fn=toggle_password_visibility,
                    inputs=[show_password_checkbox],
                    outputs=[password_input]
                )

                show_delete_password_checkbox.change(
                    fn=toggle_password_visibility,
                    inputs=[show_delete_password_checkbox],
                    outputs=[delete_password_input]
                )

                # Load history on startup
                app.load(fn=refresh_history, inputs=[session], outputs=[history_list, history_job_ids])

        # Connect the domain radio button to update example choices
        domain_radio.change(fn=get_example_choices, inputs=domain_radio, outputs=example_selector)
        
        # Connect the example selector to fill query and type
        example_selector.change(fn=fill_query_and_type, inputs=[example_selector, domain_radio], outputs=[input_query, question_type])
 
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

        def guarded_process(query, api_key, question_type, domain, aggregator_id, ethical_views, session_user):
            """
            A wrapper function that ensures a user session is active before processing a query.

            Args:
                *args: A variable-length argument list where the last argument is expected 
                       to be the session user.

            Returns:
                tuple: If the session user is not active, returns a tuple containing:
                    - gr.update: An update object with a message prompting the user to log in.
                    - Empty strings and zeros for other return values.
                    - gr.update objects to reset UI components to their default state.
                Otherwise, delegates the processing to the `process_query` function with 
                the provided arguments (excluding the session user).
            """
            if not session_user:
                return (
                    gr.update(value="❌ Please log in first."), "", "", "", 0,
                    None, None, None, None,
                    gr.update(value="", visible=False),
                    gr.update(label="Aggregator"),
                    gr.update(label="Model 1"),
                    gr.update(label="Model 2"),
                    gr.update(label="Model 3")
                )
            return process_query(query, api_key, question_type, domain, aggregator_id, ethical_views, session_user)

        submit_btn.click(
            fn=guarded_process,
            inputs=[input_query, api_key_input, question_type, domain_radio, aggregator_radio, ethical_view_selector, session],
            outputs=[
                output_aggregator, output_model1, output_model2, output_model3,
                consensus_score, output_heatmap, output_emotion, output_polarity,
                output_radar, plot_warning_box,
                output_aggregator, output_model1, output_model2, output_model3
            ]
        )
         
    return app

# Create .env file template
def create_env_template():
    """
    Creates a `.env` file in the current working directory if it does not already exist.
    
    The `.env` file contains default configuration settings for the OpenRouter API, 
    including placeholders for the API key and backend URL.

    Contents of the generated `.env` file:
    - OPENROUTER_API_KEY: Placeholder for the OpenRouter API key.
    - BACKEND_URL: Default backend URL set to `http://localhost:8000`.

    This function ensures that the required environment configuration file is present 
    for the application to function correctly.
    """
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("""# OpenRouter API Configuration
OPENROUTER_API_KEY=
BACKEND_URL=http://localhost:8000
""")

# Main function to run the system
def main():
    """
    The main function for the Multi-Agent LLM Client application. This function
    parses command-line arguments, updates the backend URL, creates a .env 
    template, and launches a Gradio interface.
    Command-line Arguments:
        --backend-url (str): URL of the backend server. Defaults to the global BACKEND_URL.
        --host (str): Host address for the Gradio client. Defaults to "0.0.0.0".
        --port (int): Port number to run the Gradio interface on. Defaults to 7860.
        --share (bool): Flag to create a shareable link for the Gradio interface.
    """
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