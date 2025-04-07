"""
Distributed Multi-Agent System for Math Reasoning
Authors: [Your Name]

This system implements a distributed fault-tolerant multi-agent approach for solving math problems
where multiple lightweight language models act as solver agents.
"""

import os
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer
)
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from scipy.stats import entropy
import gradio as gr

# Configuration settings
@dataclass
class Config:
    """Configuration settings for the multi-agent system."""
    max_rounds: int = 3
    similarity_threshold: float = 0.85
    use_gpu: bool = False #torch.cuda.is_available()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2"

config = Config()

class MathAgent:
    """Base class for math solving agents."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """Load the model and tokenizer."""
        raise NotImplementedError
    
    def solve(self, question: str, peer_responses: List[str] = None) -> Dict[str, Any]:
        """
        Solve a math problem using the distributed agent system.
        Agents iteratively refine answers based on peer responses until convergence or max rounds.
        """
        print(f"🧠 Solving: {question}")
        all_rounds_responses = []

        # Round 0: independent solutions
        initial_responses = [agent.solve(question) for agent in self.agents]
        for resp in initial_responses:
            print(f"{resp['agent_name']} initial answer: {resp['answer'][:80]}...")
        all_rounds_responses.append(initial_responses)

        current_answers = [resp["answer"] for resp in initial_responses]
        converged, similarity = self.aggregator.check_convergence(current_answers)
        print(f"✅ Initial similarity: {similarity:.4f}")

        round_num = 1
        while not converged and round_num < config.max_rounds:
            print(f"\n🔁 Round {round_num + 1}")
            refined_responses = []
            for i, agent in enumerate(self.agents):
                peers = [resp["answer"] for j, resp in enumerate(all_rounds_responses[-1]) if j != i]
                refined = agent.solve(question, peers)
                print(f"{agent.name} refined answer: {refined['answer'][:80]}...")
                refined_responses.append(refined)

            all_rounds_responses.append(refined_responses)
            current_answers = [resp["answer"] for resp in refined_responses]
            converged, similarity = self.aggregator.check_convergence(current_answers)
            print(f"🔍 Round {round_num + 1} similarity: {similarity:.4f}")
            round_num += 1

        final_responses = all_rounds_responses[-1]
        final_result = self.aggregator.synthesize_final_answer(question, final_responses)
        final_result.update({
            "rounds_completed": round_num,
            "converged": converged,
            "all_rounds": all_rounds_responses,
            "agent_responses": final_responses
        })
        self.question_history.append(question)
        self.response_history.append(final_result)
        return final_result

def format_prompt(self, question: str, peer_responses: List[str] = None) -> str:
    """Create a thoughtful CoT prompt, optionally using peer feedback."""
    if not peer_responses:
        prompt = f"""Solve the following math problem step by step:
Question: {question}
Answer:"""
    else:
        peer_solutions = "\n\n".join([f"Peer Agent {i+1} says:\n{resp}" for i, resp in enumerate(peer_responses)])
        prompt = f"""A group of agents are solving this math problem collaboratively.
Question: {question}

Here are their suggestions:
{peer_solutions}

Please analyze the answers above, identify flaws if any, and provide your refined step-by-step solution:
Answer:"""
    return prompt.strip()


class QwenMathAgent(MathAgent):
    """Agent using Qwen2.5-Math-1.5B-Instruct model."""
    
    def __init__(self, name: str = "QwenMath"):
        super().__init__(name)
        self.model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    
    def load_model(self):
        print(f"Loading {self.name} agent...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if config.use_gpu else torch.float32,
            device_map="auto" if config.use_gpu else None
        )
        print(f"{self.name} agent loaded.")
    
    def solve(self, question: str, peer_responses: List[str] = None) -> Dict[str, Any]:
        if self.model is None:
            self.load_model()
            
        prompt = self.format_prompt(question, peer_responses)
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(config.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.2,
                num_beams=3
            )
        
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        answer = response.replace(prompt, "").strip()
        
        # Simple confidence estimation based on the agent's certainty phrases
        confidence = 0.9  # Default high confidence
        if "not sure" in answer.lower() or "uncertain" in answer.lower():
            confidence = 0.6
        elif "possibly" in answer.lower() or "maybe" in answer.lower():
            confidence = 0.7
            
        return {
            "answer": answer,
            "confidence": confidence,
            "agent_name": self.name
        }


class GemmaMathAgent(MathAgent):
    """Agent using gemma-2b-instruct-ft-omni-Math model."""
    
    def __init__(self, name: str = "GemmaMath"):
        super().__init__(name)
        self.model_name = "google/gemma-2b-instruct-ft-omni-Math"
    
    def load_model(self):
        print(f"Loading {self.name} agent...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if config.use_gpu else torch.float32,
            device_map="auto" if config.use_gpu else None
        )
        print(f"{self.name} agent loaded.")
    
    def solve(self, question: str, peer_responses: List[str] = None) -> Dict[str, Any]:
        if self.model is None:
            self.load_model()
            
        prompt = self.format_prompt(question, peer_responses)
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(config.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.2,
                num_beams=3
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.replace(prompt, "").strip()
        
        # Confidence estimation
        confidence = 0.85
        if "uncertain" in answer.lower() or "not confident" in answer.lower():
            confidence = 0.5
            
        return {
            "answer": answer,
            "confidence": confidence,
            "agent_name": self.name
        }


class MathQAAgent(MathAgent):
    """Agent fine-tuned for MathQA tasks."""
    
    def __init__(self, name: str = "MathQA"):
        super().__init__(name)
        self.model_name = "path/to/your/MathQA/model"  # Replace with actual model path
        
    def load_model(self):
        print(f"Loading {self.name} agent...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if config.use_gpu else torch.float32,
            device_map="auto" if config.use_gpu else None
        )
        print(f"{self.name} agent loaded.")
    
    def solve(self, question: str, peer_responses: List[str] = None) -> Dict[str, Any]:
        if self.model is None:
            self.load_model()
            
        prompt = self.format_prompt(question, peer_responses)
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(config.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.2,
                num_beams=3
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.replace(prompt, "").strip()
        
        # Confidence estimation
        confidence = 0.8
        if "unsure" in answer.lower():
            confidence = 0.6
            
        return {
            "answer": answer,
            "confidence": confidence,
            "agent_name": self.name
        }


class AggregatorAgent:
    """
    Aggregator agent that compiles final answers and determines consensus.
    Uses Flan-T5 to synthesize responses.
    """
    
    def __init__(self):
        self.model_name = "google/flan-t5-base"
        self.model = None
        self.tokenizer = None
        self.similarity_model = None
    
    def load_models(self):
        print("Loading Aggregator agent models...")
        # Load Flan-T5 for response synthesis
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if config.use_gpu else torch.float32,
            device_map="auto" if config.use_gpu else None
        )
        
        # Load sentence transformer for similarity computation
        self.similarity_model = SentenceTransformer(config.similarity_model)
        self.similarity_model.to(config.device)
        print("Aggregator agent models loaded.")
    
    def compute_similarity_matrix(self, responses: List[str]) -> Tuple[np.ndarray, float]:
        """Compute pairwise cosine similarity and return mean similarity."""
        if self.similarity_model is None:
            self.load_models()

        embeddings = self.similarity_model.encode(responses, convert_to_tensor=True)
        similarity_matrix = F.cosine_similarity(
            embeddings.unsqueeze(1),
            embeddings.unsqueeze(0),
            dim=2
        ).cpu().numpy()

        n = len(responses)
        mean_similarity = (similarity_matrix.sum() - n) / (n * (n - 1)) if n > 1 else 1.0
        return similarity_matrix, mean_similarity

    
    def check_convergence(self, responses: List[str]) -> Tuple[bool, float]:
        """Check if responses have converged based on similarity threshold."""
        _, mean_similarity = self.compute_similarity_matrix(responses)
        return mean_similarity > config.similarity_threshold, mean_similarity
    
    def synthesize_final_answer(self, question: str, agent_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize a final answer from agent responses using Flan-T5."""
        if self.model is None:
            self.load_models()
            
        # Extract answers and create prompt
        answers = [resp["answer"] for resp in agent_responses]
        confidences = [resp["confidence"] for resp in agent_responses]
        agent_names = [resp["agent_name"] for resp in agent_responses]
        
        # Create a formatted prompt with all agent responses
        formatted_answers = "\n\n".join([
            f"Agent {agent_names[i]} (confidence: {confidences[i]:.2f}):\n{answers[i]}"
            for i in range(len(answers))
        ])
        
        prompt = f"""
        Question: {question}
        
        Agent Responses:
        {formatted_answers}
        
        Synthesize these responses into a single, correct answer. Identify the most reliable solution 
        and explain why it's correct. If there are disagreements, analyze them and resolve the conflict.
        """
        
        # Generate the synthesized response
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = inputs.to(config.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                num_beams=4
            )
            
        synthesized_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Calculate final confidence as weighted average of agent confidences
        # with higher weight for agents whose answers are more similar to final answer
        final_confidence = sum(confidences) / len(confidences)
        
        # Compute agreement level based on similarity
        _, agreement_level = self.compute_similarity_matrix(answers)
        
        return {
            "final_answer": synthesized_answer,
            "confidence": final_confidence,
            "agreement_level": agreement_level,
            "agent_responses": agent_responses
        }


class DistributedMathSystem:
    """Main system that coordinates the agents."""
    
    def __init__(self):
        # Initialize solver agents
        self.agents = [
            QwenMathAgent(),
            #GemmaMathAgent(),
            #MathQAAgent()
        ]
        
        # Initialize aggregator agent
        self.aggregator = AggregatorAgent()
        
        # Initialize history for tracking
        self.question_history = []
        self.response_history = []
    
    def load_all_models(self):
        """Preload all models."""
        for agent in self.agents:
            agent.load_model()
        self.aggregator.load_models()
    
    def solve_problem(self, question: str) -> Dict[str, Any]:
        """
        Solve a math problem using the distributed agent system.
        
        Args:
            question: The math question to solve
            
        Returns:
            Dict containing final answer and metadata
        """
        print(f"Solving problem: {question}")
        
        # Keep track of responses for each round
        all_rounds_responses = []
        
        # Initial round - agents solve independently
        initial_responses = []
        for agent in self.agents:
            response = agent.solve(question)
            print(f"{agent.name} initial answer: {response['answer'][:50]}...")
            initial_responses.append(response)
        
        all_rounds_responses.append(initial_responses)
        
        # Extract just the answer texts for similarity computation
        current_answers = [resp["answer"] for resp in initial_responses]
        
        # Check for early convergence
        converged, similarity = self.aggregator.check_convergence(current_answers)
        print(f"Initial agreement level: {similarity:.4f}")
        
        round_num = 1
        while not converged and round_num < config.max_rounds:
            print(f"\nStarting round {round_num + 1}")
            
            # Refinement round - agents see others' responses and refine
            refined_responses = []
            for i, agent in enumerate(self.agents):
                # Get all other agents' answers
                peer_responses = [resp["answer"] for j, resp in enumerate(all_rounds_responses[-1]) if j != i]
                
                # Agent refines based on peer responses
                refined = agent.solve(question, peer_responses)
                refined_responses.append(refined)
                print(f"{agent.name} refined answer: {refined['answer'][:50]}...")
            
            all_rounds_responses.append(refined_responses)
            
            # Check if we've converged
            current_answers = [resp["answer"] for resp in refined_responses]
            converged, similarity = self.aggregator.check_convergence(current_answers)
            print(f"Round {round_num + 1} agreement level: {similarity:.4f}")
            
            round_num += 1
        
        # Use the final round responses for aggregation
        final_responses = all_rounds_responses[-1]
        
        # Let aggregator synthesize the final answer
        final_result = self.aggregator.synthesize_final_answer(question, final_responses)
        
        # Add metadata
        final_result["rounds_completed"] = round_num
        final_result["converged"] = converged
        final_result["all_rounds"] = all_rounds_responses
        
        # Store in history
        self.question_history.append(question)
        self.response_history.append(final_result)
        
        return final_result


# UI Implementation using Gradio
def create_ui(system):
    """Create a Gradio UI for interacting with the math system."""
    
    def solve_and_display(question):
        result = system.solve_problem(question)
        
        # Format the output for display
        final_answer = result["final_answer"]
        confidence = result["confidence"]
        agreement = result["agreement_level"]
        rounds = result["rounds_completed"]
        
        # Format agent responses for display
        agent_responses = result["agent_responses"]
        agent_outputs = "\n\n".join([
            f"### {resp['agent_name']} (confidence: {resp['confidence']:.2f})\n{resp['answer']}"
            for resp in agent_responses
        ])
        
        output = f"""
        ## Final Answer
        {final_answer}
        
        ## Metadata
        - **Confidence**: {confidence:.2f}
        - **Agreement Level**: {agreement:.4f}
        - **Rounds Completed**: {rounds}
        
        ## Individual Agent Responses
        {agent_outputs}
        """
        
        return output
    
    interface = gr.Interface(
        fn=solve_and_display,
        inputs=gr.Textbox(lines=5, label="Math Problem"),
        outputs=gr.Markdown(label="Solution"),
        title="Distributed Multi-Agent Math Problem Solver",
        description="This system uses multiple AI agents to collaboratively solve math problems."
    )
    
    return interface


# Example usage
if __name__ == "__main__":
    # Create system
    math_system = DistributedMathSystem()
    
    # Option 1: Run a single problem
    # result = math_system.solve_problem("If x + y = 10 and x * y = 24, what is the value of x^2 + y^2?")
    # print(f"Final answer: {result['final_answer']}")
    
    # Option 2: Launch UI
    ui = create_ui(math_system)
    ui.launch()