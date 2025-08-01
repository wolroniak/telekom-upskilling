
import sys
import os
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

sys.path.insert(0, os.path.join(project_root, 'src', '02_rag'))
sys.path.insert(0, os.path.join(project_root, 'src', '03_agent_decision_logic'))

from llm_agent_new import LLMAgent
from rag_pipeline import RAGPipeline
from decision_agent import DecisionAgent

def load_complaints(file_path="src/01_prompt_engineering/complaints.json"):
    """Loads the customer complaints from the specified JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def run_agent_on_complaints(agent, complaints):
    """Runs the decision agent on a list of complaints."""
    for complaint_data in complaints:
        complaint = complaint_data["vars"]["complaint"]
        print("-" * 80)
        print(f"Processing Complaint: {complaint}")
        
        final_response, source = agent.execute_query(complaint)
        
        print(f"\nSource Used: {source}")
        print(f"Final Response: {final_response}\n")

def main():
    """
    Initializes the agent with the fine-tuned model and runs it on the complaints dataset.
    """
    print("--- Initializing Final Application with Fine-Tuned Model ---")
    
    llm_agent = LLMAgent(model_name="Qwen3-0.6B-fine-tuned")
    
    knowledge_base_path = "src/02_rag/knowledge_base"
    rag_pipeline = RAGPipeline(knowledge_base_path)
    
    decision_agent = DecisionAgent(llm_agent, rag_pipeline, threshold=1.0)

    complaints = load_complaints()
    run_agent_on_complaints(decision_agent, complaints)

if __name__ == "__main__":
    main()
