import json
import os
import sys

# Add the specific path to the 00_setup directory to allow for package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_setup')))

from prompt_templates import PROMPT_TEMPLATES
from llm_agent import LLMAgent


def run_template_examples():
    # Load the complaints from the JSON file
    try:
        with open("src/01_prompt_engineering/complaints.json", "r") as f:
            complaints = json.load(f)
    except FileNotFoundError:
        print("Error: complaints.json not found. Make sure the file exists in the 'src/01_prompt_engineering' directory.")
        return

    # Use the first complaint as an example
    example_complaint = complaints[0]["complaint"]

    # Initialize the LLM agent (defaults to Qwen)
    agent = LLMAgent()

    print(f"--- Using Complaint: '{example_complaint}' ---\n")

    # Generate a response for each prompt template
    for name, template in PROMPT_TEMPLATES.items():
        # Replace the placeholder with the actual complaint
        prompt = template.replace("{{complaint}}", example_complaint)
        
        print(f"--- Generating Response with '{name.capitalize()}' Template ---")
        
        # Define the system prompt
        system_prompt = "You are a customer support assistant for a telecommunications company."
        
        # Generate the response using the agent
        answer, _ = agent(prompt=prompt, system_prompt=system_prompt, max_new_tokens=256)
        
        print(f"Generated Response:\n{answer}\n")

if __name__ == "__main__":
    run_template_examples()
