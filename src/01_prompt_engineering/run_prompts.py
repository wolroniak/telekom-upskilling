import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_setup')))

from prompt_templates import PROMPT_TEMPLATES
from llm_agent import LLMAgent


def run_template_examples():
    try:
        with open("src/01_prompt_engineering/complaints.json", "r") as f:
            complaints = json.load(f)
    except FileNotFoundError:
        print("Error: complaints.json not found. Make sure the file exists in the 'src/01_prompt_engineering' directory.")
        return

    example_complaint = complaints[0]["complaint"]

    agent = LLMAgent()

    print(f"--- Using Complaint: '{example_complaint}' ---\n")

    for name, template in PROMPT_TEMPLATES.items():
        prompt = template.replace("{{complaint}}", example_complaint)
        
        print(f"--- Generating Response with '{name.capitalize()}' Template ---")
        
        system_prompt = "You are a customer support assistant for a telecommunications company."
        
        answer, _ = agent(prompt=prompt, system_prompt=system_prompt, max_new_tokens=256)
        
        print(f"Generated Response:\n{answer}\n")

if __name__ == "__main__":
    run_template_examples()
