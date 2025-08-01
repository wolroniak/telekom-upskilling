
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', '00_setup')))
from llm_agent import LLMAgent

def call_agent(prompt: str, options: dict | None = None, context: dict | None = None) -> str:
    """
    Provider function for the 'Prompt-Only' system variant.
    """
    agent = LLMAgent(model_name="Qwen/Qwen3-0.6B")
    system_prompt = "You are a customer support agent. Respond to the following complaint."
    
    if isinstance(prompt, dict) and 'complaint' in prompt:
        complaint = prompt['complaint']
    else:
        complaint = prompt
    
    response, _ = agent(complaint, system_prompt=system_prompt)
    return response

if __name__ == "__main__":
    if not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
        print(call_agent(prompt))
