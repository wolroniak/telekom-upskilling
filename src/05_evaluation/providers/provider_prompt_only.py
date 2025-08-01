
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', '00_setup')))
from llm_agent import LLMAgent

# promptfoo will call this function with the prompt
def call_agent(prompt: str, options: dict | None = None, context: dict | None = None) -> str:
    """
    Provider function for the 'Prompt-Only' system variant.
    """
    agent = LLMAgent(model_name="Qwen/Qwen3-0.6B")
    system_prompt = "You are a customer support agent. Respond to the following complaint."
    
    # Extract the actual complaint from the prompt context
    # promptfoo passes variables as a formatted string, we need to extract 'complaint'
    if isinstance(prompt, dict) and 'complaint' in prompt:
        complaint = prompt['complaint']
    else:
        # If it's just a string, use it directly
        complaint = prompt
    
    response, _ = agent(complaint, system_prompt=system_prompt)
    return response

# This part is for manual testing, not used by promptfoo
if __name__ == "__main__":
    if not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
        print(call_agent(prompt))
