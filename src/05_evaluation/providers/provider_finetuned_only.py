
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', '04_finetuning', 'new_application')))
from llm_agent_new import LLMAgent

def call_agent(prompt: str, options: dict | None = None, context: dict | None = None) -> str:
    """
    Provider function for the 'Fine-Tuned Only' system variant.
    """

    agent = LLMAgent(model_name="Qwen3-0.6B-fine-tuned")
    system_prompt = "You are a helpful customer support agent. Please respond to the following customer complaint with empathy and provide a clear solution."
    
    response, _ = agent(prompt, system_prompt=system_prompt)
    return response

if __name__ == "__main__":
    if not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
        print(call_agent(prompt))
