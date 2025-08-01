import sys
import os
import importlib.util

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Dynamically import the llm_agent module
spec = importlib.util.spec_from_file_location("llm_agent", "src/00_setup/llm_agent.py")
llm_agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm_agent_module)
LLMAgent = llm_agent_module.LLMAgent

# A dictionary to cache agent instances so we don't reload models unnecessarily
agents = {}

def get_agent(model_name):
    if model_name not in agents:
        print(f"Loading model for the first time: {model_name}")
        agents[model_name] = LLMAgent(model_name=model_name)
    return agents[model_name]

def call_agent(prompt, options=None, context=None):
    """
    A bridge between promptfoo and our LLMAgent that can handle different models.
    """
    model_name = options.get('model', 'Qwen/Qwen3-0.6B') # Default to Qwen
    agent = get_agent(model_name)

    system_prompt = """You are a professional and empathetic customer support assistant for a major telecommunications company. 
Your primary goal is to help users solve their problems efficiently while making them feel heard and valued. 
You cannot perform account actions directly, but you can provide information, guide users through troubleshooting steps, and explain how to escalate issues."""
    
    answer, _ = agent(
        prompt=prompt,
        system_prompt=system_prompt,
        max_new_tokens=256
    )
    
    return {
        'output': answer
    }
