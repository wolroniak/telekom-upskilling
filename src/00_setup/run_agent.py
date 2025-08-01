from llm_agent import LLMAgent

def run_qwen_examples():
    print("--- Initializing Default Qwen Agent ---")
    agent = LLMAgent() # Default = Qwen

    # Simple prompt
    print("\n--- Example 1: Simple Prompt (Qwen) ---")
    user_prompt_1 = "Give me a short introduction to large language models."
    answer, _ = agent(prompt=user_prompt_1)
    print(f"User Prompt: {user_prompt_1}")
    print(f"Model Answer: {answer}")

    # Prompt with system prompt
    print("\n--- Example 2: With System Prompt (Qwen) ---")
    system_prompt_2 = "You are a helpful assistant that provides concise answers."
    user_prompt_2 = "What are the main components of a transformer architecture?"
    answer, _ = agent(prompt=user_prompt_2, system_prompt=system_prompt_2, max_new_tokens=256)
    print(f"System Prompt: {system_prompt_2}")
    print(f"User Prompt: {user_prompt_2}")
    print(f"Model Answer: {answer}")

def run_t5_example():
    print("\n\n--- Initializing T5 Agent ---")
    agent = LLMAgent(model_name="google/flan-t5-large")

    # T5 Translation
    print("\n--- Example 3: Translation (T5) ---")
    user_prompt_3 = "translate English to German: How old are you?"
    answer, _ = agent(prompt=user_prompt_3)
    print(f"User Prompt: {user_prompt_3}")
    print(f"Model Answer: {answer}")

def run_gemma_example():
    print("\n\n--- Initializing Gemma Agent ---")
    agent = LLMAgent(model_name="google/gemma-3-1b-it")

    # Gemma Chat
    print("\n--- Example 4: Chat (Gemma) ---")
    user_prompt_4 = "Write a poem on Hugging Face, the company"
    answer, _ = agent(prompt=user_prompt_4, max_new_tokens=256)
    print(f"User Prompt: {user_prompt_4}")
    print(f"Model Answer:\n{answer}")

    # Gemma Chat with System Prompt
    print("\n--- Example 5: Chat with System Prompt (Gemma) ---")
    system_prompt_5 = "You are a helpful assistant that provides concise answers."
    user_prompt_5 = "What are the main components of a transformer architecture?"
    answer, _ = agent(prompt=user_prompt_5, system_prompt=system_prompt_5, max_new_tokens=128)
    print(f"System Prompt: {system_prompt_5}")
    print(f"User Prompt: {user_prompt_5}")
    print(f"Model Answer:\n{answer}")

if __name__ == "__main__":
    run_qwen_examples()
    run_t5_example()
    run_gemma_example()
