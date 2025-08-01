import os
import sys
import importlib.util

# This is necessary because the script is in a subdirectory and needs to import from other subdirectories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# --- Dynamic import of DecisionAgent ---
# This is required because the directory starts with a number, which is not a valid package name.
try:
    # 1. Create the spec for the module from its file path.
    spec = importlib.util.spec_from_file_location(
        "decision_agent",
        os.path.join(os.path.dirname(__file__), 'decision_agent.py')
    )
    # 2. Create a new, empty module object from the spec.
    decision_agent_module = importlib.util.module_from_spec(spec)
    # 3. Execute the module's code to populate the module object.
    spec.loader.exec_module(decision_agent_module)
    # 4. Now that the module is loaded, we can import the class from it.
    DecisionAgent = decision_agent_module.DecisionAgent
except (FileNotFoundError, AttributeError) as e:
    print(f"Error importing DecisionAgent: {e}")
    print("Please ensure 'decision_agent.py' exists in the same directory.")
    sys.exit(1)


def main():
    # This value allows relevant queries (scores < 1.3) to use the RAG path.
    agent = DecisionAgent(threshold=1.3) 

    # --- Test Case 1: Relevant Query (should use RAG) ---
    relevant_query = "My internet is very slow in the evenings. What can I do?"
    response, decision = agent.execute_query(relevant_query)
    
    print(f"\nResponse (from {decision} path):")
    print("-" * 20)
    print(response)
    print("-" * 20)

    # --- Test Case 2: Irrelevant Query (should use LLM-only) ---
    irrelevant_query = "Can you tell me a fun fact about the Roman Empire?"
    response, decision = agent.execute_query(irrelevant_query)
    
    print(f"\nResponse (from {decision} path):")
    print("-" * 20)
    print(response)
    print("-" * 20)
    
    # --- Test Case 3: Missed installation appointment (should use RAG) ---
    missed_appointment_query = "The technician never showed up for my installation today."
    response, decision = agent.execute_query(missed_appointment_query)

    print(f"\nResponse (from {decision} path):")
    print("-" * 20)
    print(response)
    print("-" * 20)


if __name__ == "__main__":
    main()
