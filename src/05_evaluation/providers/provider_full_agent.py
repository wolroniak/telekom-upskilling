
import sys
import os

def call_agent(prompt: str, context: dict | None = None, options: dict | None = None) -> str:
    """
    Provider function for the final, complete 'Agent + RAG + Fine-tuning' system.
    """
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', '04_finetuning', 'new_application')))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', '02_rag')))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', '03_agent_decision_logic')))
    
    from llm_agent_new import LLMAgent
    from rag_pipeline import RAGPipeline
    from decision_agent import DecisionAgent

    llm_agent = LLMAgent(model_name="Qwen3-0.6B-fine-tuned")
    knowledge_base_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', '02_rag', 'knowledge_base')
    rag_pipeline = RAGPipeline(knowledge_base_path=knowledge_base_path)
    decision_agent = DecisionAgent(llm_agent, rag_pipeline, threshold=1.0)
    
    final_response, _ = decision_agent.execute_query(prompt)
    return final_response

if __name__ == "__main__":
    if not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
        print(call_agent(prompt))
