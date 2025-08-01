import sys
import os

def call_agent(prompt: str, options: dict | None = None, context: dict | None = None) -> str:
    """
    Provider function for the 'RAG-Only' system variant.
    """
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', '00_setup')))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', '02_rag')))
    from llm_agent import LLMAgent
    from rag_pipeline import RAGPipeline

    if isinstance(prompt, dict) and 'complaint' in prompt:
        complaint = prompt['complaint']
    else:
        complaint = prompt

    agent = LLMAgent(model_name="Qwen/Qwen3-0.6B")
    knowledge_base_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', '02_rag', 'knowledge_base')
    rag_pipeline = RAGPipeline(knowledge_base_dir=knowledge_base_dir)

    retrieved_docs, _ = rag_pipeline.retrieve_with_scores(complaint)
    context_str = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    
    rag_prompt = (
        f"Based on the following context, please answer the user's complaint.\n"
        f"Context:\n{context_str}\n\n"
        f"Complaint:\n{complaint}"
    )

    response, _ = agent(rag_prompt)
    return response

if __name__ == "__main__":
    if not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
        print(call_agent(prompt))