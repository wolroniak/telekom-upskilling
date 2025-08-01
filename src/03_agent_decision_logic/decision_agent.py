
import os
import sys
import importlib.util

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

llm_agent_module = import_from_path("llm_agent", "src/00_setup/llm_agent.py")
LLMAgent = llm_agent_module.LLMAgent

rag_pipeline_module = import_from_path("rag_pipeline", "src/02_rag/rag_pipeline.py")
RAGPipeline = rag_pipeline_module.RAGPipeline

class DecisionAgent:
    def __init__(self, llm_agent: LLMAgent, rag_pipeline: RAGPipeline, threshold: float = 0.6):
        """
        Initializes the DecisionAgent.

        Args:
            llm_agent (LLMAgent): An instance of the LLM agent.
            rag_pipeline (RAGPipeline): An instance of the RAG pipeline.
            threshold (float): The relevance score threshold. Documents with a score
                               below this value will be considered relevant.
                               FAISS L2 distance is used, so lower is better.
        """
        print("--- Initializing Decision Agent ---")
        self.rag_pipeline = rag_pipeline
        self.llm_agent = llm_agent
        self.relevance_threshold = threshold
        self.rag_prompt_template = self._load_prompt_template("prompts/rag_empathetic_prompt.txt")
        self.llm_only_prompt_template = self._load_prompt_template("prompts/empathetic_prompt.txt")

    def _load_prompt_template(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _format_rag_prompt(self, complaint, context):
        return self.rag_prompt_template.replace("{{context}}", context).replace("{{complaint}}", complaint)
        
    def _format_llm_only_prompt(self, complaint):
        return self.llm_only_prompt_template.replace("{{complaint}}", complaint)

    def execute_query(self, query):
        print(f"\n--- Processing Query: \"{query}\" ---")
        
        retrieved_docs, retrieved_scores = self.rag_pipeline.retrieve_with_scores(query)
        
        if not retrieved_docs:
            print("Decision: No documents found in knowledge base. Using LLM-only path.")
            final_prompt = self._format_llm_only_prompt(query)
            response, _ = self.llm_agent(final_prompt)
            return response, "LLM_ONLY"

        best_score = min(retrieved_scores)
        print(f"Best retrieval score: {best_score:.4f} (Threshold: {self.relevance_threshold})")

        if best_score < self.relevance_threshold:
            print("Decision: Relevant context found. Using RAG path.")
            context_str = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
            final_prompt = self._format_rag_prompt(query, context_str)
            response, _ = self.llm_agent(final_prompt)
            return response, "RAG"
        else:
            print("Decision: Context not relevant enough. Using LLM-only path.")
            final_prompt = self._format_llm_only_prompt(query)
            response, _ = self.llm_agent(final_prompt)
            return response, "LLM_ONLY"
