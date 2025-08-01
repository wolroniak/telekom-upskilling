import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class RAGPipeline:
    def __init__(self, knowledge_base_dir="src/02_rag/knowledge_base"):
        self.knowledge_base_dir = knowledge_base_dir
        self.vector_store = self._build_vector_store()

    def _load_documents(self):
        loader_kwargs = {'encoding': 'utf-8'}
        loader = DirectoryLoader(
            self.knowledge_base_dir,
            glob="**/*.txt",
            loader_cls=lambda path: TextLoader(path, **loader_kwargs),
            show_progress=True,
            use_multithreading=True
        )
        return loader.load()

    def _split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.split_documents(documents)

    def _get_embeddings(self):
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda'}
        )

    def _build_vector_store(self):
        documents = self._load_documents()
        if not documents:
            return None
            
        chunks = self._split_documents(documents)
        embeddings = self._get_embeddings()
        
        print("Building vector store... This may take a moment.")
        # FAISS - L2 distance (lower score = more relevant)
        vector_store = FAISS.from_documents(chunks, embeddings)
        print("Vector store built successfully.")
        
        return vector_store

    def retrieve_with_scores(self, query, k=3):
        """Retrieves documents and their relevance scores."""
        if self.vector_store is None:
            return [], []
            
        docs_and_scores = self.vector_store.similarity_search_with_score(query, k=k)
        
        documents = [doc for doc, score in docs_and_scores]
        scores = [score for doc, score in docs_and_scores]
        
        return documents, scores

if __name__ == '__main__':
    pipeline = RAGPipeline()
    
    example_query = "My internet has been down since yesterday. I work from home and this is very frustrating!"
    retrieved_docs, retrieved_scores = pipeline.retrieve_with_scores(example_query)
    
    print(f"\n--- Query ---")
    print(example_query)
    print("\n--- Retrieved Documents & Scores ---")
    for doc, score in zip(retrieved_docs, retrieved_scores):
        print(f"Score: {score:.4f}")
        print(f"Content: {doc.page_content[:100]}...")
        print("---")

    # Example of a high (bad) score
    irrelevant_query = "What is the capital of France?"
    irrelevant_docs, irrelevant_scores = pipeline.retrieve_with_scores(irrelevant_query)
    
    print(f"\n--- Irrelevant Query ---")
    print(irrelevant_query)
    print("\n--- Retrieved Documents & Scores ---")
    for doc, score in zip(irrelevant_docs, irrelevant_scores):
        print(f"Score: {score:.4f}")
        print(f"Content: {doc.page_content[:100]}...")
        print("---")
