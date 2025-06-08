import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever.document_processor import DocumentProcessor
from retriever.embedding_store import EmbeddingStore
from local_llm import LocalLLM
from loguru import logger

def main():
    # Initialize components
    model_path = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    data_dir = "data"
    
    # Load and process documents
    processor = DocumentProcessor(data_dir)
    documents = processor.load_documents()
    chunks = processor.process_documents(documents)
    
    # Create embeddings and index
    embedder = EmbeddingStore()
    embeddings = embedder.create_embeddings(chunks)
    embedder.create_index(embeddings, chunks)
    
    # Initialize LLM
    llm = LocalLLM(model_path)
    
    # Test query
    query = "What are the key components and applications of deep learning? Please provide specific examples."
    
    # Search for relevant context
    results = embedder.search(query, k=3, score_threshold=0.7)
    context = [doc.page_content for doc, score in results]
    
    # Generate response
    response = llm.generate_response(
        query=query,
        context=context,
        max_tokens=512,
        temperature=0.1,
        top_p=0.9,
        repetition_penalty=1.2
    )
    
    print("\nQuery:", query)
    print("\nResponse:", response)

if __name__ == "__main__":
    main() 