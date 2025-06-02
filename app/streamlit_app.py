"""
Streamlit web interface for the RAG pipeline.
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from retriever.embedder import DocumentEmbedder
from retriever.faiss_store import FAISSStore
from llm.ollama_wrapper import OllamaWrapper

# Initialize components
@st.cache_resource
def init_components():
    """Initialize the RAG pipeline components."""
    embedder = DocumentEmbedder()
    faiss_store = FAISSStore()
    llm = OllamaWrapper()
    return embedder, faiss_store, llm

def main():
    st.title("Local RAG Pipeline")
    
    # Initialize components
    embedder, faiss_store, llm = init_components()
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    num_results = st.sidebar.slider("Number of results", 1, 5, 3)
    
    # Main content
    st.header("Ask a question")
    query = st.text_input("Enter your question:")
    
    if query:
        with st.spinner("Processing..."):
            # Embed query
            query_embedding = embedder.embed_query(query)
            
            # Retrieve relevant documents
            results = faiss_store.search(query_embedding, k=num_results)
            
            # Display retrieved documents
            st.subheader("Retrieved Documents")
            for doc, score in results:
                st.text_area("Document", doc, height=100)
                st.text(f"Relevance score: {score:.4f}")
            
            # Generate response
            context = [doc for doc, _ in results]
            response = llm.generate(query, context)
            
            # Display response
            st.subheader("Generated Response")
            st.write(response)

if __name__ == "__main__":
    main() 