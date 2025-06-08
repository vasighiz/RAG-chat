import streamlit as st
import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from retriever.document_processor import DocumentProcessor
from retriever.embedding_store import EmbeddingStore
from llm.local_llm import LocalLLM
from loguru import logger

# Set page config
st.set_page_config(
    page_title="RAG System",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.llm = None
    st.session_state.embedder = None

def initialize_system():
    """Initialize the RAG system components."""
    try:
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
        
        # Store in session state
        st.session_state.llm = llm
        st.session_state.embedder = embedder
        st.session_state.initialized = True
        
        return True
    except Exception as e:
        logger.error(f"Error initializing system: {str(e)}")
        return False

def main():
    st.title("🤖 RAG System")
    st.markdown("""
    This is a Retrieval-Augmented Generation (RAG) system that uses local LLMs to answer questions based on your documents.
    """)
    
    # Initialize system automatically
    if not st.session_state.initialized:
        with st.spinner("Initializing system..."):
            if initialize_system():
                st.success("System initialized successfully!")
            else:
                st.error("Failed to initialize system. Please check the logs.")
                return
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This RAG system uses:
        - Mistral 7B Instruct model
        - FAISS for vector search
        - Sentence Transformers for embeddings
        """)
    
    # Main content
    # Parameters
    st.markdown("### Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.max_tokens = st.slider("Max Tokens", 100, 1024, 512)
    with col2:
        st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
    with col3:
        st.session_state.top_k = st.slider("Number of Context Chunks", 1, 5, 3)

    st.markdown("---")

    # Query input with callback
    def on_enter():
        if st.session_state.query:
            with st.spinner("Generating answer..."):
                try:
                    # Search for relevant context
                    results = st.session_state.embedder.search(
                        st.session_state.query,
                        k=st.session_state.top_k,
                        score_threshold=0.7
                    )
                    context = [doc.page_content for doc, score in results]
                    
                    # Generate response
                    response = st.session_state.llm.generate_response(
                        query=st.session_state.query,
                        context=context,
                        max_tokens=st.session_state.max_tokens,
                        temperature=st.session_state.temperature,
                        top_p=0.9,
                        repetition_penalty=1.2
                    )
                    
                    # Store response in session state
                    st.session_state.response = response
                    st.session_state.context_results = results
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
        else:
            st.warning("Please enter a question.")

    # Query input with Enter key support
    st.text_input(
        "Enter your question (press Enter to generate answer):",
        placeholder="What would you like to know?",
        key="query",
        on_change=on_enter
    )

    # Display answer if available
    if hasattr(st.session_state, 'response'):
        st.markdown("### Answer")
        st.write(st.session_state.response)
        
        # Display context
        with st.expander("View Retrieved Context"):
            for i, (doc, score) in enumerate(st.session_state.context_results, 1):
                st.markdown(f"**Chunk {i}** (Similarity: {score:.2f})")
                st.write(doc.page_content)
                st.markdown("---")

if __name__ == "__main__":
    main() 