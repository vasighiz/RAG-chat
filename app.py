import streamlit as st
from llm.local_llm import LocalLLM
from retriever.document_processor import DocumentProcessor
from retriever.embedding_store import EmbeddingStore
import os
from loguru import logger

# Set page config
st.set_page_config(
    page_title="RAG Chat Interface",
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
        st.session_state.embedder = embedder
        st.session_state.llm = llm
        st.session_state.initialized = True
        
        return True
    except Exception as e:
        logger.error(f"Error initializing system: {str(e)}")
        return False

def generate_response(query: str) -> str:
    """Generate a response using the RAG system."""
    try:
        # Search for relevant context
        results = st.session_state.embedder.search(
            query,
            k=3,
            score_threshold=0.7
        )
        context = [doc.page_content for doc, score in results]
        
        # Generate response
        response = st.session_state.llm.generate_response(
            query=query,
            context=context,
            max_tokens=512,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.2
        )
        
        return response
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Error: {str(e)}"

# Main UI
st.title("🤖 RAG Chat Interface")
st.markdown("""
This interface allows you to interact with our RAG (Retrieval-Augmented Generation) system.
Ask questions about the documents in our knowledge base, and the system will provide relevant answers.
""")

# Initialize system if not already done
if not st.session_state.initialized:
    with st.spinner("Initializing system..."):
        if initialize_system():
            st.success("System initialized successfully!")
        else:
            st.error("Failed to initialize system. Please check the logs.")
            st.stop()

# Chat interface
st.markdown("### 💬 Chat")
query = st.text_input("Enter your question:", key="query_input")

if query:
    with st.spinner("Generating response..."):
        response = generate_response(query)
        
        # Display response in a nice format
        st.markdown("### Response")
        st.markdown(response)
        
        # Display context used
        with st.expander("View Context Used"):
            results = st.session_state.embedder.search(
                query,
                k=3,
                score_threshold=0.7
            )
            for i, (doc, score) in enumerate(results, 1):
                st.markdown(f"**Context {i}** (Similarity: {score:.2f})")
                st.markdown(doc.page_content)
                st.markdown("---")

# Sidebar with information
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    This RAG system uses:
    - Mistral 7B Instruct model
    - FAISS for efficient similarity search
    - Sentence Transformers for embeddings
    """)
    
    st.markdown("### ⚙️ Parameters")
    st.markdown("""
    - Temperature: 0.1
    - Top-p: 0.9
    - Repetition Penalty: 1.2
    - Context Chunks: 3
    - Similarity Threshold: 0.7
    """) 