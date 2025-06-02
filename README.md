# Local RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline built using local tools and open-source models. This project implements a complete RAG system without relying on paid APIs or cloud services.

## Features

- Document loading and processing from local files
- Document embedding using Hugging Face's sentence-transformers
- Vector storage using FAISS
- Local LLM integration via Ollama
- Optional Streamlit web interface
- Query logging and analysis

## Project Structure

```
.
├── data/               # Document storage
├── retriever/         # Embedding and retrieval logic
│   ├── __init__.py
│   ├── embedder.py    # Document embedding
│   └── faiss_store.py # FAISS index management
├── llm/              # LLM integration
│   ├── __init__.py
│   └── ollama_wrapper.py
├── app/              # Web interface
│   ├── __init__.py
│   └── streamlit_app.py
├── requirements.txt   # Project dependencies
└── README.md         # This file
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Ollama:
- Follow instructions at https://ollama.ai/download
- Pull required models: `ollama pull mistral`

## Usage

1. Place your documents in the `data/` directory
2. Run the Streamlit app:
```bash
streamlit run app/streamlit_app.py
```

Or use the command-line interface:
```bash
python -m app.cli
```

## Dependencies

- sentence-transformers: For document and query embedding
- FAISS: For efficient similarity search
- Ollama: For local LLM inference
- Streamlit: For web interface (optional)

## License

MIT 