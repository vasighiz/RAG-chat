# RAG Chat Interface

A Retrieval-Augmented Generation (RAG) system with a user-friendly web interface built using Streamlit.

## Features

- 🤖 Powered by Mistral 7B Instruct model
- 🔍 Efficient document retrieval using FAISS
- 📝 Clean and intuitive web interface
- 🔄 Real-time response generation
- 📊 Context visualization

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

3. Download the model:
```powershell
Invoke-WebRequest -Uri "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf" -OutFile "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
```

4. Place your documents in the `data` directory (supports .txt files)

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the provided URL (usually http://localhost:8501)

3. Enter your questions in the chat interface and get AI-powered responses based on your documents

## System Parameters

- Temperature: 0.1 (for focused responses)
- Top-p: 0.9 (for controlled randomness)
- Repetition Penalty: 1.2 (to reduce repetition)
- Context Chunks: 3 (for comprehensive context)
- Similarity Threshold: 0.7 (for relevant context)

## Project Structure

```
.
├── app.py              # Streamlit web interface
├── llm/               # Language model components
│   ├── local_llm.py   # Local LLM implementation
│   └── test_llm.py    # LLM testing script
├── retriever/         # Document retrieval components
│   ├── document_processor.py  # Document processing
│   └── embedding_store.py     # Embedding and search
├── data/              # Document storage
├── models/            # Model storage
└── requirements.txt   # Project dependencies
```

## Contributing

Feel free to submit issues and enhancement requests! 