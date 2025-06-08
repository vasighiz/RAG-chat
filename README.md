# RAG System with Local LLM

A Retrieval-Augmented Generation (RAG) system that uses local language models to answer questions based on your documents. This project implements a complete RAG pipeline with a user-friendly web interface.

## Features

-  Local LLM integration with Mistral 7B
-  Document processing and chunking
-  Semantic search using FAISS
-  Interactive web interface with Streamlit
-  Adjustable parameters for fine-tuning responses

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vasighiz/RAG.git
cd RAG
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the model:
```bash
# Create models directory
mkdir models

# Download Mistral model (Windows PowerShell)
Invoke-WebRequest -Uri "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf" -OutFile "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
```

## Usage

1. Place your documents in the `data` directory (supports .txt files)

2. Run the Streamlit app:
```bash
streamlit run app/app.py
```

3. Open your browser and navigate to the provided URL (usually http://localhost:8501)

4. Click "Initialize System" in the sidebar

5. Enter your question and adjust parameters as needed

## Project Structure

```
RAG/
├── app/
│   └── app.py              # Streamlit web interface
├── llm/
│   ├── local_llm.py        # Local LLM implementation
│   └── test_llm.py         # LLM testing script
├── retriever/
│   ├── document_processor.py # Document processing
│   └── embedding_store.py    # Vector store implementation
├── data/                   # Document storage
├── models/                 # Model files
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Configuration

The system can be configured through various parameters:

- **Max Tokens**: Control response length (100-1024)
- **Temperature**: Adjust response creativity (0.0-1.0)
- **Context Chunks**: Number of relevant chunks to use (1-5)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
