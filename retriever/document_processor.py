import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader,
    UnstructuredPDFLoader,
)
from langchain.schema import Document
from loguru import logger

class DocumentProcessor:
    def __init__(self, data_dir: str):
        """Initialize the document processor with the data directory."""
        self.data_dir = data_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,  # Smaller chunks for better context
            chunk_overlap=50,  # Overlap to maintain context between chunks
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]  # More granular splitting
        )

    def load_documents(self) -> List[str]:
        """Load documents from the data directory."""
        documents = []
        try:
            for filename in os.listdir(self.data_dir):
                if filename.endswith('.txt'):
                    file_path = os.path.join(self.data_dir, filename)
                    loader = TextLoader(file_path)
                    documents.extend(loader.load())
                    logger.info(f"Loaded {filename}")
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise
        return documents

    def process_documents(self, documents: List[str]) -> List[str]:
        """Process documents into chunks."""
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split documents into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise 