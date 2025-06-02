"""
Document embedding module using sentence-transformers.
"""

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np

class DocumentEmbedder:
    """Handles document embedding using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the document embedder.
        
        Args:
            model_name (str): Name of the sentence-transformer model to use.
                             Defaults to "all-MiniLM-L6-v2" for its good balance of speed and quality.
        """
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        Embed a list of documents.
        
        Args:
            documents (List[str]): List of document texts to embed.
            
        Returns:
            np.ndarray: Array of document embeddings.
        """
        return self.model.encode(documents, show_progress_bar=True)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query.
        
        Args:
            query (str): Query text to embed.
            
        Returns:
            np.ndarray: Query embedding.
        """
        return self.model.encode([query])[0] 