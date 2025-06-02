"""
FAISS index management for document storage and retrieval.
"""

import faiss
import numpy as np
from typing import List, Tuple, Optional
import pickle
import os

class FAISSStore:
    """Manages document storage and retrieval using FAISS."""
    
    def __init__(self, dimension: int = 384):
        """
        Initialize the FAISS store.
        
        Args:
            dimension (int): Dimension of the embeddings. Defaults to 384 for all-MiniLM-L6-v2.
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents: List[str] = []
        
    def add_documents(self, embeddings: np.ndarray, documents: List[str]) -> None:
        """
        Add documents and their embeddings to the index.
        
        Args:
            embeddings (np.ndarray): Document embeddings.
            documents (List[str]): List of document texts.
        """
        self.index.add(embeddings)
        self.documents.extend(documents)
        
    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding (np.ndarray): Query embedding.
            k (int): Number of results to return.
            
        Returns:
            List[Tuple[str, float]]: List of (document, score) tuples.
        """
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):  # Ensure index is valid
                results.append((self.documents[idx], float(distance)))
        return results
    
    def save(self, directory: str) -> None:
        """
        Save the FAISS index and documents to disk.
        
        Args:
            directory (str): Directory to save the files.
        """
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "faiss.index"))
        with open(os.path.join(directory, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
            
    @classmethod
    def load(cls, directory: str) -> 'FAISSStore':
        """
        Load a FAISS index and documents from disk.
        
        Args:
            directory (str): Directory containing the saved files.
            
        Returns:
            FAISSStore: Loaded FAISS store instance.
        """
        store = cls()
        store.index = faiss.read_index(os.path.join(directory, "faiss.index"))
        with open(os.path.join(directory, "documents.pkl"), "rb") as f:
            store.documents = pickle.load(f)
        return store 