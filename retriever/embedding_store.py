import os
from typing import List, Dict, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from loguru import logger

class EmbeddingStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding store with the specified model."""
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = None

    def create_embeddings(self, documents: List[Document]) -> np.ndarray:
        """Create embeddings for the documents."""
        try:
            texts = [doc.page_content for doc in documents]
            embeddings = self.model.encode(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise

    def create_index(self, embeddings: np.ndarray, documents: List[Document]):
        """Create a FAISS index from the embeddings."""
        try:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype('float32'))
            self.documents = documents
            logger.info(f"Created and saved FAISS index with {len(documents)} documents")
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            raise

    def search(self, query: str, k: int = 3, score_threshold: float = 0.7) -> List[Tuple[Document, float]]:
        """Search for similar documents using the query."""
        try:
            query_embedding = self.model.encode([query])
            distances, indices = self.index.search(query_embedding.astype('float32'), k)
            
            # Convert distances to similarity scores (0 to 1)
            max_distance = np.max(distances)
            similarities = 1 - (distances / max_distance)
            
            # Filter results based on similarity threshold
            results = []
            for idx, similarity in zip(indices[0], similarities[0]):
                if similarity >= score_threshold:
                    results.append((self.documents[idx], float(similarity)))
            
            return results
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            raise

    def load_index(self) -> bool:
        """Load existing FAISS index."""
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                logger.info("Loaded existing FAISS index")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False
            
    def create_index_path(self, index_path: str):
        self.index_path = index_path
        
    def save_index(self):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Save index
        faiss.write_index(self.index, self.index_path)
        logger.info(f"Created and saved FAISS index with {len(self.documents)} documents")
        
    def get_index(self):
        return self.index

    def get_documents(self):
        return self.documents 