"""
Ollama wrapper for local LLM inference.
"""

import requests
from typing import List, Dict, Any, Optional
import json
import logging

class OllamaWrapper:
    """Wrapper for Ollama API to interact with local LLMs."""
    
    def __init__(self, model_name: str = "mistral", base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama wrapper.
        
        Args:
            model_name (str): Name of the Ollama model to use.
            base_url (str): Base URL for Ollama API.
        """
        self.model_name = model_name
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)
        
    def generate(self, prompt: str, context: Optional[List[str]] = None) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            prompt (str): The prompt to send to the model.
            context (Optional[List[str]]): List of context documents to include.
            
        Returns:
            str: Generated response.
        """
        full_prompt = self._build_prompt(prompt, context)
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": full_prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raise
            
    def _build_prompt(self, prompt: str, context: Optional[List[str]] = None) -> str:
        """
        Build the full prompt with context.
        
        Args:
            prompt (str): User prompt.
            context (Optional[List[str]]): List of context documents.
            
        Returns:
            str: Full prompt with context.
        """
        if not context:
            return prompt
            
        context_str = "\n\n".join(context)
        return f"""Context information is below.
---------------------
{context_str}
---------------------
Given the context information, please answer the following question:
{prompt}

Answer:""" 