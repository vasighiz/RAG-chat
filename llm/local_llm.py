from typing import List, Dict, Any
from ctransformers import AutoModelForCausalLM
from loguru import logger
import re

class LocalLLM:
    def __init__(self, model_path: str):
        """Initialize the local LLM with the specified model."""
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type="mistral",
            gpu_layers=0,
            context_length=2048,
            threads=4
        )
        self.repetitive_patterns = [
            r"there are there are",
            r"you can you can",
            r"it is it is",
            r"this is this is",
            r"that is that is",
            r"they are they are",
            r"we can we can",
            r"i am i am",
            r"he is he is",
            r"she is she is"
        ]

    def format_prompt(self, query: str, context: List[str]) -> str:
        """Format the prompt with the query and context."""
        context_str = "\n".join(context)
        return f"""<s>[INST] You are a helpful AI assistant. Use the following context to answer the question. 
If you cannot answer from the context, say so. Avoid unnecessary repetition and maintain a clear, professional tone.

Context:
{context_str}

Question: {query}

Answer: [/INST]"""

    def clean_response(self, response: str) -> str:
        """Clean the response by removing artifacts and improving formatting."""
        # Remove instruction tokens
        response = re.sub(r"\[/INST\]|\[INST\]", "", response)
        
        # Remove repetitive patterns
        for pattern in self.repetitive_patterns:
            response = re.sub(pattern, pattern.split()[0], response, flags=re.IGNORECASE)
        
        # Remove multiple newlines
        response = re.sub(r"\n{3,}", "\n\n", response)
        
        # Remove extra spaces
        response = re.sub(r" {2,}", " ", response)
        
        # Remove any remaining repetitive words
        response = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", response)
        
        # Remove incomplete sentences at the end
        response = re.sub(r"[^.!?]+\s*$", "", response)
        
        return response.strip()

    def generate_response(
        self,
        query: str,
        context: List[str],
        max_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2
    ) -> str:
        """Generate a response using the local LLM."""
        try:
            prompt = self.format_prompt(query, context)
            response = self.model(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stop=["</s>", "[INST]", "\n\n"]
            )
            return self.clean_response(response)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise 