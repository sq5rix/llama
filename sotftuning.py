import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from nomic.embed import Embedder
import ollama
import numpy as np
from typing import List, Tuple, Optional
import json
from tqdm import tqdm

class SoftPromptTuner:
    def __init__(
        self,
        model_name: str = "llama2:3b",
        prompt_length: int = 20,
        learning_rate: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the soft prompt tuner.
        
        Args:
            model_name: Name of the Llama model to use
            prompt_length: Length of the soft prompt in tokens
            learning_rate: Learning rate for optimization
            device: Device to run the model on
        """
        self.device = device
        self.model_name = model_name
        self.prompt_length = prompt_length
        self.learning_rate = learning_rate
        
        # Initialize Nomic embedder for semantic similarity
        self.embedder = Embedder(model="nomic-embed-text-v1")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("huggingface/llama2-3b")
        
        # Initialize soft prompt
        embedding_dim = 4096  # Llama's hidden dimension
        self.soft_prompt = nn.Parameter(
            torch.randn(prompt_length, embedding_dim).to(device)
        )
        self.optimizer = torch.optim.AdamW([self.soft_prompt], lr=learning_rate)

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text using Nomic embeddings."""
        embeddings = self.embedder.embed(text)
        return torch.tensor(embeddings).to(self.device)

    def get_llama_response(self, prompt: str) -> str:
        """Get response from Llama model using Ollama."""
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            temperature=0.7,
            max_tokens=100
        )
        return response['response']

    def compute_loss(
        self,
        input_text: str,
        target_text: str,
        alpha: float = 0.5
    ) -> torch.Tensor:
        """
        Compute loss combining semantic similarity and language model likelihood.
        
        Args:
            input_text: Input prompt
            target_text: Target response
            alpha: Weight for semantic similarity vs language model loss
        """
        # Get embeddings
        input_emb = self.encode_text(input_text)
        target_emb = self.encode_text(target_text)
        
        # Semantic similarity loss
        sim_loss = 1 - F.cosine_similarity(input_emb, target_emb, dim=0)
        
        # Get LLM response with soft prompt
        soft_prompt_text = self.tokenizer.decode(
            self.soft_prompt.argmax(dim=-1)
        )
        model_response = self.get_llama_response(
            soft_prompt_text + " " + input_text
        )
        response_emb = self.encode_text(model_response)
        
        # Language model loss
        lm_loss = 1 - F.cosine_similarity(response_emb, target_emb, dim=0)
        
        # Combined loss
        total_loss = alpha * sim_loss + (1 - alpha) * lm_loss
        return total_loss

    def train(
        self,
        train_data: List[Tuple[str, str]],
        epochs: int = 10,
        alpha: float = 0.5
    ):
        """
        Train the soft prompt.
        
        Args:
            train_data: List of (input_text, target_text) pairs
            epochs: Number of training epochs
            alpha: Weight for loss combination
        """
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(train_data, desc=f"Epoch {epoch + 1}/{epochs}")
            
            for input_text, target_text in progress_bar:
                self.optimizer.zero_grad()
                loss = self.compute_loss(input_text, target_text, alpha)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": total_loss / len(train_data)})

    def save_soft_prompt(self, path: str):
        """Save the trained soft prompt."""
        torch.save(self.soft_prompt.state_dict(), path)

    def load_soft_prompt(self, path: str):
        """Load a trained soft prompt."""
        state_dict = torch.load(path)
        self.soft_prompt.load_state_dict(state_dict)

    def generate_with_soft_prompt(self, input_text: str) -> str:
        """Generate text using the trained soft prompt."""
        soft_prompt_text = self.tokenizer.decode(
            self.soft_prompt.argmax(dim=-1)
        )
        return self.get_llama_response(soft_prompt_text + " " + input_text)