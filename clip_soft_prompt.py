import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from typing import List, Optional

class SoftPromptConverter:
    def __init__(
        self,
        num_tokens: int = 10,  # number of soft tokens to learn
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.num_tokens = num_tokens
        
        # Initialize CLIP text encoder and tokenizer
        self.clip = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # Initialize random soft prompt tokens (starts as random noise)
        # These will be trained to capture the essence of your hard prompt
        embedding_dim = self.clip.get_input_embeddings().weight.shape[1]
        self.soft_tokens = nn.Parameter(
            torch.randn(num_tokens, embedding_dim).to(device)
        )
        
    def encode_hard_prompt(self, prompt: str) -> torch.Tensor:
        """Get CLIP embeddings for the hard prompt"""
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            prompt_embeddings = self.clip(
                input_ids=tokens.input_ids
            ).last_hidden_state
            
        return prompt_embeddings

    def train_soft_prompt(
        self,
        hard_prompt: str,
        num_iterations: int = 1000,
        learning_rate: float = 1e-4
    ):
        """
        Train soft tokens to capture the essence of the hard prompt.
        Example:
        hard_prompt = "image of serene landscape at sunrise with trees mountains and lake"
        Will train soft tokens like [v1][v2][v3]... to capture this meaning
        """
        # Get target embeddings from the hard prompt
        target_embeddings = self.encode_hard_prompt(hard_prompt)
        
        # Initialize optimizer for soft tokens
        optimizer = torch.optim.AdamW([self.soft_tokens], lr=learning_rate)
        
        # Training loop
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Expand soft tokens to match batch size
            soft_tokens_expanded = self.soft_tokens.unsqueeze(0)
            
            # Get current soft prompt embeddings
            soft_embeddings = self.clip(
                inputs_embeds=soft_tokens_expanded
            ).last_hidden_state
            
            # Compute loss: try to match the semantic meaning of hard prompt
            loss = nn.functional.mse_loss(
                soft_embeddings,
                target_embeddings[:, :self.num_tokens, :]
            )
            
            # Add cosine similarity loss to maintain semantic direction
            cos_sim = nn.functional.cosine_similarity(
                soft_embeddings.mean(dim=1),
                target_embeddings.mean(dim=1)
            ).mean()
            
            total_loss = loss - cos_sim  # Minimize MSE while maximizing similarity
            
            # Update soft tokens
            total_loss.backward()
            optimizer.step()
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {total_loss.item():.4f}")

    def get_soft_prompt_tokens(self) -> List[str]:
        """
        Convert trained soft tokens to string representation.
        Returns something like ['v1', 'v2', 'v3'...] where each token
        captures part of the original prompt's meaning
        """
        # Get closest tokens in CLIP's vocabulary (for visualization)
        with torch.no_grad():
            vocab_embeddings = self.clip.get_input_embeddings().weight
            soft_tokens_norm = self.soft_tokens / self.soft_tokens.norm(dim=1, keepdim=True)
            vocab_embeddings_norm = vocab_embeddings / vocab_embeddings.norm(dim=1, keepdim=True)
            
            similarity = torch.mm(soft_tokens_norm, vocab_embeddings_norm.t())
            closest_tokens = similarity.argmax(dim=1)
            
        # Convert to token strings (these are just representations, not the actual learned embeddings)
        token_strings = [f"<v{i}>" for i in range(self.num_tokens)]
        return token_strings

    def save_soft_prompt(self, path: str):
        """Save the trained soft prompt tokens"""
        torch.save(self.soft_tokens, path)

    def load_soft_prompt(self, path: str):
        """Load trained soft prompt tokens"""
        self.soft_tokens = torch.load(path)

# Example usage
def train_landscape_soft_prompt():
    # Initialize converter
    converter = SoftPromptConverter(num_tokens=10)
    
    # Original hard prompt
    hard_prompt = "image of serene landscape at sunrise with trees mountains and lake"
    
    # Train soft prompt tokens
    converter.train_soft_prompt(
        hard_prompt=hard_prompt,
        num_iterations=1000
    )
    
    # Get the trained tokens
    soft_tokens = converter.get_soft_prompt_tokens()
    print("Trained soft prompt tokens:", " ".join(soft_tokens))
    
    # Save the trained soft prompt
    converter.save_soft_prompt("landscape_soft_prompt.pt")
    
    return soft_tokens

# To use the trained soft prompt with an image generator:
def generate_with_soft_prompt(generator, soft_prompt_path: str, additional_text: str = ""):
    converter = SoftPromptConverter()
    converter.load_soft_prompt(soft_prompt_path)
    
    # Combine soft prompt with any additional text
    soft_tokens = converter.get_soft_prompt_tokens()
    full_prompt = " ".join(soft_tokens) + " " + additional_text
    
    # Generate image
    return generator.generate(full_prompt)
