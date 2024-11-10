from typing import List, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from peft import get_peft_config, PeftModel, PrefixTuningConfig
import numpy as np
from tqdm import tqdm
import flux

class CLIPTuner:
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        num_prefix_tokens: int = 8,
        num_virtual_tokens: int = 8,
        use_prefix: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize CLIP tuner for image generation guidance.
        
        Args:
            clip_model_name: Name of CLIP model to use
            num_prefix_tokens: Number of prefix tokens for prefix tuning
            num_virtual_tokens: Number of virtual tokens for soft prompting
            use_prefix: Whether to use prefix tuning (True) or soft prompt (False)
            device: Device to run the model on
        """
        self.device = device
        self.use_prefix = use_prefix
        
        # Initialize CLIP text encoder and tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(clip_model_name).to(device)
        
        if use_prefix:
            # Configure prefix tuning
            peft_config = PrefixTuningConfig(
                task_type="CAUSAL_LM",
                num_virtual_tokens=num_prefix_tokens,
                token_dim=self.text_encoder.config.hidden_size,
                num_transformer_submodules=1,  # CLIP usually has 1 attention block per layer
                num_attention_heads=self.text_encoder.config.num_attention_heads,
                num_layers=self.text_encoder.config.num_hidden_layers
            )
            self.text_encoder = PeftModel.from_pretrained(
                self.text_encoder,
                peft_config
            )
        else:
            # Initialize soft prompt
            self.soft_prompt = nn.Parameter(
                torch.randn(num_virtual_tokens, self.text_encoder.config.hidden_size)
            ).to(device)

    class ImagePromptDataset(Dataset):
        def __init__(self, image_prompt_pairs: List[Dict[str, str]]):
            """
            Dataset for image-prompt pairs.
            
            Args:
                image_prompt_pairs: List of dicts with 'image' and 'prompt' keys
            """
            self.pairs = image_prompt_pairs

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            return self.pairs[idx]

    def prepare_flux_model(self):
        """Initialize and prepare Flux diffusion model."""
        self.flux_model = flux.load_diffusion_model()
        self.flux_model.to(self.device)
        self.flux_model.eval()

    def encode_text(self, prompt: str) -> torch.Tensor:
        """
        Encode text prompt with tuned CLIP.
        
        Args:
            prompt: Text prompt to encode
        """
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        if self.use_prefix:
            # Use prefix tuning
            text_embeddings = self.text_encoder(**tokens).last_hidden_state
        else:
            # Use soft prompt
            token_embeddings = self.text_encoder.get_input_embeddings()(tokens.input_ids)
            soft_prompt_expanded = self.soft_prompt.unsqueeze(0).expand(
                tokens.input_ids.shape[0], -1, -1
            )
            combined_embeddings = torch.cat([soft_prompt_expanded, token_embeddings], dim=1)
            text_embeddings = self.text_encoder(inputs_embeds=combined_embeddings).last_hidden_state

        return text_embeddings

    def train(
        self,
        train_data: List[Dict[str, str]],
        num_epochs: int = 10,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        save_path: Optional[str] = None
    ):
        """
        Train the tuned CLIP model.
        
        Args:
            train_data: List of image-prompt pairs
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            save_path: Path to save the model
        """
        dataset = self.ImagePromptDataset(train_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize optimizer
        if self.use_prefix:
            optimizer = torch.optim.AdamW(self.text_encoder.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.AdamW([self.soft_prompt], lr=learning_rate)
        
        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                optimizer.zero_grad()
                
                # Get text embeddings
                text_embeddings = self.encode_text(batch['prompt'])
                
                # Generate images using Flux
                generated_images = self.flux_model.generate(
                    text_embeddings=text_embeddings,
                    num_inference_steps=20
                )
                
                # Compute loss using CLIP's image-text similarity
                with torch.no_grad():
                    target_images = self.flux_model.preprocess_images(batch['image'])
                    target_features = self.flux_model.get_image_features(target_images)
                
                generated_features = self.flux_model.get_image_features(generated_images)
                loss = nn.functional.mse_loss(generated_features, target_features)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}, Average Loss: {total_loss/len(dataloader):.4f}")
            
            if save_path:
                self.save_model(f"{save_path}_epoch_{epoch+1}")

    def generate_image(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5
    ):
        """
        Generate image using tuned CLIP guidance.
        
        Args:
            prompt: Text prompt for image generation
            num_inference_steps: Number of denoising steps
            guidance_scale: Scale for classifier-free guidance
        """
        # Encode prompt with tuned CLIP
        text_embeddings = self.encode_text(prompt)
        
        # Generate image using Flux
        image = self.flux_model.generate(
            text_embeddings=text_embeddings,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        
        return image

    def save_model(self, path: str):
        """Save the tuned model."""
        if self.use_prefix:
            self.text_encoder.save_pretrained(path)
        else:
            torch.save(self.soft_prompt.state_dict(), f"{path}_soft_prompt.pt")
        
    def load_model(self, path: str):
        """Load a tuned model."""
        if self.use_prefix:
            self.text_encoder = PeftModel.from_pretrained(
                self.text_encoder,
                path
            )
        else:
            self.soft_prompt.load_state_dict(torch.load(f"{path}_soft_prompt.pt"))

# Example usage
def example_usage():
    # Initialize tuner
    tuner = CLIPTuner(
        use_prefix=True,  # Use prefix tuning
        num_prefix_tokens=16
    )
    
    # Prepare training data
    train_data = [
        {
            "image": "path/to/image1.jpg",
            "prompt": "a beautiful sunset over mountains"
        },
        {
            "image": "path/to/image2.jpg",
            "prompt": "a serene lake reflection"
        }
    ]
    
    # Train the model
    tuner.train(train_data, num_epochs=10)
    
    # Generate new image
    generated_image = tuner.generate_image(
        prompt="a misty forest at dawn",
        num_inference_steps=50
    )
    
    # Save the tuned model
    tuner.save_model("tuned_clip_model")

    if __name__=="__main__":
        example_usage()

