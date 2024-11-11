import torch
from torch import nn, optim
from torch.nn.functional import cosine_similarity
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
import torch.nn.functional as F

IMAGES = 'images/'
EPOCHS = 50  # Increased from 10
INIT_RND_VALUE = 0.5  # Reduced from 2.9
LEARNING_RATE = 1e-4  # Reduced from 0.7
NUM_TOKENS = 10
PROMPT = "Create an enchanting, whimsical landscape with balanced warm golden hues and soothing pastel blues..."
TITLE = "new_landscape"

def compare_embeddings(pipe, soft_prompt_embeddings, real_prompt):
    """Compare soft prompt embeddings with real prompt embeddings."""
    input_ids = pipe.tokenizer(real_prompt, return_tensors="pt").input_ids.to(pipe.device)
    real_prompt_embeddings = pipe.text_encoder(input_ids).last_hidden_state
    
    # Normalize embeddings before computing similarity
    soft_prompt_norm = F.normalize(soft_prompt_embeddings, p=2, dim=1)
    real_prompt_norm = F.normalize(real_prompt_embeddings, p=2, dim=2)
    
    # Compute similarity for each token
    similarities = torch.matmul(soft_prompt_norm, real_prompt_norm.squeeze(0).T)
    return similarities

def fine_tune_soft_prompt(pipe, prompt, num_virtual_tokens=NUM_TOKENS, num_steps=EPOCHS, learning_rate=LEARNING_RATE):
    device = pipe.device
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    
    # Initialize with smaller random values and proper dtype
    soft_prompt_embeddings = nn.Parameter(
        torch.randn(1, num_virtual_tokens, text_encoder.config.hidden_size, 
                   device=device, dtype=text_encoder.dtype) * INIT_RND_VALUE
    )
    
    # Get target embeddings
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        target_embeddings = text_encoder(input_ids).last_hidden_state
    
    # Freeze the text encoder
    for param in text_encoder.parameters():
        param.requires_grad = False
    
    optimizer = optim.AdamW([soft_prompt_embeddings], lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    
    best_loss = float('inf')
    best_embeddings = None
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Concatenate soft prompts with target embeddings
        combined_embeddings = torch.cat([soft_prompt_embeddings, target_embeddings], dim=1)
        
        # Generate latents
        latents = torch.randn(
            (1, pipe.unet.config.in_channels, pipe.unet.config.sample_size, pipe.unet.config.sample_size),
            device=device, dtype=text_encoder.dtype
        )
        
        # Predict noise
        timesteps = torch.tensor([999], device=device)  # Start with high noise level
        noise_pred = pipe.unet(latents, timesteps, encoder_hidden_states=combined_embeddings).sample
        
        # Multiple loss components
        # 1. Reconstruction loss
        recon_loss = F.mse_loss(noise_pred, torch.zeros_like(noise_pred))
        
        # 2. Similarity loss with target embeddings
        sim_loss = -torch.mean(cosine_similarity(
            soft_prompt_embeddings.view(-1, text_encoder.config.hidden_size),
            target_embeddings.view(-1, text_encoder.config.hidden_size)
        ))
        
        # 3. Regularization loss to prevent embeddings from growing too large
        reg_loss = 0.01 * torch.norm(soft_prompt_embeddings)
        
        # Combined loss
        loss = recon_loss + 0.1 * sim_loss + reg_loss
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([soft_prompt_embeddings], max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Save best embeddings
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_embeddings = soft_prompt_embeddings.detach().clone()
        
        if step % 10 == 0:
            print(f"Step {step}/{num_steps}, Loss: {loss.item():.4f}")
    
    return best_embeddings

def generate_image(pipe, embeddings, prompt, num_inference_steps=50, title="output"):
    input_ids = pipe.tokenizer(prompt, return_tensors="pt").input_ids.to(pipe.device)
    prompt_embeddings = pipe.text_encoder(input_ids).last_hidden_state
    
    # Concatenate soft prompts with prompt embeddings
    combined_embeddings = torch.cat([embeddings, prompt_embeddings], dim=1)
    
    # Generate image with noise scheduling
    with torch.no_grad():
        image = pipe(
            prompt_embeds=combined_embeddings,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5,  # Added guidance scale
            width=512,
            height=512,
        ).images[0]
    
    image.save(f"{IMAGES}/{title}.png")
    return f"{IMAGES}/{title}.png"

def dream_model(title="output", prompt=PROMPT, num_inference_steps=50):
    model_id = "dreamlike-art/dreamlike-diffusion-1.0"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to("cuda")
    
    # Enable attention slicing for memory efficiency
    pipe.enable_attention_slicing()
    
    # Fine-tune soft prompts
    soft_prompt_embeddings = fine_tune_soft_prompt(pipe, prompt)
    
    # Generate and save image
    image_path = generate_image(pipe, soft_prompt_embeddings, prompt, 
                              num_inference_steps=num_inference_steps, 
                              title=title)
    
    # Compare and print similarities
    similarities = compare_embeddings(pipe, soft_prompt_embeddings.squeeze(0), prompt)
    print(f"Token-wise similarities:\n{similarities}")
    
    return image_path

if __name__ == '__main__':
    dream_model(title=TITLE, prompt=PROMPT)
