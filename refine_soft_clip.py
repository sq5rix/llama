import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
from torch import nn, optim

IMAGES = 'images/'

def fine_tune_soft_prompt(pipe, prompt, num_virtual_tokens=5, num_steps=100, learning_rate=0.1):
    """
    Fine-tunes soft prompts to improve image generation.
    
    Args:
        pipe: The Stable Diffusion pipeline with a loaded model.
        prompt: The initial text prompt.
        num_virtual_tokens: Number of soft prompt tokens to prepend.
        num_steps: Number of optimization steps.
        learning_rate: Learning rate for the optimizer.
        
    Returns:
        The optimized prompt embeddings.
    """
    device = pipe.device
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # Initialize soft prompt embeddings
    soft_prompt_embeddings = nn.Parameter(
        torch.randn(num_virtual_tokens, text_encoder.config.hidden_size, device=device, dtype=torch.float16)
    )
    
    # Freeze the text encoder
    text_encoder.requires_grad_(False)
    
    # Optimizer
    optimizer = optim.AdamW([soft_prompt_embeddings], lr=learning_rate)
    
    # Optimization loop
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Get embeddings for the actual prompt
        #prompt_embeddings = text_encoder.embeddings(input_ids).last_hidden_state
        prompt_embeddings = text_encoder(input_ids).last_hidden_state
        
        # Concatenate soft prompts with actual prompt embeddings
        embeddings = torch.cat([soft_prompt_embeddings.unsqueeze(0), prompt_embeddings], dim=1)
        
        # Generate latents (dummy latents since we're only optimizing the prompt)
        latents = torch.randn((1, pipe.unet.in_channels, pipe.unet.sample_size, pipe.unet.sample_size), device=device, dtype=torch.float16)
        
        # Get the model output
        noise_pred = pipe.unet(latents, timestep=torch.tensor([0]).to(device), encoder_hidden_states=embeddings).sample
        
        # Define a dummy loss (you can customize this)
        loss = noise_pred.pow(2).mean()
        
        # Backpropagate
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step}/{num_steps}, Loss: {loss.item()}")
    
    # Return the optimized embeddings
    return soft_prompt_embeddings.detach()
    
def dream_model(title, prompt, num_inference_steps=50):
    model_id = "dreamlike-art/dreamlike-diffusion-1.0"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to("cuda")
    
    # Fine-tune soft prompts
    soft_prompt_embeddings = fine_tune_soft_prompt(pipe, prompt)
    
    # Tokenize the prompt
    tokenizer = pipe.tokenizer
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(pipe.device)
    
    # Get embeddings for the actual prompt
    #prompt_embeddings = pipe.text_encoder.embeddings(input_ids).last_hidden_state
    prompt_embeddings = text_encoder(input_ids).last_hidden_state
    
    # Concatenate soft prompts with actual prompt embeddings
    embeddings = torch.cat([soft_prompt_embeddings.unsqueeze(0), prompt_embeddings], dim=1)
    
    # Generate the image using the optimized embeddings
    with torch.no_grad():
        image = pipe(
            prompt_embeds=embeddings,
            num_inference_steps=num_inference_steps,
            width=512,
            height=512,
        ).images[0]
    
    image.save(f"{IMAGES}/{title}.png")
    return f"{IMAGES}/{title}.png"

if __name__=='__main__':
    dream_model('lake', 'a beautiful sunset over mountains with serene lake reflection', num_inference_steps=32)
