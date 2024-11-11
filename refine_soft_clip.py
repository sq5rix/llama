import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
from torch import nn, optim
from torch.nn.functional import cosine_similarity

IMAGES = 'images/'

EPOCHS = 50
INIT_RND_VALUE = 0.1
LEARNING_RATE = 1.2
LOSS_SCALE = 100.0
ADAM_EPS = 1e-7
NUM_TOKENS = 10
PROMPT = "Create an enchanting, whimsical landscape with balanced warm golden hues and soothing pastel blues. A delicate rose petal lies on a carpet of soft, velvety moss, its edges gently curled as if kissed by the whispering dusk winds. The surrounding foliage, a mix of emerald-green ferns and wispy, lavender-hued wildflowers"
TITLE = 'whispers'

def compare_embeddings(pipe, soft_prompt_embeddings, real_prompt=PROMPT):
    # Get embeddings for a real prompt
    real_prompt = "a beautiful sunset over mountains with serene lake reflection"
    input_ids = pipe.tokenizer(real_prompt, return_tensors="pt").input_ids.to(pipe.device)
    real_prompt_embeddings = pipe.text_encoder(input_ids).last_hidden_state
    # Compute cosine similarity between soft prompts and real prompt
    similarity = cosine_similarity(soft_prompt_embeddings, real_prompt_embeddings.mean(dim=1))
    print(f"Similarity: {similarity}")

def fine_tune_soft_prompt(pipe, prompt, num_virtual_tokens=NUM_TOKENS, num_steps=EPOCHS, learning_rate=LEARNING_RATE):
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
    pipe.unet.enable_gradient_checkpointing()
    if hasattr(pipe, "text_encoder") and hasattr(pipe.text_encoder, "gradient_checkpointing_enable"):
        pipe.text_encoder.gradient_checkpointing_enable()
    
    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    # Initialize soft prompt embeddings
    soft_prompt_embeddings = nn.Parameter(
        torch.randn(num_virtual_tokens, text_encoder.config.hidden_size, device=device, dtype=torch.float16) * INIT_RND_VALUE
    )
    torch.nn.utils.clip_grad_norm_([soft_prompt_embeddings], max_norm=1.0)
    
    # Freeze the text encoder
    text_encoder.requires_grad_(False)
    
    # Optimizer
    optimizer = optim.AdamW([soft_prompt_embeddings], lr=learning_rate, eps=ADAM_EPS)
    
    # Optimization loop
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Get embeddings for the actual prompt
        #prompt_embeddings = text_encoder.embeddings(input_ids).last_hidden_state
        prompt_embeddings = text_encoder(input_ids).last_hidden_state
        
        # Concatenate soft prompts with actual prompt embeddings
        embeddings = torch.cat([soft_prompt_embeddings.unsqueeze(0), prompt_embeddings], dim=1)
        
        #latents = torch.randn((1, pipe.unet.in_channels, pipe.unet.sample_size, pipe.unet.sample_size), device=device, dtype=torch.float16)
        latents = torch.randn((1, pipe.unet.config.in_channels, pipe.unet.config.sample_size, pipe.unet.config.sample_size), device=device, dtype=torch.float16)
        
        # Get the model output
        noise_pred = pipe.unet(latents, timestep=torch.tensor([0]).to(device), encoder_hidden_states=embeddings).sample
        
        # Define a dummy loss (you can customize this)
        loss = noise_pred.pow(2).mean() #* LOSS_SCALE
        
        # Backpropagate
        loss.backward()

        # scale back
        #for param in [soft_prompt_embeddings]:
            #param.grad /= LOSS_SCALE

        optimizer.step()
        
        print(f"Step {step}/{num_steps}, Loss: {loss.item()}")
    
    # Return the optimized embeddings
    return soft_prompt_embeddings.detach()
    
def dream_model(title=TITLE, prompt=PROMPT, num_inference_steps=50):
    model_id = "dreamlike-art/dreamlike-diffusion-1.0"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to("cuda")

    image = pipe(
        prompt=PROMPT,
        num_inference_steps=num_inference_steps,
        width=512,
        height=512,
    ).images[0]
    image.save(f"{IMAGES}/{title}_orig.png")
    
    # Fine-tune soft prompts
    soft_prompt_embeddings = fine_tune_soft_prompt(pipe, prompt)
    
    # Tokenize the prompt
    tokenizer = pipe.tokenizer
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(pipe.device)
    
    # Get embeddings for the actual prompt
    #prompt_embeddings = pipe.text_encoder.embeddings(input_ids).last_hidden_state
    prompt_embeddings = pipe.text_encoder(input_ids).last_hidden_state
    
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
    compare_embeddings(pipe, soft_prompt_embeddings)
    return f"{IMAGES}/{title}.png"

if __name__=='__main__':
    dream_model()
