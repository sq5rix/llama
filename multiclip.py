from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import StableDiffusionPipeline
import torch

def generate_image_from_long_prompt(model_name="CompVis/stable-diffusion-v1-4", 
                                    clip_model_name="openai/clip-vit-large-patch14", 
                                    prompt=None, 
                                    aggregation="average"):
    """
    Generates an image from a long prompt using CLIP in chunks and aggregation.
    
    Parameters:
    - model_name (str): The name of the Stable Diffusion model to load.
    - clip_model_name (str): The name of the CLIP model for text encoding.
    - prompt (str): The long prompt text to use.
    - aggregation (str): Aggregation method for embeddings ("average" or "concatenate").
    
    Returns:
    - images (List[PIL.Image]): Generated image(s) from the long prompt.
    """
    
    # Initialize CLIP tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    text_encoder = CLIPTextModel.from_pretrained(clip_model_name).to("cuda")
    
    # Tokenize the prompt and split into chunks of 77 tokens each
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=77)
    input_ids = inputs.input_ids.squeeze()
    chunk_size = 77
    input_chunks = [input_ids[i:i + chunk_size] for i in range(0, len(input_ids), chunk_size)]
    
    # Encode each chunk independently
    chunk_embeddings = []
    for chunk in input_chunks:
        chunk = chunk.unsqueeze(0).to("cuda")  # Add batch dimension and move to GPU
        with torch.no_grad():
            embedding = text_encoder(chunk).last_hidden_state
        chunk_embeddings.append(embedding)
    
    # Aggregate the embeddings based on the specified method
    if aggregation == "average":
        combined_embedding = torch.mean(torch.stack(chunk_embeddings), dim=0)
    elif aggregation == "concatenate":
        combined_embedding = torch.cat(chunk_embeddings, dim=1)
    else:
        raise ValueError("Aggregation method must be 'average' or 'concatenate'")
    
    # Initialize the Stable Diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe.to("cuda")
    
    # Generate the image using the combined embedding
    with torch.no_grad():
        images = pipe(prompt_embeds=combined_embedding).images
    
    return images

# Example usage:
prompt = "A very detailed description that goes beyond 77 tokens... (insert your long prompt here)"
images = generate_image_from_long_prompt(prompt=prompt, aggregation="average")

# Save the first image to a file
img_path = "generated_image.png"
images[0].save(img_path)