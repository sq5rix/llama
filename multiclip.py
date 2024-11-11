from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import StableDiffusionPipeline
import torch

def create_pipeline_with_long_prompt(model_name="CompVis/stable-diffusion-v1-4", 
                                     clip_model_name="openai/clip-vit-large-patch14", 
                                     prompt=None, 
                                     aggregation="average"):
    """
    Creates a Stable Diffusion pipeline with a long prompt using CLIP, chunked and aggregated.
    
    Parameters:
    - model_name (str): The name of the Stable Diffusion model to load.
    - clip_model_name (str): The name of the CLIP model for text encoding.
    - prompt (str): The long prompt text to use.
    - aggregation (str): Aggregation method for embeddings ("average" or "concatenate").
    
    Returns:
    - pipe (StableDiffusionPipeline): A Stable Diffusion pipeline with a custom generate method for long prompts.
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
    
    # Add a custom generate method to use the combined embedding
    def generate_with_custom_embedding(pipe, embedding, **kwargs):
        with torch.no_grad():
            images = pipe(prompt_embeds=embedding, **kwargs).images
        return images
    
    # Attach the generate method to the pipeline and set custom embedding
    pipe.generate_with_custom_embedding = lambda **kwargs: generate_with_custom_embedding(pipe, combined_embedding, **kwargs)
    
    return pipe

def main():
    # Example usage:
    prompt = "A very detailed description that goes beyond 77 tokens... (insert your long prompt here)"
    pipe = create_pipeline_with_long_prompt(prompt=prompt, aggregation="average")

    # Generate an image using the custom embedding
    image = pipe.generate_with_custom_embedding()