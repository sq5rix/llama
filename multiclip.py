from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import StableDiffusionPipeline
import torch

# Initialize CLIP tokenizer and text encoder
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")

# Define a long prompt
prompt = "A very detailed description that goes beyond 77 tokens... (insert your long prompt here)"

# Tokenize the prompt and split into chunks of 77 tokens each
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=77)
input_ids = inputs.input_ids.squeeze()

# Split input IDs into chunks
chunk_size = 77
input_chunks = [input_ids[i:i + chunk_size] for i in range(0, len(input_ids), chunk_size)]

# Encode each chunk independently
chunk_embeddings = []
for chunk in input_chunks:
    chunk = chunk.unsqueeze(0).to("cuda")  # Add batch dimension and move to GPU
    with torch.no_grad():
        embedding = text_encoder(chunk).last_hidden_state
    chunk_embeddings.append(embedding)

# Aggregate the embeddings (e.g., by averaging or concatenation)
# Option 1: Average embeddings
combined_embedding = torch.mean(torch.stack(chunk_embeddings), dim=0)

# Option 2: Concatenate embeddings (if your model can handle the increased dimension)
# combined_embedding = torch.cat(chunk_embeddings, dim=1)

# Load the Stable Diffusion model and set the custom text encoder embedding
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe.to("cuda")

# Replace the default text embedding with the combined embedding
pipe.text_encoder = text_encoder

# Use combined embedding for image generation
with torch.no_grad():
    images = pipe(prompt_embeds=combined_embedding).images  # Pass combined embedding to pipeline