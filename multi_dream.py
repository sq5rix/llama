import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer

# MODEL = "CompVis/stable-diffusion-v1-4"
MODEL = "dreamlike-art/dreamlike-diffusion-1.0"
CLIP = "openai/clip-vit-large-patch14"

PROMPT = """
    Envision a serene, warm-lit scene: a delicate, hand-woven tapestry 
    suspended in mid-air above a shallow, moonlit pond. 
    The golden hues of dawn softly caress the surroundings, 
    with gentle wisps of fog dancing across the water’s surface. 
    A lone, gnarled tree stands sentinel on the far bank, 
    its branches stretching upwards like nature’s own cathedral. 
    Place the central axis of the canvas at a 45-degree angle 
    to the right edge of the frame, allowing its intricate patterns 
    to spill diagonally across the composition. 
    Balance the palette with complementary earthy tones: 
    rich soil, weathered stone, and the deep blues of a summer sky, 
    subtly nuanced in the shadows.
    """


def tokenize_extended_clip(prompt, aggregation="concatenate"):
    # Initialize CLIP tokenizer and text encoder
    clip_model_name = CLIP
    tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    text_encoder = CLIPTextModel.from_pretrained(clip_model_name).to("cuda")

    # Tokenize the prompt and split into chunks of 77 tokens each
    inputs = tokenizer(
        prompt, return_tensors="pt", padding=True, truncation=True, max_length=77
    )
    input_ids = inputs.input_ids.squeeze()
    chunk_size = 77
    input_chunks = [
        input_ids[i : i + chunk_size] for i in range(0, len(input_ids), chunk_size)
    ]
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
    return combined_embedding


def dream_model(
    title,
    prompt,
    num_inference_steps=50,
    aggregation="concatenate",
):
    model_id = "dreamlike-art/dreamlike-diffusion-1.0"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    combined_embedding = tokenize_extended_clip(prompt, aggregation)

    image = pipe(
        prompt_embeds=combined_embedding,
        # prompt=PROMPT,
        num_inference_steps=num_inference_steps,
        width=512,
        height=512,
        guidance_scale=7.5,
    ).images[0]
    image.save(f"images/{title}.png")
    return f"images/{title}.png"


_ = dream_model("paint_no_grad", PROMPT, num_inference_steps=50)
