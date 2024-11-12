import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer

from utils import count_words

# MODEL = "CompVis/stable-diffusion-v1-4"
MODEL = "dreamlike-art/dreamlike-diffusion-1.0"
CLIP = "openai/clip-vit-large-patch14"

PROMPT_SHORT = """
    Vibrant autumn landscape: warm sunlight filters through golden leaves 
    of deciduous trees, illuminating a carpet of crimson and amber hues 
    on the forest floor. A few wispy clouds drift lazily across a blue sky, 
    casting dappled shadows on the terrain. 
    """

PROMPT_LONG = """
A young woman stands at the center of a vibrant autumn landscape, 
her figure radiating warmth and energy against the cool, blue-green 
hues of the forest. The warm sunlight filters through the golden 
leaves above, casting a golden glow around her, but also creates 
a dramatic backlight that sets her features ablaze with depth and 
dimension. Her long, raven-black hair is blown back by the gentle 
breeze, framing her face and highlighting the sharp angles of her 
cheekbones. She wears a rich, crimson cloak cinched at the waist 
with a wide, leather belt, its bold color popping against the muted 
tones of the forest floor. As she walks through the landscape, 
the camera's attention is drawn to her figure, isolating her from 
the surroundings and making her the focal point. The wooden bridge 
in the background recedes into the distance, becoming a subtle 
element that adds depth and context to the scene without competing 
with the woman's presence. The entire composition is balanced around 
the axis of her body, creating a sense of symmetry and harmony. 
The warm sunlight casts long shadows across the forest floor, 
leading the viewer's eye to the woman and drawing attention 
to her striking features.
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


def dream_model_long(
    title,
    prompt,
    width=512,
    height=512,
    num_inference_steps=50,
    aggregation="concatenate",
):
    model_id = "dreamlike-art/dreamlike-diffusion-1.0"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    combined_embedding = tokenize_extended_clip(prompt, aggregation)

    image = pipe(
        prompt_embeds=combined_embedding,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        guidance_scale=7.5,
    ).images[0]
    image.save(f"images/{title}.png")
    return f"images/{title}.png"


def dream_model_short(
    title,
    prompt,
    width=512,
    height=512,
    num_inference_steps=50,
):
    model_id = "dreamlike-art/dreamlike-diffusion-1.0"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    image = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        guidance_scale=7.5,
    ).images[0]
    image.save(f"images/{title}.png")
    return f"images/{title}.png"


print("short prompt len: ", count_words(PROMPT_SHORT))
print("long prompt len: ", count_words(PROMPT_LONG))

_ = dream_model_short("test_cut_prompt", PROMPT_SHORT, width=1024, height=1024)
_ = dream_model_long(
    "test_long_prompt_concat", PROMPT_SHORT + PROMPT_LONG, width=1024, height=1024
)
_ = dream_model_long(
    "test_long_prompt_average",
    PROMPT_SHORT + PROMPT_LONG,
    aggregation="average",
    width=1024,
    height=1024,
)
