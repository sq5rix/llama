import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer

from utils import count_words

# MODEL = "CompVis/stable-diffusion-v1-4"
MODEL = "dreamlike-art/dreamlike-diffusion-1.0"
CLIP = "openai/clip-vit-large-patch14"

DINOS_SHORT = """
A lush, vibrant Jurassic jungle sprawls before us, its dense foliage and towering canopy creating a tapestry of greens that stretches as far as the eye can see. The air is thick with humidity, and the scent of blooming flowers and decaying vegetation hangs heavy over the landscape.
"""

DINOS = """

A lush, vibrant Jurassic jungle sprawls before us, its dense foliage and towering canopy creating a tapestry of greens that stretches as far as the eye can see. The air is thick with humidity, and the scent of blooming flowers and decaying vegetation hangs heavy over the landscape.
To our left, a herd of massive Apatosaurs grazes on the lush undergrowth, their long necks bent as they reach for the treetops. Their scaly skin glistens in the dappled sunlight filtering through the canopy above, and their gentle lowing echoes through the jungle. Nearby, a smaller group of Camptosaurs feeds on the tender shoots of ferns and cycads, their more compact bodies weaving between the Apatosaur's larger forms.
Deeper in the jungle, a trio of Allosaurs stalks its prey, their sharp eyes scanning the underbrush for any sign of movement. These apex predators are built for speed and stealth, their sleek, muscular bodies capable of reaching incredible velocities as they pursue their unsuspecting quarry. A lone Olorotitan wanders through the jungle, its massive body and distinctive crest marking it out from other hadrosaurs.
In a sun-dappled clearing, a pair of Stegosaurs basks in the warmth, their plates glistening with dew and their spiky tails swishing lazily behind them. Nearby, a lone Ceratosaur patrols the edge of the jungle, its distinctive horns and crested head making it a formidable sight to behold.
As we venture deeper into the jungle, the sounds of distant roaring grow louder. A group of massive Tyrannosaurs moves through the undergrowth, their sharp eyes fixed intently on some unseen target. The air seems to vibrate with tension as they stalk their prey, their massive feet barely making a sound as they move.
In the distance, a flock of Pteranodons soars overhead, their wings beating in unison as they ride the thermals above the jungle. A lone Oviraptor stalks its prey through the underbrush, its sharp eyes scanning for any sign of movement.
The light begins to fade as the sun dips below the horizon, casting long shadows across the jungle floor. The air cools, and the sounds of the jungle begin to change, as nocturnal creatures stir from their daytime slumber. The scent of blooming flowers gives way to the musky aroma of nocturnal predators, and the jungle transforms into a world of mystery and danger.
The camera's eye pans across this vibrant, teeming ecosystem, taking in the intricate web of life that exists within the Jurassic jungle. We see the delicate balance between predator and prey, the adaptability of species to their environment, and the sheer diversity of life that thrives in this ancient world.

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


print("short prompt len: ", count_words(DINOS_SHORT))
print("long prompt len: ", count_words(DINOS))

_ = dream_model_short("test_cut_prompt", DINOS_SHORT, width=1024, height=1024)
_ = dream_model_long("test_long_prompt_concat", DINOS, width=1024, height=1024)
_ = dream_model_long(
    "test_long_prompt_average",
    DINOS,
    aggregation="average",
    width=1024,
    height=1024,
)
