import os
import gc
import torch
from diffusers import StableDiffusion3Pipeline, StableDiffusionPipeline
from huggingface_hub import login
from numpy import size
from utils import count_words

IMAGES = 'images/'

PROMPT = """Imagine a serene moonlit garden, where a majestic cat 
    with shimmering silver fur is perched 
    on a toadstool throne. Its eyes gleam like 
    lanterns in the dark, as it gazes up at a 
    giant crescent moon hanging low in the sky. 
    """
P2=""""
Two golden retriever puppies frolic in a sun-drenched clearing with beautiful blonde girl. 
Crimson and gold leaves swirl around them as they chase each other, their fluffy fur 
catching the warm light. A blanket of vibrant autumn colors envelops the scene.
"""


def login_hugging():
    login()


def medium_model(title, prompt):
    torch.cuda.empty_cache()
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        # torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")

    image = pipe(
        prompt=prompt,
        guidance_scale=3.0,
        num_inference_steps=50,
    ).images[0]
    image.save("{IMAGES}/{title}.png")


def dream_model(title, prompt, num_inference_steps=50):
    model_id = "dreamlike-art/dreamlike-diffusion-1.0"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    image = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        width=1024,
        height=1024,
    ).images[0]
    image.save(f"{IMAGES}/{title}.png")
    return f"{IMAGES}/{title}.png"


def main():
    """
    test
    """
    dream_model('dogs', P2)


if __name__ == "__main__":
    main()
