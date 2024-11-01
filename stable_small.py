import random

import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline

#MODEL = "stabilityai/stable-diffusion-3-medium-diffusers"
MODEL = "stabilityai/stable-diffusion-3.5-medium"

MAX_SEED = np.iinfo(np.int32).max

PROMPT = """Imagine a serene moonlit garden, where a majestic cat 
    with shimmering silver fur is perched 
    on a toadstool throne. Its eyes gleam like 
    lanterns in the dark, as it gazes up at a 
    giant crescent moon hanging low in the sky. 
    """


def infer(
    pipe,
    title,
    prompt="hello cat",
    guidance_scale=3,
    negative_prompt="human figure",
    num_inference_steps=30,
    width=1024,
    height=1024,
):
    seed = random.randint(0, MAX_SEED)

    generator = torch.Generator().manual_seed(seed)
    pipe.to("cuda")
    # pipe.enable_sequential_cpu_offload()
    # pipe.enable_model_cpu_offload()
    image = pipe(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        generator=generator,
    ).images[0]
    image.save(f"pix/{title}.png")
    return title


def create_stable_image(title, prompt, width=512, height=512, num_inference_steps=50):
    """
    square image from stable medium
    """
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL,
        text_encoder_3=None,
        tokenizer_3=None,
        torch_dtype=torch.float16,
    )

    image_name = infer(
        pipe,
        title,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        negative_prompt='human figure, face, hands, double subject',
        width=width,
        height=height,
    )
    return image_name

if __name__ == "__main__":
    create_stable_image('cats', 'cats')
