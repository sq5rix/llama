from diffusers import AuraFlowPipeline
import torch

class AuraFlow():
    """
    class to keep aura flow data
    """
    def __init__(self):
        self.pipeline = AuraFlowPipeline.from_pretrained(
            "fal/AuraFlow-v0.3",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda")

    def generate_image(self, title, prompt, num_inference_steps=50, width=1008,height=672,guidance_scale=3.5,):
        image = self.pipeline(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps, 
            generator=torch.Generator().manual_seed(1),
            guidance_scale=guidance_scale,
            text_encoder_3=None,
            tokenizer_3=None,
        ).images[0]
        image.save(f"pix/{title}.png")

def main():
    """
    test
    """
    aura = AuraFlow()
    aura.generate_image('image', 'white cat and beatiful blonde girl')


if __name__ == "__main__":
    main()
