import argparse
import requests
from io import BytesIO
from PIL import Image

from aura_sr import AuraSR

def load_image_from_url(url):
    response = requests.get(url)
    image_data = BytesIO(response.content)
    return Image.open(image_data)

def aura_upscale(image_name):
    """
    upscaler
    """
    aura_sr = AuraSR.from_pretrained("fal/AuraSR-v2")
    image = Image.open(f'pix/{image_name}')
    upscaled_image = aura_sr.upscale_4x(image)
    upscaled_image.save(f'pix/upscaylr/{image_name}') 

def main():
    """
    main testing
    """
    parser = argparse.ArgumentParser(description="Process and rescale images based on an SQLite database table.")
    parser.add_argument('image_name', type=str, help="The name of the image to process.")
    args = parser.parse_args()
    aura_upscale(args.image_name)
    print(f"Table name received: {args.image_name}")

if __name__ == "__main__":
    main()
