import requests
from io import BytesIO
from PIL import Image

from aura_sr import AuraSR

IMAGES = 'images'

def load_image_from_url(url):
    response = requests.get(url)
    image_data = BytesIO(response.content)
    return Image.open(image_data)

def aura_upscale(image_name):
    """
    upscaler
    """
    aura_sr = AuraSR.from_pretrained("fal/AuraSR-v2")
    image = Image.open(f'{IMAGES}/{image_name}')
    upscaled_image = aura_sr.upscale_4x_overlapped(image)
    upscaled_image_16x = aura_sr.upscale_4x_overlapped(upscaled_image)
    upscaled_image_16x.save(f'{IMAGES}/ups16x{image_name}') 

def main():
    """
    main testing
    """
    aura_upscale('cats.png')

if __name__ == "__main__":
    main()
