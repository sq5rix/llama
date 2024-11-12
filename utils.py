"""
utils
"""

import math
import random
import uuid

import requests
from PIL import Image


def generate_uuid():
    """
    generates random uuid hex
    """
    random_uuid = str(uuid.uuid4())
    return random_uuid


def get_random_item(item_list):
    """
    returns weighted choice according to weights
    """
    return random.choices(item_list)[0]


def get_random_item_weighted(item_list, weights):
    """
    returns weighted choice according to weights
    """
    if len(weights) != len(item_list):
        raise ValueError("Weights and item_list must be the same length")
    return random.choices(item_list, weights=weights)[0]


def get_poisson_item(item_list, lam=0.2):
    """
    Number of items in the list
    """
    n = len(item_list)
    poisson_weights = [math.exp(-lam * i) for i in range(n)]
    total_weight = sum(poisson_weights)
    normalized_weights = [w / total_weight for w in poisson_weights]
    return random.choices(item_list, weights=normalized_weights, k=1)[0]


def download_image(url, save_path):
    try:
        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Open the file in write-binary mode and save the content
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"Image downloaded and saved to {save_path}")
        else:
            print(f"Failed to download image. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")


def count_words(text):
    # Split the text into words using whitespace as the delimiter
    words = text.split()

    # Count the number of words in the list
    word_count = len(words)

    return word_count


def resize_image(image_path, scale, output_image_name):
    im = Image.open(image_path)
    width, height = im.size
    new_width = int(width * scale)
    new_height = int(height * scale)
    im_small = im.resize((new_width, new_height), Image.LANCZOS)
    im_small.save(output_image_name)


def main():
    """
    main
    """
    # result = generate_uuid()
    # item_list = ["apple", "banana", "cherry"]
    # weights = [0.2, 0.4, 0.4]
    # print("random: ", get_random_item(item_list))
    # print("weight: ", get_random_item_weighted(item_list, weights))
    # print("poisson: ", get_poisson_item(item_list))
    url = "https://file.io/RkKxYSwwmRxF"  # Replace with your image URL
    save_path = "downloaded_image.jpg"  # Specify where you want to save the image
    download_image(url, save_path)
    resize_image(save_path, 0.5, "pix/x.jpg")


if __name__ == "__main__":
    main()
