import subprocess
import ollama
from pprint import pprint
from utils import resize_image, encode_image_to_base64

MODEL = "llava:13b"
CLEAN_GPU = f"ollama stop {MODEL}"

PIX_PATH = '/home/tom/MaxBounty/maxbounty/wp_pix/f78c9f5a-7c27-40f5-8531-93efabb7ae65.jpeg'
PIX_OUT = 'tmp/x.jpeg'
PROMPT = 'tell me what is in the picture, dont use words like appears, seems, just say what is in the image '

# Function to send a prompt to the llama3.1 model
def describe_picture(image_path): 
    """
    describe_picture using llava
    """
    pix_code = encode_image_to_base64(PIX_OUT)
    messages = [ 
        {
            'role': 'user', 
            'prompt': PROMPT,
            'images': [pix_code],
        }, ]
    response = ollama.chat(model=MODEL, messages=messages)
    return response['message']['content']

def main():
    """
    test
    """
    resp = describe_picture(PIX_PATH)
    print(resp)


if __name__ == "__main__":
    main()
