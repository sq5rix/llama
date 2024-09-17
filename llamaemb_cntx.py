import glob
import json
import os
from pprint import pprint
import numpy as np
import ollama
from numpy.linalg import norm

SYSTEM_PROMPT = """You are a creative helpful artistic souls who answers questions
    based on snippets of text provided and all your information. 
    Context: 
"""
EMBEDDINGS_DIR = "embeddings"
DATA_DIR = "data"
EMBEDDINGS_FILE = "all_embeddings.json"

def parse_file(filename):
    with open(filename, encoding="utf-8-sig") as f:
        paragraphs = []
        buffer = []
        for line in f.readlines():
            line = line.strip()
            if line:
                buffer.append(line)
            elif len(buffer):
                paragraphs.append((" ").join(buffer))
                buffer = []
        if len(buffer):
            paragraphs.append((" ").join(buffer))
        return paragraphs


def save_embeddings(embeddings):
    # Create directory if it doesn't exist
    if not os.path.exists("embeddings"):
        os.makedirs(EMBEDDINGS_DIR)
    # Dump embeddings to JSON
    with open("embeddings/all_embeddings.json", "w") as f:
        json.dump(embeddings, f)


def load_embeddings():
    # Check if embeddings file exists
    if not os.path.exists("embeddings/all_embeddings.json"):
        return False
    # Load embeddings from JSON
    with open(f"{EMBEDDINGS_DIR}/{EMBEDDINGS_FILE}", "r") as f:
        return json.load(f)


def get_embeddings(modelname, chunks):
    # Check if embeddings are already saved
    if (embeddings := load_embeddings()) is not False:
        return embeddings
    # Get embeddings from Ollama
    embeddings = [
        ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
        for chunk in chunks
    ]
    # Save embeddings
    save_embeddings(embeddings)
    return embeddings


def find_most_similar(needle, haystack):
    """
    znajdź chunki tekstu z RAG (haystack)
    najbliżej promptu (needle)
    """
    needle_norm = norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)


def get_all_paragraphs_from_data():
    paragraphs = []
    files = glob.glob(DATA_DIR + "/*")
    for filename in files:
        paragraphs.extend(parse_file(filename))
    return paragraphs


def ask_llama(prompt, context=None):
    if context is None:
        paragraphs = get_all_paragraphs_from_data()
        embeddings = get_embeddings("nomic-embed-text", paragraphs)
        prompt_embedding = ollama.embeddings(model="nomic-embed-text", prompt=prompt)[
            "embedding"
        ]
        most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:6]
        local_context = "\n".join(paragraphs[item[1]] for item in most_similar_chunks)
    else:
        local_context = context

    full_context = SYSTEM_PROMPT + local_context
    response = ollama.generate(model="llama3.1:latest", prompt=prompt, system=full_context)
    return response, local_context

if __name__ == "__main__":
    prompt = input("what do you want to know? -> ")
    response, context = ask_llama(prompt)
    print("\n\n")
    print(response["response"])
    #print("\n\n")
    #print(context)
