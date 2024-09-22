import requests
from bs4 import BeautifulSoup
from duck import find_relevant_websites 
import ollama

def scrape_website(url):
    """
    Scrape the given website and extract relevant text information.
    """
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return "Error fetching the website."
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract all paragraph text as an example
        paragraphs = soup.find_all('p')
        text_content = "\n".join([p.get_text() for p in paragraphs])
        
        return text_content
    except Exception as e:
        return f"Error during scraping: {e}"

def process_with_llama(scraped_content):
    """
    Simulate processing with a LLM like Llama.
    In reality, this would involve calling an LLM API.
    """
    # Placeholder for actual LLM processing
    return f"Llama's response to the scraped content:\n{scraped_content[:500]}..."  # Truncate for example

def main(prompt):
    # Step 1: Find a relevant website for the prompt
    website_urls = find_relevant_websites(prompt)
    
    # Step 2: Scrape the websites
    scraped_content = ""
    for website in website_urls[:3]:
        scraped_content += scrape_website(website)
    
    # Step 3: Pass the scraped content to the LLM (Llama in this case)
    llama_response = process_with_llama(scraped_content)
    
    return llama_response

# Example usage (uncomment below to run)
# result = main("Tell me about web scraping")
# print(result)