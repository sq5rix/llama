from duckduckgo_search import ddg

def find_relevant_websites(query, max_results=5):
    """
    Search DuckDuckGo for a list of relevant websites.

    Parameters:
    - query (str): The search query to find relevant websites.
    - max_results (int): Maximum number of websites to return.

    Returns:
    - List of dictionaries containing the 'title' and 'href' of each result.
    """
    try:
        # Perform the search with DuckDuckGo
        search_results = ddg(query, max_results=max_results)
        
        # Extract the title and link for each result
        websites = [{"title": result['title'], "url": result['href']} for result in search_results]
        
        return websites
    
    except Exception as e:
        return f"Error occurred during search: {e}"

# Example usage
query = "Python web scraping tutorials"
relevant_websites = find_relevant_websites(query)

print(relevant_websites)  # This will return a list of relevant websites for the given query