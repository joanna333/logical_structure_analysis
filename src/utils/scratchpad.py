from wikipedia import wikipedia
import pandas as pd
from typing import List, Dict

def show_wiki_search(query: str, results_limit: int = 5) -> List[Dict]:
    """
    Search Wikipedia articles based on a query.
    
    Args:
        query: Search term
        results_limit: Maximum number of results to return (default=5)
    
    Returns:
        List of dictionaries containing article titles and summaries
    """
    try:
        # Search Wikipedia
        search_results = wikipedia.search(query, results=results_limit)
        articles = []
        
        for title in search_results:
            try:
                # Get page content
                page = wikipedia.page(title)
                articles.append({
                    'title': page.title,
                    'summary': page.summary,
                    'url': page.url,
                    'content': page.content
                })
            except wikipedia.exceptions.DisambiguationError as e:
                # Handle disambiguation pages
                continue
            except wikipedia.exceptions.PageError:
                # Handle non-existent pages
                continue
                
        return articles
        
    except Exception as e:
        print(f"Error during Wikipedia search: {str(e)}")
        return []

# Search for medical topics
results = show_wiki_search("cardiac physiology", results_limit=3)

# Print results
for article in results:
    print(f"Title: {article['title']}")
    print(f"Summary: {article['summary'][:200]}...")
    print(f"URL: {article['url']}")
    print("-" * 50)