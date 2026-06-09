from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable info on a topic. Returns titles, urls and snippets."""
    results = tavily.search(query=query, max_results=5)

    out = []
    for r in results.get('results', []):
        # FIX: Changed r['URL'] to lowercase r['url']. Tavily keys are strictly lowercase.
        out.append(
            f"Title: {r.get('title', 'N/A')}\nURL: {r.get('url', 'N/A')}\nSnippet: {r.get('content', '')[:300]}\n"
        )

    return "\n-----\n".join(out)

@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    try:
        # FIX: Corrected "Morzilla" typo to "Mozilla"
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
            
        # FIX: Corrected the typo 'separayor' to 'separator'
        return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as e:
        # FIX: Corrected "scrap" to "scrape"
        return f"Could not scrape the URL: {str(e)}"