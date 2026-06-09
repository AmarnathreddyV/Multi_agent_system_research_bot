from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def get_web_search_tool(tavily_key: str = None):
    # Fallback to os.getenv if st.secrets isn't found locally
    key = tavily_key or os.getenv("TAVILY_API_KEY")
    if not key:
        raise ValueError("TAVILY_API_KEY is missing! Please configure it in Secrets.")
        
    tavily = TavilyClient(api_key=key)

    @tool("web_search")
    def web_search(query: str) -> str:
        """Search the web for recent and reliable info on a topic. Returns titles, urls and snippets."""
        results = tavily.search(query=query, max_results=5)
        out = []
        for r in results.get('results', []):
            out.append(
                f"Title: {r.get('title', 'N/A')}\nURL: {r.get('url', 'N/A')}\nSnippet: {r.get('content', '')[:300]}\n"
            )
        return "\n-----\n".join(out)
        
    return web_search

def get_scrape_url_tool():
    @tool("scrape_url")
    def scrape_url(url: str) -> str:
        """Scrape and return clean text content from a given URL for deeper reading."""
        try:
            resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
                
            return soup.get_text(separator=" ", strip=True)[:3000]
        except Exception as e:
            return f"Could not scrape the URL: {str(e)}"
            
    return scrape_url
