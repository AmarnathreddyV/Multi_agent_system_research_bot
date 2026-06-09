from langchain.tools import tool
import requests
import streamlit as st
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

# Securely retrieve the Tavily key from Streamlit Cloud secrets or local environment variables
tavily_key = st.secrets.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")

if not tavily_key:
    raise ValueError("TAVILY_API_KEY is missing! Please configure it in your Streamlit Secrets box or .env file.")

# Initialize TavilyClient with the verified key
tavily = TavilyClient(api_key=tavily_key)

@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable info on a topic. Returns titles, urls and snippets."""
    results = tavily.search(query=query, max_results=5)

    out = []
    for r in results.get('results', []):
        out.append(
            f"Title: {r.get('title', 'N/A')}\nURL: {r.get('url', 'N/A')}\nSnippet: {r.get('content', '')[:300]}\n"
        )

    return "\n-----\n".join(out)

@tool
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
