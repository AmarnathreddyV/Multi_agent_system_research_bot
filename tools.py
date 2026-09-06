import os

import requests

from bs4 import BeautifulSoup

from tavily import TavilyClient

from dotenv import load_dotenv

from langchain.tools import tool


load_dotenv()


# ==================================================
# WEB SEARCH TOOL
# ==================================================

def get_web_search_tool(
    tavily_key: str = None
):

    key = (
        tavily_key
        or os.getenv("TAVILY_API_KEY")
    )

    if not key:

        raise ValueError(
            "TAVILY_API_KEY is missing! "
            "Please configure it in Streamlit Secrets."
        )


    tavily = TavilyClient(
        api_key=key
    )


    @tool("web_search")
    def web_search(query: str) -> str:
        """
        Search the web for recent and reliable
        information about a topic.

        Returns titles, URLs and snippets.
        """

        results = tavily.search(
            query=query,
            max_results=5
        )


        output = []


        for result in results.get(
            "results",
            []
        ):

            output.append(
                f"Title: {result.get('title', 'N/A')}\n"
                f"URL: {result.get('url', 'N/A')}\n"
                f"Snippet: {result.get('content', '')[:500]}\n"
            )


        if not output:

            return "No search results found."


        return "\n-----\n".join(
            output
        )


    return web_search


# ==================================================
# URL SCRAPER TOOL
# ==================================================

def get_scrape_url_tool():


    @tool("scrape_url")
    def scrape_url(url: str) -> str:
        """
        Scrape a web page and return clean text
        content for deeper research.
        """

        try:

            response = requests.get(
                url,
                timeout=10,
                headers={
                    "User-Agent":
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                }
            )


            response.raise_for_status()


            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )


            # Remove unnecessary HTML elements
            for tag in soup(
                [
                    "script",
                    "style",
                    "nav",
                    "footer",
                    "header",
                    "noscript"
                ]
            ):

                tag.decompose()


            text = soup.get_text(
                separator=" ",
                strip=True
            )


            # Limit content sent to the LLM
            return text[:5000]


        except Exception as e:

            return (
                f"Could not scrape the URL: {str(e)}"
            )


    return scrape_url
