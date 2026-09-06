import os
from dotenv import load_dotenv

from langgraph.prebuilt import create_react_agent
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from tools import get_web_search_tool, get_scrape_url_tool

load_dotenv()


def get_llm(mistral_key: str = None):
    key = mistral_key or os.getenv("MISTRAL_API_KEY")

    if not key:
        raise ValueError(
            "MISTRAL_API_KEY is missing! Please configure it in Secrets."
        )

    return ChatMistralAI(
        model="mistral-small-2506",
        temperature=0,
        api_key=key
    )


# --------------------------------------------------
# SEARCH AGENT
# --------------------------------------------------

def build_search_agent(
    mistral_key: str = None,
    tavily_key: str = None
):
    llm = get_llm(mistral_key)

    search_tool = get_web_search_tool(tavily_key)

    return create_react_agent(
        model=llm,
        tools=[search_tool],
        prompt=(
            "You are a web research search agent. "
            "Use the web_search tool to find recent, reliable information. "
            "Perform only the necessary search. "
            "After receiving search results, summarize the most useful "
            "results and include their URLs. "
            "Do not perform unnecessary additional searches."
        )
    )


# --------------------------------------------------
# READER AGENT
# --------------------------------------------------

def build_reader_agent(mistral_key: str = None):
    llm = get_llm(mistral_key)

    scrape_tool = get_scrape_url_tool()

    return create_react_agent(
        model=llm,
        tools=[scrape_tool],
        prompt=(
            "You are a research reading agent. "
            "Select the most relevant URL from the provided search results "
            "and use scrape_url once to obtain deeper content. "
            "Do not repeatedly scrape URLs. "
            "After scraping, summarize the useful information clearly."
        )
    )


# --------------------------------------------------
# WRITER
# --------------------------------------------------

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research writer. "
        "Write clear, structured and factual research reports."
    ),
    (
        "human",
        """Write a research report on the topic below.

Topic:
{topic}

Research Gathered:
{research}

Use this structure:

# Introduction

# Key Findings
Explain at least 3 important findings.

# Conclusion

# Sources
List the URLs available in the research.

Be factual, professional and concise.
Do not invent information or URLs."""
    ),
])


def get_writer_chain(mistral_key: str = None):
    return writer_prompt | get_llm(mistral_key) | StrOutputParser()


# --------------------------------------------------
# CRITIC
# --------------------------------------------------

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a sharp and constructive research critic. "
        "Evaluate the report for accuracy, completeness, clarity and "
        "evidence quality."
    ),
    (
        "human",
        """Review the research report below.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...
- ...

Areas to Improve:
- ...
- ...
- ...

One line verdict:
...

Be specific and constructive."""
    ),
])


def get_critic_chain(mistral_key: str = None):
    return critic_prompt | get_llm(mistral_key) | StrOutputParser()
