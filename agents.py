import os
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from tools import get_web_search_tool, get_scrape_url_tool

load_dotenv()


# -----------------------------
# Mistral LLM Configuration
# -----------------------------
def get_llm(mistral_key: str = None):
    key = mistral_key or os.getenv("MISTRAL_API_KEY")

    if not key:
        raise ValueError(
            "MISTRAL_API_KEY is missing! Please configure it in Streamlit Secrets."
        )

    return ChatMistralAI(
        model="mistral-small-2506",
        temperature=0,
        api_key=key
    )


# -----------------------------
# Search Agent
# -----------------------------
def build_search_agent(
    mistral_key: str = None,
    tavily_key: str = None
):
    llm = get_llm(mistral_key)

    return create_react_agent(
        model=llm,
        tools=[
            get_web_search_tool(tavily_key)
        ]
    )


# -----------------------------
# Reader / Scraper Agent
# -----------------------------
def build_reader_agent(mistral_key: str = None):
    llm = get_llm(mistral_key)

    return create_react_agent(
        model=llm,
        tools=[
            get_scrape_url_tool()
        ]
    )


# -----------------------------
# Writer Agent
# -----------------------------
writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research writer. "
        "Write clear, structured, factual and insightful reports."
    ),
    (
        "human",
        """Write a detailed research report on the topic below.

Topic:
{topic}

Research Gathered:
{research}

Structure the report as:

1. Introduction
2. Key Findings
   - Explain at least 3 important findings in detail
3. Conclusion
4. Sources
   - List all relevant URLs found in the research

Requirements:
- Be factual and professional.
- Use only the information available in the research.
- Do not invent facts.
- Clearly explain important findings.
- Make the report easy to read."""
    ),
])


def get_writer_chain(mistral_key: str = None):
    return (
        writer_prompt
        | get_llm(mistral_key)
        | StrOutputParser()
    )


# -----------------------------
# Critic Agent
# -----------------------------
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a sharp and constructive research critic. "
        "Be honest, specific and objective."
    ),
    (
        "human",
        """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in exactly this format:

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

Check the report for:
- Accuracy
- Clarity
- Completeness
- Organization
- Quality of sources
- Missing or weak information"""
    ),
])


def get_critic_chain(mistral_key: str = None):
    return (
        critic_prompt
        | get_llm(mistral_key)
        | StrOutputParser()
    )
