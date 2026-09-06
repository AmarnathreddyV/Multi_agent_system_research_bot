import os

from dotenv import load_dotenv

from langgraph.prebuilt import create_react_agent

from langchain_mistralai import ChatMistralAI

from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import ChatPromptTemplate

from tools import (
    get_web_search_tool,
    get_scrape_url_tool
)


load_dotenv()


# -----------------------------
# Mistral LLM
# -----------------------------
def get_llm(mistral_key: str = None):

    key = (
        mistral_key
        or os.getenv("MISTRAL_API_KEY")
    )

    if not key:

        raise ValueError(
            "MISTRAL_API_KEY is missing! "
            "Please configure it in Streamlit Secrets."
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

    search_tool = get_web_search_tool(
        tavily_key
    )

    return create_react_agent(
        model=llm,
        tools=[search_tool]
    )


# -----------------------------
# Reader / Scraper Agent
# -----------------------------
def build_reader_agent(
    mistral_key: str = None
):

    llm = get_llm(
        mistral_key
    )

    scrape_tool = get_scrape_url_tool()

    return create_react_agent(
        model=llm,
        tools=[scrape_tool]
    )


# ==================================================
# WRITER AGENT
# ==================================================

writer_prompt = ChatPromptTemplate.from_messages(
    [

        (
            "system",
            """
You are an expert research writer.

Your job is to create clear, structured,
factual and insightful research reports.

Use only the information provided in
the research gathered by the agents.

Do not invent facts.
"""
        ),

        (
            "human",
            """
Write a detailed research report on the topic below.

Topic:
{topic}

Research Gathered:
{research}


Structure the report as:

1. Introduction

2. Key Findings
   - Explain at least 3 important findings
   - Provide clear explanations

3. Conclusion

4. Sources
   - List relevant URLs found in the research


Requirements:

- Be factual.
- Be professional.
- Be easy to understand.
- Do not invent information.
- Use the research provided.
- Clearly explain the important findings.
"""
        )

    ]
)


def get_writer_chain(
    mistral_key: str = None
):

    return (
        writer_prompt
        | get_llm(mistral_key)
        | StrOutputParser()
    )


# ==================================================
# CRITIC AGENT
# ==================================================

critic_prompt = ChatPromptTemplate.from_messages(
    [

        (
            "system",
            """
You are a sharp and constructive research critic.

Evaluate the report honestly and objectively.

Identify both strengths and weaknesses.
"""
        ),

        (
            "human",
            """
Review the research report below.

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


Evaluate the report based on:

- Accuracy
- Clarity
- Completeness
- Organization
- Quality of sources
- Missing or weak information
"""
        )

    ]
)


def get_critic_chain(
    mistral_key: str = None
):

    return (
        critic_prompt
        | get_llm(mistral_key)
        | StrOutputParser()
    )
