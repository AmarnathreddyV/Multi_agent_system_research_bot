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
        raise ValueError("MISTRAL_API_KEY is missing!")
    return ChatMistralAI(model="mistral-small-2506", temperature=0, api_key=key)

def build_search_agent(mistral_key: str = None, tavily_key: str = None):
    llm = get_llm(mistral_key)
    return create_react_agent(
        model=llm,
        tools=[get_web_search_tool(tavily_key)]
    )

def build_reader_agent(mistral_key: str = None, tavily_key: str = None):
    llm = get_llm(mistral_key)
    return create_react_agent(
        model=llm,
        tools=[get_scrape_url_tool()]
    )

# --- Writer Chain Layout ---
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

# Default fallback chain using environment variables
default_llm = ChatMistralAI(model="mistral-small-2506", temperature=0) if os.getenv("MISTRAL_API_KEY") else None
writer_chain = writer_prompt | (default_llm or ChatMistralAI(model="mistral-small-2506", temperature=0)) | StrOutputParser()

# --- Critic Chain Layout ---
critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | (default_llm or ChatMistralAI(model="mistral-small-2506", temperature=0)) | StrOutputParser()
