import os
import streamlit as st
from dotenv import load_dotenv
# 1. Fixed the import to use LangGraph's prebuilt agent creator
from langgraph.prebuilt import create_react_agent 
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tools import web_search, scrape_url

load_dotenv()

# Securely retrieve the API key from Streamlit Cloud secrets or local environment variables
mistral_key = st.secrets.get("MISTRAL_API_KEY") or os.getenv("MISTRAL_API_KEY")

if not mistral_key:
    raise ValueError("MISTRAL_API_KEY is missing! Please configure it in your Streamlit Secrets box or .env file.")

# Initialize LLM with the verified key
llm = ChatMistralAI(
    model="mistral-small-2506", 
    temperature=0, 
    api_key=mistral_key
)

# 2. Updated agent creators to use create_react_agent correctly
def build_search_agent():
    return create_react_agent(
        model=llm,
        tools=[web_search]
    )

def build_reader_agent():
    return create_react_agent(
        model=llm,
        tools=[scrape_url]
    )

# --- Writer Chain ---
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

writer_chain = writer_prompt | llm | StrOutputParser()

# --- Critic Chain ---
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

critic_chain = critic_prompt | llm | StrOutputParser()
