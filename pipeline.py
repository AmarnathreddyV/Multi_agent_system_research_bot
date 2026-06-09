from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain

def run_research_pipeline(topic: str, mistral_key: str = None, tavily_key: str = None) -> dict:
    state = {}

    # Pass the keys down to the agent builders
    search_agent = build_search_agent(mistral_key=mistral_key, tavily_key=tavily_key)
    search_result = search_agent.invoke({
        "messages": [("user", f"find recent, reliable and detailed info about: {topic}")]
    })
    state["search_results"] = search_result['messages'][-1].content
    print("\n search result", state['search_results'])

    reader_agent = build_reader_agent(mistral_key=mistral_key, tavily_key=tavily_key)
    reader_result = reader_agent.invoke({
        "messages": [("user",
                     f"based on following search results about '{topic}', "
                     f"pick the most relevant URL and scrape it for deeper content.\n\n"
                     f"search results:\n{state['search_results'][:800]}"
                     )]
    })

    state['scraped_content'] = reader_result['messages'][-1].content
    print("\n scraped content:\n", state['scraped_content'])

    research_combined = (
        f"SEARCH RESULTS:\n {state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )

    # Configure the chains to use the dynamic key if available
    w_chain = writer_chain.with_config(configurable={"api_key": mistral_key}) if mistral_key else writer_chain
    state["report"] = w_chain.invoke({
        "topic": topic,
        "research": research_combined
    })

    print("final report", state['report'])

    c_chain = critic_chain.with_config(configurable={"api_key": mistral_key}) if mistral_key else critic_chain
    state["feedback"] = c_chain.invoke({
        "report": state['report']
    })

    print("\n critic report:", state['feedback'])

    return state

if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    run_research_pipeline(topic)
