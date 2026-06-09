from agents import build_reader_agent, build_search_agent, get_writer_chain, get_critic_chain

def run_research_pipeline(topic: str, mistral_key: str = None, tavily_key: str = None) -> dict:
    state = {}

    # Gather search metrics
    search_agent = build_search_agent(mistral_key=mistral_key, tavily_key=tavily_key)
    search_result = search_agent.invoke({
        "messages": [("user", f"find recent, reliable and detailed info about: {topic}")]
    })
    state["search_results"] = search_result['messages'][-1].content

    # Deep scrape analysis
    reader_agent = build_reader_agent(mistral_key=mistral_key)
    reader_result = reader_agent.invoke({
        "messages": [("user",
                     f"based on following search results about '{topic}', "
                     f"pick the most relevant URL and scrape it for deeper content.\n\n"
                     f"search results:\n{state['search_results'][:800]}"
                     )]
    })
    state['scraped_content'] = reader_result['messages'][-1].content

    research_combined = (
        f"SEARCH RESULTS:\n {state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )

    # Invoke writing pipeline
    writer_chain = get_writer_chain(mistral_key=mistral_key)
    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })

    # Invoke criticism evaluation
    critic_chain = get_critic_chain(mistral_key=mistral_key)
    state["feedback"] = critic_chain.invoke({
        "report": state['report']
    })

    return state

if __name__ == "__main__":
    import os
    topic = input("\n Enter a research topic: ")
    run_research_pipeline(topic)
