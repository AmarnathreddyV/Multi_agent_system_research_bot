from agents import (
    build_reader_agent,
    build_search_agent,
    get_writer_chain,
    get_critic_chain
)


def run_research_pipeline(
    topic: str,
    mistral_key: str = None,
    tavily_key: str = None
) -> dict:

    state = {}


    # ==================================================
    # 1. SEARCH AGENT
    # ==================================================

    search_agent = build_search_agent(
        mistral_key=mistral_key,
        tavily_key=tavily_key
    )


    search_result = search_agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"""
Find recent, reliable and detailed
information about:

{topic}
"""
                )
            ]
        }
    )


    state["search_results"] = (
        search_result["messages"][-1].content
    )


    # ==================================================
    # 2. READER / SCRAPER AGENT
    # ==================================================

    reader_agent = build_reader_agent(
        mistral_key=mistral_key
    )


    reader_result = reader_agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"""
Based on the following search results
about '{topic}', identify the most
relevant URL and scrape it for deeper
information.

Search Results:

{state["search_results"][:1500]}
"""
                )
            ]
        }
    )


    state["scraped_content"] = (
        reader_result["messages"][-1].content
    )


    # ==================================================
    # 3. COMBINE RESEARCH
    # ==================================================

    research_combined = (
        "SEARCH RESULTS:\n"
        + state["search_results"]
        + "\n\n"
        + "DETAILED SCRAPED CONTENT:\n"
        + state["scraped_content"]
    )


    # ==================================================
    # 4. WRITER AGENT
    # ==================================================

    writer_chain = get_writer_chain(
        mistral_key=mistral_key
    )


    state["report"] = writer_chain.invoke(
        {
            "topic": topic,
            "research": research_combined
        }
    )


    # ==================================================
    # 5. CRITIC AGENT
    # ==================================================

    critic_chain = get_critic_chain(
        mistral_key=mistral_key
    )


    state["feedback"] = critic_chain.invoke(
        {
            "report": state["report"]
        }
    )


    # ==================================================
    # RETURN RESULTS
    # ==================================================

    return state


# ==================================================
# LOCAL TESTING
# ==================================================

if __name__ == "__main__":

    topic = input(
        "\nEnter a research topic: "
    )


    result = run_research_pipeline(
        topic
    )


    print(
        "\n========== RESEARCH REPORT ==========\n"
    )

    print(
        result["report"]
    )


    print(
        "\n========== CRITIC FEEDBACK ==========\n"
    )

    print(
        result["feedback"]
    )
