from agents import (
    build_reader_agent,
    build_search_agent,
    get_writer_chain,
    get_critic_chain,
)


def run_research_pipeline(
    topic: str,
    mistral_key: str = None,
    tavily_key: str = None
) -> dict:

    state = {}

    # -----------------------------
    # 1. Search Agent
    # -----------------------------
    search_agent = build_search_agent(
        mistral_key=mistral_key,
        tavily_key=tavily_key
    )

    search_result = search_agent.invoke({
        "messages": [
            (
                "user",
                f"Find recent, reliable and detailed information about: {topic}"
            )
        ]
    })

    state["search_results"] = search_result["messages"][-1].content


    # -----------------------------
    # 2. Reader / Scraper Agent
    # -----------------------------
    reader_agent = build_reader_agent(
        mistral_key=mistral_key
    )

    reader_result = reader_agent.invoke({
        "messages": [
            (
                "user",
                f"""
Based on the following search results about '{topic}',
identify the most relevant URL and scrape it for deeper information.

Search results:

{state["search_results"][:800]}
"""
            )
        ]
    })

    state["scraped_content"] = reader_result["messages"][-1].content


    # -----------------------------
    # 3. Combine Research
    # -----------------------------
    research_combined = (
        f"SEARCH RESULTS:\n"
        f"{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n"
        f"{state['scraped_content']}"
    )


    # -----------------------------
    # 4. Writer Agent
    # -----------------------------
    writer_chain = get_writer_chain(
        mistral_key=mistral_key
    )

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })


    # -----------------------------
    # 5. Critic Agent
    # -----------------------------
    critic_chain = get_critic_chain(
        mistral_key=mistral_key
    )

    state["feedback"] = critic_chain.invoke({
        "report": state["report"]
    })


    return state


# -----------------------------
# Local Testing
# -----------------------------
if __name__ == "__main__":

    topic = input(
        "\nEnter a research topic: "
    )

    result = run_research_pipeline(topic)

    print("\n===== RESEARCH REPORT =====\n")
    print(result["report"])

    print("\n===== CRITIC FEEDBACK =====\n")
    print(result["feedback"])
