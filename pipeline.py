import time

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
    # 1. SEARCH
    # ==================================================

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

    state["search_results"] = (
        search_result["messages"][-1].content
    )

    # Small delay between API operations
    time.sleep(2)

    # ==================================================
    # 2. READER
    # ==================================================

    reader_agent = build_reader_agent(
        mistral_key=mistral_key
    )

    reader_result = reader_agent.invoke({
        "messages": [
            (
                "user",
                f"""
Based on the following search results about "{topic}",
select the most relevant URL and scrape it for deeper information.

Search results:

{state["search_results"][:4000]}
"""
            )
        ]
    })

    state["scraped_content"] = (
        reader_result["messages"][-1].content
    )

    time.sleep(2)

    # ==================================================
    # 3. PREPARE RESEARCH
    # ==================================================

    research_combined = (
        f"SEARCH RESULTS:\n"
        f"{state['search_results'][:5000]}\n\n"
        f"DETAILED SCRAPED CONTENT:\n"
        f"{state['scraped_content'][:10000]}"
    )

    # ==================================================
    # 4. WRITER
    # ==================================================

    writer_chain = get_writer_chain(
        mistral_key=mistral_key
    )

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })

    time.sleep(2)

    # ==================================================
    # 5. CRITIC
    # ==================================================

    critic_chain = get_critic_chain(
        mistral_key=mistral_key
    )

    state["feedback"] = critic_chain.invoke({
        "report": state["report"][:12000]
    })

    return state


if __name__ == "__main__":

    topic = input("\nEnter a research topic: ")

    result = run_research_pipeline(topic)

    print("\n===== REPORT =====")
    print(result["report"])

    print("\n===== CRITIC =====")
    print(result["feedback"])
