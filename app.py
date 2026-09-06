import streamlit as st

from pipeline import run_research_pipeline


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🤖",
    layout="wide"
)


# -----------------------------
# Title & Description
# -----------------------------
st.title("🤖 Multi-Agent Research Assistant")

st.markdown(
    "Enter a topic below to unleash a squad of specialized AI agents. "
    "They will search the web, scrape relevant content, synthesize a report, "
    "and critique the final output."
)

st.divider()


# -----------------------------
# User Input
# -----------------------------
topic = st.text_input(
    "What topic do you want to research today?",
    placeholder="e.g., Key breakthroughs in Solid-State Batteries (2026)"
)


# -----------------------------
# Get API Keys from Streamlit Secrets
# -----------------------------
mistral_key = st.secrets.get("MISTRAL_API_KEY")
tavily_key = st.secrets.get("TAVILY_API_KEY")


# -----------------------------
# Execute Research Pipeline
# -----------------------------
if st.button("Launch Research Team", type="primary"):

    if not topic.strip():

        st.warning(
            "⚠️ Please provide a valid research topic before running the pipeline!"
        )

    elif not mistral_key or not tavily_key:

        st.error(
            "❌ API keys are missing. Please configure "
            "MISTRAL_API_KEY and TAVILY_API_KEY in Streamlit Secrets."
        )

    else:

        with st.status(
            "🚀 Agents are working... This may take a minute.",
            expanded=True
        ) as status:

            try:

                # -----------------------------
                # Search Agent
                # -----------------------------
                st.write(
                    "🔍 **Search Agent** is searching for reliable information..."
                )

                results = run_research_pipeline(
                    topic,
                    mistral_key=mistral_key,
                    tavily_key=tavily_key
                )


                # -----------------------------
                # Pipeline Completed
                # -----------------------------
                status.update(
                    label="✅ Research Complete!",
                    state="complete",
                    expanded=False
                )

                st.success("🎉 Your research report is ready!")


                # -----------------------------
                # Tabs
                # -----------------------------
                tab1, tab2, tab3 = st.tabs(
                    [
                        "📝 Final Report",
                        "🧐 Critic Feedback",
                        "🗂️ Collected Raw Data"
                    ]
                )


                # -----------------------------
                # Final Report
                # -----------------------------
                with tab1:

                    st.subheader("Generated Research Report")

                    st.markdown(
                        results.get(
                            "report",
                            "No report text generated."
                        )
                    )


                # -----------------------------
                # Critic Feedback
                # -----------------------------
                with tab2:

                    st.subheader("Critic Evaluation")

                    st.info(
                        results.get(
                            "feedback",
                            "No feedback recorded."
                        )
                    )


                # -----------------------------
                # Raw Data
                # -----------------------------
                with tab3:

                    st.subheader("Agent Grounding Data")

                    col1, col2 = st.columns(2)


                    with col1:

                        st.markdown(
                            "**Search Results Summary**"
                        )

                        st.text_area(
                            "Raw Search Results",
                            value=results.get(
                                "search_results",
                                ""
                            ),
                            height=350
                        )


                    with col2:

                        st.markdown(
                            "**Deep Scraped Content**"
                        )

                        st.text_area(
                            "Extracted Web Content",
                            value=results.get(
                                "scraped_content",
                                ""
                            ),
                            height=350
                        )


            except Exception as e:

                status.update(
                    label="💥 Pipeline Interrupted",
                    state="error"
                )

                st.error(
                    f"An error occurred while executing the "
                    f"multi-agent system: {e}"
                )
