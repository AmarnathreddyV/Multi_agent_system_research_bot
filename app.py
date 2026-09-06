
import streamlit as st
from pipeline import run_research_pipeline

# Page Configuration
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🤖",
    layout="wide"
)

# Title & Description
st.title("🤖 Multi-Agent Research Assistant")

st.markdown(
    "Enter a topic below to unleash a squad of specialized AI agents. "
    "They will search the web, scrape relevant content, synthesize a report, "
    "and critique the final output."
)

st.divider()

# User Input
topic = st.text_input(
    "What topic do you want to research today?",
    placeholder="e.g., Key breakthroughs in Solid-State Batteries (2026)"
)

# Securely extract secrets from Streamlit Cloud
mistral_key = st.secrets.get("MISTRAL_API_KEY")
tavily_key = st.secrets.get("TAVILY_API_KEY")

# Execution Trigger
if st.button("🚀 Launch Research Team", type="primary"):

    if not topic.strip():
        st.warning("⚠️ Please provide a valid research topic.")
        st.stop()

    if not mistral_key:
        st.error("❌ MISTRAL_API_KEY is missing from Streamlit Secrets.")
        st.stop()

    if not tavily_key:
        st.error("❌ TAVILY_API_KEY is missing from Streamlit Secrets.")
        st.stop()

    with st.status(
        "🚀 Agents are working... This may take a minute.",
        expanded=True
    ) as status:

        try:
            st.write("🔍 **Search Agent:** Finding reliable sources...")

            results = run_research_pipeline(
                topic=topic,
                mistral_key=mistral_key,
                tavily_key=tavily_key
            )

            st.write("📖 **Reader Agent:** Analyzing relevant content...")
            st.write("✍️ **Writer Agent:** Generating the research report...")
            st.write("🧐 **Critic Agent:** Evaluating the report...")

            status.update(
                label="✅ Research Complete!",
                state="complete",
                expanded=False
            )

        except Exception as e:

            status.update(
                label="❌ Pipeline Interrupted",
                state="error",
                expanded=True
            )

            if "429" in str(e):
                st.error(
                    "⚠️ Mistral API rate limit exceeded. "
                    "Please wait and try again."
                )
            else:
                st.error(
                    f"An error occurred while executing the pipeline:\n\n{e}"
                )

            st.stop()

    # Results
    st.success("🎉 Your report is ready!")

    tab1, tab2, tab3 = st.tabs(
        [
            "📝 Final Report",
            "🧐 Critic Feedback",
            "🗂️ Collected Raw Data"
        ]
    )

    # Final Report
    with tab1:
        st.subheader("Generated Research Report")

        st.markdown(
            results.get(
                "report",
                "No report text generated."
            )
        )

    # Critic Feedback
    with tab2:
        st.subheader("Critic Evaluation")

        st.info(
            results.get(
                "feedback",
                "No feedback recorded."
            )
        )

    # Raw Data
    with tab3:

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔍 Search Results")

            st.text_area(
                "Raw Search Results",
                value=results.get(
                    "search_results",
                    "No search results available."
                ),
                height=350
            )

        with col2:
            st.markdown("### 📖 Scraped Content")

            st.text_area(
                "Extracted Web Content",
                value=results.get(
                    "scraped_content",
                    "No scraped content available."
                ),
                height=350
            )
