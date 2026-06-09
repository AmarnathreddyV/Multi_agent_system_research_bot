import streamlit as st
import sys
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
    "They will search the web, scrape relevant contents, synthesize a report, and critique the final output."
)

st.divider()

# User Input Layout
topic = st.text_input(
    "What topic do you want to research today?", 
    placeholder="e.g., Key breakthroughs in Solid-State Batteries (2026)"
)

# Pull secrets safely to pass down into the backend module context
mistral_key = st.secrets.get("MISTRAL_API_KEY")
tavily_key = st.secrets.get("TAVILY_API_KEY")

# Execution Trigger
if st.button("Launch Research Team", type="primary"):
    if not topic.strip():
        st.warning("⚠️ Please provide a valid research topic before running the pipeline!")
    elif not mistral_key or not tavily_key:
        st.error("❌ Configuration Secret Error: Double check that your Streamlit secrets are saved properly!")
    else:
        # Visual status loader for the backend agent execution
        with st.status("🚀 Agents are working... (This may take a minute)", expanded=True) as status:
            try:
                st.write("🔍 **Search Agent** is hunting for reliable links...")
                # Run the pipeline function passing the keys explicitly down
                results = run_research_pipeline(topic, mistral_key=mistral_key, tavily_key=tavily_key)
                
                status.update(label="✅ Research Complete!", state="complete", expanded=False)
                st.success("🎉 Your report is ready!")
                
                # Dynamic Tabbed Interface to view outputs neatly
                tab1, tab2, tab3 = st.tabs([
                    "📝 Final Report", 
                    "🧐 Critic Feedback", 
                    "🗂️ Collected Raw Data"
                ])
                
                with tab1:
                    st.subheader("Generated Research Paper")
                    st.markdown(results.get("report", "No report text generated."))
                    
                with tab2:
                    st.subheader("Critic Evaluation")
                    st.info(results.get("feedback", "No feedback recorded."))
                    
                with tab3:
                    st.subheader("Agent Grounding Data")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Search Results Summary**")
                        st.text_area("Raw Snippets", value=results.get("search_results", ""), height=300)
                    with col2:
                        st.markdown("**Deep Scraped Content**")
                        st.text_area("Extracted Web Content", value=results.get("scraped_content", ""), height=300)
                        
            except Exception as e:
                status.update(label="💥 Pipeline Interrupted", state="error")
                st.error(f"An error occurred while executing the multi-agent system: {e}")
