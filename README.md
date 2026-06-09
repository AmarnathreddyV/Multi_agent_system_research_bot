# 🤖 Multi-Agent Research Assistant

An AI-powered research pipeline that deploys a collaborative squad of specialized LangGraph agents and LangChain chains to automate open-ended market or academic research. The system searches the web, deeply scrapes highly relevant content, synthesizes a structured report, and subjects it to an honest critique—all controlled through an intuitive Streamlit interface.

## 🌟 Features

*   🔍 **Search Agent:** Uses LangGraph's prebuilt ReAct agent paired with Tavily Search to hunt down reliable web links and summaries.
*   📖 **Reader Agent:** Dynamically extracts and cleans deep web content from selected sources using `BeautifulSoup4`[cite: 1, 3, 5].
*   📝 **Writer Chain:** Synthesizes raw findings into a clear, publication-ready research report with strict formatting constraints.
*   🧐 **Critic Chain:** Evaluates the generated report, providing score breakdowns, constructive feedback, and a final verdict.
*   💻 **Streamlit UI:** Provides a clean dashboard with live status updates and a dynamic tabbed view to inspect logs, raw text, feedback, and the final report.

---

## 🛠️ Project Structure

```bash
├── agents.py          # Configures the LangGraph ReAct agents & Writer/Critic chains
├── pipeline.py        # Connects the agents sequentially to handle execution flow
├── tools.py           # Contains customized web_search and scrape_url tools
├── app.py             # Streamlit graphical UI interface
├── requirements.txt   # Complete list of dependencies
└── .env               # (To be created) Stored API environment secrets
