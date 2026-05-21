# NewsAgent

An agentic AI news summariser built with **LangGraph**, **Streamlit**, and your choice of LLM (Anthropic Claude, OpenAI, or DeepSeek). Given a topic, it automatically searches for recent news, gathers articles, summarises them, scores sentiment, and saves a clean Markdown report locally.

---

## How it works

```
Topic ──► Planner ──► Executor ──► Finalizer ──► Streamlit UI + Markdown Report
```

1. **Planner** — uses the LLM to generate 1–2 targeted search queries for the topic
2. **Executor** — runs an agentic search loop using NewsAPI and DuckDuckGo (via MCP), collecting articles with rate-limited tool calls
3. **Finalizer** — summarises all gathered articles, computes per-article sentiment, produces an overall news summary, and triggers a replan if results are insufficient
4. **Streamlit UI** — displays live agent logs, article cards with sentiment scores, and auto-saves a Markdown report to `results/`

---

## Features

- **Multi-source search** — NewsAPI for structured metadata + DuckDuckGo MCP for web fallback
- **Multi-LLM support** — switch between Anthropic, OpenAI, or DeepSeek from the sidebar
- **Auto-replan** — if results are insufficient, the agent critiques itself and retries with new queries
- **Auto-save Markdown report** — results are saved to `results/news_report_<topic>_<timestamp>.md` after every run
- **In-browser download** — download the report directly from the Streamlit UI
- **Rate-limit guardrails** — hard caps on search/fetch tool calls and recursion depth to protect API credits

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd news-agent
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and fill in your API keys:

| Variable | Description |
| :--- | :--- |
| `NEWS_API_KEY` |  [newsapi.org](https://newsapi.org) free tier key |
| `ANTHROPIC_API_KEY` |  For Claude models |
| `OPENAI_API_KEY` |  For GPT models |
| `DEEPSEEK_API_KEY` |  For DeepSeek models |

### 3. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), enter a topic in the sidebar, and click **Run Agent**.

---

## Project structure

```
news-agent/
├── app.py               ← Streamlit UI + live agent log streaming + Markdown export
├── graph.py             ← LangGraph StateGraph (Planner → Executor → Finalizer)
├── agents/
│   ├── planner.py       ← LLM generates targeted search queries
│   ├── executor.py      ← Agentic search loop (NewsAPI + DuckDuckGo MCP)
│   └── finalizer.py     ← Summarise, score sentiment, replan if needed
├── tools/
│   ├── mcp_tools.py     ← DuckDuckGo MCP client config
│   ├── summarizer.py    ← Per-article LLM summariser
│   └── sentiment.py     ← Per-article LLM sentiment scorer
├── models/
│   └── schemas.py       ← Pydantic models (ArticleResult, FinalOutput, OverallState)
├── results/             ← Auto-generated Markdown reports (gitignored)
├── .env.example         ← Environment variable template
└── requirements.txt
```

---

## Agentic AI concepts demonstrated

| Concept | Where |
| :--- | :--- |
| **Planning** | `planner_node` uses LLM to decompose a topic into targeted search queries |
| **Tool use** | Executor calls NewsAPI and DuckDuckGo MCP tools autonomously |
| **Agentic loop** | Executor runs agent → tools → agent until it decides to stop |
| **Dynamic replan** | `finalizer_node` evaluates quality and triggers replanning if needed |
| **Safe state merging** | `Annotated[List, operator.add]` reducer prevents state overwrites |
| **Fault tolerance** | Failed article fetches are isolated; agent continues with remaining results |
| **Structured output** | Pydantic `FinalOutput` validates the final JSON before rendering |
| **Rate-limit guardrails** | Hard caps on tool calls + recursion limit protect API credits |
