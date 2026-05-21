import os
import asyncio
import re
import datetime
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from graph import app
import importlib
import tools.summarizer, tools.sentiment

st.set_page_config(page_title="NewsAgent", page_icon="📰", layout="wide")
st.title("NewsAgent")


with st.sidebar:
    topic = st.text_input("Topic", placeholder="e.g. AI in Malaysia")
    max_replans = st.slider("Max replan rounds", 0, 3, 1,
                             help="If fewer than 50% of articles succeed, the planner retries with new queries.")

    st.divider()
    st.subheader("LLM Settings")

    provider = st.selectbox(
        "Provider",
        ["anthropic", "openai", "deepseek"],
        index=2,
    )

    _model_defaults = {
        "anthropic": "claude-sonnet-4-20250514",
        "openai":    "gpt-4o-mini",
        "deepseek":  "deepseek-chat",
    }
    model = st.text_input("Model", value=_model_defaults[provider])

    run = st.button("Run Agent", type="primary", use_container_width=True)

os.environ["LLM_PROVIDER"] = provider
os.environ["LLM_MODEL"]    = model

for mod in (tools.summarizer, tools.sentiment):
    importlib.reload(mod)

SENTIMENT_BADGE = {
    "Positive": "Positive",
    "Negative": "Negative",
    "Neutral":  "Neutral",
}

def _extract_text(msg) -> str:
    if hasattr(msg, "content"):
        content = msg.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "".join(parts)
    return ""

def _extract_tool_calls(msg) -> list[dict]:
    raw = getattr(msg, "tool_calls", None) or getattr(msg, "tool_call_chunks", None) or []
    tool_calls = []
    for tc in raw:
        if isinstance(tc, dict):
            name = tc.get("name") or ""
            args = tc.get("args") or tc.get("arguments") or ""
            if isinstance(args, dict):
                import json as _json
                args = _json.dumps(args, ensure_ascii=False)
            if name or args:
                tool_calls.append({"name": name, "args": str(args)})
    return tool_calls

def clean_log_text(text: str) -> str:
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())
    return text.strip()

def tool_call_box(name: str, args: str) -> str:
    args_html = (
        f"<br><pre style='margin:4px 0 0 0; white-space:pre-wrap; font-size:0.85em'>{args}</pre>"
        if args else ""
    )
    return (
        "<div style='background-color: rgba(255, 200, 80, 0.12); padding: 8px 12px; "
        "border-radius: 5px; border-left: 4px solid #f0a500; margin: 6px 0; font-size: 0.9em;'>"
        f"<strong>Tool call:</strong> <code>{name}</code>{args_html}"
        "</div>"
    )

def tool_result_box(tool_name: str, content: str) -> str:
    preview = content[:600] + ("…" if len(content) > 600 else "")
    name_html = f" · <code>{tool_name}</code>" if tool_name else ""
    return (
        "<div style='background-color: rgba(80, 200, 120, 0.10); padding: 8px 12px; "
        "border-radius: 5px; border-left: 4px solid #2ecc71; margin: 6px 0; font-size: 0.9em;'>"
        f"<strong>Tool result</strong>{name_html}"
        f"<br><pre style='margin:4px 0 0 0; white-space:pre-wrap; font-size:0.85em'>{preview}</pre>"
        "</div>"
    )

def generate_markdown_report(output) -> str:
    md = []
    md.append(f"# NewsAgent Report: {output.topic}")
    md.append("")
    md.append(f"- **Generated on**: {output.fetched_at}")
    md.append(f"- **Topic**: {output.topic}")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Summary Metrics")
    md.append("")
    md.append("| Metric | Value |")
    md.append("| :--- | :--- |")
    md.append(f"| **Total Articles Processed** | {output.total_articles} |")
    md.append(f"| **Successful Retrievals** | {output.successful} |")
    md.append(f"| **Failed Retrievals** | {output.failed} |")
    md.append(f"| **Overall Sentiment** | {output.overall_sentiment} |")
    md.append("")
    md.append("---")
    md.append("")
    if output.overall_summary:
        md.append("## Overall Summary")
        md.append("")
        md.append(output.overall_summary)
        md.append("")
        md.append("---")
        md.append("")
    
    md.append("## Articles Directory")
    md.append("")
    md.append("| Title | Source | Published Date | Sentiment | Status |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")
    for art in output.articles:
        title_link = f"[{art.title}]({art.url})" if art.title else f"[Link]({art.url})"
        published = art.published_at[:10] if art.published_at else "N/A"
        sentiment_str = f"{art.sentiment} ({art.sentiment_score:.2f})" if art.status == "complete" else "N/A"
        md.append(f"| {title_link} | {art.source} | {published} | {sentiment_str} | **{art.status.upper()}** |")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Detailed Articles Analysis")
    md.append("")
    
    for i, art in enumerate(output.articles, 1):
        published = art.published_at[:10] if art.published_at else "N/A"
        md.append(f"### {i}. [{art.title}]({art.url})")
        md.append(f"- **Source**: {art.source}")
        md.append(f"- **Published Date**: {published}")
        md.append(f"- **Status**: `{art.status}`")
        if art.status == "complete":
            md.append(f"- **Sentiment**: {art.sentiment} (Score: {art.sentiment_score:.2f})")
            md.append("")
            md.append("#### **Summary**")
            md.append(art.summary)
        else:
            md.append("")
            md.append("#### **Error Details**")
            md.append(f"> {art.summary}")
        md.append("")
        md.append("---")
        md.append("")
        
    return "\n".join(md)


if run and topic:
    status = st.empty()
    status.info(f"Starting agent... (provider: {provider}, model: {model})")

    log_expander = st.expander("Agent Logs", expanded=True)
    log_area = log_expander.empty()

    log_lines: list[str] = []

    def render_logs():
        content = "\n\n".join(log_lines) if log_lines else "_Waiting for agent output..._"
        log_area.markdown(content, unsafe_allow_html=True)

    def sys_box(text: str) -> str:
        return (
            "<div style='background-color: rgba(128, 128, 128, 0.15); padding: 8px 12px; "
            "border-radius: 5px; margin: 15px 0; font-family: monospace; font-size: 0.9em;'>"
            f"<strong>{text}</strong>"
            "</div>"
        )

    async def run_with_streaming(inputs):
        final_state = inputs.copy()

        node_buffers: dict[str, str] = {}

        try:
            async for part in app.astream(
                inputs,
                stream_mode=["messages", "updates"],
            ):
                kind = part[0] if isinstance(part, tuple) else None

                if kind == "messages":
                    _, payload = part
                    msg, metadata = payload if isinstance(payload, tuple) else (payload, {})
                    text = _extract_text(msg)
                    if not text:
                        continue

                    node = metadata.get("langgraph_node", "agent") if isinstance(metadata, dict) else "agent"

                    if node not in node_buffers:
                        node_buffers[node] = ""
                        log_lines.append(f"SYSBOX_{node}")

                    node_buffers[node] += text

                    display_text = clean_log_text(node_buffers[node])

                    for i in range(len(log_lines) - 1, -1, -1):
                        if log_lines[i].startswith(f"SYSBOX_{node}"):
                            log_lines[i] = f"SYSBOX_{node}<!-- -->\n{sys_box(f'[{node}]')}\n\n{display_text}"
                            break
                    render_logs()

                elif kind == "updates":
                    _, state_update = part
                    for node, update in state_update.items():
                        if node == "planner":
                            status.info(f"Planner generated search plan. Executing...")
                            log_lines.append(sys_box(f"[planner done] Search plan ready."))
                            render_logs()

                        elif node == "executor":
                            results = update.get("results", [])
                            status.info(f"Executor finished gathering {len(results)} articles.")
                            log_lines.append(sys_box(f"[executor done] {len(results)} articles gathered."))
                            render_logs()

                        elif node == "joiner":
                            if update.get("search_plan") is not None:
                                round_num = update.get("replan_count", "?")
                                status.warning(f"Results insufficient. Replan round {round_num}...")
                                log_lines.append(sys_box(f"[joiner] Replanning (round {round_num})."))
                            else:
                                status.success("Joining complete. Finalizing...")
                                log_lines.append(sys_box(f"[joiner] Complete."))
                            render_logs()

                        final_state.update(update)

        except Exception as e:
            st.error(f"Agent failed: {e}")
            st.stop()

        return final_state

    result = asyncio.run(run_with_streaming({
        "topic": topic,
        "max_replans": max_replans,
        "replan_count": 0,
        "replan_feedback": "",
        "search_plan": "",
        "results": [],
        "final_output": None,
    }))

    if not result:
        st.stop()

    output = result["final_output"]

    replan_rounds = result.get("replan_count", 0)
    replan_msg = f" · {replan_rounds} replan round{'s' if replan_rounds != 1 else ''}" if replan_rounds > 0 else ""
    status.success(f"Done — {output.successful}/{output.total_articles} articles processed{replan_msg}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", output.total_articles)
    col2.metric("Successful", output.successful)
    col3.metric("Failed", output.failed)
    col4.metric("Overall Sentiment", SENTIMENT_BADGE.get(output.overall_sentiment, output.overall_sentiment))

    if replan_rounds > 0:
        st.info(f"Agent replanned {replan_rounds} time{'s' if replan_rounds != 1 else ''} to improve result quality.")

    if output.overall_summary:
        st.subheader(f"Overall Summary: {topic}")
        st.markdown(output.overall_summary)
        st.divider()

    # Generate and save Markdown report to local results folder
    try:
        report_md = generate_markdown_report(output)
        
        # Ensure results directory exists
        os.makedirs("results", exist_ok=True)
        
        # Prepare filesystem-safe filename
        safe_topic = re.sub(r'[^a-zA-Z0-9_\-]+', '_', topic.lower().strip())
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"news_report_{safe_topic}_{timestamp}.md"
        filepath = os.path.join("results", filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_md)
            
        st.info(f"📁 **Saved report locally to**: `results/{filename}`")
        st.download_button(
            label="📥 Download Markdown Report",
            data=report_md,
            file_name=filename,
            mime="text/markdown",
            use_container_width=True
        )
        st.divider()
    except Exception as e:
        st.error(f"Failed to save Markdown report: {e}")
        st.divider()

    st.subheader("Extracted Articles & Attempts")
    for article in output.articles:
        with st.container(border=True):
            if article.status == "complete":
                st.markdown(f"### [{article.title}]({article.url})")
                st.caption(f"**{article.source}** · {article.published_at[:10] if article.published_at else 'N/A'}")
                st.write(article.summary)
                badge = SENTIMENT_BADGE.get(article.sentiment, article.sentiment)
                st.markdown(f"{badge} `{article.sentiment_score:.2f}`", unsafe_allow_html=True)
            else:
                st.markdown(f"### [{article.title}]({article.url})")
                st.caption(f"**{article.source}** · {article.published_at[:10] if article.published_at else 'N/A'}")
                st.error(article.summary)

elif run and not topic:
    st.warning("Enter a topic first.")