import json
import os
from typing import Annotated, Literal, Optional
import operator
import datetime
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.errors import GraphRecursionError
from llm import get_chat_model
from models.schemas import OverallState, ArticleResult
from tools.mcp_tools import DDG_MCP_CONFIG

import re
import asyncio
from newsapi import NewsApiClient

@tool
def news_api_search(query: str) -> str:
    """Search NewsAPI for recent articles matching the query.
    Returns a JSON list of {title, url, source, published_at, description} objects.
    Use this first - it provides structured metadata."""
    try:
        client = NewsApiClient(api_key=os.environ["NEWS_API_KEY"])
        response = client.get_everything(
            q=query,
            language="en",
            sort_by="relevancy",
            page_size=10,
        )
        articles = response.get("articles", [])
        results = [
            {
                "title": a.get("title", "No title"),
                "url": a.get("url", ""),
                "source": a.get("source", {}).get("name", "Unknown"),
                "published_at": a.get("publishedAt", ""),
                "description": a.get("description", ""),
            }
            for a in articles
            if a.get("url")
        ]
        return json.dumps(results) if results else "No articles found via NewsAPI."
    except Exception as e:
        return f"ERROR: NewsAPI failed — {e}"

def parse_ddg_results(text: str) -> str:
    results = []
    blocks = text.split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) >= 3 and lines[0].strip().split(". ", 1)[0].isdigit():
            title = lines[0].strip().split(". ", 1)[-1]
            url = lines[1].replace("URL:", "").strip()
            summary = lines[2].replace("Summary:", "").strip()
            results.append({"title": title, "url": url, "description": summary, "source": "Web Search"})
    return json.dumps(results) if results else text

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    search_plan: str
    gathered_articles: list[ArticleResult]


SYSTEM_PROMPT = SystemMessage(content=(
    "You are an autonomous news curation agent. You will receive a search plan detailing what information to gather.\n"
    "Your goal is to execute the search plan using your search tools (NewsAPI preferred, DuckDuckGo fallback).\n\n"
    "CRITICAL RULES:\n"
    "- When you call a search tool, all discovered articles are AUTOMATICALLY saved to your memory. You do not need to do anything to save them.\n"
    "- Because you are a curator, you should NOT do deep research on every article.\n"
    "- You may OPTIONALLY use `fetch_content(url)` on a small number (max 2-3) of the most highly relevant URLs to get the full article text. The system will automatically extract and save the raw text for the final evaluation phase.\n"
    "- Do NOT try to fetch URLs that end in .pdf.\n"
    "- STRICT RATE LIMIT RULE: DO NOT generate tool calls for `news_api_search` more than 2 times total. Limit it to 1 or 2 queries ONLY per session. You MUST use the `search` tool (DuckDuckGo) for the rest of your searches.\n"
    "- Once you have executed the search plan and feel you have gathered a good breadth of articles, STOP and reply with: DONE\n"
))


def _make_agent_node(tools_list: list):
    llm = get_chat_model().bind_tools(tools_list)

    def agent_node(state: AgentState) -> dict:
        current_date = datetime.datetime.now().strftime("%B %Y")
        system_msg_content = SYSTEM_PROMPT.content + f"\n- Today's date is {current_date}. Keep this context in mind when reading news."
        system_msg = SystemMessage(content=system_msg_content)
        response = llm.invoke([system_msg] + state["messages"])
        return {"messages": [response]}

    return agent_node

_ddg_lock = None

def get_ddg_lock():
    global _ddg_lock
    # Only create the lock if it doesn't exist, to share it across the current loop
    if _ddg_lock is None:
        _ddg_lock = asyncio.Lock()
    return _ddg_lock

def _make_tools_node(tools_by_name: dict):
    async def tools_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        results = []
        updates: dict = {}

        # Count previous tool executions in history
        search_count = 0
        fetch_count = 0
        for msg in state["messages"][:-1]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tname = tc.get("name")
                    if tname in ("news_api_search", "search"):
                        search_count += 1
                    elif tname == "fetch_content":
                        fetch_count += 1

        for tc in last.tool_calls:
            name = tc["name"]
            args = tc.get("args", {})
            output = f"ERROR: unknown tool '{name}'"
            
            # Check limits
            if name in ("news_api_search", "search"):
                if search_count >= 3:
                    output = "ERROR: Search tool call limit reached (max 3 search queries allowed per session to save API costs). Please finalize the process and reply with DONE."
                    results.append(ToolMessage(content=output, tool_call_id=tc["id"]))
                    continue
                else:
                    search_count += 1
            elif name == "fetch_content":
                if fetch_count >= 2:
                    output = "ERROR: Fetch content limit reached (max 2 fetches allowed per session to save API costs). Please rely on existing summaries/descriptions and reply with DONE."
                    results.append(ToolMessage(content=output, tool_call_id=tc["id"]))
                    continue
                else:
                    fetch_count += 1

            try:
                tool_fn = tools_by_name.get(name)
                if tool_fn is not None:
                    if name in ("search", "fetch_content"):
                        async with get_ddg_lock():
                            await asyncio.sleep(1.5)  
                            output = await tool_fn.ainvoke(args)
                    else:
                        output = await tool_fn.ainvoke(args)

                # Post-process tool outputs
                gathered = state.get("gathered_articles", []).copy()
                
                if name in ("news_api_search", "search") and not str(output).startswith("ERROR"):
                    raw_out = str(output)
                    if name == "search":
                        if isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict):
                            raw_out = output[0].get("text", str(output))
                        elif hasattr(output, "content"):
                            raw_out = output.content
                        raw_out = parse_ddg_results(raw_out)
                    try:
                        articles = json.loads(raw_out)
                        for a in articles:
                            # Avoid duplicates
                            if not any(existing.url == a.get("url") for existing in gathered):
                                res = ArticleResult(
                                    task_id=1,
                                    title=a.get("title", "Unknown"),
                                    url=a.get("url", ""),
                                    source=a.get("source", "Unknown"),
                                    published_at=a.get("publishedAt", a.get("published_at", "")),
                                    summary=a.get("description", a.get("content", "")),
                                    sentiment="Neutral",
                                    sentiment_score=0.5,
                                    status="pending"
                                )
                                gathered.append(res)
                        updates["gathered_articles"] = gathered
                        output = f"Successfully saved {len(articles)} articles to memory. You can continue searching or reply DONE."
                    except Exception as e:
                        pass
                
                elif name == "fetch_content" and not str(output).startswith("ERROR"):
                    if isinstance(output, list) and len(output) > 0 and isinstance(output[0], dict):
                        text = output[0].get("text", str(output))[:10000]
                    elif hasattr(output, "content"):
                        text = str(output.content)[:10000]
                    else:
                        text = str(output)[:10000]
                    
                    found = False
                    for a in gathered:
                        if a.url == args.get("url"):
                            a.full_content = text
                            a.status = "fetched"
                            found = True
                            break
                    if found:
                        updates["gathered_articles"] = gathered
                        output = f"Successfully fetched and stored raw article content in memory. You may search for more or reply DONE."
                    else:
                        output = f"Fetched article, but URL not found in gathered list."
                        
            except Exception as e:
                output = f"ERROR: {e}"

            results.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))

        return {"messages": results, **updates}
    return tools_node


def should_continue(state: AgentState) -> Literal["tools", "finalize"]:
    if state["messages"][-1].tool_calls:
        return "tools"
    return "finalize"


def finalize_node(state: AgentState) -> dict:
    return {"results": state.get("gathered_articles", [])}


async def run_executor_async(search_plan: str) -> list[ArticleResult]:
    mcp_client = MultiServerMCPClient(DDG_MCP_CONFIG)
    mcp_tools = await mcp_client.get_tools()
    all_tools = [news_api_search] + mcp_tools
    tools_by_name = {t.name: t for t in all_tools}

    builder = StateGraph(AgentState)
    builder.add_node("agent", _make_agent_node(all_tools))
    builder.add_node("tools", _make_tools_node(tools_by_name))
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, ["tools", "finalize"])
    builder.add_edge("tools", "agent")
    builder.add_edge("finalize", END)

    agent = builder.compile()

    try:
        final_state = None
        async for s in agent.astream({
            "messages": [HumanMessage(content=f"Search Plan:\n{search_plan}")],
            "search_plan": search_plan,
            "gathered_articles": [],
        }, config={"recursion_limit": 20}, stream_mode="values"):
            final_state = s
        
        return final_state.get("gathered_articles", []) if final_state else []
    except GraphRecursionError:
        if final_state:
            return final_state.get("gathered_articles", [])
        return []

async def executor_node(state: OverallState) -> dict:
    plan = state.get("search_plan", "")
    results = await run_executor_async(plan)
    return {"results": results}
