from models.schemas import OverallState
from llm import get_chat_model
from datetime import datetime


def planner_node(state: OverallState) -> dict:
    llm = get_chat_model()
    topic = state["topic"]
    replan_count = state.get("replan_count", 0)
    feedback = state.get("replan_feedback", "")

    current_date = datetime.now().strftime("%Y-%m-%d")
    current_year = datetime.now().year

    if replan_count == 0:
        prompt = (
            f"You are a news research planner. Today's date is {current_date}. Your goal is to draft a highly efficient and cohesive search plan "
            f"to find diverse, high-quality recent news articles about: \"{topic}\".\n\n"
            "Rules:\n"
            "- Outline exactly 1 to 2 specific search queries the executor agent should run (keep it to a maximum of 2 to save search credits).\n"
            "- Each query should approach the topic from a distinct angle (e.g. latest developments, expert opinions, regional impact, data/statistics).\n"
            "- The queries should be highly precise and targeted to find real articles via a search engine.\n"
            "- Tell the executor to rely primarily on article titles, sources, and descriptions/snippets. DO NOT instruct the executor to fetch the full content of all articles (at most 1 highly relevant article should be fetched, if any).\n"
            "- DO NOT instruct the executor on how to format its output or ask it to compile a document. The executor only searches and fetches; another agent handles formatting.\n"
            f"- You may use the current year ({current_year}) in some queries if the topic requires recent news.\n"
            "- Output your plan as a clear, concise Markdown document containing only the queries and search angles.\n"
        )
    else:
        prompt = (
            f"You are a news research planner on REPLAN round {replan_count}. Today's date is {current_date}.\n"
            f"Topic: \"{topic}\"\n"
            f"Previous queries failed or returned poor results. Evaluator feedback:\n{feedback}\n\n"
            f"Generate a highly targeted NEW search plan in Markdown that:\n"
            "- Avoids the same angles as before.\n"
            "- Outlines exactly 1 to 2 new search queries (trying different keywords, sources, or perspectives).\n"
            "- Instructs the executor to rely primarily on article descriptions and snippets instead of fetching full contents.\n"
            "- DO NOT instruct the executor on how to format its output or ask it to compile a document.\n"
            f"- Uses the current year ({current_year}) only if strictly necessary.\n"
        )

    response = llm.invoke(prompt)
    plan = str(response.content)

    return {"search_plan": plan, "results": []}
