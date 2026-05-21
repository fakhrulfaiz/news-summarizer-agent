from datetime import datetime, timezone
from collections import Counter
from models.schemas import OverallState, FinalOutput
from llm import get_chat_model
from tools.summarizer import summarize as _summarize
from tools.sentiment import analyse_sentiment as _analyse_sentiment



def finalizer_node(state: OverallState) -> dict:
    results = state["results"]
    replan_count = state.get("replan_count", 0)
    max_replans = state.get("max_replans", 2)

    current_round_results = results

    for r in current_round_results:
        if r.status == "fetched" and r.full_content:
            try:
                r.summary = _summarize(r.full_content)
                s_data = _analyse_sentiment(r.full_content)
                r.sentiment = s_data.get("sentiment", "Neutral")
                r.sentiment_score = float(s_data.get("score", 0.5))
                r.status = "complete"
            except Exception:
                r.status = "failed"
        elif r.status == "pending":
            r.status = "complete"  # standard search snippets are considered complete

    successful = [r for r in current_round_results if r.status == "complete"]
    failed = [r for r in current_round_results if r.status != "complete"]
    total = len(current_round_results)
    llm = get_chat_model()

    should_replan = False
    if replan_count < max_replans:
        if not successful:
            should_replan = True
        else:
            context = "\n\n".join(
                [f"Title: {r.title}\nSummary: {r.summary}" for r in successful]
            )
            eval_prompt = (
                f"You are a research evaluator. The user's topic is: '{state['topic']}'.\n\n"
                f"We have gathered the following articles:\n{context}\n\n"
                f"Are these articles at least broadly related to the topic (e.g., AI, technology, or the main subject)?\n"
                f"Do NOT be overly strict. If there is ANY relevant information at all, reply with exactly 'YES'.\n"
                f"ONLY reply with exactly 'NO' if the articles are COMPLETELY unrelated to the topic or if the search totally failed."
            )
            try:
                res = llm.invoke(eval_prompt)
                if "NO" in str(res.content).upper() and "YES" not in str(res.content).upper():
                    should_replan = True
            except Exception:
                pass # Default to not replanning if eval fails but we have some successful articles

    if should_replan:
        # Generate feedback to guide the planner's next round
        failed_queries = [r.title for r in failed]  # title holds query on failure
        context = (
            f"Topic: {state['topic']}\n"
            f"Replan round: {replan_count + 1}\n"
            f"Results so far: {len(successful)} successful, {len(failed)} failed out of {total}\n"
            f"Failed tasks: {failed_queries}\n"
        )
        feedback_prompt = (
            "You are evaluating a news research agent that returned poor results.\n\n"
            f"{context}\n"
            "In 2-3 sentences, explain why the queries likely failed and suggest what "
            "different search angles or keywords the planner should try next. "
            "Be specific and actionable."
        )
        try:
            response = llm.invoke(feedback_prompt)
            feedback = str(response.content)
        except Exception as e:
            feedback = f"Previous queries returned insufficient results. Try different keywords. Error: {e}"

        return {
            "replan_count": replan_count + 1,
            "replan_feedback": feedback,
            "search_plan": "",
        }

    sentiments = [r.sentiment for r in successful]
    overall = Counter(sentiments).most_common(1)[0][0] if sentiments else "Neutral"

    if successful:
        deep_reads = [r for r in successful if r.full_content]
        search_snippets = [r for r in successful if not r.full_content]
        
        context_parts = []
        if deep_reads:
            context_parts.append("--- DEEP READ ARTICLES (FETCHED FOR HIGH RELEVANCE) ---")
            for r in deep_reads:
                context_parts.append(f"Title: {r.title}\nSource: {r.source}\nURL: {r.url}\nSummary: {r.summary}")
        if search_snippets:
            context_parts.append("--- OTHER GATHERED ARTICLES (SEARCH SNIPPETS) ---")
            for r in search_snippets:
                context_parts.append(f"Title: {r.title}\nSource: {r.source}\nURL: {r.url}\nSummary: {r.summary}")
        
        context = "\n\n".join(context_parts)
        
        current_date = datetime.now().strftime("%B %d, %Y")
        
        deep_read_prompt = ""
        if deep_reads:
            deep_read_prompt = (
                f"#### Highly Related Articles (Deep Dives)\n"
                f"[List ONLY the articles provided under the 'DEEP READ ARTICLES' section above]\n"
                f"**Item 1:** [Title/Summary]\n"
                f"**Source:** [Source]\n"
                f"**Link:** [URL]\n"
                f"(Repeat for all deep read items)\n"
            )

        prompt = (
            f"You are a news summariser. Summarise the following articles into a clear, well-structured news summary about '{state['topic']}'.\n\n"
            f"Articles Context:\n{context}\n\n"
            f"Write your summary in a journalistic style. Structure it naturally based on what the data warrants — you do not need to force a fixed number of trends or sections. "
            f"Include the following where relevant:\n\n"
            f"- **Date:** {current_date}\n"
            f"- **Key Statistics**: a brief table or bullet list (e.g. total sources, date range covered, overall sentiment direction)\n"
            f"- **Core Themes & Insights**: identify the most important themes, developments, or narratives emerging from the articles. "
            f"Group related articles naturally — there may be 2, 3, or 4 themes depending on the data. For each theme, briefly describe what happened and why it matters.\n"
            f"- **Deep Dive Insights**: if any articles were fetched in full, weave those richer insights into the narrative with inline citations linking to the source URL (e.g. [Source Name](URL)).\n"
            f"- **Summary Insight**: a concise concluding paragraph summarising the overall picture and any near-term implications.\n\n"
            f"Write in clear, professional prose. Avoid mechanical bullet-point repetition. Prioritise readability and analytical depth over rigid formatting.\n"
        )
        try:
            res = llm.invoke(prompt)
            overall_summary = str(res.content)
        except Exception as e:
            overall_summary = f"Failed to generate overall summary: {e}"
    else:
        # User requested a summary of what they already got/tried if everything failed
        fb = state.get("replan_feedback", "No specific feedback.")
        failed_titles = [r.title for r in failed]
        prompt = (
            f"You are a news analyst. The research agent failed to process any full articles about '{state['topic']}'.\n"
            f"It attempted to search for the following angles/titles: {failed_titles}\n"
            f"Internal feedback: {fb}\n\n"
            f"Provide a brief 2-3 sentence summary explaining to the user that very few or no accessible news articles could be found for this topic. "
            f"Mention what angles were attempted."
        )
        try:
            res = llm.invoke(prompt)
            overall_summary = str(res.content)
        except Exception as e:
            overall_summary = f"No articles were successfully processed. Attempted angles: {failed_titles}"
    all_successful = [r for r in results if r.status == "complete"]
    all_failed = [r for r in results if r.status != "complete"]
    all_total = len(results)

    replan_note = f" (after {replan_count} replan round{'s' if replan_count != 1 else ''})" if replan_count > 0 else ""
    output = FinalOutput(
        topic=state["topic"] + replan_note,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        total_articles=all_total,
        successful=len(all_successful),
        failed=len(all_failed),
        overall_sentiment=overall,
        overall_summary=overall_summary,
        articles=results,
    )
    return {"final_output": output}
