from pydantic import BaseModel
from typing import List, Annotated, Optional
import operator
from typing_extensions import TypedDict



class ArticleResult(BaseModel):
    task_id: int
    title: str
    url: str
    source: str
    published_at: str
    summary: str
    full_content: str = ""
    sentiment: str
    sentiment_score: float
    status: str


class FinalOutput(BaseModel):
    topic: str
    fetched_at: str
    total_articles: int
    successful: int
    failed: int
    overall_sentiment: str
    overall_summary: str
    articles: List[ArticleResult]


class OverallState(TypedDict):
    topic: str
    max_replans: int                    # cap on replan rounds (e.g. 2)
    replan_count: int                   # incremented each time joiner triggers a replan
    replan_feedback: str                # joiner critique passed back to planner
    search_plan: str
    results: Annotated[List[ArticleResult], operator.add]
    final_output: Optional[FinalOutput]


