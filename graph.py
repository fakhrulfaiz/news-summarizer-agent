from langgraph.graph import StateGraph, START, END
from models.schemas import OverallState
from agents.planner import planner_node
from agents.executor import executor_node
from agents.finalizer import finalizer_node

def should_replan(state: OverallState) -> str:
    if state.get("final_output") is None:
        return "planner"
    return END

graph = StateGraph(OverallState)

graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node("finalizer", finalizer_node)

graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")
graph.add_edge("executor", "finalizer")
graph.add_conditional_edges("finalizer", should_replan, ["planner", END])

app = graph.compile()
