"""
graph.py — LangGraph StateGraph definition, compiled once at module level.

Graph topology:
    START
      │
      ▼
  classify_query          ← decides: relational / semantic / visual / hybrid
      │
      ▼
  run_tools               ← calls graph_search, vector_search, or vlm_analyze
      │
      ├── documents found  → generate   → END
      │
      └── no documents     → no_answer  → END

The compiled `app` is cached across Lambda warm starts.
"""

from langgraph.graph import StateGraph, START, END

from nodes import GraphRAGState, classify_query, run_tools, generate, no_answer


def route_after_tools(state: GraphRAGState) -> str:
    """Send to 'generate' if we have context, otherwise 'no_answer'."""
    return "generate" if state.get("documents") else "no_answer"


# ── Build the graph ───────────────────────────────────────────────────────────

_builder = StateGraph(GraphRAGState)

_builder.add_node("classify_query", classify_query)
_builder.add_node("run_tools",      run_tools)
_builder.add_node("generate",       generate)
_builder.add_node("no_answer",      no_answer)

_builder.add_edge(START,            "classify_query")
_builder.add_edge("classify_query", "run_tools")
_builder.add_conditional_edges("run_tools", route_after_tools)
_builder.add_edge("generate",       END)
_builder.add_edge("no_answer",      END)

# Compiled once — reused across Lambda warm starts
app = _builder.compile()
