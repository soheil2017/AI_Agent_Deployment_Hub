"""
graph.py — LangGraph StateGraph definition, compiled once at module level.

Graph topology (fast path — no evaluation blocking):
    START → retrieve → grade_documents → [YES] → generate → END
                                       → [NO]  → no_answer → END

Evaluation (Layers 1–3) runs asynchronously after the response is delivered.
See evaluator_handler.py for the background evaluation Lambda.

The compiled app is cached across Lambda warm starts because this module
is imported once when the container initialises.
"""

from langgraph.graph import StateGraph, START, END

from nodes import RAGState, retrieve, grade_documents, generate, no_answer


def route_after_grade(state: RAGState) -> str:
    """Conditional edge: route to 'generate' if docs are relevant, else 'no_answer'."""
    return "generate" if state.get("relevant") else "no_answer"


# Build the graph
_builder = StateGraph(RAGState)

_builder.add_node("retrieve",        retrieve)
_builder.add_node("grade_documents", grade_documents)
_builder.add_node("generate",        generate)
_builder.add_node("no_answer",       no_answer)

_builder.add_edge(START,     "retrieve")
_builder.add_edge("retrieve", "grade_documents")
_builder.add_conditional_edges("grade_documents", route_after_grade)
_builder.add_edge("generate",  END)
_builder.add_edge("no_answer", END)

# Compile once — this is cached across warm starts
app = _builder.compile()
