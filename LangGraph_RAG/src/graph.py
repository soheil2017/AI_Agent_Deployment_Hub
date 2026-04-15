"""
graph.py — LangGraph StateGraph definition, compiled once at module level.

Graph topology:
    START → retrieve → generate → END

The compiled app is cached across Lambda warm starts because this module
is imported once when the container initialises.
"""

from langgraph.graph import StateGraph, START, END

from nodes import RAGState, retrieve, generate

# Build the graph
_builder = StateGraph(RAGState)

_builder.add_node("retrieve", retrieve)
_builder.add_node("generate", generate)

_builder.add_edge(START, "retrieve")
_builder.add_edge("retrieve", "generate")
_builder.add_edge("generate", END)

# Compile once — this is cached across warm starts
app = _builder.compile()
