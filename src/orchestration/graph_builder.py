builder.add_node(agent, name="Supervisor")
graph = builder.compile(checkpointer=memory)
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.prompts.prompt_manager import PromptManager

from src.agents.agent_nodes import (
    supervisor_node,
    literature_search_node,
    summarizer_node,
    gap_finder_node,
    synthesizer_writer_node,
)
from src.orchestration.state import AppState
from src.orchestration.vector_index import vector_index_node

pm = PromptManager("src/prompts/templates")
memory = MemorySaver()
builder = StateGraph(AppState)


builder.add_node("Supervisor", supervisor_node(pm))
builder.add_node("LiteratureSearch", literature_search_node(pm))
builder.add_node("Summarizer", summarizer_node(pm))
builder.add_node("GapFinder", gap_finder_node(pm))
builder.add_node("SynthesizerWriter", synthesizer_writer_node(pm))
builder.add_node("VectorIndex", vector_index_node())


# Define transitions (linear workflow with Synthesizer/Writer)
builder.add_edge(START, "Supervisor")
builder.add_edge("Supervisor", "LiteratureSearch")
builder.add_edge("LiteratureSearch", "Summarizer")
builder.add_edge("Summarizer", "GapFinder")
builder.add_edge("GapFinder", "SynthesizerWriter")
builder.add_edge("SynthesizerWriter", "VectorIndex")
builder.add_edge("VectorIndex", END)

graph = builder.compile(checkpointer=memory)