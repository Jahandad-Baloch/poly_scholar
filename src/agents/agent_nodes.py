"""PolyScholar - Refactored agent node definitions

This module centralises all LangGraph node factory functions for agent roles
and aligns them with the helper utilities provided in
``src/orchestration/llm_model.py`` (``initialize_llm``, ``parse_llm_response``,
``handle_agent_response``).  Each factory returns a *callable node* ready to be
registered in a ``StateGraph``.

The key improvements over the previous per-file implementations are:
--------------------------------------------------------------------
• **Single-source LLM initialisation** - all nodes call ``initialize_llm`` to
  ensure consistent temperature / token limits across the workflow.
• **Unified response handling** - agent-specific post-processing is delegated
  to ``handle_agent_response``, guaranteeing state updates follow a standard
  schema recognised by downstream components.
• **Reduced boilerplate** - common logic (building prompts, invoking the LLM,
  parsing the response) lives in the private helper ``_invoke_and_route``.
• **Easier maintenance** - adding a new role requires only a specialised
  ``build_prompt`` callback (if the default signature isn't sufficient).
"""
from __future__ import annotations

from typing import Callable, Any, Dict

from src.orchestration.llm_model import (
    initialize_llm,
    handle_agent_response,
)
from src.prompts.prompt_manager import PromptManager
from src.orchestration.state import AppState, format_dynamic_block

# Optional tool imports - only initialised inside the node to avoid heavy
# imports when not required.
from langchain_openai import ChatOpenAI  # type: ignore  # used for type hints

# --------------------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------------------

def _invoke_and_route(
    agent_name: str,
    llm: ChatOpenAI,
    prompt: str,
    extra_update: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run the LLM *once* and map its output to a LangGraph state update.

    Parameters
    ----------
    agent_name : str
        Semantic identifier understood by ``handle_agent_response``.
    llm : ChatOpenAI
        A *pre-initialised* model instance (usually via ``initialize_llm``).
    prompt : str
        The fully-rendered prompt to feed to the LLM.
    extra_update : dict | None
        Optional additional mutations to merge into ``handle_agent_response``
        output (e.g. raw search results).
    """
    raw_response = llm.invoke(prompt)
    update_dict = handle_agent_response(agent_name, raw_response)

    if extra_update:
        # Merge without clobbering existing keys.
        target = update_dict.setdefault("update", {})
        for k, v in extra_update.items():
            if k == "artifacts" and k in target:
                target["artifacts"].update(v)  # type: ignore[arg-type]
            else:
                target[k] = v

    return update_dict


# --------------------------------------------------------------------------------------
# Public factory functions
# --------------------------------------------------------------------------------------

def supervisor_node(prompt_manager: PromptManager) -> Callable[[AppState], Dict]:
    llm = initialize_llm()

    def node(state: AppState):
        prompt = prompt_manager.build(
            role="expert_supervisor",
            dynamic_state=format_dynamic_block(state),
        )
        return _invoke_and_route("supervisor", llm, prompt)

    return node


def summarizer_node(prompt_manager: PromptManager) -> Callable[[AppState], Dict]:
    llm = initialize_llm()

    def node(state: AppState):
        prompt = prompt_manager.build(
            role="synthesizer_writer",
            dynamic_state=format_dynamic_block(state),
            content=state.get("artifacts", {}).get("to_summarize", ""),
        )
        return _invoke_and_route("summarizer", llm, prompt)

    return node


def gap_finder_node(prompt_manager: PromptManager) -> Callable[[AppState], Dict]:
    llm = initialize_llm()

    def node(state: AppState):
        prompt = prompt_manager.build(
            role="screening_specialist",
            dynamic_state=format_dynamic_block(state),
            content=state.get("artifacts", {}).get("to_analyze", ""),
            research_topic=state.get("topic", ""),
            existing_research=state.get("artifacts", {}).get("existing_research", ""),
            desired_outcome=state.get("artifacts", {}).get("desired_outcome", ""),
        )
        return _invoke_and_route("gap_finder", llm, prompt)

    return node


def synthesizer_writer_node(prompt_manager: PromptManager) -> Callable[[AppState], Dict]:
    llm = initialize_llm()

    def node(state: AppState):
        prompt = prompt_manager.build(
            role="synthesizer_writer",
            dynamic_state=format_dynamic_block(state),
            content=state.get("artifacts", {}).get("extracted_data", ""),
            literature_summary=state.get("artifacts", {}).get("literature_summary", ""),
            gaps=state.get("artifacts", {}).get("gaps", ""),
        )
        return _invoke_and_route("synthesizer_writer", llm, prompt)

    return node


# ------------------------
# Special - Literature Search
# ------------------------


def literature_search_node(prompt_manager: PromptManager) -> Callable[[AppState], Dict]:
    """The only agent that *also* calls an external search tool before the LLM."""
    llm = initialize_llm()
    from src.tools.arxiv_tool import ArxivTool
    arxiv_tool = ArxivTool()

    def node(state: AppState):
        query = state.get("research_question", "")
        results = arxiv_tool.run(query)

        prompt = prompt_manager.build(
            role="search_specialist",
            dynamic_state=format_dynamic_block(state),
            content=str(results),
        )

        # Use handle_agent_response to structure the summary; then merge in raw results.
        extra = {"artifacts": {"literature_results": results}}
        return _invoke_and_route("literature_search", llm, prompt, extra_update=extra)

    return node


# --------------------------------------------------------------------------------------
# Convenience export list
# --------------------------------------------------------------------------------------
__all__ = [
    "supervisor_node",
    "literature_search_node",
    "summarizer_node",
    "gap_finder_node",
    "synthesizer_writer_node",
]
