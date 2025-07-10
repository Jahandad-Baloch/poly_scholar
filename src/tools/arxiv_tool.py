
# Arxiv tool integration for LangGraph agents
# Uses the latest LangChain community API (see arxiv_docs.txt)
from typing import Annotated, Any
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_core.tools import tool

class ArxivTool:
    """
    Wrapper for the ArxivAPIWrapper to fetch academic paper metadata from arXiv.
    Usage:
        arxiv_tool = ArxivTool()
        result = arxiv_tool.run("1605.08386")
    """
    def __init__(self, params: dict = None):
        
        # Initialize the ArxivAPIWrapper with parameters.
        # If no parameters are provided, use default values.
        if params is None:
            params = {
                "top_k_results": 3,
                "ARXIV_MAX_QUERY_LENGTH": 300,
                "load_max_docs": 3,
                "load_all_available_meta": False,
                "doc_content_chars_max": 40000
            }
        self.api = ArxivAPIWrapper(
            arxiv_search=Any,
            arxiv_exceptions=Any,
            continue_on_failure=False,
            **params
        )
        

    def run(self, query: str) -> str:
        """
        Query arXiv for papers or authors. Returns formatted metadata string or error message.
        """
        return self.api.run(query)