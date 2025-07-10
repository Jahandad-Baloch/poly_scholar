# from langchain import TavilySearch
from langchain_tavily import TavilySearch

class TavilyTool:
    def __init__(self, api_key: str = None, params: dict = None):
        # self.api_key = api_key
        # self.search_client = TavilySearch(api_key=self.api_key)
        if params is None:
            params = {
                "top_k_results": 10,
                "TAVILY_MAX_QUERY_LENGTH": 300,
                "load_max_docs": 10,
                "load_all_available_meta": False,
                "doc_content_chars_max": 40000
            }
        self.search_client = TavilySearch(
            api_key=api_key,
            top_k_results=params.get("top_k_results", 10),
            TAVILY_MAX_QUERY_LENGTH=params.get("TAVILY_MAX_QUERY_LENGTH", 300),
            load_max_docs=params.get("load_max_docs", 10),
            load_all_available_meta=params.get("load_all_available_meta", False),
            doc_content_chars_max=params.get("doc_content_chars_max", 40000)
        )

    def search(self, query: str, num_results: int = 10):
        results = self.search_client.search(query, num_results=num_results)
        return results

    def get_result_details(self, result_id: str):
        details = self.search_client.get_details(result_id)
        return details
