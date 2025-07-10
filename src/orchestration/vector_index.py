from src.tools.faiss_tool import FAISSTool
from langchain.embeddings import OpenAIEmbeddings
from src.orchestration.state import format_dynamic_block

def vector_index_node(embeddings_model="text-embedding-3-large"):
    embeddings = OpenAIEmbeddings(embeddings_model)
    faiss_tool = FAISSTool(embeddings)
    def node(state):
        # Add or query documents based on state
        action = state.get("vector_action", "query")
        result = None
        if action == "add":
            documents = state.get("artifacts", {}).get("documents", [])
            ids = state.get("artifacts", {}).get("doc_ids", None)
            result = faiss_tool.add_documents(documents, ids=ids)
            log = "Documents added to vector index."
        else:
            query_text = state.get("artifacts", {}).get("query_text", "")
            k = state.get("artifacts", {}).get("k", 5)
            filter = state.get("artifacts", {}).get("filter", None)
            result = faiss_tool.similarity_search(query_text, k=k, filter=filter)
            log = "Vector index query completed."
        return {"update": {"artifacts": {"vector_index_result": result}, "progress_log": [log]}}
    return node