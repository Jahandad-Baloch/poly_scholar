# Example usage for agents:
# from src.memory.memory_store import memory_store
# memory_store.save('agent_name', 'session_id', 'some text')
# data = memory_store.get('agent_name', 'session_id')
# memory_store.delete('agent_name', 'session_id')
# keys = memory_store.list_keys('agent_name')

from langgraph.store.memory import InMemoryStore
from langchain.embeddings import OpenAIEmbeddings

class MemoryStore:
    """
    MemoryStore provides both short-term (in-memory) and long-term (persistent) memory for agents.
    Supports saving, retrieving, updating, and deleting memory by namespace and key.
    """
    def __init__(self, embedding_model="text-embedding-3-large"):
        self.store = InMemoryStore()
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

    def save(self, namespace: str, key: str, text: str):
        vec = self.embeddings.embed_query(text)
        self.store.set(namespace, key, {"text": text, "vector": vec})

    def get(self, namespace: str, key: str):
        return self.store.get(namespace, key)

    def delete(self, namespace: str, key: str):
        self.store.delete(namespace, key)

    def update(self, namespace: str, key: str, new_text: str):
        vec = self.embeddings.embed_query(new_text)
        self.store.set(namespace, key, {"text": new_text, "vector": vec})

    def list_keys(self, namespace: str):
        return self.store.list_keys(namespace)

# Singleton instance for use by agents
memory_store = MemoryStore()