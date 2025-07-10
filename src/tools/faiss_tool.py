
# FAISS vector store integration for LangGraph agents
# Uses langchain_community FAISS and InMemoryDocstore
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

class FAISSTool:
    """
    Wrapper for FAISS vector store using LangChain community toolkit.
    Usage:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        faiss_tool = FAISSTool(embeddings)
        faiss_tool.add_documents([Document(page_content="foo", metadata={"baz": "bar"})])
        results = faiss_tool.similarity_search("foo")
    """
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.index = faiss.IndexFlatL2(len(self.embeddings.embed_query("hello world")))
        self.docstore = InMemoryDocstore({})
        self.index_to_docstore_id = {}
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=self.index,
            docstore=self.docstore,
            index_to_docstore_id=self.index_to_docstore_id
        )

    def add_documents(self, documents, ids=None):
        """
        Add a list of LangChain Document objects to the vector store.
        Optionally provide a list of string IDs.
        """
        return self.vector_store.add_documents(documents=documents, ids=ids)

    def similarity_search(self, query, k=5, filter=None):
        """
        Perform a similarity search for the query string.
        Optionally filter by metadata.
        Returns a list of Document objects.
        """
        return self.vector_store.similarity_search(query=query, k=k, filter=filter)

    def similarity_search_with_score(self, query, k=5, filter=None):
        """
        Perform a similarity search and return (Document, score) tuples.
        """
        return self.vector_store.similarity_search_with_score(query=query, k=k, filter=filter)

    def delete(self, ids):
        """
        Delete documents by their IDs.
        """
        return self.vector_store.delete(ids=ids)

    def save_local(self, folder_path):
        """
        Save the FAISS index and docstore to disk.
        """
        return self.vector_store.save_local(folder_path)

    @classmethod
    def load_local(cls, folder_path, embeddings):
        """
        Load a FAISS vector store from disk.
        """
        vector_store = FAISS.load_local(folder_path, embeddings, allow_dangerous_deserialization=True)
        tool = cls(embeddings)
        tool.vector_store = vector_store
        return tool