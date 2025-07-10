import unittest
from unittest.mock import patch, MagicMock
from src.tools.tavily_tool import TavilyTool
from src.tools.arxiv_tool import ArxivTool
from src.tools.faiss_tool import FAISSTool

class DummyEmbeddings:
    def embed_query(self, text):
        # Return a fixed-size dummy vector
        return [0.1] * 10

class TestTools(unittest.TestCase):

    @patch('src.tools.tavily_tool.TavilySearch')
    def test_tavily_tool(self, MockTavilySearch):
        # Mock TavilySearch.search to return a list of dummy results
        mock_instance = MockTavilySearch.return_value
        mock_instance.search.return_value = [{'title': 'Result', 'link': 'http://example.com'}]
        tool = TavilyTool(api_key="dummy")
        results = tool.search("example query")
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        self.assertIn('title', results[0])

    @patch('src.tools.arxiv_tool.ArxivAPIWrapper')
    def test_arxiv_tool(self, MockArxivAPIWrapper):
        # Mock ArxivAPIWrapper.run to return a dummy string
        mock_instance = MockArxivAPIWrapper.return_value
        mock_instance.run.return_value = "Dummy arxiv result"
        tool = ArxivTool()
        result = tool.run("1605.08386")
        self.assertIsInstance(result, str)
        self.assertIn("Dummy arxiv result", result)

    @patch('src.tools.faiss_tool.FAISS')
    @patch('src.tools.faiss_tool.faiss')
    def test_faiss_tool(self, mock_faiss, mock_FAISS):
        # Mock FAISS vector store
        mock_index = MagicMock()
        mock_faiss.IndexFlatL2.return_value = mock_index
        mock_vector_store = MagicMock()
        mock_vector_store.similarity_search.return_value = [MagicMock()]
        mock_FAISS.return_value = mock_vector_store
        embeddings = DummyEmbeddings()
        tool = FAISSTool(embeddings)
        results = tool.similarity_search("test query")
        self.assertIsInstance(results, list)

if __name__ == '__main__':
    unittest.main()