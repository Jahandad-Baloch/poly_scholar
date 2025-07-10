import unittest
from src.agents.supervisor import Supervisor
from src.agents.literature_search import LiteratureSearch
from src.agents.vector_index import VectorIndex
from src.agents.summarizer import Summarizer
from src.agents.gap_finder import GapFinder

class TestAgents(unittest.TestCase):

    def setUp(self):
        self.supervisor = Supervisor()
        self.literature_search = LiteratureSearch()
        self.vector_index = VectorIndex()
        self.summarizer = Summarizer()
        self.gap_finder = GapFinder()

    def test_supervisor_initialization(self):
        self.assertIsNotNone(self.supervisor)

    def test_literature_search_functionality(self):
        query = "Recent advancements in AI"
        results = self.literature_search.search(query)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_vector_index_creation(self):
        documents = ["Document 1", "Document 2"]
        self.vector_index.create_index(documents)
        self.assertTrue(self.vector_index.is_indexed())

    def test_summarizer_functionality(self):
        text = "This is a long text that needs to be summarized."
        summary = self.summarizer.summarize(text)
        self.assertIsInstance(summary, str)
        self.assertLess(len(summary), len(text))

    def test_gap_finder_functionality(self):
        literature = ["Study A", "Study B"]
        gaps = self.gap_finder.find_gaps(literature)
        self.assertIsInstance(gaps, list)

if __name__ == '__main__':
    unittest.main()