import unittest
from src.prompts.prompt_manager import PromptManager
import os

class TestPromptManager(unittest.TestCase):
    def setUp(self):
        # Use the actual templates directory
        self.template_dir = os.path.join(os.path.dirname(__file__), '../src/prompts/templates')
        self.manager = PromptManager(self.template_dir)

    def test_load_templates(self):
        # Should load all templates
        self.assertIn('summary_prompt', self.manager.templates)
        self.assertIn('search_prompt', self.manager.templates)
        self.assertIn('gap_prompt', self.manager.templates)

    def test_build_summary_prompt(self):
        summary = self.manager.build('summary_prompt', content="Test content.")
        self.assertIn('Summarize the following content', summary)
        self.assertIn('Test content.', summary)

    def test_build_search_prompt(self):
        # The search prompt is a schema, not a string template, so just check loading
        search = self.manager.templates['search_prompt']
        self.assertIn('template', search)
        self.assertIn('fields', search)

    def test_build_gap_prompt(self):
        # The gap prompt is a schema, not a string template, so just check loading
        gap = self.manager.templates['gap_prompt']
        self.assertIn('description', gap['template'])
        self.assertIn('fields', gap['template'])

if __name__ == '__main__':
    unittest.main()
