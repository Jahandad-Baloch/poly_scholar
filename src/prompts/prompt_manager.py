from pathlib import Path
import json

class PromptManager:
    def __init__(self, template_dir: str):
        self.templates = {
            t.stem: json.loads(t.read_text())
            for t in Path(template_dir).glob("*.json")
        }

    def build(self, name: str, **kwargs) -> str:
        tpl = self.templates[name]
        return tpl["template"].format(**kwargs)