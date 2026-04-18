import yaml
import os

BASE_DIR = os.path.dirname(__file__)
PROMPT_DIR = os.path.join(BASE_DIR, "prompts")

def load_prompt(name):
    path = os.path.join(PROMPT_DIR, f"{name}.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)