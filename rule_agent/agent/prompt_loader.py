import yaml
import os

BASE_DIR = os.path.dirname(__file__)
PROMPT_DIR = os.path.join(BASE_DIR, "prompts")

def load_prompt(name):
    with open(os.path.join(PROMPT_DIR, f"{name}.yaml")) as f:
        return yaml.safe_load(f)