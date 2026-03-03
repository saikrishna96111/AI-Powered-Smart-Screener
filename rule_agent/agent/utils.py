import json
import re

def safe_json_parse(response):

    raw = response.content

    # If Gemini returns list
    if isinstance(raw, list):
        raw = raw[0]

    if not isinstance(raw, str):
        raw = str(raw)

    # Remove markdown code blocks
    raw = re.sub(r"```json", "", raw)
    raw = re.sub(r"```", "", raw)

    raw = raw.strip()

    # Extract JSON object only
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)

    try:
        return json.loads(raw)

    except json.JSONDecodeError:
        # Try to fix common issues

        # Replace single quotes with double
        raw = raw.replace("'", '"')

        # Add quotes around unquoted keys
        raw = re.sub(r'(\w+):', r'"\1":', raw)

        return json.loads(raw)
    
def extract_text(response):
    """
    Normalize Gemini / OpenAI response content
    into plain string.
    """

    content = response.content

    # Case 1: Gemini returns list of content blocks
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                texts.append(block["text"])
        return "\n".join(texts)

    # Case 2: Normal string
    if isinstance(content, str):
        return content

    return str(content)