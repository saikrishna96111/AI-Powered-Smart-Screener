def extract_text(response) -> str:
    content = response.content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                texts.append(block["text"])
        return "\n".join(texts)
    if isinstance(content, str):
        return content
    return str(content)
