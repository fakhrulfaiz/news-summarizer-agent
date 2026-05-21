from llm import get_chat_model

_client = None

def summarize(text: str) -> str:
    global _client
    if _client is None:
        _client = get_chat_model()
    prompt = (
        "Summarize this news article in exactly 3 sentences. "
        "Be factual and concise.\n\n"
        f"{text[:3000]}"
    )
    return _client.chat(prompt, max_tokens=200)
