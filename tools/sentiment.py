import json
from llm import get_chat_model

_client = None


def analyse_sentiment(text: str) -> dict:
    global _client
    if _client is None:
        _client = get_chat_model()
    prompt = (
        'Analyse the sentiment of this article. '
        'Reply in JSON only: {"sentiment": "Positive|Negative|Neutral", "score": 0.0-1.0}\n\n'
        f"{text[:2000]}"
    )
    raw = _client.chat(prompt, max_tokens=60)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"sentiment": "Neutral", "score": 0.5}
