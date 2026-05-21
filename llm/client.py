from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Any


PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI    = "openai"
PROVIDER_DEEPSEEK  = "deepseek"

_DEFAULT_MODELS: dict[str, str] = {
    PROVIDER_ANTHROPIC: "claude-sonnet-4-20250514",
    PROVIDER_OPENAI:    "gpt-4o-mini",
    PROVIDER_DEEPSEEK:  "deepseek-chat",
}

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


@dataclass
class LLMClient:
 
    provider: str
    model: str
    _raw: object = field(repr=False, default=None)

    def chat(self, prompt: str, max_tokens: int = 512) -> str:
       
        if self.provider == PROVIDER_ANTHROPIC:
            return self._chat_anthropic(prompt, max_tokens)
        elif self.provider in (PROVIDER_OPENAI, PROVIDER_DEEPSEEK):
            return self._chat_openai_compat(prompt, max_tokens)
        else:
            raise ValueError(f"Unknown provider: {self.provider!r}")

    def _chat_anthropic(self, prompt: str, max_tokens: int) -> str:
        msg = self._raw.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()

    def _chat_openai_compat(self, prompt: str, max_tokens: int) -> str:
        response = self._raw.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()


def get_llm_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> LLMClient:
  
    provider = (provider or os.environ.get("LLM_PROVIDER", PROVIDER_ANTHROPIC)).lower()
    model    = model or os.environ.get("LLM_MODEL") or _DEFAULT_MODELS.get(provider)

    if provider == PROVIDER_ANTHROPIC:
        import anthropic as _anthropic
        raw = _anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    elif provider == PROVIDER_OPENAI:
        import openai as _openai
        raw = _openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    elif provider == PROVIDER_DEEPSEEK:
        import openai as _openai
        raw = _openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=DEEPSEEK_BASE_URL,
        )

    else:
        raise ValueError(
            f"Unsupported provider: {provider!r}. "
            f"Choose one of: anthropic, openai, deepseek"
        )

    return LLMClient(provider=provider, model=model, _raw=raw)


def get_chat_model(provider: Optional[str] = None, model: Optional[str] = None) -> Any:
    provider = (provider or os.environ.get("LLM_PROVIDER", PROVIDER_ANTHROPIC)).lower()
    model = model or os.environ.get("LLM_MODEL") or _DEFAULT_MODELS.get(provider)

    if provider == PROVIDER_ANTHROPIC:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, api_key=os.environ["ANTHROPIC_API_KEY"])
    elif provider == PROVIDER_OPENAI:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=os.environ["OPENAI_API_KEY"])
    elif provider == PROVIDER_DEEPSEEK:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=os.environ["DEEPSEEK_API_KEY"], base_url=DEEPSEEK_BASE_URL)
    else:
        raise ValueError(f"Unsupported provider: {provider!r}")
