import time
from openai import OpenAI
from config import (
    OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL,
)

_openrouter = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
_groq        = OpenAI(api_key=GROQ_API_KEY,        base_url=GROQ_BASE_URL)


def _call_with_retry(client, model, messages, max_tokens, retries=3) -> str:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                wait = 10 * (attempt+1)
                print(f"[Rate limit] waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


def generate(messages: list[dict]) -> str:
    """Main answer generation — OpenRouter llama-3.3-70b-instruct."""
    return _call_with_retry(_openrouter, LLM_MODEL, messages, LLM_MAX_TOKENS)


def critique(prompt: str) -> str:
    """Yes/no reflection decisions — Groq llama-3.3-70b-versatile."""
    messages = [{"role": "user", "content": prompt}]
    return _call_with_retry(_groq, GROQ_MODEL, messages, max_tokens=20)