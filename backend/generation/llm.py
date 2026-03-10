from openai import OpenAI
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS


_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
)


def generate(messages: list[dict]) -> str:
    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    return response.choices[0].message.content