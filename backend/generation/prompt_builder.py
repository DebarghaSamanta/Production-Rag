from config import SYSTEM_PROMPT, PROMPT_TEMPLATE


def build_prompt(query: str, chunks: list[dict]) -> list[dict]:
    context_blocks = []
    for i, chunk in enumerate(chunks):
        context_blocks.append(f"[Chunk {i+1}] (source: {chunk['source']})\n{chunk['text']}")

    context = "\n\n".join(context_blocks)
    user_message = PROMPT_TEMPLATE.format(context=context, question=query)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]