import os
import yaml
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

PROMPTS_DIR = Path(__file__).parent / "prompts"
GENERATION_MODEL  = "openai/gpt-oss-20b"
MAX_TOKENS        = 1024


def _load_prompt(filename: str) -> dict:
    """
    Loads a prompt template from the prompts/ directory by filename.
    Returns the full yaml dict so callers can access version, changelog, etc.
    """
    path = PROMPTS_DIR / filename
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _build_context_block(context_chunks: list[dict]) -> str:
    """
    Formats the retrieved parent chunks into a numbered block for the prompt.
    Numbering is what enables citation grounding — the LLM cites [1], [2], [3]
    and we map those back to real metadata (source, section_id) after generation.

    Output looks like:
        [1] Source: paper.pdf | Section: 4
        <text of chunk 1>

        [2] Source: paper.pdf | Section: 7
        <text of chunk 2>
    """
    lines = []
    for i, chunk in enumerate(context_chunks, start=1):
        meta   = chunk.get("metadata", {})
        source = meta.get("source", "unknown")
        sec    = meta.get("section_id", "?")
        text   = chunk.get("text", "")
        lines.append(f"[{i}] Source: {source} | Section: {sec}\n{text}")
    return "\n\n".join(lines)


class AnswerGenerator:
    def __init__(self, prompt_file: str = "v1_generation.yaml"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.prompt = _load_prompt(prompt_file)
        print(f"[Generator] Loaded prompt '{self.prompt['name']}' "
              f"v{self.prompt['version']}")

    def generate(self, question: str, context_chunks: list[dict]) -> dict:
        """
        Generates a grounded answer from the final context chunks.

        Returns:
            {
                "answer":          str,   # the raw LLM answer with [1][2] citations
                "context_block":   str,   # the formatted context sent to the LLM
                "prompt_version":  str,
                "citation_map":    dict   # {1: {source, section_id}, 2: {...}, ...}
            }
        """
        context_block = _build_context_block(context_chunks)

        system_msg = self.prompt["system"]
        user_msg   = (self.prompt["user"]
                      .replace("{context_block}", context_block)
                      .replace("{question}", question))

        response = self.client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,   # near-deterministic for factual research answers
            max_tokens=MAX_TOKENS,
        )

        answer = response.choices[0].message.content.strip()

        # Build citation map so the API response can return
        # which [number] maps to which real source
        citation_map = {}
        for i, chunk in enumerate(context_chunks, start=1):
            meta = chunk.get("metadata", {})
            citation_map[i] = {
                "source":     meta.get("source", "unknown"),
                "section_id": meta.get("section_id", -1),
                "parent_id":  meta.get("parent_id", ""),
            }

        return {
            "answer":         answer,
            "context_block":  context_block,
            "prompt_version": self.prompt["version"],
            "citation_map":   citation_map,
        }