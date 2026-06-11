import os
import re
import yaml
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

PROMPTS_DIR  = Path(__file__).parent / "prompts"
CRITIC_MODEL = "llama-3.3-70b-versatile"   # cheap fast model — critic doesn't need power

# If both faithfulness AND completeness score >= this, we pass without re-retrieval
PASS_THRESHOLD = 3


def _load_prompt(filename: str) -> dict:
    path = PROMPTS_DIR / filename
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _parse_critique(raw: str) -> dict:
    """
    Parses the structured critique output from the LLM.
    Expected format (enforced by prompt):
        FAITHFULNESS: 4
        COMPLETENESS: 3
        VERDICT: PASS
        REASON: All claims are directly supported by the retrieved context.
    """
    result = {
        "faithfulness":  0,
        "completeness":  0,
        "verdict":       "FAIL",
        "reason":        "Could not parse critique output.",
        "raw":           raw,
    }

    faith_match = re.search(r"FAITHFULNESS:\s*(\d)", raw, re.IGNORECASE)
    comp_match  = re.search(r"COMPLETENESS:\s*(\d)", raw, re.IGNORECASE)
    verdict     = re.search(r"VERDICT:\s*(PASS|FAIL)", raw, re.IGNORECASE)
    reason      = re.search(r"REASON:\s*(.+)", raw, re.IGNORECASE)

    if faith_match:  result["faithfulness"] = int(faith_match.group(1))
    if comp_match:   result["completeness"] = int(comp_match.group(1))
    if verdict:      result["verdict"]      = verdict.group(1).upper()
    if reason:       result["reason"]       = reason.group(1).strip()

    return result


class AnswerCritic:
    def __init__(self, prompt_file: str = "v1_critique.yaml"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.prompt = _load_prompt(prompt_file)
        print(f"[Critic] Loaded prompt '{self.prompt['name']}' "
              f"v{self.prompt['version']}")

    def critique(
        self,
        question:       str,
        answer:         str,
        context_chunks: list[dict],
        context_block:  str,
    ) -> dict:
        """
        Scores the generated answer against the retrieved context.

        Returns:
            {
                "faithfulness": int (1-5),
                "completeness": int (1-5),
                "verdict":      "PASS" | "FAIL",
                "reason":       str,
                "raw":          str   # raw LLM output for debugging
            }
        """
        user_msg = (self.prompt["user"]
                    .replace("{context_block}", context_block)
                    .replace("{question}", question)
                    .replace("{answer}", answer))

        response = self.client.chat.completions.create(
            model=CRITIC_MODEL,
            messages=[
                {"role": "system", "content": self.prompt["system"]},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,   # fully deterministic — critique must be consistent
            max_tokens=150,
        )

        raw = response.choices[0].message.content.strip()
        return _parse_critique(raw)