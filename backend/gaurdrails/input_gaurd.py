import re
from dataclasses import dataclass

# Minimum meaningful query length
MIN_QUERY_LENGTH = 5

# If query matches these patterns it's likely a prompt injection attempt
INJECTION_PATTERNS = [
    r"ignore (previous|all|above) instructions",
    r"you are now",
    r"disregard your",
    r"forget everything",
    r"act as (a|an)",
    r"jailbreak",
]

# If NONE of these domain signals are present in a long query, it's off-topic.
# Short queries (< 6 words) skip this check — too aggressive for vague but valid queries.
DOMAIN_SIGNALS = [
    "model", "loss", "training", "neural", "network", "paper", "dataset",
    "accuracy", "learning", "attention", "transformer", "embedding", "layer",
    "gradient", "function", "algorithm", "method", "approach", "result",
    "performance", "architecture", "classification", "detection", "image",
    "text", "audio", "video", "multimodal", "feature", "representation",
    "encoder", "decoder", "baseline", "benchmark", "evaluation", "metric",
    "experiment", "ablation", "propose", "novel", "framework", "system",
    "what", "how", "why", "explain", "describe", "compare", "difference",
]


@dataclass
class GuardResult:
    passed:  bool
    reason:  str   # empty string if passed


class InputGuard:
    """
    Validates raw user queries before they enter the retrieval pipeline.
    Cheap, deterministic, no LLM calls — runs in microseconds.
    """

    def check(self, query: str) -> GuardResult:
        query = query.strip()

        # 1. Empty / too short
        if len(query) < MIN_QUERY_LENGTH:
            return GuardResult(
                passed=False,
                reason=f"Query too short (min {MIN_QUERY_LENGTH} chars). "
                       f"Please ask a complete question."
            )

        # 2. Prompt injection attempt
        query_lower = query.lower()
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, query_lower):
                return GuardResult(
                    passed=False,
                    reason="Query contains disallowed instruction patterns."
                )

        # 3. Off-topic check (only for queries >= 6 words)
        word_count = len(query.split())
        if word_count >= 6:
            has_domain_signal = any(sig in query_lower for sig in DOMAIN_SIGNALS)
            if not has_domain_signal:
                return GuardResult(
                    passed=False,
                    reason="Query does not appear to be related to academic research papers. "
                           "Please ask a question about the indexed papers."
                )

        return GuardResult(passed=True, reason="")