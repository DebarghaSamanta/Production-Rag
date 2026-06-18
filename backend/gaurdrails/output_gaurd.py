import re
from dataclasses import dataclass

# Cross-encoder rerank score below this = retrieved context was probably irrelevant.
# Tune this after you run RAGAS — start conservative.
MIN_RERANK_SCORE = 0.0   # cross-encoder scores are logits; 0.0 is a safe floor

# Faithfulness score from self-RAG critic below this = likely hallucination
MIN_FAITHFULNESS_SCORE = 2

# If answer contains these phrases the LLM is signaling it couldn't answer
ABSTENTION_PHRASES = [
    "does not contain sufficient information",
    "i don't have enough information",
    "i cannot answer",
    "not mentioned in the context",
    "no information available",
]

# Minimum answer length — single word responses are useless
MIN_ANSWER_LENGTH = 30


@dataclass
class OutputGuardResult:
    passed:       bool
    reason:       str
    should_retry: bool   # True = caller should trigger self-RAG re-retrieval
                         # False = hard failure, return error to user


class OutputGuard:
    """
    Validates generated answers before returning them to the user.
    Catches hallucinations, empty answers, and low-confidence outputs.
    """

    def check(
        self,
        answer:           str,
        critique:         dict,   # from AnswerCritic — has faithfulness, verdict
        context_chunks:   list[dict],
    ) -> OutputGuardResult:

        # 1. Empty or too-short answer
        if len(answer.strip()) < MIN_ANSWER_LENGTH:
            return OutputGuardResult(
                passed=False,
                reason="Generated answer is too short to be useful.",
                should_retry=True
            )

        # 2. LLM explicitly abstained (couldn't answer from context)
        answer_lower = answer.lower()
        for phrase in ABSTENTION_PHRASES:
            if phrase in answer_lower:
                return OutputGuardResult(
                    passed=False,
                    reason=f"Model abstained: '{phrase}' detected in answer. "
                           f"The indexed papers may not cover this topic.",
                    should_retry=True
                )

        # 3. Faithfulness check from self-RAG critic
        faithfulness = critique.get("faithfulness", 0)
        if faithfulness < MIN_FAITHFULNESS_SCORE:
            return OutputGuardResult(
                passed=False,
                reason=f"Faithfulness score {faithfulness}/5 is below threshold "
                       f"{MIN_FAITHFULNESS_SCORE}. Answer may contain hallucinations.",
                should_retry=True
            )

        # 4. No citations present (LLM ignored the citation instruction)
        has_citation = bool(re.search(r'\[\d+\]', answer))
        if not has_citation:
            return OutputGuardResult(
                passed=False,
                reason="Answer contains no citations [1][2][3]. "
                       "Cannot verify grounding.",
                should_retry=False   # retry won't fix a prompt-following failure
            )

        # 5. Low rerank scores across all retrieved chunks
        if context_chunks:
            top_rerank = max(
                c.get("rerank_score", 0.0) for c in context_chunks
            )
            if top_rerank < MIN_RERANK_SCORE:
                return OutputGuardResult(
                    passed=False,
                    reason=f"Top rerank score {top_rerank:.3f} below threshold. "
                           f"Retrieved context may be irrelevant.",
                    should_retry=True
                )

        return OutputGuardResult(passed=True, reason="", should_retry=False)