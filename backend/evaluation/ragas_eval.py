import json
import sqlite3
import yaml
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from core.config import MONITOR_DB_PATH


def _get_conn():
    conn = sqlite3.connect(MONITOR_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def load_prompt(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def compute_relevancy(query, answer):
    q_emb = model.encode(query)
    a_emb = model.encode(answer)
    return float(cosine_similarity([q_emb], [a_emb])[0][0])


def compute_context_precision(query, contexts):
    if not contexts:
        return 0.0

    q_emb = model.encode(query)
    ctx_embs = model.encode(contexts)

    sims = cosine_similarity([q_emb], ctx_embs)[0]

    relevant = sum(1 for s in sims if s > 0.5)

    return relevant / len(contexts)


def build_prompt_from_yaml(prompt_yaml, question, answer, contexts):
    context_block = "\n\n".join(contexts)

    system_prompt = prompt_yaml["system"]
    user_prompt = prompt_yaml["user"].format(
        question=question,
        answer=answer,
        context_block=context_block
    )

    return system_prompt, user_prompt


def compute_faithfulness_llm(llm, prompt_yaml, question, answer, contexts):
    contexts = contexts[:3]  # limit

    system_prompt, user_prompt = build_prompt_from_yaml(
        prompt_yaml,
        question,
        answer,
        contexts
    )

    try:
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        text = response.content if hasattr(response, "content") else str(response)
        print("RAW LLM FAITHFULNESS RESPONSE:", repr(text))
        match = re.search(r"score:\s*([0-9.]+)", text, re.IGNORECASE)

        if match:
            score = float(match.group(1))
            score = max(0.0, min(1.0, score))
        else:
            score = 0.0

        return {
            "score": score,
            "reason": text.strip()
        }

    except Exception as e:
        print(f"[FAITHFULNESS ERROR] {str(e)}")
        return {
            "score": 0.0,
            "reason": f"ERROR: {str(e)}"
        }


def run_manual_eval(llm, prompt_path):
    prompt_yaml = load_prompt(prompt_path)

    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT id, query, answer, context_text
            FROM query_logs
            WHERE manual_faithfulness IS NULL
              AND context_text IS NOT NULL
              AND answer IS NOT NULL
        """).fetchall()

    if not rows:
        print("[EVAL] No rows to evaluate.")
        return

    print(f"[EVAL] Evaluating {len(rows)} rows...")

    for row in rows:
        try:
            contexts = json.loads(row["context_text"])[:3]

            if not contexts:
                continue

            query = row["query"]
            answer = row["answer"]

            # ---- compute metrics ---- #
            relevance = compute_relevancy(query, answer)
            precision = compute_context_precision(query, contexts)

            faith_result = compute_faithfulness_llm(
                llm=llm,
                prompt_yaml=prompt_yaml,
                question=query,
                answer=answer,
                contexts=contexts
            )

            faith_score = faith_result["score"]
            faith_reason = faith_result["reason"]

            # ---- update DB ---- #
            with _get_conn() as conn:
                conn.execute("""
                    UPDATE query_logs
                    SET manual_faithfulness = ?,
                        manual_relevancy = ?,
                        manual_context_precision = ?,
                        manual_faithfulness_reason = ?
                    WHERE id = ?
                """, (
                    faith_score,
                    relevance,
                    precision,
                    faith_reason,
                    row["id"]
                ))
                conn.commit()

            print(f"[EVAL] ID={row['id']} | F={faith_score:.2f} R={relevance:.2f} P={precision:.2f}")

        except Exception as e:
            print(f"[EVAL ERROR] Row {row['id']}: {str(e)}")

if __name__ == "__main__":
    from langchain_groq import ChatGroq
    import os
    from dotenv import load_dotenv
    load_dotenv()

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant",
        temperature=0.0,
    )

    run_manual_eval(
        llm=llm,
        prompt_path="generation/prompts/v1_evaluation.yaml"  # adjust path
    )