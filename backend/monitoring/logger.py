import os
import json
import sqlite3
import datetime
from core.config import MONITOR_DB_PATH

os.makedirs(os.path.dirname(MONITOR_DB_PATH), exist_ok=True)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(MONITOR_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_logs (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp               TEXT    NOT NULL,
                query                   TEXT    NOT NULL,
                answer                  TEXT,
                context_text            TEXT,   -- JSON list of retrieved chunk texts
                self_rag_iterations     INTEGER,
                faithfulness_score      INTEGER,
                completeness_score      INTEGER,
                critique_verdict        TEXT,
                input_guard_passed      INTEGER,
                output_guard_passed     INTEGER,
                guard_reason            TEXT,
                context_chunks_used     INTEGER,
                dense_raw_count         INTEGER,
                sparse_raw_count        INTEGER,
                top_rerank_score        REAL,
                prompt_version          TEXT,
                manual_faithfulness      REAL,
                manual_relevancy  REAL,
                manual_context_precision REAL,
                manual_faithfulness_reason TEXT,
                latency_ms              INTEGER
            )
        """)
        conn.commit()
    print(f"[Monitor] DB initialized at {MONITOR_DB_PATH}")


def log_query(
    query:               str,
    answer:              str,
    context_chunks:      list[dict],   # ← new: pass the actual chunks
    self_rag_iterations: int,
    critique:            dict,
    input_guard_passed:  bool,
    output_guard_passed: bool,
    guard_reason:        str,
    telemetry:           dict,
    prompt_version:      str,
    latency_ms:          int,
):
    # Extract just the text from each chunk and store as JSON list
    context_texts = [c.get("text", "") for c in context_chunks]
    context_json  = json.dumps(context_texts)

    top_rerank = telemetry.get("top_rerank_score", None)

    with _get_conn() as conn:
        conn.execute("""
            INSERT INTO query_logs (
                timestamp, query, answer, context_text,
                self_rag_iterations,
                faithfulness_score, completeness_score, critique_verdict,
                input_guard_passed, output_guard_passed, guard_reason,
                context_chunks_used, dense_raw_count, sparse_raw_count,
                top_rerank_score, prompt_version, latency_ms
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            datetime.datetime.utcnow().isoformat(),
            query,
            answer,
            context_json,
            self_rag_iterations,
            critique.get("faithfulness", 0),
            critique.get("completeness", 0),
            critique.get("verdict", "UNKNOWN"),
            int(input_guard_passed),
            int(output_guard_passed),
            guard_reason,
            telemetry.get("context_chunks_used", 0),
            telemetry.get("dense_raw_count", 0),
            telemetry.get("sparse_raw_count", 0),
            top_rerank,
            prompt_version,
            latency_ms,
        ))
        conn.commit()


def fetch_all_logs() -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM query_logs ORDER BY timestamp DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def fetch_stats() -> dict:
    with _get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM query_logs").fetchone()[0]
        if total == 0:
            return {"total_queries": 0}

        row = conn.execute("""
            SELECT
                COUNT(*)                                              AS total,
                ROUND(AVG(faithfulness_score), 2)                    AS avg_faithfulness,
                ROUND(AVG(completeness_score), 2)                    AS avg_completeness,
                ROUND(100.0 * SUM(input_guard_passed)  / COUNT(*), 1) AS input_pass_pct,
                ROUND(100.0 * SUM(output_guard_passed) / COUNT(*), 1) AS output_pass_pct,
                ROUND(AVG(self_rag_iterations), 2)                   AS avg_rag_iterations,
                ROUND(AVG(latency_ms), 0)                            AS avg_latency_ms,
                ROUND(AVG(top_rerank_score), 4)                      AS avg_rerank_score,
                ROUND(AVG(manual_faithfulness), 3)       AS avg_ragas_faithfulness,
                ROUND(AVG(manual_relevancy), 3)          AS avg_ragas_relevancy,
                ROUND(AVG(manual_context_precision), 3)             AS avg_ragas_precision,
                SUM(CASE WHEN critique_verdict='PASS' THEN 1 ELSE 0 END) AS critique_pass_count
            FROM query_logs
        """).fetchone()
    return dict(row)