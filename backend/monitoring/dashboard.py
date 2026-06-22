
import time
import os
from monitoring.logger import fetch_all_logs, fetch_stats


def _clear():
    os.system("cls" if os.name == "nt" else "clear")


def _bar(value: float, max_val: float = 5.0, width: int = 20) -> str:
    """ASCII progress bar."""
    if value is None:
        return "[" + "-" * width + "] N/A"
    filled = int((value / max_val) * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {value}"


def render_dashboard():
    stats = fetch_stats()
    logs  = fetch_all_logs()

    _clear()
    print("=" * 65)
    print("   arXiv RAG  —  Monitoring Dashboard")
    print("=" * 65)

    if stats.get("total", 0) == 0:
        print("\n  No queries logged yet. Run some queries first.\n")
        return

    # ── Overview ─────────────────────────────────────────────────
    print(f"\n  Total Queries       : {stats['total']}")
    print(f"  Avg Latency         : {stats['avg_latency_ms']} ms")
    print(f"  Avg Self-RAG Loops  : {stats['avg_rag_iterations']}")
    print(f"  Critique PASS rate  : {stats['critique_pass_count']}/{stats['total']}")

    # ── Guardrails ───────────────────────────────────────────────
    print(f"\n  {'─'*40}")
    print("  GUARDRAILS")
    print(f"  Input  Guard Pass % : {stats['input_pass_pct']}%")
    print(f"  Output Guard Pass % : {stats['output_pass_pct']}%")

    # ── Self-RAG Quality ─────────────────────────────────────────
    print(f"\n  {'─'*40}")
    print("  SELF-RAG CRITIC SCORES  (1–5)")
    print(f"  Avg Faithfulness    : {_bar(stats['avg_faithfulness'])}")
    print(f"  Avg Completeness    : {_bar(stats['avg_completeness'])}")

    # ── RAGAS ────────────────────────────────────────────────────
    print(f"\n  {'─'*40}")
    print("  RAGAS SCORES  (0.0–1.0)  [run ragas_eval.py to populate]")
    f  = stats.get("avg_ragas_faithfulness")
    r  = stats.get("avg_ragas_relevancy")
    p  = stats.get("avg_ragas_precision")
    print(f"  Faithfulness        : {_bar(f, max_val=1.0) if f else 'Not scored yet'}")
    print(f"  Answer Relevancy    : {_bar(r, max_val=1.0) if r else 'Not scored yet'}")
    print(f"  Context Precision   : {_bar(p, max_val=1.0) if p else 'Not scored yet'}")

    # ── Recent queries ───────────────────────────────────────────
    print(f"\n  {'─'*40}")
    print("  RECENT QUERIES  (last 5)\n")
    for log in logs[:5]:
        verdict_symbol = "✓" if log["critique_verdict"] == "PASS" else "✗"
        guard_symbol   = "✓" if log["output_guard_passed"] else "✗"
        print(f"  [{log['timestamp'][:19]}]")
        print(f"  Q: {log['query'][:60]}")
        print(f"  Critic: {verdict_symbol}  |  Output Guard: {guard_symbol}  "
              f"|  Faith: {log['faithfulness_score']}/5  "
              f"|  Loops: {log['self_rag_iterations']}")
        print()

    print("=" * 65)
    print("  Refreshes every 10s  |  Ctrl+C to exit")
    print("=" * 65)


if __name__ == "__main__":
    while True:
        try:
            render_dashboard()
            time.sleep(10)
        except KeyboardInterrupt:
            print("\n[Dashboard] Stopped.")
            break