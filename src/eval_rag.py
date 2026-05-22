"""
eval_rag.py
-----------
RAG evaluation script using LLM-as-judge for faithfulness and relevance scoring.

What it does:
1. Loads test questions from test_questions.json
2. For each question:
   - Calls your RAG pipeline (handle_rag_query)
   - Retrieves the chunks the RAG used (for transparency)
   - Asks a separate LLM call to score the answer for FAITHFULNESS and RELEVANCE (1-5)
3. Writes per-question results to eval_results.csv
4. Prints a summary with average scores

Why LLM-as-judge:
- A real RAGAS-style framework requires retrieved-context comparisons, which need
  ground-truth answers per question. That's a lot of manual work upfront.
- LLM-as-judge with a clear rubric is the lightest-weight method that still gives
  defensible numbers. Industry uses this widely (e.g., MT-Bench, Arena-Hard).

How to use it:
  $ python -m src.eval_rag                 # uses test_questions.json in repo root
  $ python -m src.eval_rag --questions path/to/questions.json
  $ python -m src.eval_rag --sample 5      # only run first 5 questions (quick test)

Output:
  eval_results.csv — per-question scores and answers
  Console summary — averages, distribution

What to say in interviews:
  "I built a 30-question eval set with LLM-as-judge scoring on a 1-5 rubric for both
  faithfulness (is the answer supported by retrieved context?) and relevance (does
  it actually answer the question?). My current baseline is faithfulness X.X/5 and
  relevance X.X/5. I run it as a regression check whenever I change the chunking
  strategy, embedding model, or retrieval parameters."
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

# Make src importable when run from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_groq import ChatGroq
from src.rag_module import handle_rag_query, _get_vectorstore


# ─── Config ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_QUESTIONS_PATH = os.path.join(BASE_DIR, "test_questions.json")
DEFAULT_RESULTS_PATH = os.path.join(BASE_DIR, "eval_results.csv")

# Use a slightly larger model for judging — Groq is free so this is fine
JUDGE_MODEL = "llama-3.1-8b-instant"

# Delay between calls to avoid rate-limiting on Groq free tier
DELAY_SECONDS = 1.0


# ─── Judge prompt ────────────────────────────────────────────────────
JUDGE_PROMPT = """You are an impartial judge evaluating a RAG system's answer.

QUESTION:
{question}

RETRIEVED CONTEXT (what the RAG system used to answer):
{context}

ANSWER GIVEN BY THE RAG SYSTEM:
{answer}

Score the answer on TWO dimensions, each from 1 to 5:

1. FAITHFULNESS: Is the answer supported by the retrieved context?
   - 5 = Every claim is directly supported by context
   - 4 = Mostly supported, minor unsupported details
   - 3 = Half supported, half inferred or unclear
   - 2 = Mostly unsupported claims (hallucination)
   - 1 = Entirely fabricated, no relation to context
   - N/A = If the system correctly refused to answer ("I don't know") because context didn't have the info, score 5 (this is correct behavior)

2. RELEVANCE: Does the answer actually address the question?
   - 5 = Directly and completely answers the question
   - 4 = Mostly answers, missing minor details
   - 3 = Partially answers, missing key info
   - 2 = Tangentially related but doesn't really answer
   - 1 = Off-topic or refuses without justification

Respond in EXACTLY this format (no extra text):
FAITHFULNESS: <1-5>
RELEVANCE: <1-5>
REASONING: <one sentence>
"""


# ─── Judge ────────────────────────────────────────────────────────────
_judge = None


def get_judge():
    global _judge
    if _judge is None:
        _judge = ChatGroq(
            model=JUDGE_MODEL,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,  # Deterministic judgments
        )
    return _judge


def get_retrieved_context(question, k=2):
    """Pull the same chunks the RAG would retrieve, so we can show the judge."""
    try:
        vectorstore = _get_vectorstore()
        docs = vectorstore.similarity_search(question, k=k)
        return "\n\n---\n\n".join([d.page_content for d in docs])
    except Exception as e:
        return f"[Could not retrieve context: {e}]"


def parse_judge_response(text):
    """Parse FAITHFULNESS: X / RELEVANCE: X / REASONING: ... format."""
    faithfulness = None
    relevance = None
    reasoning = ""

    for line in text.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("FAITHFULNESS:"):
            try:
                faithfulness = int(line.split(":", 1)[1].strip()[0])
            except (ValueError, IndexError):
                faithfulness = None
        elif line.upper().startswith("RELEVANCE:"):
            try:
                relevance = int(line.split(":", 1)[1].strip()[0])
            except (ValueError, IndexError):
                relevance = None
        elif line.upper().startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()

    return faithfulness, relevance, reasoning


def judge_answer(question, context, answer):
    """Send the (question, context, answer) tuple to the LLM judge."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        context=context[:3000],  # Cap context length to avoid token overflow
        answer=answer,
    )
    try:
        response = get_judge().invoke(prompt)
        return parse_judge_response(response.content)
    except Exception as e:
        print(f"  [Judge error: {e}]")
        return None, None, f"Judge failed: {e}"


# ─── Main eval loop ──────────────────────────────────────────────────
def run_eval(questions_path, results_path, sample_size=None):
    """Run the full evaluation and write results to CSV."""
    # Sanity checks
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY environment variable not set.")
        print("Set it with: set GROQ_API_KEY=your_key_here  (Windows)")
        print("         or: export GROQ_API_KEY=your_key   (macOS/Linux)")
        return

    with open(questions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    questions = data["questions"]

    if sample_size:
        questions = questions[:sample_size]

    print(f"\nRunning RAG eval on {len(questions)} questions")
    print(f"Results will be saved to: {results_path}\n")

    results = []
    failed = 0

    for i, q in enumerate(questions, 1):
        qid = q["id"]
        question = q["question"]
        category = q.get("category", "general")

        print(f"[{i}/{len(questions)}] {qid} — {question[:60]}...")

        # Step 1: Get the RAG's answer
        try:
            answer = handle_rag_query(question, session_id=f"eval_{qid}")
        except Exception as e:
            print(f"  RAG ERROR: {e}")
            failed += 1
            continue

        # Step 2: Get the retrieved context for the judge
        context = get_retrieved_context(question)

        # Step 3: Judge the answer
        faithfulness, relevance, reasoning = judge_answer(question, context, answer)

        print(f"  Faithfulness: {faithfulness} | Relevance: {relevance}")

        results.append({
            "id": qid,
            "category": category,
            "question": question,
            "answer": answer[:500],  # Truncate for CSV readability
            "faithfulness": faithfulness,
            "relevance": relevance,
            "reasoning": reasoning,
        })

        # Be nice to the Groq rate limits
        time.sleep(DELAY_SECONDS)

    # ─── Write CSV ────────────────────────────────────────────────
    if results:
        with open(results_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # ─── Print summary ────────────────────────────────────────────
    print_summary(results, failed)


def print_summary(results, failed):
    """Print aggregate statistics to console."""
    print("\n" + "=" * 60)
    print("EVAL SUMMARY")
    print("=" * 60)

    valid_faith = [r["faithfulness"] for r in results if r["faithfulness"] is not None]
    valid_rel = [r["relevance"] for r in results if r["relevance"] is not None]

    if not valid_faith:
        print("No valid scores produced. Check that the judge model is responding.")
        return

    avg_faith = sum(valid_faith) / len(valid_faith)
    avg_rel = sum(valid_rel) / len(valid_rel)

    print(f"\nQuestions evaluated: {len(results)}")
    print(f"RAG call failures:   {failed}")
    print(f"\nAverage FAITHFULNESS: {avg_faith:.2f} / 5  (n={len(valid_faith)})")
    print(f"Average RELEVANCE:    {avg_rel:.2f} / 5  (n={len(valid_rel)})")

    # Distribution
    print("\nFaithfulness distribution:")
    for score in [5, 4, 3, 2, 1]:
        n = sum(1 for s in valid_faith if s == score)
        bar = "█" * n
        print(f"  {score}: {bar} ({n})")

    print("\nRelevance distribution:")
    for score in [5, 4, 3, 2, 1]:
        n = sum(1 for s in valid_rel if s == score)
        bar = "█" * n
        print(f"  {score}: {bar} ({n})")

    # Per-category breakdown
    by_cat = {}
    for r in results:
        cat = r["category"]
        if cat not in by_cat:
            by_cat[cat] = {"faith": [], "rel": []}
        if r["faithfulness"] is not None:
            by_cat[cat]["faith"].append(r["faithfulness"])
        if r["relevance"] is not None:
            by_cat[cat]["rel"].append(r["relevance"])

    print("\nPer-category averages:")
    for cat in sorted(by_cat):
        f_avg = sum(by_cat[cat]["faith"]) / max(len(by_cat[cat]["faith"]), 1)
        r_avg = sum(by_cat[cat]["rel"]) / max(len(by_cat[cat]["rel"]), 1)
        print(f"  {cat:25s} F={f_avg:.2f}  R={r_avg:.2f}  (n={len(by_cat[cat]['faith'])})")

    print("\nLow-scoring questions (FAITHFULNESS <= 2):")
    bad = [r for r in results if r["faithfulness"] is not None and r["faithfulness"] <= 2]
    if not bad:
        print("  None — no hallucinations detected!")
    else:
        for r in bad:
            print(f"  - [{r['id']}] {r['question']}")
            print(f"    Reasoning: {r['reasoning']}")

    print("\n" + "=" * 60)
    print(f"Full results: {DEFAULT_RESULTS_PATH}")
    print("=" * 60 + "\n")


# ─── CLI ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument(
        "--questions",
        default=DEFAULT_QUESTIONS_PATH,
        help="Path to test_questions.json",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_RESULTS_PATH,
        help="Path to write eval_results.csv",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only run first N questions (for quick smoke tests)",
    )
    args = parser.parse_args()

    start = datetime.now()
    run_eval(args.questions, args.out, args.sample)
    elapsed = (datetime.now() - start).total_seconds()
    print(f"Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
