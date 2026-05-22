# RAG Evaluation

A lightweight evaluation harness for the RAG pipeline. Uses **LLM-as-judge** scoring on a fixed test set of 30 questions to track faithfulness and relevance regressions over time.

## Files

- `test_questions.json` — 30 test questions with expected topics, organized by category (CS2/Valorant comparison, Minecraft basics, gameplay mechanics, etc.)
- `src/eval_rag.py` — eval runner: calls RAG → calls judge → writes CSV + prints summary
- `eval_results.csv` — output file (created on each run)

## How to run

Prerequisites: `GROQ_API_KEY` set, FAISS index built, dependencies installed.

```bash
# Full run (~30 questions, ~2-3 minutes)
python -m src.eval_rag

# Quick smoke test (first 5 questions only)
python -m src.eval_rag --sample 5

# Custom question file
python -m src.eval_rag --questions my_custom_questions.json
```

## What you get

**Per-question CSV** with columns: `id, category, question, answer, faithfulness (1-5), relevance (1-5), reasoning`.

**Console summary**:
- Average faithfulness and relevance scores
- Score distribution (how many 5s, 4s, etc.)
- Per-category breakdown — useful for spotting which document areas RAG handles well or poorly
- Low-scoring questions called out individually

## Scoring rubric

Each answer is judged by a separate Groq LLM call (temperature 0, deterministic) on:

**Faithfulness (1-5):** Is the answer supported by the retrieved chunks?
- 5: Every claim supported
- 1: Entirely fabricated
- An honest "I don't know" when context lacks the info scores 5 — refusing to hallucinate is correct behavior

**Relevance (1-5):** Does the answer actually address the question?
- 5: Directly and completely answers
- 1: Off-topic

## Interpreting results

| Avg faithfulness | What it means |
|---|---|
| 4.5+ | Strong baseline — RAG is grounded in retrieved context |
| 3.5–4.5 | Acceptable but watch for over-extrapolation |
| <3.5 | RAG is hallucinating; tighten chunking, retrieval k, or prompt |

| Avg relevance | What it means |
|---|---|
| 4.5+ | RAG addresses questions directly |
| 3.5–4.5 | Some answers miss key parts of the question |
| <3.5 | Retrieval is missing the right chunks; check chunk size or try reranking |

## Using this in interviews

> "I built a 30-question eval set with LLM-as-judge scoring on a 1–5 rubric for both faithfulness and relevance. My current baseline is faithfulness **X.X/5** and relevance **X.X/5** across categories like comparison, basics, and gameplay. I run it as a regression check whenever I change chunking strategy, embedding model, or retrieval k."

This replaces the older "~40% hallucination reduction" claim with a real methodology and reproducible number. If asked "how do you know your changes improved things?", point to before/after eval CSVs.

## What this is NOT

It's not the full RAGAS framework. RAGAS requires ground-truth answers per question to compute context precision and context recall. That's a lot of manual labeling. This LLM-as-judge approach is the lightest defensible methodology — industry uses it widely (MT-Bench, Arena-Hard, OpenAI evals).

If you want to upgrade to full RAGAS later:
- Add `ground_truth` field to each question in `test_questions.json`
- Install `ragas` package
- Replace the judge call with `from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall`

## Adding more questions

Edit `test_questions.json` and append new items to the `questions` array. Schema:

```json
{
  "id": "unique_string",
  "question": "natural language question",
  "expected_topics": ["keywords", "the answer should mention"],
  "category": "grouping_label"
}
```

Aim for ~10 questions per major document or topic. Mix easy questions (single-fact) with harder ones (cross-document, multi-step).

## Tips for honest results

- **Don't curate questions to make scores look good.** Include questions you know your RAG will struggle with — that's how you find what to fix.
- **Re-run after every meaningful change** — chunking, retrieval k, embedding model swap, prompt edit. Save the CSV with a date so you have a trail of progress.
- **Watch for judge bias.** The judge LLM is the same family as the answerer; it may be lenient on its own kind. If scores look suspiciously high, manually spot-check 5 random answers.
