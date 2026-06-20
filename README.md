# DeepFake Research Assistant — Production RAG System

A production-grade question answering system built specifically for deepfake detection research papers. Ask any question about the papers you've indexed and get a precise, cited answer — grounded only in what the papers actually say, never in the model's prior knowledge.

---

## What Problem This Solves

Academic papers are dense. Finding specific architectural details, loss functions, dataset statistics, or methodology comparisons across multiple papers takes time — you have to read, cross-reference, and manually trace which paper said what.

This system lets you ask questions in plain English and get answers that:
- Come directly from the paper text, not from the model's training data
- Tell you exactly which paper and section the answer came from
- Refuse to answer when the information isn't in the papers, rather than making something up
- Work across multiple papers simultaneously

---

## What Makes This Different From Just Using ChatGPT

When you ask ChatGPT about a research paper, it answers from its training data — which may be outdated, incomplete, or simply wrong for niche academic work. It has no way to tell you which page or section its answer came from, and it will confidently hallucinate details it doesn't know.

This system only knows what is in the papers you give it. Every claim in every answer is backed by a citation pointing to a real chunk of retrieved text. If the answer isn't in the papers, it says so.

---

## How It Works — In Plain English

When you ask a question, the system runs four stages before generating any answer:

**Stage 1 — Understanding your question better**
Your question gets expanded into three variations — the original, a conceptual rephrasing, and a keyword-focused version. This increases the chance of finding all relevant content even when the paper uses different terminology than you did.

**Stage 2 — Searching the papers**
Both a semantic search (finds meaning) and a keyword search (finds exact terms) run simultaneously across all indexed papers. This is important because semantic search finds conceptually related content while keyword search finds exact model names, numbers, and acronyms that semantic search might miss.

**Stage 3 — Finding the best evidence**
Results from both searches are merged and ranked. Sections of a paper that appear repeatedly across multiple search results get a relevance boost — if three different searches all point to Section 4, that section is genuinely relevant, not a coincidence. The top candidates then go through a deeper reading pass that scores each one against your original question.

**Stage 4 — Generating and checking the answer**
An LLM generates an answer using only the retrieved text. A separate critic model then scores whether the answer is actually supported by what was retrieved. If the score is too low, the system re-searches with a broader query and tries again. The answer only goes back to you once it passes quality checks.

---

## Quality Safeguards Built In

**Before your question reaches the pipeline:**
- Queries that are too vague get flagged
- Prompt injection attempts are blocked
- Off-topic questions (unrelated to research papers) are caught early

**After an answer is generated:**
- If the answer contains claims not supported by the retrieved text, it fails the faithfulness check and triggers a re-search
- If the LLM couldn't find enough information and says so, that triggers a re-search with broader terms
- If the answer contains no citations, it is blocked — every factual claim must be traceable
- All of this is logged automatically so you can see exactly what happened for any query

---

## Evaluation — How You Know It's Working

This system doesn't just run and hope for the best. It measures itself:

**Retrieval quality** — A hand-written set of questions with known correct answers tests whether the right papers and sections are being retrieved. You can see exactly which questions the system gets right and which it misses.

**Answer faithfulness** — An LLM judge reads the retrieved context and the generated answer together and scores how well the answer is grounded. This score is stored for every query.

**Answer relevance** — Embedding similarity between the answer and your question measures whether the answer actually addresses what you asked.

All scores are stored in a database and visible in a terminal dashboard that updates in real time.

---

## Setup

**Requirements**
- Python 3.11+
- A Groq API key (free tier works — get one at console.groq.com)

**Install**
```bash
git clone <repo>
cd Production-Rag/backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

**Configure**

Create a `.env` file in `backend/`:
```
GROQ_API_KEY=your_key_here
```

**Add your papers**

Drop PDF files into `backend/raw_data/`. The system is built around deepfake detection papers but works with any academic PDFs.

**Index the papers** (run once per new paper batch)
```bash
python -m ingestion.indexer
```

**Start the API**
```bash
uvicorn main:app --reload
```

---

## Asking Questions

Send a POST request to `http://localhost:8000/api/v1/generate`:

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "What loss function is used for contrastive learning?"}'
```

The response includes the answer, which paper and section each claim came from, how many re-search iterations were needed, and quality scores from the critic.

You can also use the Swagger UI at `http://localhost:8000/docs` to test queries interactively.

---

## Monitoring and Evaluation

**Live dashboard** — shows query history, quality scores, and guardrail pass rates:
```bash
python -m monitoring.dashboard
```

**Faithfulness evaluation** — scores stored queries against retrieved context using an LLM judge:
```bash
python -m evaluation.ragas_eval
```

**Golden dataset evaluation** — tests the system against hand-written questions with known answers:
```bash
python -m tests.golden_eval
```

---

## Project Structure

```
backend/
├── raw_data/           put your PDF files here
├── ingestion/          converts PDFs into searchable indexes
├── retrieval/          finds relevant content for each query
├── generation/         produces cited answers from retrieved content
├── gaurdrails/         blocks bad inputs and low-quality outputs
├── monitoring/         logs every query and shows a live dashboard
├── evaluation/         measures answer quality offline
├── tests/              golden question set and evaluation runner
├── api/                FastAPI endpoints
└── core/               shared configuration
```

---

## Limitations

**Mathematical equations rendered as images** — Some papers embed equations as images rather than text. The PDF parser cannot extract these. The answer will still be generated from surrounding text but the specific formula won't be in the retrieved content.

**Cross-paper comparison questions** — Questions that require synthesizing information from two different papers in a single answer work reasonably well but are harder than single-paper questions. The system retrieves from all indexed papers simultaneously and ensures at least one chunk from each paper appears in the context, but highly specific cross-paper comparisons may return incomplete answers.

**No internet access** — The system only knows what is in the papers you've indexed. It cannot fetch new papers, check for updates, or access any external information at query time.

---

## Built With

- **FastAPI** — API layer
- **ChromaDB** — semantic vector search
- **BM25** — keyword search
- **BAAI/bge-small-en-v1.5** — embedding model
- **BAAI/bge-reranker-base** — cross-encoder reranker
- **Groq / Llama 3** — query rewriting, answer generation, and critique
- **pymupdf4llm** — PDF parsing with math preservation
- **SQLite** — query logging and monitoring

---

## Author

Built as a research tool for the deepfake detection domain. Designed to be extended to any academic domain by swapping the paper collection in `raw_data/`.
