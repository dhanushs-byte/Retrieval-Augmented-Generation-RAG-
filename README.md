# Week 7 – Day 1: Retrieval-Augmented Generation (RAG)

This repository implements the four assessment tasks from the Week 7 Day 1 PDFs.

## Assessment coverage

### Task 1 – Why RAG: See the Problem
- Uses a fictional private leave-policy document.
- Compares a no-context baseline LLM response with a context-grounded response.
- Checks grounded answers against expected facts.
- If no API key is configured, the program clearly marks the baseline as NOT RUN and uses a transparent extractive method for the grounded answer. It does not pretend that an offline response is an LLM hallucination.
- Five lines explain RAG versus fine-tuning.

### Task 2 – Indexing Pipeline
- Loads exactly five supported sample files from `data/` (PDF, DOCX and TXT).
- Parses text and keeps source, page/section and chunk metadata.
- Chunks text, embeds with Sentence Transformers, and stores points in local Qdrant.
- Stable UUIDs plus Qdrant upsert make repeated indexing of unchanged inputs idempotent.
- Prints counts and elapsed time for each stage.

### Task 3 – Query Pipeline
- Embeds the question and retrieves top-k chunks.
- Prints retrieved chunks for debugging.
- Builds the required prompt: `Answer ONLY using the context below. Context: {context} Question: {question}`
- Returns `I don't have information on that` when no chunks are found or the best score is below the configured threshold.
- Uses an LLM when an API key is configured; otherwise uses transparent offline extractive answering.

### Task 4 – Context Token Budget
- Counts system prompt, retrieved context, question and reserved output tokens.
- `fit_context(chunks, budget)` retains rank order and stops when the budget is reached.
- Runs k=2, 5 and 10 against a simulated 4,000-token context.
- Evaluates the generated answer against expected phrases, rather than checking whether the context contains a target number.
- Compares answers with the best retrieved chunk placed first and last.
- Prints a recommended 8k context split.

## Project structure

```text
Week7-Day1/
├── data/
│   ├── company_leave_policy.txt
│   ├── expense_policy.pdf
│   ├── it_support_guide.txt
│   ├── onboarding_guide.docx
│   └── security_basics.pdf
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── document_loader.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── indexing.py
│   ├── llm.py
│   ├── query_pipeline.py
│   ├── context_budget.py
│   └── task1_why_rag.py
├── tests/
│   ├── conftest.py
│   ├── test_chunking.py
│   └── test_context_budget.py
├── diagrams/
│   ├── indexing_pipeline.mmd
│   └── rag_query_flow.mmd
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Requirements and setup (Windows PowerShell)

Python 3.11 or 3.12 is recommended for broad compatibility with the ML dependencies. If you continue with Python 3.14, some dependencies may not yet provide compatible wheels.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

The first embedding run downloads `sentence-transformers/all-MiniLM-L6-v2`; internet access is required for the initial download. The model is cached for later runs.

## Optional OpenRouter LLM configuration

Open `.env` and set:

```env
OPENAI_API_KEY=your_openrouter_api_key
OPENAI_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=meta-llama/llama-3.3-70b-instruct:free
```

Use a model currently available to your OpenRouter account. Never commit `.env` or publish your key. Without a key, indexing, retrieval, fallback, token counting and offline extractive answers still work. However, the true no-context LLM baseline required to observe hallucination in Task 1 cannot be performed without an LLM endpoint.

## Run from the project root

```powershell
python -m src.task1_why_rag
python -m src.indexing
python -m src.query_pipeline
python -m src.context_budget
```

Run the tests:

```powershell
python -m pytest -v
```

## Expected indexing output shape

Exact chunk counts and timings depend on the document text, chunk size and machine:

```text
[LOAD] 5 documents selected ...
[PARSE] 5 docs -> ... pages/sections
[CHUNK] ... chunks created
[EMBED] ... embeddings in ...s
[STORE] upserted ... points in ...s
[DONE] total ...s
```

## Context budget recommendation for an 8k model

| Component | Tokens | Share |
|---|---:|---:|
| System/instructions | 800 | 10% |
| Retrieved context | 4,000 | 50% |
| Question/history | 1,200 | 15% |
| Reserved output | 2,000 | 25% |
| **Total** | **8,000** | **100%** |

This is a starting allocation, not a universal rule. Adjust it after measuring the actual model's prompt format and output needs.

## Submission table

Run `python -m src.context_budget` and copy the measured values into this table:

| k | context tokens | answer correct (Y/N) |
|---:|---:|:---:|
| 2 | Fill from run | Fill from answer evaluation |
| 5 | Fill from run | Fill from answer evaluation |
| 10 | Fill from run | Fill from answer evaluation |

Do not invent results. The answer-quality check currently uses expected phrases for the sample casual-leave question; manually verify the answer against `data/company_leave_policy.txt` and note that this is a simple evaluation, not a comprehensive semantic metric.

## GitHub submission

If this folder is not already a Git repository:

```powershell
git init
git branch -M main
git add .
git status
git commit -m "Complete Week 7 Day 1 RAG assessment"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

If the remote already exists, do not add it again. Check it with:

```powershell
git remote -v
git status
```

For later updates:

```powershell
git add .
git commit -m "Improve Week 7 Day 1 RAG assessment"
git push
```

Before pushing, verify `.env` is not staged:

```powershell
git status
```

The `.gitignore` excludes `.env`, `.venv/`, local Qdrant storage and Python cache files.

## Notes on the experiment
- The five sample documents are fictional training data, not a real company's policy.
- The lost-in-the-middle comparison uses the same question and compares best-chunk-first versus best-chunk-last ordering. Results vary by model and prompt.
- Stable IDs and upsert prevent duplicate points when the same chunk is indexed again. If you change a source document, old chunks whose IDs no longer occur may remain in the local collection; recreate the local Qdrant collection when you need a clean rebuild.
