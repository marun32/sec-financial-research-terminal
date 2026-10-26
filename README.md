# Fin AI — SEC Intelligence

A source-cited research workspace for retail investors. The project compares fixed-size and semantic chunking over SEC filings, combines dense and BM25 retrieval in Pinecone, reranks candidates, and routes questions across filings, XBRL fundamentals, and delayed market data.

## Documentation

- [Complete project documentation](docs/PROJECT_DOCUMENTATION.md) — assignment rubric, product
  scope, architecture, pipeline, implementation details, evaluation, iterations, limitations,
  security, operations, and deployment.

## What it includes

- A versioned September 30, 2026 S&P 500 Top 10 company snapshot
- Three latest 10-Ks and four latest 10-Qs filed on or before the snapshot date,
  plus amendments associated with those selected reporting periods
- Table-aware SEC HTML parsing
- Fixed and semantic chunking experiments
- Nebius embeddings and generation
- Pinecone hybrid search and local cross-encoder reranking
- LangChain embedding/retriever adapters and LCEL answer generation
- Nebius question planning and LangGraph routing with optional LangSmith tracing/export
- Live, collapsible Ask Fin AI progress showing planning, evidence routing, actual source collection,
  and answer/citation checks
- Five-page Streamlit interface
- A 30-question evaluation set and retrieval metrics

This application is for research and education. It does not provide personalized investment advice. Market data supplied by `yfinance` may be delayed.

## Setup

```bash
cp .env.example .env
uv sync --extra dev
uv run finrag preflight
```

Fill `.env` locally. It is intentionally ignored by Git. Never put real secrets in `.env.example`.

## Pipeline

```bash
uv run finrag download
uv run finrag parse
uv run finrag build-artifacts
uv run finrag index
uv run finrag evaluate
```

If only Company Facts normalization, filing metadata, or scorecards change, rebuild their compact
artifacts without recomputing chunks or Pinecone vectors:

```bash
uv run finrag build-structured
```

Or run the offline stages together:

```bash
uv run finrag all
```

The commands are idempotent. Downloads are cached under `data/raw/`, chunk corpora under
`data/processed/`, and reports under `reports/generated/`; all are ignored by Git. Compact,
non-secret deployment artifacts are written to `data/app/` and may be committed after review.

## Run the application

```bash
uv run streamlit run streamlit_app.py
```

Ask Fin AI follows `plan → route → evidence → answer`. The planning model identifies issuers,
metrics, periods, and intent; validated LangGraph routing then reads SEC XBRL facts, searches and
reranks Pinecone filing passages, checks delayed market context, or applies the unsupported-answer
guardrail as appropriate. The assistant generates from the evidence and validates source markers.
Its progress panel updates when each graph node actually finishes, collapses after completion,
and remains expandable in that chat answer. A numeric-only multi-year question can use XBRL
without searching Pinecone; the panel reflects the path that ran.

For Streamlit Community Cloud, copy the variables from `.env.example` into the deployment's Secrets panel. Do not commit `.streamlit/secrets.toml`.

## Evaluation

`finrag evaluate` compares the `fixed-v1` and `semantic-v1` namespaces using all-evidence Hit@5/10,
evidence Recall@5/10, MRR@10, and normalized nDCG@10, both before and after reranking. It reports
category breakdowns and reranking regressions. Gold questions and their topical evidence anchors
live in `data/evaluation/gold_questions.jsonl`. This command performs retrieval evaluation only;
it does not generate answers or invoke an LLM judge.

Generation evaluation is resumable and adds an LLM judge for faithfulness, answer relevancy,
citation correctness, and citation completeness. Citation-marker integrity and refusal correctness
remain deterministic checks:

```bash
uv run finrag evaluate-answers
```

LangSmith is optional and is not required by the project rubric, application, preflight, or the
two local evaluation commands. If intentionally enabled, the following command synchronizes the
same gold set and records the completed generation metrics as a named LangSmith experiment:

```bash
uv run finrag evaluate-langsmith
```

The LangSmith command reuses completed `generation-v5` answer checkpoints and only regenerates
missing cases, avoiding duplicate model-judge spend.

## Security check

```bash
uv run pytest tests/test_repository_security.py
git ls-files -z --cached --others --exclude-standard | xargs -0 uv run detect-secrets-hook --no-verify
git status --ignored --short
```

If a credential is ever committed, rotate it immediately and remove it from repository history before publishing.
