# SEC Financial Research Terminal

A source-cited research workspace for retail investors. The project compares fixed-size and semantic chunking over SEC filings, combines dense and BM25 retrieval in Pinecone, reranks candidates, and routes questions across filings, XBRL fundamentals, and delayed market data.

## What it includes

- A versioned September 30, 2026 S&P 500 Top 10 company snapshot
- Three 10-Ks and four 10-Qs per issuer, plus amendments
- Table-aware SEC HTML parsing
- Fixed and semantic chunking experiments
- Nebius embeddings and generation
- Pinecone hybrid search and local cross-encoder reranking
- LangGraph routing and LangSmith tracing
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

For Streamlit Community Cloud, copy the variables from `.env.example` into the deployment's Secrets panel. Do not commit `.streamlit/secrets.toml`.

## Evaluation

`finrag evaluate` compares the `fixed-v1` and `semantic-v1` namespaces using Recall@5/10, MRR@10, and nDCG@10, both before and after reranking. Gold questions live in `data/evaluation/gold_questions.jsonl`.

## Security check

```bash
uv run pytest tests/test_repository_security.py
uv run detect-secrets scan .env.example README.md pyproject.toml streamlit_app.py src pages tests data/manifest data/evaluation
git status --ignored --short
```

If a credential is ever committed, rotate it immediately and remove it from repository history before publishing.
