# Deployment artifacts

The offline pipeline writes `fundamentals.parquet`, `filings.parquet`, `scorecards.parquet`,
and namespace-specific `bm25_fixed.json` / `bm25_semantic.json` here. The legacy `bm25.json`
remains available as a fallback. Run `finrag build-structured` to refresh the Parquet artifacts
from cached SEC data without recomputing chunks or Pinecone vectors.
These compact derived files contain no credentials and may be committed for Streamlit deployment.
Review their contents before publication. Raw filings and chunk corpora remain ignored elsewhere.
