# Deployment artifacts

The offline pipeline writes `fundamentals.parquet`, `scorecards.parquet`, and `bm25.json` here.
These compact derived files contain no credentials and may be committed for Streamlit deployment.
Review their contents before publication. Raw filings and chunk corpora remain ignored elsewhere.
