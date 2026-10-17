# Deployment

## GitHub

1. Run the full local checks documented in the README.
2. Confirm `git status --ignored --short` lists `.env` and `.streamlit/secrets.toml` as ignored.
3. Create an empty public GitHub repository and add it as `origin`.
4. Push the `main` branch only after the CI workflow passes locally.

Never use `git add -f` for an ignored credential or data file. If a secret enters history,
rotate it immediately and remove it from all published history before continuing.

## Streamlit Community Cloud

1. Run the full ingestion and indexing pipeline locally.
2. Commit the compact files produced under `data/app/`; do not commit `data/raw/` or
   `data/processed/`.
3. Connect the public GitHub repository in Streamlit Community Cloud.
4. Select `streamlit_app.py` as the entry point.
5. Paste the required variables from `.env.example` into Streamlit's Secrets panel.
6. Deploy and verify every page plus a cited chat response.

The hosted application performs queries only. It does not download filings or rebuild Pinecone
indexes at startup.
