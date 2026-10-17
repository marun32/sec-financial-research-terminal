import json

import pandas as pd
import streamlit as st

from sec_fin_rag.config import get_settings
from sec_fin_rag.ui import page_setup

page_setup("Methodology and Evaluation")
st.markdown(
    """
### Retrieval experiment

- **Fixed:** 512-token chunks with 64-token overlap.
- **Semantic:** embedding-distance boundaries at the 90th percentile, bounded to 200–1,000 tokens.
- **Hybrid:** Nebius dense embeddings plus corpus-fitted BM25 sparse vectors in Pinecone.
- **Reranking:** top 20 candidates reranked locally; top eight are supplied to generation.

The same documents, embedding model, query set, and retrieval parameters are used for
both strategies.
"""
)
report_path = get_settings().reports_dir / "retrieval_evaluation.json"
if not report_path.exists():
    st.info("Run `uv run finrag evaluate` after indexing to populate the comparison report.")
else:
    report = json.loads(report_path.read_text())
    rows = []
    for strategy, payload in report["strategies"].items():
        for stage in ("before_rerank", "after_rerank"):
            rows.append({"strategy": strategy, "stage": stage, **payload[stage]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.subheader("Acceptance targets")
st.markdown(
    "- Recall@10 ≥ 80%\n- Citation correctness and groundedness ≥ 90%\n"
    "- Correct refusal behavior ≥ 80%\n"
    "- Median chat latency below eight seconds, excluding cold starts"
)
