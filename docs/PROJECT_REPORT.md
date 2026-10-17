# Project Report: SEC Financial Research Terminal

## Project overview

Document the retail-investor problem, the ten-company snapshot, application workflow, and
research-only boundary.

## Dataset

Record the ingestion date, issuer manifest, selected accessions, amendments, excluded documents,
and SEC/XBRL coverage gaps.

## Prompts and graph instructions

Include the grounded-generation instruction, refusal behavior, routes, and citation format.

## Chunking comparison

Insert fixed-versus-semantic Recall@5/10, MRR@10, nDCG@10, latency, and index-size results from
`finrag evaluate`.

## Reranking impact

Report aggregate and per-category metrics before and after reranking, including regressions.

## Answer-quality review

Score faithfulness, citation correctness, numeric accuracy, and refusal correctness against the
30-question set.

## Iterations and learnings

Explain parser changes, chunking decisions, retrieval failures, table behavior, unavailable XBRL
concepts, and cross-sector scorecard limitations.

## Reproduction

Link the public repository, Streamlit deployment, demo recording, model identifiers, snapshot date,
and setup instructions.
