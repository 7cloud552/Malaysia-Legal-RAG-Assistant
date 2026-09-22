# Malaysian Legal RAG Assistant

A retrieval-augmented Q&A system over Malaysian legislation (Employment Act 1955,
Companies Act 2016, Personal Data Protection Act 2010), built to practice evaluating
RAG systems rigorously, not just building them.

**[Live demo](your-streamlit-url)** — may take ~30s to wake if idle.

## Architecture
- RAG-Fusion: generates 4 query variations per question, retrieves top-k per
  query, fuses results with reciprocal rank fusion
- LangGraph-orchestrated: query generation → retrieval → fusion → cited answer generation
- pgvector via `langchain-postgres` (`PGVectorStore`/`PGEngine`) for storage
- Answers cite the specific retrieved chunk they're grounded in

## Corpus
Employment Act 1955 (Act 265) · Companies Act 2016 (Act 777) · PDPA 2010 (Act 709)
— sourced from Malaysia's Federal Legislation Portal (lom.agc.gov.my)

## Evaluation
[N]-question hand-written golden set , each grounded in a specific
section citation. Scored on retrieval hit rate, correctness, and groundedness
(LLM-as-judge).

### Results
| chunk_size | overlap | retrieval hit | correctness | groundedness |
|---|---|---|---|---|
| 800 | 240 | 79% | 97% | 100% |


## Known limitations
- One question — whether a First Schedule exclusion overrides a general annual-leave provision — isn't reliably resolved: the query-generation step doesn't consistently search exclusion schedules unless the question hints at one. When unresolved, the system hedges rather than asserting a wrong answer.
