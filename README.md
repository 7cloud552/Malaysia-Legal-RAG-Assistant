# ⚖️ Malaysian Legal RAG Assistant

A retrieval-augmented Q&A system over Malaysian legislation (Employment Act 1955,
Companies Act 2016, Personal Data Protection Act 2010).

**[Live demo](https://rag-fusion-malaysia-legal-assistant.streamlit.app)** — may take ~30s to open webpage.

## 🛠️ Running locally
1. Create a Postgres database with pgvector (local via Docker, or Supabase)
2. Install libraries and packages: `pip install -r requirements.txt`
3. Set `OPENAI_API_KEY` and `CONNECTION_STRING` in `.env`
4. Run ingestion script to embed the corpus to Postgres database: `python ingest.py`
5. Run the app locally: `streamlit run app.py`

## 🏗️ Architecture
- RAG-Fusion: generates 4 query variations per question, retrieves top-k per
  query, fuses results with reciprocal rank fusion
- LangGraph-orchestrated: query generation → retrieval → fusion → cited answer generation
- pgvector via `langchain-postgres` (`PGVectorStore`/`PGEngine`) for storage
- Answers cite the specific retrieved chunk they're grounded in

## 📚 Corpus
Employment Act 1955 (Act 265) · Companies Act 2016 (Act 777) · PDPA 2010 (Act 709)
— sourced from Malaysia's Federal Legislation Portal (lom.agc.gov.my)

## 📊 Evaluation
30-question hand-written golden set , each grounded in a specific
section citation. Scored on retrieval hit rate, correctness, and groundedness
(LLM-as-judge).

### Results
- **Retrieval hit**: did the correct section actually get retrieved, checked against each answer's cited section marker — a strict text-matching heuristic that can undercount (see Limitations)
- **Correctness**: measures whether the generated answer matched the referenced answer
- **Groundedness**: measures whether the answer was actually supported by the retrieved document

Both correctectness and groundedness are graded by an LLM-as-judge against the reference answer and the retrieved context.

| chunk_size | overlap | retrieval hit | correctness | groundedness |
|---|---|---|---|---|
| 800 | 240 | 79% | 97% | 100% |


## ⚠️ Known limitations
- One question — whether a First Schedule exclusion overrides a general annual-leave provision — isn't reliably resolved: the query-generation step doesn't consistently search exclusion schedules unless the question hints at one. When unresolved, the system hedges rather than asserting a wrong answer.
