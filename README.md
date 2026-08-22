# Insight Analyst

An advanced RAG system for business research and reporting — hybrid search,
reranking, and RAG evaluation. Runs entirely locally on Ollama.

> **Status: Phase 1 (ingestion) complete.** Retrieval, reranking and the answer
> chain are not implemented yet.

## Layout

```
data/documents/     source corpus (PDF, Markdown, text)
data/tabular/       CSVs (Phase 3)
src/config.py       paths, model names, chunk sizes
src/ingestion/      loading, chunking, index building
src/retrieval/      hybrid retrieval + reranking (Phase 2)
notebooks/          experimentation
storage/            built indexes (gitignored, rebuildable)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # optional; defaults work as-is
ollama pull nomic-embed-text
```

## Build the indexes

```bash
python -m src.ingestion.build_index
```

Useful flags: `--preview N` (sample chunks), `--skip-vectors` (chunking/BM25
only, for tuning), `--backend sentence-transformers`.

Chunking alone, for a quick sanity check:

```bash
python -m src.ingestion.loader
```

## Roadmap

- [x] **Phase 1** — ingestion: loading, semantic-aware chunking, Chroma + BM25
- [ ] **Phase 2** — hybrid retrieval (dense + BM25 fusion), cross-encoder reranking
- [ ] **Phase 3** — tabular/CSV analysis
- [ ] **Phase 4** — RAG evaluation harness
- [ ] **Phase 5** — report generation, served via LangServe
