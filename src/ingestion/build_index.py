"""End-to-end ingestion: load -> chunk -> vector index -> BM25 index -> stats.

    python -m src.ingestion.build_index
    python -m src.ingestion.build_index --preview 5
    python -m src.ingestion.build_index --backend sentence-transformers
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config
from src.ingestion import bm25_index, loader, vectorstore


def _fmt(seconds: float) -> str:
    return f"{seconds:.1f}s" if seconds < 60 else f"{seconds // 60:.0f}m {seconds % 60:.0f}s"


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Insight Analyst indexes.")
    parser.add_argument("--preview", type=int, default=3,
                        help="how many sample chunks to print (0 to skip)")
    parser.add_argument("--backend", default=None,
                        help="override EMBEDDING_BACKEND (ollama | sentence-transformers)")
    parser.add_argument("--skip-vectors", action="store_true",
                        help="chunk and build BM25 only; useful when tuning chunking")
    args = parser.parse_args(argv)

    config.enable_utf8_stdout()

    _rule("Configuration")
    print(config.describe())

    # --- Load ---------------------------------------------------------------
    _rule("Loading documents")
    t0 = time.perf_counter()
    documents = loader.load_documents()
    load_time = time.perf_counter() - t0

    if not documents:
        print(f"\nNo supported documents found in {config.DOCUMENTS_DIR}")
        print("Add .pdf, .md or .txt files there and rerun.")
        return 1

    sources = sorted({d.metadata["source"] for d in documents})
    print(f"{len(sources)} file(s), {len(documents)} page-level documents in {_fmt(load_time)}")
    for name in sources:
        pages = sum(1 for d in documents if d.metadata["source"] == name)
        print(f"  - {name} ({pages} page{'s' if pages != 1 else ''})")

    # --- Chunk --------------------------------------------------------------
    _rule("Chunking")
    t0 = time.perf_counter()
    chunks = loader.chunk_documents(documents)
    chunk_time = time.perf_counter() - t0

    counter = loader.build_token_counter()
    token_lengths = [counter(c.page_content) for c in chunks]
    char_lengths = [len(c.page_content) for c in chunks]

    print(f"{len(chunks)} chunks in {_fmt(chunk_time)}")
    print(f"  tokens/chunk : avg {statistics.mean(token_lengths):.0f} | "
          f"median {statistics.median(token_lengths):.0f} | "
          f"min {min(token_lengths)} | max {max(token_lengths)}")
    print(f"  chars/chunk  : avg {statistics.mean(char_lengths):.0f} | "
          f"min {min(char_lengths)} | max {max(char_lengths)}")

    if args.preview:
        _rule(f"Sample chunks ({args.preview})")
        loader.preview_chunks(chunks, n=args.preview)

    # --- Vector index -------------------------------------------------------
    vector_time = 0.0
    vector_count = 0
    if args.skip_vectors:
        _rule("Vector index")
        print("skipped (--skip-vectors)")
    else:
        _rule("Vector index (Chroma)")
        t0 = time.perf_counter()
        try:
            embeddings = vectorstore.get_embeddings(args.backend)
            store = vectorstore.build_vectorstore(chunks, embeddings=embeddings)
            vector_count = vectorstore.count_vectors(store)
        except Exception as exc:  # noqa: BLE001 - report clearly instead of a traceback
            print(f"FAILED: {type(exc).__name__}: {exc}")
            if "ollama" in str(exc).lower() or "connect" in str(exc).lower():
                print("\nIs Ollama running? Try `ollama serve`, and confirm the model "
                      f"is pulled with `ollama pull {config.OLLAMA_EMBED_MODEL}`.")
            return 2
        vector_time = time.perf_counter() - t0
        print(f"{vector_count} vectors persisted to {config.CHROMA_DIR} in {_fmt(vector_time)}")

    # --- BM25 index ---------------------------------------------------------
    _rule("BM25 index (rank_bm25)")
    t0 = time.perf_counter()
    bundle = bm25_index.build_bm25_index(chunks)
    bm25_time = time.perf_counter() - t0
    size_kb = config.BM25_PATH.stat().st_size / 1024
    print(f"{len(bundle)} documents indexed in {_fmt(bm25_time)} "
          f"({size_kb:.0f} KB at {config.BM25_PATH})")

    # --- Summary ------------------------------------------------------------
    total = load_time + chunk_time + vector_time + bm25_time
    _rule("Summary")
    print(f"  source files      : {len(sources)}")
    print(f"  page documents    : {len(documents)}")
    print(f"  chunks            : {len(chunks)}")
    print(f"  avg chunk length  : {statistics.mean(token_lengths):.0f} tokens "
          f"({statistics.mean(char_lengths):.0f} chars)")
    print(f"  vectors           : {vector_count}")
    print(f"  bm25 documents    : {len(bundle)}")
    print(f"  total build time  : {_fmt(total)}")
    print("\nIndexes are ready. Phase 2 (hybrid retrieval + reranking) goes in src/retrieval/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
