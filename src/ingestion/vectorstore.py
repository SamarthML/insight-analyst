"""Chroma vector index — build, persist and reload.

Embeddings run locally: either through Ollama (default, uses a model you have
already pulled) or sentence-transformers. Neither needs an API key.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from langchain_core.documents import Document

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config


# --- Embeddings ------------------------------------------------------------
def get_embeddings(backend: str | None = None):
    """Return a LangChain Embeddings object for the configured local backend."""
    backend = (backend or config.EMBEDDING_BACKEND).lower()

    if backend == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "langchain-ollama is not installed. Either `pip install "
                "langchain-ollama` or set EMBEDDING_BACKEND=sentence-transformers."
            ) from exc
        return OllamaEmbeddings(
            model=config.OLLAMA_EMBED_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )

    if backend in {"sentence-transformers", "sentence_transformers", "huggingface"}:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=config.SENTENCE_TRANSFORMERS_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )

    raise ValueError(
        f"Unknown EMBEDDING_BACKEND {backend!r}; "
        "expected 'ollama' or 'sentence-transformers'."
    )


def _chroma_cls():
    try:
        from langchain_chroma import Chroma
    except ImportError as exc:  # pragma: no cover
        raise ImportError("langchain-chroma is not installed. Run `pip install langchain-chroma`.") from exc
    return Chroma


# --- Build / load ----------------------------------------------------------
def build_vectorstore(
    chunks: list[Document],
    persist_dir: Path | None = None,
    collection: str | None = None,
    embeddings=None,
    overwrite: bool = True,
):
    """Embed `chunks` and persist them to a Chroma collection on disk.

    With overwrite=True (the default) any existing collection directory is
    removed first, so rebuilding after a corpus change does not leave stale
    chunks behind — silently accumulating duplicates across rebuilds is a
    nasty source of bogus retrieval results.
    """
    if not chunks:
        raise ValueError("No chunks to index.")

    persist_dir = persist_dir or config.CHROMA_DIR
    collection = collection or config.CHROMA_COLLECTION
    embeddings = embeddings or get_embeddings()
    Chroma = _chroma_cls()

    if overwrite and persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    store = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    # Chroma's client caps how many records one add_documents call may carry.
    # Batching also gives useful progress output on a slow local embedder.
    batch = config.CHROMA_MAX_BATCH
    for start in range(0, len(chunks), batch):
        window = chunks[start:start + batch]
        store.add_documents(window)
        print(f"  embedded {min(start + batch, len(chunks))}/{len(chunks)} chunks")

    return store


def load_vectorstore(
    persist_dir: Path | None = None,
    collection: str | None = None,
    embeddings=None,
):
    """Reopen a previously persisted Chroma collection."""
    persist_dir = persist_dir or config.CHROMA_DIR
    collection = collection or config.CHROMA_COLLECTION

    if not persist_dir.exists():
        raise FileNotFoundError(
            f"No Chroma index at {persist_dir}. Run src/ingestion/build_index.py first."
        )

    Chroma = _chroma_cls()
    return Chroma(
        collection_name=collection,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(persist_dir),
    )


def count_vectors(store) -> int:
    """Number of vectors currently in the collection."""
    return store._collection.count()
