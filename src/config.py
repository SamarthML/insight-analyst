"""Central configuration for the Insight Analyst pipeline.

Every path and model name lives here so the ingestion, retrieval and evaluation
phases all agree on where the corpus and the indexes are. Anything that might
reasonably differ per machine is overridable via the environment (see
.env.example).
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def enable_utf8_stdout() -> None:
    """Make stdout UTF-8 safe.

    The Windows console defaults to cp1252, which raises or mangles when a
    chunk preview contains an em-dash, curly quote or ellipsis — i.e. most real
    business documents. Called from the CLI entrypoints only, never on import.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001 - cosmetic only, never fatal
                pass

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
TABULAR_DIR = DATA_DIR / "tabular"

STORAGE_DIR = PROJECT_ROOT / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
BM25_PATH = STORAGE_DIR / "bm25.pkl"

# --- Chunking --------------------------------------------------------------
# Measured in tokens, not characters. See loader.py for why these numbers.
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "800"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "100"))

# --- Embeddings ------------------------------------------------------------
# "ollama" uses a model you have pulled locally; "sentence-transformers"
# downloads a small model from HuggingFace once and then runs offline.
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "ollama")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

SENTENCE_TRANSFORMERS_MODEL = os.getenv(
    "SENTENCE_TRANSFORMERS_MODEL", "BAAI/bge-small-en-v1.5"
)

# --- Vector store ----------------------------------------------------------
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "insight_analyst")

# Chroma's underlying client rejects oversized single inserts; we batch below
# its ceiling rather than relying on the default.
CHROMA_MAX_BATCH = int(os.getenv("CHROMA_MAX_BATCH", "4000"))

# Sentinel used for the `page` metadata field on formats that have no pages
# (markdown, plain text). Chroma only accepts scalar metadata values, so we
# cannot store None here.
NO_PAGE = -1


def describe() -> str:
    """Human-readable summary of the active configuration."""
    if EMBEDDING_BACKEND == "ollama":
        embed = f"ollama:{OLLAMA_EMBED_MODEL} @ {OLLAMA_BASE_URL}"
    else:
        embed = f"sentence-transformers:{SENTENCE_TRANSFORMERS_MODEL}"
    return (
        f"documents : {DOCUMENTS_DIR}\n"
        f"chroma    : {CHROMA_DIR} (collection={CHROMA_COLLECTION})\n"
        f"bm25      : {BM25_PATH}\n"
        f"chunking  : {CHUNK_SIZE_TOKENS} tokens / {CHUNK_OVERLAP_TOKENS} overlap\n"
        f"embeddings: {embed}"
    )
