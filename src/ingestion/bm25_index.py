"""BM25 keyword index over the same chunks that go into the vector store.

The vector index is good at paraphrase and concept matching but weak on exact
tokens — ticker symbols, product SKUs, section numbers, surnames. BM25 covers
exactly that gap, which is why the hybrid retriever in Phase 2 will query both
and fuse the rankings.

Persisted with pickle: the index is a derived artefact rebuilt from the corpus
on demand, so portability of the format is not a concern.
"""

from __future__ import annotations

import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config

# Keep digits and intra-word apostrophes/hyphens together so "Q3", "2024",
# "year-over-year" and "O'Brien" survive as single searchable tokens.
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")

# Trailing possessive only: "apple's" and "employees'" have a stem worth
# indexing, but "o'brien" has no trailing "'s" and is left alone.
_POSSESSIVE_RE = re.compile(r"(?<=.)'s?$")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer used for both indexing and querying.

    Phase 2 must call this same function on incoming queries — a BM25 index is
    only meaningful if both sides are tokenized identically.

    A possessive is emitted as both forms: "Apple's" yields ["apple's",
    "apple"]. Without this, the possessive is a token that appears nowhere in
    the corpus, so the most discriminative word in a query like "what was
    Apple's revenue" contributes nothing and BM25 ranks on the filler terms
    instead — observed returning NVIDIA filings for an Apple question.

    Trade-off: emitting both forms slightly inflates document length and term
    counts for BM25's normalisation, which is preferable to stripping every
    apostrophe and merging distinct tokens like "o'brien" into "obrien".
    """
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        token = raw.lower()
        tokens.append(token)
        stem = _POSSESSIVE_RE.sub("", token)
        if stem and stem != token:
            tokens.append(stem)
    return tokens


@dataclass
class BM25Bundle:
    """Everything needed to score a query and map hits back to chunks."""

    bm25: object                      # rank_bm25.BM25Okapi
    texts: list[str]
    metadatas: list[dict]

    def __len__(self) -> int:
        return len(self.texts)

    def to_documents(self, indices) -> list[Document]:
        """Rehydrate Documents for a list of corpus positions."""
        return [
            Document(page_content=self.texts[i], metadata=self.metadatas[i])
            for i in indices
        ]


def build_bm25_index(
    chunks: list[Document],
    path: Path | None = None,
) -> BM25Bundle:
    """Build a BM25Okapi index over `chunks` and pickle it to `path`.

    Chunk order is preserved, so a BM25 hit at position i refers to chunks[i] —
    that positional link is what lets the hybrid retriever fuse BM25 ranks with
    vector hits.
    """
    if not chunks:
        raise ValueError("No chunks to index.")

    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:  # pragma: no cover
        raise ImportError("rank_bm25 is not installed. Run `pip install rank_bm25`.") from exc

    path = path or config.BM25_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    texts = [c.page_content for c in chunks]
    metadatas = [dict(c.metadata) for c in chunks]
    corpus = [tokenize(t) for t in texts]

    bundle = BM25Bundle(bm25=BM25Okapi(corpus), texts=texts, metadatas=metadatas)

    with path.open("wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return bundle


def load_bm25_index(path: Path | None = None) -> BM25Bundle:
    """Load a previously persisted BM25 index."""
    path = path or config.BM25_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No BM25 index at {path}. Run src/ingestion/build_index.py first."
        )
    with path.open("rb") as fh:
        return pickle.load(fh)
