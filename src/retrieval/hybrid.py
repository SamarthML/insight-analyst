"""Hybrid retrieval: dense vector search + BM25, fused with Reciprocal Rank Fusion.

The two retrievers fail in opposite directions. Dense search matches paraphrase
("supply chain disruption" against "difficulties securing components") but is
weak on literal tokens, because a fiscal year or a ticker contributes almost
nothing to a 384-dimensional sentence embedding. BM25 is the mirror image:
exact on "2023" and "NVDA", blind to anything worded differently.

Reciprocal Rank Fusion combines them without needing their scores to be
comparable, which matters here because Chroma returns a cosine *distance* and
BM25 returns an unbounded relevance score -- there is no principled way to add
those directly. RRF discards magnitudes and keeps only positions:

    score(chunk) = sum over lists of  1 / (rank_constant + rank)

with rank 1-based. A chunk ranked well by both retrievers accumulates from both
terms and outranks a chunk that only one retriever liked, which is exactly the
behaviour we want from a hybrid.

    python -m src.retrieval.hybrid "what was Apple revenue in fiscal 2023"
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config
from src.ingestion import bm25_index, vectorstore
from src.retrieval import metadata as MD

# Cormack et al. (2009), "Reciprocal Rank Fusion outperforms Condorcet and
# individual Rank Learning Methods". The constant damps the influence of the
# very top ranks so a single retriever cannot dominate the fusion on its own.
DEFAULT_RANK_CONSTANT = 60

# How deep to go in each retriever before fusing. Fusing only the final top-k
# would defeat the point: a chunk ranked 8th by both retrievers is a strong
# consensus candidate that neither top-5 would ever surface.
DEFAULT_FETCH_K = 20


ChunkKey = tuple[str, int, int]


def chunk_key(doc: Document) -> ChunkKey:
    """Stable identity for a chunk, used to match it across retrievers.

    Verified unique across all chunks in the corpus: the vector store and the
    BM25 index hold the same chunks but return different object instances, so
    fusion needs a value key rather than object identity.
    """
    meta = doc.metadata
    return (
        str(meta.get("source", "?")),
        int(meta.get("page", config.NO_PAGE)),
        int(meta.get("chunk_index", -1)),
    )


@dataclass
class Retrievers:
    """The two loaded indexes, kept together so callers load them once."""

    store: object
    bm25: bm25_index.BM25Bundle

    def __len__(self) -> int:
        return len(self.bm25)


_CACHE: Retrievers | None = None


def load_retrievers(force: bool = False) -> Retrievers:
    """Open the existing Chroma and BM25 indexes. Never rebuilds them.

    Cached at module level because loading the embedding model costs a few
    seconds and a comparison run queries the same indexes repeatedly.
    """
    global _CACHE
    if _CACHE is None or force:
        _CACHE = Retrievers(
            store=vectorstore.load_vectorstore(),
            bm25=bm25_index.load_bm25_index(),
        )
    return _CACHE


# --- Individual retrievers -------------------------------------------------
def vector_search(
    query: str, k: int, retrievers: Retrievers | None = None,
    spec: MD.QuerySpec | None = None,
) -> list[tuple[Document, float]]:
    """Dense top-k. The float is Chroma's distance, so lower is closer.

    When `spec` names a year, the edition filter is pushed down into Chroma as
    a `where` clause, so wrong-edition chunks are excluded before the nearest
    neighbour search rather than dropped from its output.
    """
    retrievers = retrievers or load_retrievers()
    where = MD.chroma_where(spec) if spec else None
    if where:
        hits = retrievers.store.similarity_search_with_score(query, k=k, filter=where)
        if hits:
            return hits
        # An over-narrow filter that removes everything is worse than none.
        # Falling back keeps the failure mode "unfiltered", never "empty".
    return retrievers.store.similarity_search_with_score(query, k=k)


def bm25_search(
    query: str, k: int, retrievers: Retrievers | None = None,
    spec: MD.QuerySpec | None = None,
) -> list[tuple[Document, float]]:
    """BM25 top-k, scored highest-first.

    Tokenized with the indexer's own `tokenize`, since a BM25 index is only
    meaningful when both sides are tokenized identically. Zero-scoring chunks
    share no query term at all and are dropped rather than padded into the
    list, where they would occupy fusion ranks they have not earned.
    """
    retrievers = retrievers or load_retrievers()
    bundle = retrievers.bm25
    scores = bundle.bm25.get_scores(bm25_index.tokenize(query))

    # Restrict the candidate pool to matching editions before selecting the
    # top k, so a wrong-year chunk cannot occupy a slot it would then have to
    # be re-ranked out of.
    candidates = range(len(scores))
    if spec and (spec.has_year or spec.entity):
        allowed = [i for i in candidates
                   if MD.chunk_matches(bundle.metadatas[i], spec)]
        if allowed:
            candidates = allowed

    ranked = sorted(candidates, key=lambda i: scores[i], reverse=True)
    hits = [(i, float(scores[i])) for i in ranked[:k] if scores[i] > 0]
    return [
        (Document(page_content=bundle.texts[i], metadata=dict(bundle.metadatas[i])), s)
        for i, s in hits
    ]


def search_both(
    query: str,
    k: int = DEFAULT_FETCH_K,
    retrievers: Retrievers | None = None,
    year_filter: bool = True,
) -> tuple[list[tuple[Document, float]], list[tuple[Document, float]]]:
    """Run both retrievers independently and return their separate ranked lists.

    The edition filter is applied inside each retriever, before ranking and
    therefore before fusion. Filtering after fusion would be too late: the
    wrong-edition chunks would already have consumed the candidate slots.
    """
    retrievers = retrievers or load_retrievers()
    spec = MD.parse_query(query) if year_filter else None
    if spec is not None and not (spec.has_year or spec.entity):
        spec = None                      # nothing to filter on: behaviour unchanged
    return (
        vector_search(query, k, retrievers, spec),
        bm25_search(query, k, retrievers, spec),
    )


# --- Fusion ----------------------------------------------------------------
@dataclass
class FusedHit:
    """One chunk in the fused ranking, with where each retriever placed it."""

    document: Document
    score: float
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def sources(self) -> str:
        """e.g. 'bm25#7 + vector#3', or 'vector#3' if only one found it."""
        return " + ".join(f"{name}#{rank}" for name, rank in sorted(self.ranks.items()))


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[Document]],
    k: int,
    rank_constant: int = DEFAULT_RANK_CONSTANT,
) -> list[FusedHit]:
    """Fuse named ranked lists into one, highest RRF score first.

    Every chunk appearing in *any* input list gets a score; a chunk missing
    from a list simply contributes no term for it, which is what lets RRF
    handle lists of differing length (BM25 may return fewer than k when few
    chunks share a query term).
    """
    fused: dict[ChunkKey, FusedHit] = {}

    for name, docs in ranked_lists.items():
        for rank, doc in enumerate(docs, start=1):
            key = chunk_key(doc)
            hit = fused.get(key)
            if hit is None:
                hit = fused[key] = FusedHit(document=doc, score=0.0)
            hit.score += 1.0 / (rank_constant + rank)
            hit.ranks[name] = rank

    ordered = sorted(fused.values(), key=lambda h: h.score, reverse=True)
    return ordered[:k]


def hybrid_search(
    query: str,
    k: int = 5,
    fetch_k: int = DEFAULT_FETCH_K,
    rank_constant: int = DEFAULT_RANK_CONSTANT,
    retrievers: Retrievers | None = None,
    year_filter: bool = True,
) -> list[FusedHit]:
    """Retrieve `fetch_k` from each retriever, fuse, and return the top `k`.

    Source metadata (source, page, chunk_index) rides along on each Document
    so the caller can cite the answer back to a location in the corpus.
    """
    retrievers = retrievers or load_retrievers()
    dense, sparse = search_both(query, fetch_k, retrievers, year_filter)
    return reciprocal_rank_fusion(
        {"vector": [d for d, _ in dense], "bm25": [d for d, _ in sparse]},
        k=k,
        rank_constant=rank_constant,
    )


def citation(doc: Document) -> str:
    """Short human-readable provenance string for a retrieved chunk."""
    meta = doc.metadata
    page = meta.get("page", config.NO_PAGE)
    where = f"p.{page}" if page != config.NO_PAGE else "no pages"
    return f"{meta.get('source', '?')} | {where} | chunk {meta.get('chunk_index', '?')}"


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Hybrid retrieval over the built indexes."
    )
    parser.add_argument("query", help="the search query")
    parser.add_argument("-k", type=int, default=5, help="results to return")
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=DEFAULT_FETCH_K,
        help="candidates pulled from each retriever before fusing",
    )
    parser.add_argument("--no-year-filter", action="store_true",
                        help="disable edition filtering (A/B against the default)")
    args = parser.parse_args(argv)

    config.enable_utf8_stdout()
    hits = hybrid_search(args.query, k=args.k, fetch_k=args.fetch_k,
                         year_filter=not args.no_year_filter)
    print(f"\n{len(hits)} hybrid result(s) for {args.query!r}\n")
    for i, hit in enumerate(hits, start=1):
        print(f"{i}. [{hit.sources}] rrf={hit.score:.5f}")
        print(f"   {citation(hit.document)}")
        print(f"   {' '.join(hit.document.page_content.split())[:160]}...\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
