"""Cross-encoder reranking on top of hybrid retrieval.

Fusion ranks a chunk by *where* two retrievers placed it, never by reading the
chunk against the query. A cross-encoder does exactly that: it encodes query
and chunk together in one forward pass, so attention runs across both. That is
strictly more informative than comparing two independently-computed vectors,
and strictly more expensive -- there is no index to precompute, every candidate
costs a full model pass at query time.

That cost drives the defaults here. Measured on this CPU-only machine:

    ms-marco-MiniLM-L-6-v2  ( 23M)   139 ms/pair   ~1.2s for 10 candidates
    BAAI/bge-reranker-base  (278M)  1183 ms/pair  ~23.7s for 20 candidates

So the small model reranking a short candidate list is the interactive default,
and the larger, stronger model is available by name for offline work such as
Phase 4 evaluation, where a 24-second query costs nothing.

    python -m src.retrieval.rerank "supply chain disruption risk"
    python -m src.retrieval.rerank --model BAAI/bge-reranker-base "..."
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config
from src.retrieval import hybrid

# Small and fast: the interactive default.
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Stronger and ~9x slower. Practical only when latency does not matter.
EVAL_RERANKER = "BAAI/bge-reranker-base"

# How many fused candidates to rerank. The dominant latency lever: cost is
# linear in this number, so 10 costs half what 20 does. The trade-off is reach
# -- a chunk fusion ranked 15th can never be promoted if only 10 are rescored.
DEFAULT_RERANK_CANDIDATES = 10

# Chunks average ~1000 characters, which fits comfortably. Lowering this would
# buy little speed and would silently truncate the tail of a chunk before the
# model ever reads it -- the same failure the chunk sizing was chosen to avoid.
MAX_LENGTH = 512


_MODELS: dict[str, object] = {}


def use_all_cores() -> None:
    """Let torch use every logical core.

    Worth ~12% on reranking. Called from CLI entrypoints only, never on
    import, since changing global torch state is not a library's business.
    """
    try:
        import os

        import torch

        torch.set_num_threads(os.cpu_count() or 1)
    except Exception:  # noqa: BLE001 - a speed hint, never fatal
        pass


def get_reranker(model_name: str = DEFAULT_RERANKER):
    """Load (and cache) a CrossEncoder. First call downloads the model once."""
    if model_name not in _MODELS:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is not installed. Run "
                "`pip install sentence-transformers`."
            ) from exc
        _MODELS[model_name] = CrossEncoder(model_name, max_length=MAX_LENGTH)
    return _MODELS[model_name]


@dataclass
class RerankedHit:
    """A chunk after reranking, carrying where it stood before.

    Keeping `fused_rank` is what makes the reranker auditable: without it you
    cannot tell whether the cross-encoder reordered anything or simply passed
    fusion's ranking straight through.
    """

    document: Document
    score: float
    fused_rank: int
    fused_score: float
    retriever_ranks: dict[str, int]
    rank: int = 0

    @property
    def movement(self) -> str:
        """e.g. '+3' if promoted three places by reranking, '0' if unmoved."""
        return f"{self.fused_rank - self.rank:+d}" if self.rank else "?"

    @property
    def sources(self) -> str:
        return " + ".join(f"{n}#{r}" for n, r in sorted(self.retriever_ranks.items()))


def rerank(
    query: str,
    hits: list[hybrid.FusedHit],
    top_k: int = 5,
    model_name: str = DEFAULT_RERANKER,
) -> list[RerankedHit]:
    """Rescore `hits` against `query` with a cross-encoder, best first."""
    if not hits:
        return []

    model = get_reranker(model_name)
    scores = model.predict([(query, h.document.page_content) for h in hits])

    scored = [
        RerankedHit(
            document=h.document,
            score=float(s),
            fused_rank=i,
            fused_score=h.score,
            retriever_ranks=dict(h.ranks),
        )
        for i, (h, s) in enumerate(zip(hits, scores), start=1)
    ]
    scored.sort(key=lambda r: r.score, reverse=True)
    for position, hit in enumerate(scored, start=1):
        hit.rank = position
    return scored[:top_k]


def retrieve_and_rerank(
    query: str,
    k: int = 5,
    candidates: int = DEFAULT_RERANK_CANDIDATES,
    fetch_k: int = hybrid.DEFAULT_FETCH_K,
    model_name: str = DEFAULT_RERANKER,
    year_filter: bool = True,
) -> list[RerankedHit]:
    """Full pipeline: dense + BM25 -> RRF -> cross-encoder -> top `k`."""
    fused = hybrid.hybrid_search(query, k=candidates, fetch_k=fetch_k,
                                 year_filter=year_filter)
    return rerank(query, fused, top_k=k, model_name=model_name)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Hybrid retrieval + cross-encoder rerank.")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5, help="final results")
    parser.add_argument("--candidates", type=int, default=DEFAULT_RERANK_CANDIDATES,
                        help="fused candidates to rerank (latency scales with this)")
    parser.add_argument("--model", default=DEFAULT_RERANKER,
                        help=f"cross-encoder model (eval-grade: {EVAL_RERANKER})")
    parser.add_argument("--no-year-filter", action="store_true",
                        help="disable edition filtering")
    args = parser.parse_args(argv)

    config.enable_utf8_stdout()
    use_all_cores()

    t0 = time.perf_counter()
    hits = retrieve_and_rerank(args.query, k=args.k, candidates=args.candidates,
                               model_name=args.model,
                               year_filter=not args.no_year_filter)
    elapsed = time.perf_counter() - t0

    print(f"\n{len(hits)} result(s) for {args.query!r}")
    print(f"model={args.model}  candidates={args.candidates}  {elapsed:.2f}s\n")
    for hit in hits:
        print(f"{hit.rank}. ce={hit.score:+.3f}  (fusion #{hit.fused_rank}, "
              f"moved {hit.movement})  [{hit.sources}]")
        print(f"   {hybrid.citation(hit.document)}")
        print(f"   {' '.join(hit.document.page_content.split())[:160]}...\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
