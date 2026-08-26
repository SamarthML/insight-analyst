"""Side-by-side comparison of vector-only, BM25-only, hybrid and reranked retrieval.

The point of this script is qualitative inspection before trusting the stack.
If the hybrid column is simply one of the other two columns rewritten, fusion
is buying nothing; if the reranked column is the hybrid column unchanged, the
cross-encoder is not earning its ~1.2s. What we want to see is fusion promoting
chunks both retrievers liked moderately, and reranking then reordering those on
the actual query-chunk relationship.

    python -m src.retrieval.compare_retrieval
    python -m src.retrieval.compare_retrieval "your own query"
    python -m src.retrieval.compare_retrieval --no-rerank --snippets
    python -m src.retrieval.compare_retrieval --rerank-model BAAI/bge-reranker-base
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from langchain_core.documents import Document

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config
from src.retrieval import hybrid, rerank as rerank_mod, router

# Deliberately split between queries that should favour keyword matching
# (literal years, tickers, named financial line items) and queries that should
# favour dense matching (concepts the filings express in different words).
DEFAULT_QUERIES = [
    ("what was Apple's revenue in fiscal 2023", "exact-term -> expect BM25"),
    ("what risks does the company face from supply chain disruption",
     "conceptual -> expect vector"),
    ("NVDA data center segment revenue growth", "ticker + exact -> expect BM25"),
    ("how might rising interest rates affect bank profitability",
     "conceptual -> expect vector"),
    ("common equity tier 1 capital ratio", "exact financial term -> expect BM25"),
    ("what structural headwinds face the global economy over the long run",
     "conceptual -> expect vector"),
]

COL = 29


def short_label(doc: Document) -> str:
    """Compact chunk identifier that fits in a comparison column."""
    meta = doc.metadata
    name = str(meta.get("source", "?"))
    for prefix in ("sec_10k_", "worldbank_global_economic_prospects_",
                   "bis_annual_economic_report_", "fed_beige_book_", "sample_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    name = name.rsplit(".", 1)[0]
    page = meta.get("page", config.NO_PAGE)
    where = f"p{page}" if page != config.NO_PAGE else ""
    return f"{name[:16]:<16}{where:>5} #{meta.get('chunk_index', '?')}"


def _cell(text: str) -> str:
    return text[:COL].ljust(COL)


def compare(query: str, note: str, k: int, fetch_k: int, snippets: bool,
            do_rerank: bool, candidates: int, model_name: str,
            year_filter: bool = True) -> None:
    retrievers = hybrid.load_retrievers()
    dense, sparse = hybrid.search_both(query, fetch_k, retrievers, year_filter)

    # Fuse deep enough to feed the reranker, then slice for display, so the
    # hybrid column and the reranker see a consistent candidate pool.
    pool = hybrid.reciprocal_rank_fusion(
        {"vector": [d for d, _ in dense], "bm25": [d for d, _ in sparse]},
        k=max(k, candidates),
    )
    fused = pool[:k]

    reranked = []
    elapsed = 0.0
    if do_rerank:
        t0 = time.perf_counter()
        reranked = rerank_mod.rerank(query, pool[:candidates], top_k=k,
                                     model_name=model_name)
        elapsed = time.perf_counter() - t0

    vec_top = [d for d, _ in dense][:k]
    bm_top = [d for d, _ in sparse][:k]
    width = COL * (4 if do_rerank else 3)

    print("\n" + "=" * (width + 6))
    print(f"QUERY: {query}")
    print(f"       ({note})")
    print("=" * (width + 6))
    header = (_cell(f"VECTOR-only top-{k}") + _cell(f"BM25-only top-{k}")
              + _cell(f"HYBRID (RRF) top-{k}"))
    if do_rerank:
        header += _cell(f"+RERANKED top-{k}")
    print("  " + header)
    print("  " + "-" * width)

    for i in range(k):
        row = (_cell(short_label(vec_top[i]) if i < len(vec_top) else "-")
               + _cell(short_label(bm_top[i]) if i < len(bm_top) else "-")
               + _cell(short_label(fused[i].document) if i < len(fused) else "-"))
        if do_rerank:
            row += _cell(short_label(reranked[i].document) if i < len(reranked) else "-")
        print(f"{i + 1}. " + row)

    print("  " + "-" * width)
    print("  hybrid provenance:")
    for i, hit in enumerate(fused, start=1):
        print(f"    {i}. {hit.sources:<24} rrf={hit.score:.5f}  {short_label(hit.document)}")

    if do_rerank:
        print(f"  rerank ({model_name.split('/')[-1]}, {candidates} candidates, "
              f"{elapsed:.2f}s):")
        for hit in reranked:
            print(f"    {hit.rank}. ce={hit.score:+8.3f}  from fusion #{hit.fused_rank} "
                  f"(moved {hit.movement:>3})  {short_label(hit.document)}")

    vec_keys = {hybrid.chunk_key(d) for d in vec_top}
    bm_keys = {hybrid.chunk_key(d) for d in bm_top}
    hyb_keys = [hybrid.chunk_key(h.document) for h in fused]
    both = len([1 for h in fused if len(h.ranks) == 2])
    new = len([kk for kk in hyb_keys if kk not in vec_keys and kk not in bm_keys])

    print("  " + "-" * width)
    print(f"  vector n BM25 (top-{k}) : {len(vec_keys & bm_keys)}"
          f"   |  hybrid hits found by BOTH retrievers: {both}/{len(fused)}")
    print(f"  hybrid hits in neither displayed top-{k} (promoted from deeper): {new}")
    if do_rerank and reranked:
        moved = sum(1 for h in reranked if h.fused_rank != h.rank)
        promoted = [h for h in reranked if h.fused_rank > k]
        print(f"  rerank reordered {moved}/{len(reranked)} of the top-{k}; "
              f"{len(promoted)} pulled up from below fusion's top-{k}")

    if snippets:
        source = reranked if do_rerank else fused
        print(f"\n  --- {'reranked' if do_rerank else 'hybrid'} snippets ---")
        for i, hit in enumerate(source, start=1):
            body = " ".join(hit.document.page_content.split())[:200]
            print(f"    {i}. [{hybrid.citation(hit.document)}]\n       {body}...")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare vector-only, BM25-only, hybrid and reranked retrieval."
    )
    parser.add_argument("queries", nargs="*", help="queries (default: the built-in set)")
    parser.add_argument("-k", type=int, default=5, help="results per column")
    parser.add_argument("--fetch-k", type=int, default=hybrid.DEFAULT_FETCH_K,
                        help="candidates pulled from each retriever before fusing")
    parser.add_argument("--candidates", type=int,
                        default=rerank_mod.DEFAULT_RERANK_CANDIDATES,
                        help="fused candidates passed to the reranker")
    parser.add_argument("--rerank-model", default=rerank_mod.DEFAULT_RERANKER,
                        help=f"cross-encoder (eval-grade: {rerank_mod.EVAL_RERANKER})")
    parser.add_argument("--no-rerank", action="store_true",
                        help="omit the reranked column")
    parser.add_argument("--snippets", action="store_true", help="print chunk text")
    parser.add_argument("--no-year-filter", action="store_true",
                        help="disable edition filtering (A/B against the default)")
    args = parser.parse_args(argv)

    config.enable_utf8_stdout()
    rerank_mod.use_all_cores()

    pairs = ([(q, "custom") for q in args.queries] if args.queries else DEFAULT_QUERIES)
    do_rerank = not args.no_rerank

    retrievers = hybrid.load_retrievers()
    print(f"indexes: {len(retrievers)} chunks | embeddings: "
          f"{config.SENTENCE_TRANSFORMERS_MODEL} | fetch_k={args.fetch_k}, "
          f"rrf_k={hybrid.DEFAULT_RANK_CONSTANT}")
    if do_rerank:
        # Warm the model so the first query's timing is not the download.
        rerank_mod.get_reranker(args.rerank_model)
        print(f"reranker: {args.rerank_model} | candidates={args.candidates}")

    for query, note in pairs:
        compare(query, note, args.k, args.fetch_k, args.snippets,
                do_rerank, args.candidates, args.rerank_model,
                not args.no_year_filter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
