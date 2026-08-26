"""Add structured document metadata to the existing indexes, in place.

New metadata is derived entirely from each chunk's `source` filename, so no
text needs re-reading and no vector needs recomputing. Chroma stores metadata
beside the embedding and can update it on its own, and the BM25 bundle is a
pickle of plain dicts. Rebuilding instead would cost ~77 minutes of re-embedding
to arrive at byte-identical vectors.

    python -m src.ingestion.backfill_metadata
    python -m src.ingestion.backfill_metadata --check   # report, change nothing
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config
from src.ingestion import bm25_index
from src.retrieval import metadata as MD

NEW_FIELDS = ("doc_type", "entity", "doc_year", "doc_month")


def backfill_bm25(check: bool = False) -> int:
    bundle = bm25_index.load_bm25_index()
    missing = sum(1 for m in bundle.metadatas if "doc_year" not in m)
    print(f"BM25   : {len(bundle.metadatas)} chunks, {missing} lacking metadata")
    if check or not missing:
        return missing

    for m in bundle.metadatas:
        m.update(MD.parse_source(m["source"]).as_chunk_metadata())

    with config.BM25_PATH.open("wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"         updated and re-pickled -> {config.BM25_PATH}")
    return missing


def backfill_chroma(check: bool = False, batch: int = 2000) -> int:
    import chromadb

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    coll = client.get_collection(config.CHROMA_COLLECTION)
    total = coll.count()

    got = coll.get(include=["metadatas"], limit=total)
    ids, metas = got["ids"], got["metadatas"]
    missing = sum(1 for m in metas if "doc_year" not in m)
    print(f"Chroma : {total} vectors, {missing} lacking metadata")
    if check or not missing:
        return missing

    updated = [{**m, **MD.parse_source(m["source"]).as_chunk_metadata()} for m in metas]
    for start in range(0, len(ids), batch):
        coll.update(ids=ids[start:start + batch],
                    metadatas=updated[start:start + batch])
        print(f"         updated {min(start + batch, len(ids))}/{len(ids)}")
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill document metadata in place.")
    parser.add_argument("--check", action="store_true", help="report only")
    args = parser.parse_args(argv)

    config.enable_utf8_stdout()
    backfill_bm25(args.check)
    backfill_chroma(args.check)

    if not args.check:
        print("\nverifying ...")
        b = bm25_index.load_bm25_index()
        sample = b.metadatas[0]
        print(f"  bm25 sample: {({k: sample.get(k) for k in NEW_FIELDS})}")
        import chromadb

        coll = chromadb.PersistentClient(path=str(config.CHROMA_DIR)).get_collection(
            config.CHROMA_COLLECTION)
        one = coll.get(limit=1, include=["metadatas"])["metadatas"][0]
        print(f"  chroma sample: {({k: one.get(k) for k in NEW_FIELDS})}")
        print(f"  chroma vectors intact: {coll.count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
