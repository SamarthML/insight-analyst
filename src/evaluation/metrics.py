"""Deterministic retrieval metrics scored against ground-truth chunk citations.

RAGAS computes context precision and recall by asking an LLM whether each
retrieved chunk was useful. That is necessary when you have no labels. Here we
*do* have labels -- questions.yaml records the exact chunks that contain each
answer -- so these can be computed by set comparison instead: exactly, in
milliseconds, with no LLM, and with no judge error to argue about.

This matters beyond convenience. The question "was reranking worth building?"
is a retrieval question, and these metrics answer it directly. The LLM-judged
metrics (faithfulness, answer relevancy) measure the *generator*, which is a
different component and is unaffected by rerank ordering except through the
context it receives.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ChunkKey = tuple[str, int, int]


def key_of(doc: Document) -> ChunkKey:
    m = doc.metadata
    return (str(m.get("source", "?")), int(m.get("page", -1)), int(m.get("chunk_index", -1)))


def key_of_truth(entry: dict) -> ChunkKey:
    return (str(entry["source"]), int(entry["page"]), int(entry["chunk_index"]))


@dataclass
class RetrievalScore:
    """Per-question retrieval quality against labelled ground truth."""

    hit: bool                 # was any ground-truth chunk retrieved at all
    precision: float          # fraction of retrieved chunks that are ground truth
    recall: float             # fraction of ground-truth chunks that were retrieved
    reciprocal_rank: float    # 1/rank of the first ground-truth chunk, else 0
    first_rank: int | None    # where the first ground-truth chunk landed
    n_retrieved: int
    n_truth: int

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def score_retrieval(retrieved: list[Document], truth: list[dict]) -> RetrievalScore | None:
    """Score one question. Returns None when the question has no ground truth.

    Adversarial questions carry no ground truth by design -- there is nothing
    correct to retrieve -- so they are excluded rather than scored as zero,
    which would drag the aggregate down for questions that were never meant to
    have a right answer.
    """
    truth_keys = {key_of_truth(t) for t in truth}
    if not truth_keys:
        return None

    got = [key_of(d) for d in retrieved]
    got_set = set(got)
    matched = truth_keys & got_set

    first_rank = next((i for i, k in enumerate(got, start=1) if k in truth_keys), None)

    return RetrievalScore(
        hit=bool(matched),
        precision=len(matched) / len(got) if got else 0.0,
        recall=len(matched) / len(truth_keys),
        reciprocal_rank=(1.0 / first_rank) if first_rank else 0.0,
        first_rank=first_rank,
        n_retrieved=len(got),
        n_truth=len(truth_keys),
    )


def aggregate(scores: list[RetrievalScore]) -> dict[str, float]:
    """Mean of each metric over the questions that had ground truth."""
    if not scores:
        return {"hit_rate": 0.0, "precision": 0.0, "recall": 0.0, "mrr": 0.0, "f1": 0.0, "n": 0}
    n = len(scores)
    return {
        "hit_rate": sum(s.hit for s in scores) / n,
        "precision": sum(s.precision for s in scores) / n,
        "recall": sum(s.recall for s in scores) / n,
        "mrr": sum(s.reciprocal_rank for s in scores) / n,
        "f1": sum(s.f1 for s in scores) / n,
        "n": n,
    }
