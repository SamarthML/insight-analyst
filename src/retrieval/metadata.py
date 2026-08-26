"""Structured document metadata parsed from source filenames, and year filtering.

WHY THIS EXISTS
Evaluation traced 10 of 22 retrieval misses to one failure: neither retriever
can tell two editions of the same document apart. Asked about Microsoft's
fiscal 2024, retrieval returns the FY2026 and FY2025 filings; asked for the
January 2025 Beige Book it returns five Beige Books, none of them January.

The reason is that the distinguishing token is a year, and a year is nearly
invisible to both retrievers. In a 384-dimensional sentence embedding "fiscal
2024" and "fiscal 2025" are almost collinear -- the surrounding boilerplate
dominates. For BM25, "2024" appears across most of the corpus, so its IDF is
near zero. The information *is* present, but not in a form ranking can use.

It is, however, sitting in the filename. `sec_10k_msft_2024-07-30.txt` and
`fed_beige_book_20250115.pdf` both state their edition unambiguously. Parsing
that into structured metadata and filtering on it turns a ranking problem into
a lookup, which is the right shape for the job.

THE TRAP: FILING DATE IS NOT FISCAL YEAR
A filename carries the date the document was *published*, which is not the
period it *reports on*, and the gap is not constant. Verified against the
labelled eval set:

    NVDA / MSFT / AAPL 10-K   fiscal year == filing year        offset  0
    JPM / KO 10-K             calendar-year filers, file in Feb offset +1
    Beige Book                edition date is the subject       offset  0
    World Bank / BIS          year in query is the SUBJECT year offset -1..+3

So a strict `filename_year == query_year` filter would fix the Microsoft and
Beige Book questions while breaking every JPMorgan and Coca-Cola question, plus
the World Bank ones -- a net regression. Filtering therefore has to know what
kind of document it is looking at, which is what `year_window` below encodes.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# --- Document types --------------------------------------------------------
SEC_10K = "sec_10k"
BEIGE_BOOK = "beige_book"
WORLDBANK = "worldbank_gep"
BIS = "bis_annual"
SAMPLE = "sample"
UNKNOWN = "unknown"

# How a query year relates to the document's own date, per type.
#
#   (lo, hi) means: a query about year Y can be answered by a document dated
#   Y+lo .. Y+hi. None means year filtering is not meaningful for this type.
#
# SEC filings report the year before they are published, or the same year,
# depending on where the company's fiscal year ends -- so both are allowed.
# For research reports the year in a query is usually the year being discussed
# ("growth in 2025", "projection for 2027"), not the year of publication, and
# the observed offsets range from -1 to +3. There is no window that helps, so
# those are left unfiltered rather than filtered wrongly.
YEAR_WINDOW: dict[str, tuple[int, int] | None] = {
    SEC_10K: (0, 1),
    BEIGE_BOOK: (0, 0),
    WORLDBANK: None,
    BIS: None,
    SAMPLE: None,
    UNKNOWN: None,
}

_SEC_RE = re.compile(r"^sec_10k_(?P<ticker>[a-z]+)_(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})")
_BEIGE_RE = re.compile(r"^fed_beige_book_(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})")
_WB_RE = re.compile(r"^worldbank_global_economic_prospects_(?P<y>\d{4})_(?P<m>\d{2})")
_BIS_RE = re.compile(r"^bis_annual_economic_report_(?P<y>\d{4})")


@dataclass(frozen=True)
class DocMeta:
    doc_type: str
    entity: str          # ticker for filings, issuer otherwise, "" if n/a
    year: int            # 0 when the filename carries no year
    month: int           # 0 when unknown
    day: int             # 0 when unknown

    def as_chunk_metadata(self) -> dict:
        """Flat, scalar-only dict -- Chroma rejects nested or None values."""
        return {
            "doc_type": self.doc_type,
            "entity": self.entity,
            "doc_year": self.year,
            "doc_month": self.month,
        }


def parse_source(source: str) -> DocMeta:
    """Parse a corpus filename into structured document metadata."""
    name = str(source)

    m = _SEC_RE.match(name)
    if m:
        return DocMeta(SEC_10K, m["ticker"], int(m["y"]), int(m["m"]), int(m["d"]))

    m = _BEIGE_RE.match(name)
    if m:
        return DocMeta(BEIGE_BOOK, "fed", int(m["y"]), int(m["m"]), int(m["d"]))

    m = _WB_RE.match(name)
    if m:
        return DocMeta(WORLDBANK, "worldbank", int(m["y"]), int(m["m"]), 0)

    m = _BIS_RE.match(name)
    if m:
        return DocMeta(BIS, "bis", int(m["y"]), 0, 0)

    if name.startswith("sample_"):
        return DocMeta(SAMPLE, "", 0, 0, 0)
    return DocMeta(UNKNOWN, "", 0, 0, 0)


# --- Query-side detection --------------------------------------------------
_YEAR_RE = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}
_MONTH_RE = re.compile(r"\b(" + "|".join(_MONTHS) + r")\b", re.I)

# Tickers and company names, so a query can be narrowed to one filer.
_ENTITY_ALIASES = {
    "aapl": ("apple", "aapl"),
    "msft": ("microsoft", "msft"),
    "nvda": ("nvidia", "nvda"),
    "jpm": ("jpmorgan", "jpmorganchase", "jpmorgan chase", "jpm"),
    "ko": ("coca-cola", "coca cola", "cocacola", "the coca-cola company"),
}


@dataclass(frozen=True)
class QuerySpec:
    """What a query says about which document edition it wants."""

    years: tuple[int, ...] = ()
    month: int = 0
    entity: str = ""

    @property
    def has_year(self) -> bool:
        return bool(self.years)


def parse_query(query: str) -> QuerySpec:
    """Extract year(s), month and company from a natural-language query.

    Deliberately literal. A four-digit year and a spelled-out month cover the
    phrasing in this corpus's questions ("in fiscal year 2024", "the January
    2025 Beige Book"); anything cleverer would need its own evaluation before
    being trusted to *remove* candidates from retrieval.
    """
    text = query.lower()
    years = tuple(sorted({int(y) for y in _YEAR_RE.findall(text)}))

    mm = _MONTH_RE.search(text)
    month = _MONTHS[mm.group(1).lower()] if mm else 0

    entity = ""
    for ticker, aliases in _ENTITY_ALIASES.items():
        if any(a in text for a in aliases):
            entity = ticker
            break

    return QuerySpec(years=years, month=month, entity=entity)


# --- Matching --------------------------------------------------------------
def chunk_matches(meta: dict, spec: QuerySpec) -> bool:
    """Could the chunk's document plausibly answer a query about `spec`?

    Permissive by design. This runs *before* ranking, so a false negative
    silently deletes the right answer, while a false positive merely leaves a
    candidate for BM25, the embedding and the reranker to sort out -- which is
    what they are good at. Any document type without a meaningful year window
    is therefore always kept.
    """
    if not spec.has_year and not spec.entity:
        return True

    doc_type = meta.get("doc_type", UNKNOWN)

    # A query naming a company cannot be answered by another company's filing.
    # Applied only within SEC filings: a Beige Book has no ticker, and saying
    # "Microsoft" does not mean Fed commentary is irrelevant.
    if spec.entity and doc_type == SEC_10K:
        if str(meta.get("entity", "")) != spec.entity:
            return False

    if not spec.has_year:
        return True

    window = YEAR_WINDOW.get(doc_type)
    if window is None:
        return True                      # year filtering not meaningful here

    doc_year = int(meta.get("doc_year", 0) or 0)
    if not doc_year:
        return True

    lo, hi = window
    if not any(y + lo <= doc_year <= y + hi for y in spec.years):
        return False

    # Beige Books publish eight times a year, so the year alone leaves most of
    # them in. Month is applied only when the query actually names one.
    if doc_type == BEIGE_BOOK and spec.month:
        doc_month = int(meta.get("doc_month", 0) or 0)
        if doc_month and abs(doc_month - spec.month) > 1:
            return False

    return True


def prefer_latest_edition(docs: list, spec: QuerySpec) -> list:
    """Float the newest edition's chunks to the front when no year was asked for.

    WHY THIS EXISTS
    `chunk_matches` returns True for everything when a query carries no year,
    which is right for filtering -- there is nothing to filter on. But it means
    a question like "revenue in its most recent annual filing" retrieves freely
    across all three NVIDIA 10-Ks, and each such question independently lands
    on whichever edition happened to rank highest. In a multi-section report
    that produces sections resting on *different* editions while reading as one
    continuous account: observed with a FY2026 revenue figure in one section
    and a FY2024 growth rate in the next, presented as its year-over-year.

    Ordering, not filtering. A wrong-year chunk demoted to position 5 is
    recoverable; one deleted before ranking is not, and the module docstring's
    whole argument is that removing candidates is the dangerous direction. The
    relative order within each group is preserved, so relevance still decides
    everything except which edition leads.

    Only document types whose date *is* their edition are moved. World Bank and
    BIS reports carry a subject year rather than an edition year -- the same
    reason `YEAR_WINDOW` leaves them unfiltered -- so they are left alone.
    """
    if spec.has_year or not docs:
        return docs                      # an explicit year already decided this

    def edition(doc) -> tuple[int, int] | None:
        meta = getattr(doc, "metadata", {}) or {}
        if YEAR_WINDOW.get(meta.get("doc_type", UNKNOWN)) is None:
            return None                  # date is not this type's edition
        year = int(meta.get("doc_year", 0) or 0)
        if not year:
            return None
        return (year, int(meta.get("doc_month", 0) or 0))

    editions = [e for e in (edition(d) for d in docs) if e is not None]
    if not editions:
        return docs
    latest = max(editions)

    newest = [d for d in docs if edition(d) == latest]
    rest = [d for d in docs if edition(d) != latest]
    return newest + rest


def chroma_where(spec: QuerySpec) -> dict | None:
    """A Chroma `where` clause implementing exactly the same rule as
    `chunk_matches`, so the dense and sparse sides filter identically.

    An earlier version approximated this with one global year range across all
    document types, which quietly diverged from the BM25 path: it ignored the
    month, so a query for the January Beige Book still admitted all eight
    editions of that year, and the dense side returned the same wrong documents
    as before filtering. The clause is therefore built per document type.
    """
    if not spec.has_year and not spec.entity:
        return None

    branches: list[dict] = []

    for doc_type, window in YEAR_WINDOW.items():
        if window is None:
            branches.append({"doc_type": {"$eq": doc_type}})
            continue

        clauses: list[dict] = [{"doc_type": {"$eq": doc_type}}]

        if spec.has_year:
            lo, hi = window
            years = sorted({y + off for y in spec.years for off in range(lo, hi + 1)})
            clauses.append({"doc_year": {"$in": years}})

        if doc_type == SEC_10K and spec.entity:
            clauses.append({"entity": {"$eq": spec.entity}})

        if doc_type == BEIGE_BOOK and spec.month:
            months = [m for m in (spec.month - 1, spec.month, spec.month + 1) if 1 <= m <= 12]
            clauses.append({"doc_month": {"$in": months}})

        branches.append({"$and": clauses} if len(clauses) > 1 else clauses[0])

    return {"$or": branches} if len(branches) > 1 else branches[0]
