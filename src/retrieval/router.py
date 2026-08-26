"""Query-type routing: pick a retrieval strategy per query instead of stacking all of them.

WHY THIS EXISTS
The Phase 4 evaluation produced a clean dissociation across 22 labelled
questions:

    exact-fact hit rate    conceptual hit rate
      hybrid only          0.538          0.333
      + rerank             0.538          0.444    <- reranking moves conceptual
      + filter             0.692          0.333    <- filtering moves exact-fact
      + filter + rerank    0.615          0.444    <- worse than filter alone

Each technique fixes a different failure, and stacking them is actively worse
on exact-fact questions: once edition filtering has removed the wrong-year
candidates, the cross-encoder is re-scoring a pool where topical relevance is
no longer the thing that needs discriminating, so it adds noise. The obvious
move is to route rather than stack.

DESIGN: RULES, NOT A MODEL
This is a hand-written weighted rule set, not a classifier. Three reasons.
A trained model would need its own labelled data, its own evaluation and its
own failure modes, to make a binary decision over a handful of obvious
surface signals. The signals are already computed -- `metadata.parse_query`
extracts years and companies for edition filtering, so routing is nearly free.
And a routing decision that silently sends a query down the wrong path is very
hard to debug after the fact, so every decision here carries the list of
signals that produced it (see `RoutingDecision.explain`).

FAILING SAFE
When the signals are weak or contradictory the router returns AMBIGUOUS and
the caller runs the combined configuration. That is deliberately not the
best-scoring config: filtering alone scored highest overall, but a wrong
confident route costs more than a slightly suboptimal safe one, and the
combined path is the one that never performs *badly* on either category.

    python -m src.retrieval.router "what was Apple's revenue in fiscal 2023"
    python -m src.retrieval.router --validate     # score against questions.yaml
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config
from src.retrieval import metadata as MD

# --- Routes ----------------------------------------------------------------
EXACT_FACT = "exact-fact"
CONCEPTUAL = "conceptual"
AMBIGUOUS = "ambiguous"

# Strategy per route. Read directly from the Phase 4 numbers above.
ROUTE_STRATEGY = {
    EXACT_FACT: {"filter": True, "rerank": False},
    CONCEPTUAL: {"filter": False, "rerank": True},
    AMBIGUOUS: {"filter": True, "rerank": True},
}

# A route must win by at least this margin, otherwise the query is ambiguous.
# Set at 2 so a single weak signal on one side cannot decide a query on its
# own -- one moderate signal (weight 2) or two weak ones are required.
DECISION_MARGIN = 2

# --- Signals ---------------------------------------------------------------
# Phrases asking for a specific quantity. These are the strongest textual cue
# for exact-fact, distinct from the *topic* being financial: "how much did
# revenue increase" is exact-fact, "what drives revenue growth" is not.
_FIGURE_PHRASES = re.compile(
    r"\b(how much|how many|what was the|what were the|what is the|"
    r"by how much|how large|what percentage|what proportion)\b", re.I)

# Named financial line items that denote a specific reported figure.
_LINE_ITEMS = re.compile(
    r"\b(revenue|net income|net sales|earnings per share|eps|operating income|"
    r"gross margin|capital ratio|cet1|rotce|roe|dividend|total assets|"
    r"operating revenues|net yield)\b", re.I)

# Explicit magnitudes or currency in the query itself.
_MAGNITUDE = re.compile(r"(\$|\d+\s*(billion|million|percent|%|bps))", re.I)

# Open-ended asks. These target explanation, comparison or synthesis rather
# than a lookup.
_OPEN_PHRASES = re.compile(
    r"\b(what are the|what risks|why (?:did|do|does|has|have|is|are)|"
    r"how (?:did|has|have|was|were|is|are) .*? "
    r"(?:chang|evolv|perform|describ|develop|shift|trend)|"
    r"how might|how does|how do|what drives|what factors|describe|explain|"
    r"characteriz|outlook|headwinds|implications|impact of|role|"
    r"what challenges|compare|trend|"
    r"what did .*? (?:say|estimate|project|report)|direction of)\b", re.I)

# Words naming a theme rather than a number.
_THEME_WORDS = re.compile(
    r"\b(risks?|strategy|conditions|sentiment|policy|environment|landscape|"
    r"disruption|uncertainty|pressures?|prospects)\b", re.I)

# Institutional publishers. CORPUS-SPECIFIC, and worth stating plainly: in this
# corpus the exact-fact questions are answered from company filings and the
# conceptual ones from institutional research, so naming an institution is a
# strong conceptual cue. That is a domain signal rather than a general
# linguistic one, and it would need revisiting for a corpus where institutions
# publish the hard figures too.
_INSTITUTIONS = re.compile(
    r"\b(federal reserve|beige book|world bank|bis|"
    r"bank for international settlements|imf|central banks?)\b", re.I)


@dataclass(frozen=True)
class Signal:
    """One piece of evidence, with the text that triggered it."""

    name: str
    route: str
    weight: int
    detail: str = ""

    def __str__(self) -> str:
        d = f" ({self.detail})" if self.detail else ""
        return f"{self.name}{d} +{self.weight}->{self.route}"


@dataclass
class RoutingDecision:
    """The chosen route plus every signal that contributed to it."""

    query: str
    route: str
    signals: list[Signal] = field(default_factory=list)
    scores: dict[str, int] = field(default_factory=dict)
    spec: MD.QuerySpec | None = None

    @property
    def strategy(self) -> dict:
        return ROUTE_STRATEGY[self.route]

    @property
    def margin(self) -> int:
        return abs(self.scores.get(EXACT_FACT, 0) - self.scores.get(CONCEPTUAL, 0))

    def explain(self) -> str:
        """One-line audit trail: route, scores, and the signals behind them."""
        s = self.strategy
        plan = f"filter={'on' if s['filter'] else 'off'}, rerank={'on' if s['rerank'] else 'off'}"
        fired = "; ".join(str(x) for x in self.signals) or "no signals"
        return (f"{self.route.upper():<10} [{plan}]  "
                f"exact={self.scores.get(EXACT_FACT, 0)} "
                f"concept={self.scores.get(CONCEPTUAL, 0)} "
                f"margin={self.margin}  |  {fired}")


def classify(query: str) -> RoutingDecision:
    """Label a query as exact-fact, conceptual, or ambiguous.

    Weights are deliberately coarse. A detected year is the single most
    reliable exact-fact signal in this corpus, because the whole reason
    edition filtering exists is that year-bearing queries target one specific
    filing; open-ended phrasing is the mirror image on the conceptual side.
    Everything else is supporting evidence worth one or two points.
    """
    signals: list[Signal] = []
    spec = MD.parse_query(query)

    # -- exact-fact evidence, reusing the Phase 4 metadata extraction --------
    # A year is weak evidence of question *type*. It reliably indicates that a
    # specific edition is being targeted -- which is what edition filtering
    # keys on -- but conceptual questions name years just as often ("the
    # January 2025 Beige Book"). Weighting it strongly routed 8 of 9
    # conceptual questions to the exact-fact path.
    if spec.has_year:
        years = ", ".join(str(y) for y in spec.years)
        signals.append(Signal("year", EXACT_FACT, 1, years))
    if spec.entity:
        signals.append(Signal("company", EXACT_FACT, 3, spec.entity))
    if m := _FIGURE_PHRASES.search(query):
        signals.append(Signal("figure-phrase", EXACT_FACT, 3, m.group(0).lower()))
    if m := _LINE_ITEMS.search(query):
        signals.append(Signal("line-item", EXACT_FACT, 2, m.group(0).lower()))
    if m := _MAGNITUDE.search(query):
        signals.append(Signal("magnitude", EXACT_FACT, 1, m.group(0).lower()))

    # -- conceptual evidence -------------------------------------------------
    if m := _INSTITUTIONS.search(query):
        signals.append(Signal("institution", CONCEPTUAL, 3, m.group(0).lower()))
    if m := _OPEN_PHRASES.search(query):
        signals.append(Signal("open-phrase", CONCEPTUAL, 3, m.group(0).lower()))
    if m := _THEME_WORDS.search(query):
        signals.append(Signal("theme-word", CONCEPTUAL, 2, m.group(0).lower()))
    if not spec.has_year and not spec.entity:
        signals.append(Signal("no-year-no-company", CONCEPTUAL, 1))

    scores = {
        EXACT_FACT: sum(s.weight for s in signals if s.route == EXACT_FACT),
        CONCEPTUAL: sum(s.weight for s in signals if s.route == CONCEPTUAL),
    }

    diff = scores[EXACT_FACT] - scores[CONCEPTUAL]
    if diff >= DECISION_MARGIN:
        route = EXACT_FACT
    elif -diff >= DECISION_MARGIN:
        route = CONCEPTUAL
    else:
        route = AMBIGUOUS

    return RoutingDecision(query=query, route=route, signals=signals,
                           scores=scores, spec=spec)


# --- Retrieval entry point -------------------------------------------------
def route_and_retrieve(query: str, k: int = 5, fetch_k: int | None = None,
                       candidates: int | None = None, route: str | None = None):
    """Classify `query`, then retrieve with the strategy its route implies.

    Returns (documents, decision) so the caller can report *why* a query was
    handled the way it was. Imports are deferred to keep this module free of a
    circular dependency on rerank, which itself imports hybrid.

    `route` overrides classification when the caller already knows the question
    type from a more reliable source than the query text. Report generation is
    the motivating case: rewriting a sub-question to be self-contained forces
    the company name into it, which fires the +3 `company` signal and drags
    conceptual sub-questions onto the exact-fact path. The caller that *wrote*
    the sub-question knows what it was asking for; the text no longer does.
    The text-derived decision is still computed and kept in `signals`, so an
    override never hides what classification would have chosen.
    """
    from src.retrieval import hybrid
    from src.retrieval import rerank as rerank_mod

    decision = classify(query)
    if route is not None:
        if route not in ROUTE_STRATEGY:
            raise ValueError(f"Unknown route {route!r}; expected one of "
                             f"{list(ROUTE_STRATEGY)}")
        decision = RoutingDecision(
            query=query, route=route,
            signals=[Signal("supplied", route, 0, "by caller"),
                     Signal("text-would-be", decision.route, 0,
                            f"exact={decision.scores.get(EXACT_FACT, 0)} "
                            f"concept={decision.scores.get(CONCEPTUAL, 0)}")],
            scores=decision.scores, spec=decision.spec)
    plan = decision.strategy
    fetch_k = fetch_k or hybrid.DEFAULT_FETCH_K
    candidates = candidates or rerank_mod.DEFAULT_RERANK_CANDIDATES

    if plan["rerank"]:
        hits = rerank_mod.retrieve_and_rerank(
            query, k=k, candidates=candidates, fetch_k=fetch_k,
            year_filter=plan["filter"])
    else:
        hits = hybrid.hybrid_search(query, k=k, fetch_k=fetch_k,
                                    year_filter=plan["filter"])
    return [h.document for h in hits], decision


# --- Validation ------------------------------------------------------------
def validate(path: Path | None = None) -> dict:
    """Score the router against the labelled categories in questions.yaml.

    The evaluation set already labels every Track A question exact_fact or
    conceptual, which is exactly the distinction this router makes -- so it
    doubles as free ground truth. Adversarial questions are excluded: they are
    labelled by what the system should *refuse*, not by retrieval strategy.
    """
    import yaml

    path = path or config.PROJECT_ROOT / "data" / "eval" / "questions.yaml"
    questions = yaml.safe_load(path.read_text(encoding="utf-8"))["questions"]

    rows, correct, ambiguous = [], 0, 0
    for q in questions:
        if q["category"] not in ("exact_fact", "conceptual"):
            continue
        truth = EXACT_FACT if q["category"] == "exact_fact" else CONCEPTUAL
        d = classify(q["question"])
        hit = d.route == truth
        correct += hit
        ambiguous += d.route == AMBIGUOUS
        rows.append((q["id"], truth, d, hit))

    n = len(rows)
    return {"rows": rows, "n": n, "correct": correct, "ambiguous": ambiguous,
            "accuracy": correct / n if n else 0.0}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Classify a query and show why.")
    parser.add_argument("query", nargs="*", help="query text")
    parser.add_argument("--validate", action="store_true",
                        help="score routing against the labelled eval set")
    args = parser.parse_args(argv)

    config.enable_utf8_stdout()

    if args.validate:
        r = validate()
        print(f"{'id':<8}{'truth':<12}{'routed':<12}{'ok':<4}signals")
        print("-" * 104)
        for qid, truth, d, hit in r["rows"]:
            print(f"{qid:<8}{truth:<12}{d.route:<12}{'Y' if hit else 'n':<4}"
                  f"{'; '.join(str(s) for s in d.signals)[:64]}")
        print(f"\naccuracy {r['correct']}/{r['n']} = {r['accuracy']:.3f} "
              f"| routed AMBIGUOUS: {r['ambiguous']}")
        return 0

    if not args.query:
        parser.error("give a query, or use --validate")
    for q in args.query:
        print(f"\n{q!r}\n  {classify(q).explain()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
