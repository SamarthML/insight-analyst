"""Multi-section report generation over the routed retrieval pipeline.

WHY THIS EXISTS
The Phase 4b router answers one question at a time. A report topic ("NVIDIA's
revenue growth and risk factors") is not one question -- it is several, and they
route differently: the revenue half is exact-fact and wants edition filtering,
the risk-factor half is conceptual and wants reranking. Decomposing the topic
first and routing each sub-question independently is what lets a single report
use both strategies, which is exactly the dissociation Phase 4 measured.

The pipeline is: decompose -> route + retrieve + answer each sub-question ->
synthesise. Only the first and last steps are new; the middle one is
`router.route_and_retrieve` unchanged, so a report inherits whatever retrieval
quality the evaluation harness measured.

SHOWING GAPS RATHER THAN HIDING THEM
Retrieval hit rate on the evaluation set is 0.591. A report that silently drops
the sub-questions it could not answer would therefore look complete while
resting on a little over half its intended evidence, and the reader would have
no way to tell. Sub-answers that come back NOT_IN_CONTEXT or
NEEDS_CLARIFICATION are rendered as explicit gaps instead, and counted in the
report header. A visible gap is a finding; a hidden one is a defect.

TOKEN ACCOUNTING
Usage is recorded by the callback `judges.get_llm` attaches to the chat model,
so every call made through it lands in the shared daily counter automatically.
This module therefore must NOT call `budget.record` again on the same
completion -- that would count each call twice and drift the counter high,
refusing runs that would in fact have fit. What it does add is the pre-run
`ensure_budget` guard, which lives in `run_eval.main` and so did not cover this
path, plus a floor that records an estimate when a completion arrives with no
usage metadata at all (see `_invoke`).

    python -m src.reporting.generate_report "NVIDIA's revenue growth" \\
        --llm-provider groq
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config
from src.evaluation import budget
from src.evaluation import judges
from src.retrieval import hybrid
from src.retrieval import metadata as MD
from src.retrieval import router

# The generation prompt is imported rather than restated so the report path and
# the evaluated path cannot drift. The refusal sentinels below are only
# meaningful because this is the exact prompt the Track A/B numbers were
# measured with.
from src.evaluation.run_eval import ANSWER_PROMPT

NOT_IN_CONTEXT = "NOT_IN_CONTEXT"
NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"

DEFAULT_SUBQUESTIONS = 4
MIN_SUBQUESTIONS = 3
MAX_SUBQUESTIONS = 5

# Per-call token estimates, used for the pre-run guard and as the no-metadata
# floor. Deliberately set high: over-estimating declines a run that might have
# fit, under-estimating lets one abort halfway. Sub-question generation reuses
# the eval harness's own calibrated figure.
DECOMPOSE_TOKENS = 600
SUBQUESTION_TOKENS = budget.GEN_TOKENS_PER_QUESTION
SYNTHESIS_TOKENS = 2_000

DECOMPOSE_PROMPT = """You are a research analyst planning a report on the topic \
below. Break it into {n} focused sub-questions that together cover the topic.

Rules:
- Each sub-question must be self-contained and answerable on its own. Name the \
company or institution explicitly in every one -- do not write "its" or "that \
report".
- Include a time period ONLY if the topic states or clearly implies one. Never \
invent a year, quarter or edition the topic does not mention, and never assume \
a publication is issued quarterly or annually unless the topic says so.
- Prefer specific questions over broad ones. "What was NVIDIA's data centre \
revenue in fiscal 2026?" is better than "How did NVIDIA perform?".
- Cover different angles rather than restating one question {n} ways.
- Tag each sub-question with the kind of answer it needs:
    [exact-fact]  a specific reported figure, date or named quantity
    [conceptual]  an explanation, theme, risk, trend or comparison
- Output ONLY the sub-questions, one per line, in the form
  "1. [exact-fact] <question>" -- no preamble, no commentary, no summary.

Topic: {topic}"""

SYNTHESIS_PROMPT = """You are a research analyst assembling a report on the \
topic below. You are given the sub-questions investigated and the findings for \
each.

Write TWO things:

1. An introduction of 2-4 sentences that states what the report covers and \
previews what the findings show. Do not invent findings that are not listed.
2. For each section, one or two sentences of connective prose that introduce \
that section and relate it to the topic or to the section before it. Do not \
restate the finding itself -- it is printed directly beneath your prose.

Use EXACTLY this output format, with no other text:

INTRO: <your introduction>
LINK 1: <connective prose for section 1>
LINK 2: <connective prose for section 2>
{link_slots}

Rules:
- Never introduce a fact, figure or citation that does not appear below.
- Where a section is marked UNANSWERED, your connective prose must acknowledge \
the gap honestly rather than papering over it.
- Do not use citation markers like [1] in your prose.

Topic: {topic}

Findings:
{findings}"""


# --- Data ------------------------------------------------------------------
@dataclass
class SubAnswer:
    """One sub-question, its route, its answer and the chunks behind it."""

    question: str
    answer: str = ""
    route: str = ""
    text_route: str = ""
    route_source: str = ""
    explain: str = ""
    citations: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def unanswered(self) -> bool:
        return self.answer.strip().upper().startswith(NOT_IN_CONTEXT)

    @property
    def needs_clarification(self) -> bool:
        return self.answer.strip().upper().startswith(NEEDS_CLARIFICATION)

    @property
    def is_gap(self) -> bool:
        return self.unanswered or self.needs_clarification

    @property
    def gap_note(self) -> str:
        """How this gap should be described to the reader."""
        if self.unanswered:
            return ("**Not answerable from this corpus.** Retrieval returned "
                    "chunks that do not contain the answer, and the model "
                    "declined to answer rather than infer one.")
        detail = self.answer.split(":", 1)[1].strip() if ":" in self.answer else ""
        return ("**Ambiguous as asked.** The model asked for clarification "
                + (f"({detail}) " if detail else "")
                + "instead of answering, so this section is unresolved.")


@dataclass
class Report:
    topic: str
    markdown: str
    sub_answers: list[SubAnswer]
    sources: list[str]
    seconds: float = 0.0

    @property
    def gaps(self) -> list[SubAnswer]:
        return [s for s in self.sub_answers if s.is_gap]


# --- LLM plumbing ----------------------------------------------------------
def _invoke(llm, prompt: str, provider: str, fallback_tokens: int) -> str:
    """Call the model, guaranteeing the daily counter moves.

    `judges.get_llm` attaches a usage callback for the metered providers, so the
    normal path is already recorded and this must not record again. The counter
    is only nudged when a completion arrives carrying no usage metadata, which
    would otherwise let a run spend tokens invisibly and leave tomorrow's guard
    reading low. Local Ollama has no quota, so it is left alone entirely.
    """
    before = budget.snapshot().used
    response = llm.invoke(prompt)
    if provider != "ollama" and budget.snapshot().used == before:
        budget.record(fallback_tokens)
    return response.content.strip()


def estimate_report(n_subquestions: int, n_reports: int = 1) -> int:
    """Projected token cost, matching the shape of `budget.estimate_run`."""
    per_report = (DECOMPOSE_TOKENS
                  + n_subquestions * SUBQUESTION_TOKENS
                  + SYNTHESIS_TOKENS)
    return per_report * n_reports


# --- Stage 1: decomposition ------------------------------------------------
_NUMBERED = re.compile(
    r"^\s*\d+[.)]\s*(?:[\[\(](?P<tag>exact.?fact|conceptual)[\]\)])?\s*"
    r"(?P<q>.+?)\s*$", re.I)


def decompose(topic: str, llm, provider: str = "groq",
              n: int = DEFAULT_SUBQUESTIONS) -> list[tuple[str, str | None]]:
    """Split a report topic into self-contained sub-questions with type tags.

    Returns (question, route) pairs. `route` is the type the decomposer
    assigned, or None when it did not supply a parseable one -- in which case
    the caller falls back to `router.classify`, so a malformed tag costs
    routing quality rather than the whole report.

    The model is asked for exactly `n`, but its count is not trusted: the
    numbered lines are parsed out and clamped. A decomposition that silently
    returned nine sub-questions would cost more than twice its budgeted tokens.
    """
    raw = _invoke(llm, DECOMPOSE_PROMPT.format(topic=topic, n=n),
                  provider, DECOMPOSE_TOKENS)

    out: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        if m := _NUMBERED.match(line):
            q = m.group("q").strip().strip("*").strip()
            if not q or q in seen:
                continue
            tag = (m.group("tag") or "").lower()
            route = (router.EXACT_FACT if tag.startswith("exact")
                     else router.CONCEPTUAL if tag else None)
            seen.add(q)
            out.append((q, route))

    if not out:
        # Fall back to any non-empty line that reads like a question rather
        # than failing the whole report on a formatting miss.
        out = [(ln.strip(), None) for ln in raw.splitlines()
               if ln.strip().endswith("?")]
    if not out:
        raise SystemExit(
            f"\nCould not parse sub-questions from the decomposition of "
            f"{topic!r}.\nModel returned:\n{raw[:400]}"
        )
    return out[:MAX_SUBQUESTIONS]


# --- Stage 2: answer each sub-question through the router ------------------
def answer_subquestion(question: str, llm, provider: str = "groq",
                       k: int = 5, candidates: int | None = None,
                       route: str | None = None) -> SubAnswer:
    """Route, retrieve and answer one sub-question.

    Retrieval is `router.route_and_retrieve`, so this inherits the Phase 4b
    strategies rather than re-implementing them. `route` is the decomposer's
    own tag for the sub-question it wrote, which is more reliable than
    re-deriving the type from text that has had a company name forced into it.
    The text-derived route is recorded alongside, so a disagreement between the
    two is visible rather than silent.
    """
    t0 = time.perf_counter()
    text_route = router.classify(question).route

    # A sub-question naming no year ("its most recent annual filing") retrieves
    # across every edition, and each section then settles on whichever ranked
    # highest -- producing a report whose sections quietly describe different
    # filings. The newest edition is preferred instead.
    #
    # The pool has to be widened first. Measured on this corpus, NVIDIA's top-5
    # for such a question holds a single FY2026 chunk among four FY2024 ones,
    # so reordering the five promotes it to first and leaves the context still
    # four-fifths wrong-edition -- the model then answers from the majority.
    # At k=20 there are five FY2026 chunks, enough to fill the answer context.
    # Reordering after truncation cannot surface a chunk that never made the
    # cut, so the widening is the part that does the work.
    #
    # Year-bearing sub-questions keep the original path exactly: edition
    # filtering has already done this job, and `prefer_latest_edition` is a
    # no-op for them.
    spec = MD.parse_query(question)
    pool = k if spec.has_year else max(k * 4, 20)
    docs, decision = router.route_and_retrieve(
        question, k=pool, candidates=max(candidates or 0, pool) or None,
        route=route)
    docs = MD.prefer_latest_edition(docs, spec)[:k]

    context = "\n\n".join(
        f"[{i}] ({hybrid.citation(d)}) {' '.join(d.page_content.split())}"
        for i, d in enumerate(docs, start=1)
    )
    answer = _invoke(llm, ANSWER_PROMPT.format(context=context, question=question),
                     provider, SUBQUESTION_TOKENS)

    return SubAnswer(
        question=question,
        answer=answer,
        route=decision.route if decision else "",
        text_route=text_route,
        route_source="decomposer" if route else "classifier",
        explain=decision.explain() if decision else "",
        citations=[hybrid.citation(d) for d in docs],
        seconds=time.perf_counter() - t0,
    )


# --- Stage 3: synthesis ----------------------------------------------------
# The prompt asks for "[2]" but the model frequently emits full-width CJK
# brackets instead. Matching only ASCII silently left those markers
# unremapped, so one document could mix report-wide and per-section
# numbering with no visible difference to the reader -- worse than either
# numbering alone, since both render as a small integer.
_CITE = re.compile(r"[\[【](\d+(?:\s*,\s*\d+)*)[\]】]")


def _remap_citations(answer: str, local: list[str],
                     global_index: dict[str, int]) -> str:
    """Rewrite an answer's local [n] markers to the report-wide numbering.

    Each sub-question numbers its own context from 1, so the same marker means
    different sources in different sections. Left alone, the consolidated
    Sources list at the end would be unusable -- [2] in section 1 and [2] in
    section 3 are generally different chunks.
    """
    def sub(m: re.Match) -> str:
        out = []
        for part in m.group(1).split(","):
            try:
                idx = int(part.strip())
            except ValueError:
                continue
            if 1 <= idx <= len(local) and local[idx - 1] in global_index:
                out.append(str(global_index[local[idx - 1]]))
        return "[" + ", ".join(out) + "]" if out else ""

    return _CITE.sub(sub, answer)


def _cited_locals(answer: str, n_local: int) -> list[int]:
    """Local chunk numbers an answer actually references, in first-seen order."""
    seen: list[int] = []
    for m in _CITE.finditer(answer):
        for part in m.group(1).split(","):
            try:
                i = int(part.strip())
            except ValueError:
                continue
            if 1 <= i <= n_local and i not in seen:
                seen.append(i)
    return seen


def _build_source_index(sub_answers: list[SubAnswer]) -> dict[str, int]:
    """Assign one stable number per cited chunk, in first-seen order.

    Two exclusions matter. Sections the model declined to answer contribute
    nothing, since listing their retrieved chunks would credit the report with
    evidence it did not use. And within an answered section only the chunks the
    answer actually cites are listed: k=5 chunks are retrieved per
    sub-question but a typical answer leans on one or two, so numbering all of
    them inflates the source list severalfold and implies corroboration no one
    ever read.
    """
    index: dict[str, int] = {}
    for sa in sub_answers:
        if sa.is_gap:
            continue
        for i in _cited_locals(sa.answer, len(sa.citations)):
            c = sa.citations[i - 1]
            if c not in index:
                index[c] = len(index) + 1
    return index


_INTRO = re.compile(r"^INTRO:\s*(.+?)(?=^LINK\s+\d+:|\Z)", re.M | re.S)
_LINK = re.compile(r"^LINK\s+(\d+):\s*(.+?)(?=^LINK\s+\d+:|\Z)", re.M | re.S)


def synthesise(topic: str, sub_answers: list[SubAnswer], llm,
               provider: str = "groq") -> tuple[str, dict[int, str]]:
    """Ask for an introduction and per-section connective prose.

    The model writes the joining text only; the findings themselves are pasted
    in verbatim by `render`. Asking it to reproduce the answers as well would
    put every figure and citation through a paraphrase step, which is exactly
    where the ef-05 class of fabrication comes from.
    """
    findings = []
    for i, sa in enumerate(sub_answers, start=1):
        if sa.is_gap:
            state = "UNANSWERED -- " + (
                "not present in the corpus" if sa.unanswered
                else "the question was ambiguous as asked")
        else:
            state = " ".join(sa.answer.split())[:600]
        findings.append(f"Section {i}: {sa.question}\n  {state}")

    slots = "\n".join(f"LINK {i}: <connective prose for section {i}>"
                      for i in range(3, len(sub_answers) + 1))
    raw = _invoke(
        llm,
        SYNTHESIS_PROMPT.format(topic=topic, findings="\n\n".join(findings),
                                link_slots=slots),
        provider, SYNTHESIS_TOKENS)

    intro = ""
    if m := _INTRO.search(raw):
        intro = " ".join(m.group(1).split())
    links = {int(m.group(1)): " ".join(m.group(2).split())
             for m in _LINK.finditer(raw)}
    return intro, links


# --- Rendering -------------------------------------------------------------
def render(topic: str, sub_answers: list[SubAnswer], intro: str,
           links: dict[int, str], source_index: dict[str, int],
           seconds: float) -> str:
    """Assemble the markdown document deterministically.

    Everything the reader relies on -- figures, citations, gap notices -- is
    placed here rather than by the model, so a synthesis call that returns
    malformed output degrades the prose but cannot corrupt the findings.
    """
    answered = [s for s in sub_answers if not s.is_gap]
    gaps = [s for s in sub_answers if s.is_gap]

    out: list[str] = []
    add = out.append

    add(f"# {topic}\n")
    summary = (f"{len(answered)} of {len(sub_answers)} sub-questions answered "
               f"from the corpus")
    if gaps:
        verb = "is" if len(gaps) == 1 else "are"
        summary += f"; {len(gaps)} could not be answered and {verb} marked below"
    add(f"*{summary}. Generated in {seconds:.1f}s.*\n")

    if intro:
        add(intro + "\n")

    for i, sa in enumerate(sub_answers, start=1):
        add(f"## {i}. {sa.question}\n")
        if link := links.get(i):
            add(link + "\n")
        if sa.is_gap:
            add(sa.gap_note + "\n")
        else:
            add(_remap_citations(sa.answer, sa.citations, source_index) + "\n")

    if source_index:
        add("## Sources\n")
        add("Chunks retrieved and cited above, numbered consistently across "
            "sections.\n")
        for citation, n in sorted(source_index.items(), key=lambda kv: kv[1]):
            add(f"{n}. `{citation}`")
        add("")

    if gaps:
        add("## Gaps\n")
        add("Reported rather than dropped, so the report's coverage is "
            "visible:\n")
        for i, sa in enumerate(sub_answers, start=1):
            if sa.is_gap:
                kind = ("not in corpus" if sa.unanswered
                        else "ambiguous as asked")
                add(f"- **Section {i}** ({kind}): {sa.question}")
        add("")

    return "\n".join(out)


# --- Orchestration ---------------------------------------------------------
def generate_report(topic: str, provider: str = "groq", model: str | None = None,
                    n_subquestions: int = DEFAULT_SUBQUESTIONS, k: int = 5,
                    candidates: int | None = None, llm=None,
                    check_budget: bool = True, ignore_budget: bool = False,
                    verbose: bool = True) -> Report:
    """Decompose `topic`, answer each part through the router, synthesise.

    `llm` may be passed in so a caller generating several reports reuses one
    client -- and, with it, one rate limiter, which matters against Groq's
    per-minute bucket.
    """
    t0 = time.perf_counter()

    if check_budget:
        projected = estimate_report(n_subquestions)
        budget.ensure_budget(projected, provider, ignore_budget)

    llm = llm or judges.get_llm(provider=provider, model=model)

    if verbose:
        print(f"\n=== {topic} ===")
        print("  decomposing...")
    questions = decompose(topic, llm, provider, n_subquestions)
    if verbose:
        for i, (q, tag) in enumerate(questions, start=1):
            print(f"    {i}. [{tag or 'untagged'}] {q}")

    sub_answers: list[SubAnswer] = []
    for i, (q, tag) in enumerate(questions, start=1):
        if verbose:
            print(f"  [{i}/{len(questions)}] answering...", flush=True)
        sa = answer_subquestion(q, llm, provider, k=k, candidates=candidates,
                                route=tag)
        if verbose:
            state = ("NOT_IN_CONTEXT" if sa.unanswered else
                     "NEEDS_CLARIFICATION" if sa.needs_clarification else "ok")
            # Print both routes: a disagreement is the signal that the entity
            # name in the rewritten sub-question is dragging classification.
            agree = "" if sa.route == sa.text_route else f" (text: {sa.text_route})"
            print(f"      {sa.route:<12} via {sa.route_source:<10} "
                  f"{sa.seconds:5.1f}s  {state}{agree}")
        sub_answers.append(sa)

    if verbose:
        print("  synthesising...")
    intro, links = synthesise(topic, sub_answers, llm, provider)

    source_index = _build_source_index(sub_answers)
    seconds = time.perf_counter() - t0
    markdown = render(topic, sub_answers, intro, links, source_index, seconds)

    return Report(topic=topic, markdown=markdown, sub_answers=sub_answers,
                  sources=[c for c, _ in sorted(source_index.items(),
                                                key=lambda kv: kv[1])],
                  seconds=seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a multi-section report over the routed pipeline.")
    parser.add_argument("topics", nargs="+", help="one or more report topics")
    parser.add_argument("--llm-provider", default="groq",
                        choices=list(judges.PROVIDERS))
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("-n", "--subquestions", type=int,
                        default=DEFAULT_SUBQUESTIONS,
                        help=f"target sub-questions per report "
                             f"({MIN_SUBQUESTIONS}-{MAX_SUBQUESTIONS})")
    parser.add_argument("-k", type=int, default=5,
                        help="chunks retrieved per sub-question")
    parser.add_argument("--candidates", type=int, default=None)
    parser.add_argument("--out", default=None,
                        help="directory to write each report's markdown into")
    parser.add_argument("--ignore-budget", action="store_true",
                        help="start even if the local token counter says the "
                             "run cannot finish (use only if it is stale)")
    args = parser.parse_args(argv)

    config.enable_utf8_stdout()
    n = max(MIN_SUBQUESTIONS, min(MAX_SUBQUESTIONS, args.subquestions))

    # Guard once for the whole batch. Checking per report would let a two-topic
    # run clear the first check and then abort on the second, which is the
    # half-finished outcome the guard exists to prevent.
    projected = estimate_report(n, len(args.topics))
    budget.ensure_budget(projected, args.llm_provider, args.ignore_budget)

    llm = judges.get_llm(provider=args.llm_provider, model=args.llm_model)

    reports = []
    for topic in args.topics:
        reports.append(generate_report(
            topic, provider=args.llm_provider, model=args.llm_model,
            n_subquestions=n, k=args.k, candidates=args.candidates, llm=llm,
            check_budget=False))

    for r in reports:
        print("\n" + "=" * 78)
        print(r.markdown)

    if args.out:
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        for r in reports:
            slug = re.sub(r"[^a-z0-9]+", "-", r.topic.lower()).strip("-")[:60]
            path = outdir / f"{slug}.md"
            path.write_text(r.markdown, encoding="utf-8")
            print(f"written: {path}")

    if args.llm_provider != "ollama":
        print("\n" + budget.describe())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
