"""Two-track evaluation harness for the Insight Analyst retrieval stack.

TRACK A (exact_fact + conceptual) -- questions with a known answer.
  * Retrieval metrics scored against the ground-truth chunk citations in
    questions.yaml: hit rate, precision, recall, MRR. Deterministic, no LLM.
  * Optionally RAGAS faithfulness / answer relevancy over a generated answer.

TRACK B (adversarial) -- questions that should NOT get a confident answer.
  Scored pass/fail by an LLM judge: did the system decline (unanswerable) or
  ask for clarification (ambiguous), rather than answering confidently?

The two tracks are reported separately and never averaged. They measure
different things: Track A asks "is the right context found and used", Track B
asks "does the system know when to say no". A single blended number would let
a system that refuses everything look good.

    python -m src.evaluation.run_eval                    # both configs, retrieval only
    python -m src.evaluation.run_eval --with-llm         # add generation + RAGAS + judge
    python -m src.evaluation.run_eval --no-rerank        # single config
    python -m src.evaluation.run_eval --limit 3 --with-llm   # smoke test

Generation and judging default to a local Ollama model (no API key). See
--llm-model; a stronger hosted model can be substituted if one is configured.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config
from src.evaluation import budget
from src.evaluation import judges
from src.evaluation import metrics as M
from src.retrieval import hybrid
from src.retrieval import rerank as rerank_mod
from src.retrieval import router

QUESTIONS_PATH = config.PROJECT_ROOT / "data" / "eval" / "questions.yaml"
RESULTS_DIR = config.PROJECT_ROOT / "data" / "eval" / "results"

# name -> (use_rerank, use_year_filter). The full matrix isolates the two
# independent variables: reranking reorders a candidate pool, edition filtering
# changes what enters it, and they interact -- filtering alters which chunks
# both retrievers agree on, which changes RRF, which changes what the reranker
# even sees.
# (name, use_rerank, use_filter, use_routing). When use_routing is True the
# other two are ignored: the router decides per query.
CONFIG_MATRIX = [
    ("hybrid only", False, False, False),
    ("hybrid+rerank", True, False, False),
    ("hybrid+filter", False, True, False),
    ("hybrid+filter+rerank", True, True, False),
    ("routed", False, False, True),
]

TRACK_A = {"exact_fact", "conceptual"}
TRACK_B = {"adversarial"}

DEFAULT_LLM = "gemma:2b"

ANSWER_PROMPT = """You are a research analyst. Answer the question using ONLY the \
numbered context below. Cite the chunk numbers you used, like [2].

If the context does not contain the answer, reply exactly:
NOT_IN_CONTEXT

If the question is too vague to answer (it does not say which company or which \
year, and the context covers several), reply exactly:
NEEDS_CLARIFICATION: <what you need to know>

Context:
{context}

Question: {question}
Answer:"""

JUDGE_PROMPT = """A question was asked of a document-retrieval system. Judge \
whether the system's response was APPROPRIATE.

The response is APPROPRIATE if it declined to answer, said the information was \
not available, or asked for clarification.
The response is INAPPROPRIATE if it confidently gave a specific factual answer.

Question: {question}
Expected behaviour: {expected}
System response: {answer}

Reply with exactly one word: YES if appropriate, NO if inappropriate."""


# --- Data ------------------------------------------------------------------
@dataclass
class QuestionResult:
    qid: str
    category: str
    question: str
    retrieved: list = field(default_factory=list)
    retrieval: M.RetrievalScore | None = None
    answer: str = ""
    judge_pass: bool | None = None
    seconds: float = 0.0
    route: str = ""


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    import yaml

    if not path.exists():
        raise FileNotFoundError(f"No question set at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["questions"]


# --- Pipeline --------------------------------------------------------------
def retrieve(question: str, k: int, use_rerank: bool, candidates: int,
             year_filter: bool = True, use_routing: bool = False):
    """Run the configured retrieval path and return plain Documents in order.

    With `use_routing`, the strategy is chosen per query by `router.classify`
    rather than fixed for the whole run. The decision is returned alongside so
    the harness can record which route each question took.
    """
    if use_routing:
        docs, decision = router.route_and_retrieve(question, k=k, candidates=candidates)
        return docs, decision
    if use_rerank:
        hits = rerank_mod.retrieve_and_rerank(question, k=k, candidates=candidates,
                                              year_filter=year_filter)
    else:
        hits = hybrid.hybrid_search(question, k=k, year_filter=year_filter)
    return [h.document for h in hits], None


def get_llm(model_name: str | None, provider: str = "ollama"):
    return judges.get_llm(provider=provider, model=model_name)


def generate_answer(llm, question: str, docs: list) -> str:
    context = "\n\n".join(
        f"[{i}] ({hybrid.citation(d)}) {' '.join(d.page_content.split())}"
        for i, d in enumerate(docs, start=1)
    )
    return llm.invoke(ANSWER_PROMPT.format(context=context, question=question)).content.strip()


def judge_refusal(llm, question: str, expected: str, answer: str) -> bool:
    """LLM-judge for Track B: was declining/clarifying the response given?"""
    verdict = llm.invoke(
        JUDGE_PROMPT.format(question=question, expected=expected, answer=answer)
    ).content.strip().upper()
    return verdict.startswith("YES")


# --- Runner ----------------------------------------------------------------
def run_config(questions: list[dict], use_rerank: bool, k: int, candidates: int,
               with_llm: bool, llm_model: str | None,
               provider: str = "ollama", year_filter: bool = True,
               use_routing: bool = False) -> list[QuestionResult]:
    llm = get_llm(llm_model, provider) if with_llm else None
    results: list[QuestionResult] = []

    for q in questions:
        t0 = time.perf_counter()
        docs, decision = retrieve(q["question"], k, use_rerank, candidates,
                                  year_filter, use_routing)
        res = QuestionResult(qid=q["id"], category=q["category"], question=q["question"],
                             retrieved=docs)
        res.retrieval = M.score_retrieval(docs, q.get("ground_truth") or [])
        res.route = decision.route if decision else ""

        if with_llm:
            res.answer = generate_answer(llm, q["question"], docs)
            if q["category"] in TRACK_B:
                res.judge_pass = judge_refusal(
                    llm, q["question"], q.get("expected_answer", ""), res.answer
                )

        res.seconds = time.perf_counter() - t0
        results.append(res)
        print(f"  {res.qid:<8} {res.seconds:6.1f}s  {res.route:<11}"
              f"{'hit' if res.retrieval and res.retrieval.hit else '-':<4}"
              f"{'' if res.judge_pass is None else ('PASS' if res.judge_pass else 'FAIL')}")
    return results


def _serialise(results: list[QuestionResult]) -> list[dict]:
    return [{"id": r.qid, "category": r.category, "seconds": round(r.seconds, 2),
             "first_rank": r.retrieval.first_rank if r.retrieval else None,
             "recall": r.retrieval.recall if r.retrieval else None,
             "answer": r.answer, "judge_pass": r.judge_pass, "route": r.route,
             "retrieved": [hybrid.citation(d) for d in r.retrieved]}
            for r in results]


def checkpoint(by_config: dict[str, list[QuestionResult]], path: Path,
               ragas: dict | None = None) -> None:
    """Persist everything finished so far.

    Written after every config rather than once at the end. A hosted judge can
    fail hours in -- a daily token quota is invisible until it 429s -- and
    losing three completed configs to the fourth one's failure is avoidable.
    Merges into any existing file so a resumed run accumulates.
    """
    payload = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a corrupt checkpoint must not block
            payload = {}
    if not isinstance(payload, dict):
        payload = {}
    for name, results in by_config.items():
        payload[name] = _serialise(results)
    if ragas:
        # Merge per config, exactly as the retrieval results above do. A
        # wholesale replace silently discards the judged scores of every
        # configuration not in the current run -- and because the retrieval
        # rows do survive, the loss stays invisible until those numbers are
        # needed. An --only re-run of one config cost the routed config's
        # faithfulness and answer relevancy exactly this way.
        existing = payload.get("_ragas")
        payload["_ragas"] = {**(existing if isinstance(existing, dict) else {}),
                             **ragas}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# --- Reporting -------------------------------------------------------------
def _fmt(x: float) -> str:
    return f"{x:.3f}"


def markdown_report(by_config: dict[str, list[QuestionResult]], with_llm: bool,
                    llm_model: str, ragas_by_config: dict | None = None) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Retrieval evaluation\n")
    add("> Generated by `src.evaluation.run_eval` and **overwritten on every "
        "run**. It covers only the configurations of the run that produced it, "
        "which may be a subset of the matrix (`--only`). The standing "
        "multi-configuration comparison and its per-run provenance live in "
        "`eval_results.md`, which the harness never overwrites.\n")
    add(f"Corpus: 16,917 chunks | embeddings: `{config.SENTENCE_TRANSFORMERS_MODEL}`")
    add(f"| LLM: `{llm_model if with_llm else 'not used (retrieval-only run)'}`\n")

    # ---- Track A ----------------------------------------------------------
    add("## Track A — answerable questions (exact-fact + conceptual)\n")
    add("Retrieval scored against the ground-truth chunk citations in "
        "`questions.yaml`. Deterministic set comparison, no LLM judge.\n")
    add("| config | n | hit rate | precision | recall | MRR | F1 |")
    add("|---|---|---|---|---|---|---|")
    for name, results in by_config.items():
        scores = [r.retrieval for r in results
                  if r.category in TRACK_A and r.retrieval is not None]
        a = M.aggregate(scores)
        add(f"| {name} | {a['n']} | {_fmt(a['hit_rate'])} | {_fmt(a['precision'])} "
            f"| {_fmt(a['recall'])} | {_fmt(a['mrr'])} | {_fmt(a['f1'])} |")
    add("")

    if ragas_by_config:
        add("### Generation quality (RAGAS, LLM-judged)\n")
        add("Judged metrics over the generated answer. Context precision/recall are "
            "NOT taken from RAGAS -- they are measured exactly against ground-truth "
            "chunk labels in the table above.\n")
        add("| config | faithfulness | answer relevancy |")
        add("|---|---|---|")
        for name, sc in ragas_by_config.items():
            add(f"| {name} | {_fmt(sc.get('faithfulness', float('nan')))} "
                f"| {_fmt(sc.get('answer_relevancy', float('nan')))} |")
        add("")

    # ---- By category ------------------------------------------------------
    add("### By question category\n")
    add("| category | config | n | hit rate | recall | MRR |")
    add("|---|---|---|---|---|---|")
    for cat in ("exact_fact", "conceptual"):
        for name, results in by_config.items():
            scores = [r.retrieval for r in results
                      if r.category == cat and r.retrieval is not None]
            a = M.aggregate(scores)
            add(f"| {cat} | {name} | {a['n']} | {_fmt(a['hit_rate'])} "
                f"| {_fmt(a['recall'])} | {_fmt(a['mrr'])} |")
    add("")

    # ---- Per question -----------------------------------------------------
    add("### Per question (rank of first ground-truth chunk; lower is better)\n")
    names = list(by_config)
    add("| id | category | " + " | ".join(names) + " |")
    add("|---|---|" + "---|" * len(names))
    first = next(iter(by_config.values()), [])
    ids = [r.qid for r in first if r.category in TRACK_A]
    for qid in ids:
        cells = []
        cat = ""
        for name in names:
            r = next(x for x in by_config[name] if x.qid == qid)
            cat = r.category
            cells.append(str(r.retrieval.first_rank) if r.retrieval and r.retrieval.first_rank
                         else "miss")
        add(f"| {qid} | {cat} | " + " | ".join(cells) + " |")
    add("")

    # ---- Track B ----------------------------------------------------------
    add("## Track B — adversarial (unanswerable + ambiguous)\n")
    if not with_llm:
        add("_Not run: Track B requires answer generation and an LLM judge "
            "(`--with-llm`)._\n")
    else:
        add("Pass = the system declined or asked for clarification instead of "
            "answering confidently. **Reported separately from Track A and never "
            "averaged with it.**\n")
        add("| config | n | passed | pass rate |")
        add("|---|---|---|---|")
        for name, results in by_config.items():
            b = [r for r in results if r.category in TRACK_B and r.judge_pass is not None]
            passed = sum(1 for r in b if r.judge_pass)
            rate = passed / len(b) if b else 0.0
            add(f"| {name} | {len(b)} | {passed} | {_fmt(rate)} |")
        add("")
        add("| id | subtype | " + " | ".join(names) + " |")
        add("|---|---|" + "---|" * len(names))
        bids = [r.qid for r in next(iter(by_config.values())) if r.category in TRACK_B]
        for qid in bids:
            cells = []
            for name in names:
                r = next(x for x in by_config[name] if x.qid == qid)
                cells.append("PASS" if r.judge_pass else "FAIL")
            add(f"| {qid} | | " + " | ".join(cells) + " |")
        add("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Two-track RAG evaluation.")
    parser.add_argument("-k", type=int, default=5, help="chunks retrieved per question")
    parser.add_argument("--candidates", type=int,
                        default=rerank_mod.DEFAULT_RERANK_CANDIDATES)
    parser.add_argument("--with-llm", action="store_true",
                        help="generate answers, run the Track B judge")
    parser.add_argument("--llm-provider", default="ollama", choices=list(judges.PROVIDERS),
                        help="ollama is local and needs no key; the rest need an API key")
    parser.add_argument("--llm-model", default=None,
                        help="override the provider's default model")
    parser.add_argument("--ragas", action="store_true",
                        help="also run RAGAS faithfulness + answer relevancy (implies --with-llm)")
    parser.add_argument("--no-rerank", action="store_true",
                        help="evaluate only the no-rerank config")
    parser.add_argument("--rerank-only", action="store_true")
    parser.add_argument("--no-year-filter", action="store_true",
                        help="disable edition filtering (A/B against the default)")
    parser.add_argument("--routing", action="store_true",
                        help="run the router config alone (per-query strategy)")
    parser.add_argument("--no-routing", action="store_true",
                        help="exclude the routed config from --matrix, so routing "
                             "can be A/B'd against the fixed configs")
    parser.add_argument("--ignore-budget", action="store_true",
                        help="start even if the local token counter says the run "
                             "cannot finish (use only if the counter is stale)")
    parser.add_argument("--matrix", action="store_true",
                        help="run all four configs: rerank x edition-filter")
    parser.add_argument("--only", default=None,
                        help="comma-separated config names from the matrix, e.g. "
                             "'hybrid+filter,hybrid+filter+rerank'. Results are merged "
                             "into the existing raw file, so configs measured earlier "
                             "with the same judge and settings need not be re-run.")
    parser.add_argument("--limit", type=int, default=0, help="first N questions (smoke test)")
    parser.add_argument("--out", default=None, help="markdown output path")
    args = parser.parse_args(argv)

    config.enable_utf8_stdout()
    rerank_mod.use_all_cores()

    if args.ragas:
        args.with_llm = True
    llm_label = (f"{args.llm_provider}:{args.llm_model or judges.default_model(args.llm_provider)}"
                 if args.with_llm else "not used (retrieval-only run)")

    questions = load_questions()
    if args.limit:
        questions = questions[:args.limit]

    year_filter = not args.no_year_filter
    if args.only:
        wanted = [n.strip() for n in args.only.split(",") if n.strip()]
        known = {c[0]: c for c in CONFIG_MATRIX}
        unknown = [n for n in wanted if n not in known]
        if unknown:
            raise SystemExit(f"Unknown config(s) {unknown}; expected any of {list(known)}")
        configs = [known[n] for n in wanted]
    elif args.matrix:
        configs = [c for c in CONFIG_MATRIX if not (args.no_routing and c[3])]
    else:
        configs = [("hybrid+rerank", True, year_filter, False),
                   ("hybrid only", False, year_filter, False)]
        if args.no_rerank:
            configs = [("hybrid only", False, year_filter, False)]
        elif args.rerank_only:
            configs = [("hybrid+rerank", True, year_filter, False)]
        if args.routing:
            configs = [("routed", False, False, True)]

    print(f"{len(questions)} questions | k={args.k} | llm={llm_label}")

    # Refuse before spending anything if the run cannot finish. Groq's daily
    # cap is invisible to the API, and a run that aborts halfway produces
    # scores that silently rest on fewer samples than a clean run.
    if args.with_llm:
        n_a = sum(1 for q in questions if q["category"] in TRACK_A)
        n_b = sum(1 for q in questions if q["category"] in TRACK_B)
        projected = budget.estimate_run(
            n_questions=len(questions), n_track_a=n_a, n_track_b=n_b,
            n_configs=len(configs), with_llm=True, with_ragas=bool(args.ragas))
        budget.ensure_budget(projected, args.llm_provider, args.ignore_budget)

    by_config: dict[str, list[QuestionResult]] = {}
    # A retrieval-only run must not overwrite a raw file holding generated
    # answers: a cheap sanity check should never destroy an expensive run.
    raw_path = RESULTS_DIR / ("eval_raw.json" if args.with_llm
                              else "eval_raw_retrieval.json")

    for name, use_rerank, use_filter, use_routing in configs:
        print(f"\n--- {name}  (rerank={use_rerank}, filter={use_filter}, "
              f"routing={use_routing}) ---")
        t0 = time.perf_counter()
        try:
            by_config[name] = run_config(questions, use_rerank, args.k, args.candidates,
                                         args.with_llm, args.llm_model, args.llm_provider,
                                         use_filter, use_routing)
        except Exception as exc:  # noqa: BLE001 - keep whatever already succeeded
            print(f"  ABORTED: {type(exc).__name__}: {str(exc)[:220]}")
            checkpoint(by_config, raw_path)
            print(f"  partial results checkpointed to {raw_path}")
            if "rate_limit" in str(exc) or "429" in str(exc):
                print("  (hosted-judge quota exhausted; completed configs are saved)")
            break
        print(f"  ({time.perf_counter() - t0:.1f}s total)")
        checkpoint(by_config, raw_path)

    ragas_by_config: dict[str, dict] = {}
    if args.ragas:
        llm = get_llm(args.llm_model, args.llm_provider)
        for name, results in by_config.items():
            samples = [
                {"question": r.question, "answer": r.answer,
                 "contexts": [d.page_content for d in r.retrieved],
                 "reference": next((q.get("expected_answer", "") for q in questions
                                    if q["id"] == r.qid), "")}
                for r in results if r.category in TRACK_A and r.answer
            ]
            print(f"\n--- RAGAS: {name} ({len(samples)} samples) ---")
            ragas_by_config[name] = judges.ragas_scores(samples, llm)

    if not by_config:
        print("\nNo configuration completed -- nothing to report. Existing "
              f"results in {raw_path} were left untouched.")
        return 1

    report = markdown_report(by_config, args.with_llm, llm_label, ragas_by_config)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Tables go to eval_tables.md, NOT eval_results.md. eval_results.md is the
    # curated write-up that interprets these numbers, and a re-run would
    # otherwise silently destroy that analysis.
    out = Path(args.out) if args.out else RESULTS_DIR / "eval_tables.md"
    out.write_text(report, encoding="utf-8")

    checkpoint(by_config, raw_path, ragas_by_config or None)

    if args.with_llm and args.llm_provider != "ollama":
        print("\n" + budget.describe())

    print("\n" + report)
    print(f"\nwritten: {out}\n         {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
