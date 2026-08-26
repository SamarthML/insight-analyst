"""Local Streamlit front end over the routed retrieval pipeline.

Two modes, because the system does two genuinely different things. "Quick
answer" is one query through `router.route_and_retrieve` plus generation --
seconds. "Generate report" decomposes a topic and runs several sub-queries
before synthesising, which takes minutes against a rate-limited hosted judge.
Presenting them identically would make the slow one look broken, so each mode
sets its own expectation in the spinner.

DECLINING IS NOT AN ERROR
The pipeline answers `NOT_IN_CONTEXT` when retrieval missed and
`NEEDS_CLARIFICATION` when a question is ambiguous. Both are the system working
correctly -- Track B measures exactly this behaviour -- so they render as
informational notices, never as exceptions. Showing a refusal in red would
train the user to distrust the one thing the system reliably gets right.

Single user, no auth, no persistence. Retrieval, embeddings and reranking stay
local; only generation leaves the machine, and only when a hosted provider is
selected.

    streamlit run src/ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Streamlit executes this file as a script, not as a package member, so the
# project root has to be on the path before any `src.` import resolves.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation import budget, judges          # noqa: E402
from src.reporting import generate_report as R     # noqa: E402
from src.retrieval import hybrid, router            # noqa: E402

CORPUS_NOTE = ("Searches SEC 10-K filings (Apple, Microsoft, NVIDIA, JPMorgan, "
               "Coca-Cola), Federal Reserve Beige Books, and World Bank / BIS "
               "economic reports.")

ROUTE_EXPLAIN = {
    router.EXACT_FACT: "Edition filtering on, reranking off — the question "
                       "names a specific reported figure, so narrowing to the "
                       "right filing matters more than re-scoring relevance.",
    router.CONCEPTUAL: "Reranking on, edition filtering off — the question is "
                       "thematic, where paraphrase and semantic drift are the "
                       "dominant failure mode.",
    router.AMBIGUOUS: "Both filtering and reranking on — the routing signals "
                      "were weak or contradictory, so the safer combined path "
                      "is used.",
}


# --- Cached resources ------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_llm(provider: str, model: str | None):
    """One client per provider, reused across reruns.

    Streamlit re-executes this script top to bottom on every interaction, so
    without caching each keystroke would rebuild the chat model -- and with it
    the rate limiter, whose whole job is to hold state between calls.
    """
    return judges.get_llm(provider=provider, model=model or None)


@st.cache_resource(show_spinner=False)
def warm_retrievers():
    """Load the index once. The first call pays the embedding model's cold
    start, which is ~90s on this machine and would otherwise be charged to
    whichever query happened to be first."""
    return hybrid.load_retrievers()


# --- Shared rendering ------------------------------------------------------
def render_refusal(answer: str) -> bool:
    """Show a decline or clarification request as information, not failure.

    Returns True when the answer was a refusal and has been rendered.
    """
    text = answer.strip()
    if text.upper().startswith(R.NOT_IN_CONTEXT):
        st.info(
            "**No answer in this corpus.** Retrieval found documents, but not "
            "the passage that answers this question — so the system declined "
            "rather than inferring one. Try naming the company and the fiscal "
            "year explicitly, or rephrasing toward what a filing would state."
        )
        return True
    if text.upper().startswith(R.NEEDS_CLARIFICATION):
        detail = text.split(":", 1)[1].strip() if ":" in text else ""
        st.warning("**Needs clarification.** " +
                   (detail or "The question is ambiguous as asked.") +
                   "\n\nThe corpus covers several companies and years, so this "
                   "question has more than one valid answer.")
        return True
    return False


# --- Mode: quick answer ----------------------------------------------------
def quick_answer_tab(llm, provider: str, k: int) -> None:
    st.subheader("Quick answer")
    st.caption("One question, answered from the corpus with citations.")

    question = st.text_input(
        "Question",
        placeholder="What was NVIDIA's revenue in fiscal year 2026?",
        key="quick_q",
    )
    if not st.button("Search", key="quick_go", type="primary") or not question.strip():
        return

    with st.spinner("Searching…"):
        warm_retrievers()
        # Reuses the report path's single-question step, so a quick answer and
        # a report section are produced by identical code -- same prompt, same
        # routing, same edition preference.
        sa = R.answer_subquestion(question, llm, provider, k=k)

    st.divider()
    if not render_refusal(sa.answer):
        st.markdown(sa.answer)
        st.caption("Sources")
        for i, c in enumerate(sa.citations, start=1):
            st.caption(f"{i}. `{c}`")

    with st.expander("How this was answered"):
        st.markdown(f"**Route:** `{sa.route}` — {ROUTE_EXPLAIN[sa.route]}")
        plan = router.ROUTE_STRATEGY[sa.route]
        st.markdown(
            f"**Edition filtering:** {'on' if plan['filter'] else 'off'}  \n"
            f"**Reranking:** {'on' if plan['rerank'] else 'off'}  \n"
            f"**Chunks retrieved:** {len(sa.citations)}  \n"
            f"**Elapsed:** {sa.seconds:.1f}s"
        )
        st.caption("Routing signals")
        st.code(sa.explain, language=None)


# --- Mode: report ----------------------------------------------------------
def report_tab(llm, provider: str, k: int, n_subquestions: int) -> None:
    st.subheader("Generate report")
    st.caption("A topic is split into sub-questions, each routed and answered "
               "separately, then synthesised into one document.")

    topic = st.text_input(
        "Report topic",
        placeholder="NVIDIA's revenue growth and risk factors",
        key="report_topic",
    )
    if not st.button("Generate", key="report_go", type="primary") or not topic.strip():
        return

    projected = R.estimate_report(n_subquestions)
    try:
        budget.ensure_budget(projected, provider)
    except SystemExit as exc:
        # The guard raises to abort a CLI run; in the UI it is a message.
        st.error(str(exc))
        return

    status = st.status("Running sub-queries…", expanded=True)
    try:
        with status:
            st.write("Loading the index…")
            warm_retrievers()
            st.write(f"Decomposing the topic into {n_subquestions} sub-questions…")
            questions = R.decompose(topic, llm, provider, n_subquestions)
            for i, (q, tag) in enumerate(questions, start=1):
                st.write(f"**{i}. [{tag or 'untagged'}]** {q}")

            sub_answers = []
            for i, (q, tag) in enumerate(questions, start=1):
                st.write(f"Running sub-query {i} of {len(questions)}…")
                sub_answers.append(R.answer_subquestion(
                    q, llm, provider, k=k, route=tag))

            st.write("Synthesising the report…")
            intro, links = R.synthesise(topic, sub_answers, llm, provider)
    except Exception as exc:  # noqa: BLE001 - surface it in the page, not a traceback
        status.update(label="Report failed", state="error")
        st.error(f"{type(exc).__name__}: {exc}")
        return

    index = R._build_source_index(sub_answers)
    elapsed = sum(s.seconds for s in sub_answers)
    markdown = R.render(topic, sub_answers, intro, links, index, elapsed)
    status.update(label=f"Report ready ({len(questions)} sub-questions)",
                  state="complete", expanded=False)

    st.divider()
    st.markdown(markdown)

    gaps = [s for s in sub_answers if s.is_gap]
    if gaps:
        st.info(f"{len(gaps)} of {len(sub_answers)} sub-questions could not be "
                f"answered from the corpus. They are listed in the report "
                f"rather than dropped, so its coverage stays visible.")


# --- Page ------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Insight Analyst", page_icon="🔎",
                       layout="centered")
    st.title("Insight Analyst")

    with st.sidebar:
        st.header("Settings")
        provider = st.selectbox(
            "Generation model", list(judges.PROVIDERS),
            index=list(judges.PROVIDERS).index("groq"),
            help="`ollama` runs locally and needs no key. The rest need an "
                 "API key in .env and send prompt text off the machine.")
        model = st.text_input("Model override", value="",
                              placeholder=judges.default_model(provider))
        k = st.slider("Chunks per question", 3, 10, 5)
        n_sub = st.slider("Sub-questions per report",
                          R.MIN_SUBQUESTIONS, R.MAX_SUBQUESTIONS,
                          R.DEFAULT_SUBQUESTIONS)

        if provider != "ollama":
            u = budget.snapshot()
            st.caption(f"Token budget today: {u.used:,} / {budget.DAILY_LIMIT:,} "
                       f"used · {u.spendable:,} spendable")
            st.progress(min(1.0, u.used / budget.DAILY_LIMIT))

    try:
        llm = get_llm(provider, model)
    except SystemExit as exc:
        # get_llm raises SystemExit with setup instructions when a key is
        # missing. That is guidance, not a crash.
        st.warning(str(exc))
        st.stop()

    quick, report = st.tabs(["Quick answer", "Generate report"])
    with quick:
        quick_answer_tab(llm, provider, k)
    with report:
        report_tab(llm, provider, k, n_sub)

    st.divider()
    st.caption(CORPUS_NOTE)


main()
