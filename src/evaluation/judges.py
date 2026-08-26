"""LLM providers and RAGAS wiring for the evaluation harness.

Everything else in this project runs locally. Generation and judging are the
one exception, and only because they have to be: measured on this CPU-only
machine, gemma:2b takes 95s for a single RAG answer and still failed to extract
a figure sitting at rank 1 of its own context. RAGAS metrics additionally
require the judge to emit parseable structured output, which small local models
do not do reliably.

So the provider is pluggable. `ollama` remains the default and needs no key;
the hosted options are opt-in and are the only part of the pipeline that sends
corpus text off the machine. Retrieval, embeddings and reranking always stay
local -- including RAGAS's own embedding needs, which are wired to the same
local bge-small model rather than to a hosted embedding endpoint.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config
from src.evaluation import budget

# provider -> (env var holding the key, default model, pip package)
PROVIDERS = {
    "ollama":    (None,                "gemma:2b",                   "langchain-ollama"),
    "anthropic": ("ANTHROPIC_API_KEY", "claude-haiku-4-5-20251001",  "langchain-anthropic"),
    "openai":    ("OPENAI_API_KEY",    "gpt-4o-mini",                "langchain-openai"),
    # Model IDs are account-specific on Groq. This key exposes 13 models and
    # NOT llama-3.3-70b; gpt-oss-120b is the most capable chat model it can
    # reach (131k context). List what a key can actually use with
    # `groq.Groq().models.list()` rather than assuming an ID exists.
    "groq":      ("GROQ_API_KEY",      "openai/gpt-oss-120b",        "langchain-groq"),
}


# ~4 requests/min: 8,000 TPM divided by a typical ~2k-token request.
GROQ_REQUESTS_PER_SECOND = 0.075


class _UsageRecorder:
    """Callback that adds every completion's token usage to the local counter.

    Attached to the chat model itself rather than passed per call, so usage is
    captured wherever the model is invoked -- including inside RAGAS, which
    drives it through its own wrapper and would otherwise go uncounted. Since
    RAGAS accounts for roughly half of a run's tokens, missing those would make
    the counter useless.
    """

    def __init__(self):
        # Imported lazily so this module stays importable without langchain.
        from langchain_core.callbacks import BaseCallbackHandler
        self._base = BaseCallbackHandler

    @staticmethod
    def _tokens_from(response) -> int:
        out = getattr(response, "llm_output", None) or {}
        usage = out.get("token_usage") or out.get("usage") or {}
        total = usage.get("total_tokens")
        if total:
            return int(total)
        # Fall back to per-generation metadata when llm_output is empty.
        total = 0
        for batch in getattr(response, "generations", []) or []:
            for gen in batch:
                meta = getattr(getattr(gen, "message", None), "usage_metadata", None)
                if meta:
                    total += int(meta.get("total_tokens", 0) or 0)
        return total


def _make_usage_callback():
    """Build a LangChain callback handler recording usage into `budget`."""
    from langchain_core.callbacks import BaseCallbackHandler

    class UsageCallback(BaseCallbackHandler):
        def on_llm_end(self, response, **kwargs):  # noqa: D102
            try:
                tokens = _UsageRecorder._tokens_from(response)
                if tokens:
                    budget.record(tokens)
            except Exception:  # noqa: BLE001 - accounting must never break a run
                pass

    return UsageCallback()


def default_model(provider: str) -> str:
    return PROVIDERS[provider][1]


def get_llm(provider: str = "ollama", model: str | None = None, temperature: float = 0.0):
    """Build a chat model for generation and judging.

    Raises a directed error rather than a stack trace when a hosted provider is
    selected without its key, since that is the expected first-run failure.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}; expected one of {list(PROVIDERS)}")

    env_var, fallback_model, package = PROVIDERS[provider]
    model = model or fallback_model

    if env_var and not os.getenv(env_var):
        raise SystemExit(
            f"\n{provider} needs an API key, and {env_var} is not set.\n\n"
            f"Add it to the gitignored .env file in the project root:\n"
            f"    {env_var}=your-key-here\n\n"
            f"Do not paste the key into a chat or commit it. Use\n"
            f"    --llm-provider ollama\n"
            f"to stay fully local instead (slow, and low quality on this hardware)."
        )

    try:
        if provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(model=model, temperature=temperature, num_predict=256)
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model, temperature=temperature, max_tokens=1024,
                                 callbacks=[_make_usage_callback()])
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model, temperature=temperature,
                              callbacks=[_make_usage_callback()])
        from langchain_core.rate_limiters import InMemoryRateLimiter
        from langchain_groq import ChatGroq

        # Groq's free tier allows 8,000 tokens per minute across all models
        # (verified from x-ratelimit-limit-tokens; it is not model-specific).
        # A RAG prompt here runs ~1.5-2k tokens, so roughly four requests per
        # minute is the ceiling. The limiter paces the steady state and
        # max_retries absorbs the bursts -- the SDK's default of 2 retries is
        # not enough and fails the run outright partway through.
        return ChatGroq(
            model=model,
            temperature=temperature,
            callbacks=[_make_usage_callback()],
            max_retries=10,
            rate_limiter=InMemoryRateLimiter(
                requests_per_second=GROQ_REQUESTS_PER_SECOND,
                check_every_n_seconds=0.5,
                max_bucket_size=3,
            ),
        )
    except ImportError as exc:
        raise SystemExit(
            f"\n{provider} support needs `pip install {package}`.\n({exc})"
        ) from exc


def local_embeddings():
    """The same bge-small model the index was built with.

    RAGAS needs embeddings for answer relevancy. Left to itself it reaches for
    OpenAI, which would both require a key and score the answer in a different
    space from the one the corpus was indexed in.
    """
    from src.ingestion import vectorstore

    return vectorstore.get_embeddings()


def ragas_scores(samples: list[dict], llm, show_progress: bool = True) -> dict:
    """Run RAGAS faithfulness and answer relevancy over generated answers.

    Each sample needs: question, answer, contexts (list[str]), reference.
    Returns metric-name -> mean score. Context precision and recall are
    deliberately NOT computed here: this project has real ground-truth chunk
    labels, so they are measured exactly in metrics.py instead of being
    estimated by a judge.
    """
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, faithfulness

    # answer_relevancy defaults to strictness=3, which it implements by asking
    # for n=3 completions in ONE request. Groq (and several other providers)
    # cap n at 1 and reject that with "'n' : number must be at most 1", which
    # surfaces only as a silent NaN when raise_exceptions=False. Dropping to 1
    # generated question makes the metric slightly noisier per sample but is
    # the only way to run it on these providers.
    answer_relevancy.strictness = 1

    dataset = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=s["contexts"],
            reference=s.get("reference", ""),
        )
        for s in samples
    ])

    from ragas.run_config import RunConfig

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(local_embeddings()),
        show_progress=show_progress,
        raise_exceptions=False,   # a judge parse failure should not void the run
        # RAGAS fans out across samples by default. Against a token-per-minute
        # quota that guarantees 429s, so it is serialised here instead.
        # 300s was too short: a job sitting in rate-limit backoff would hit the
        # timeout and be discarded, losing 8 of 44 scores in one run. The wait
        # is expected behaviour against a token quota, not a hang.
        run_config=RunConfig(max_workers=1, max_retries=10, timeout=1200),
    )

    scores: dict[str, float] = {}
    for name, values in result._repr_dict.items() if hasattr(result, "_repr_dict") else []:
        scores[name] = values
    if not scores:
        try:
            df = result.to_pandas()
            for col in ("faithfulness", "answer_relevancy"):
                if col in df:
                    scores[col] = float(df[col].mean(skipna=True))
        except Exception:  # noqa: BLE001 - report what we have
            pass
    return scores
