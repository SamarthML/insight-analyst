"""Local token accounting for the hosted judge, with a pre-run budget guard.

WHY THIS EXISTS
Groq's free tier enforces 200,000 tokens per day per model. That limit appears
in no response header -- only the per-minute buckets do -- so it is invisible
until a request fails with `tokens per day (TPD): Limit 200000, Used 199...`.
Three evaluation runs were lost to it: one crashed partway through its second
configuration, one silently dropped 11 of 44 scoring jobs and produced numbers
that could not be compared against a clean run, and one aborted after seven
questions.

A trivial "is the API up?" probe does not help, and actively misleads: a
ten-token ping succeeds when a thousand tokens remain, which reads as
clearance to start a hundred-thousand-token run. Reachability and headroom are
different questions.

Since the provider will not tell us what is left, we count locally. Every call
made through the harness adds its reported usage to a per-day counter, and a
run refuses to start unless its projected cost fits in what remains.

The counter is advisory. It cannot see usage from outside this harness, and a
crash between the API call and the write loses that call's tokens, so it drifts
low over time. It is therefore compared against a *conservative* estimate with
a safety margin, and the 429 remains the real backstop.
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config

STATE_PATH = config.PROJECT_ROOT / "data" / "eval" / "results" / "token_usage.json"

# Groq free tier, per model, per day. Confirmed from the 429 body.
DAILY_LIMIT = 200_000

# Headroom kept in reserve. The local counter drifts low (see module docstring),
# and a run that dies at 95% is worse than one that never started.
SAFETY_MARGIN = 15_000

# Per-question costs, calibrated from three completed runs that each consumed
# roughly 100k tokens for one 28-question configuration with RAGAS. Set
# slightly high on purpose: over-estimating refuses a run that might have just
# fit, which costs a day; under-estimating corrupts a run mid-flight, which
# costs a day *and* produces numbers that look valid but are not comparable.
GEN_TOKENS_PER_QUESTION = 1_900      # answer generation, ~5 chunks of context
JUDGE_TOKENS_PER_QUESTION = 600      # Track B refusal judge
RAGAS_TOKENS_PER_SAMPLE = 2_200      # faithfulness + answer relevancy together

_LOCK = threading.Lock()


@dataclass
class Usage:
    date: str
    used: int
    calls: int

    @property
    def remaining(self) -> int:
        return max(0, DAILY_LIMIT - self.used)

    @property
    def spendable(self) -> int:
        """What a new run may claim, keeping the safety margin in reserve."""
        return max(0, self.remaining - SAFETY_MARGIN)


def _today() -> str:
    return _dt.date.today().isoformat()


def _read() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a corrupt counter must never block a run
        return {}


def snapshot() -> Usage:
    """Current usage, rolling over automatically when the date changes.

    The rollover is a read-time comparison rather than a scheduled reset, so
    it works regardless of whether anything ran overnight.
    """
    data = _read()
    today = _today()
    if data.get("date") != today:
        return Usage(date=today, used=0, calls=0)
    return Usage(date=today, used=int(data.get("used", 0)),
                 calls=int(data.get("calls", 0)))


def record(tokens: int, calls: int = 1) -> Usage:
    """Add one call's reported usage to today's counter."""
    if tokens <= 0:
        return snapshot()
    with _LOCK:
        data = _read()
        today = _today()
        if data.get("date") != today:
            # Keep yesterday's total for context, then start fresh.
            history = data.get("history", {})
            if data.get("date"):
                history[data["date"]] = data.get("used", 0)
            data = {"date": today, "used": 0, "calls": 0,
                    "history": dict(list(history.items())[-14:])}
        data["used"] = int(data.get("used", 0)) + int(tokens)
        data["calls"] = int(data.get("calls", 0)) + calls
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return Usage(today, data["used"], data["calls"])


# --- Estimation ------------------------------------------------------------
def estimate_run(n_questions: int, n_track_a: int, n_track_b: int,
                 n_configs: int, with_llm: bool, with_ragas: bool) -> int:
    """Projected token cost of a run. Zero when no hosted call will be made."""
    if not with_llm:
        return 0
    per_config = n_questions * GEN_TOKENS_PER_QUESTION
    per_config += n_track_b * JUDGE_TOKENS_PER_QUESTION
    if with_ragas:
        per_config += n_track_a * RAGAS_TOKENS_PER_SAMPLE
    return per_config * n_configs


def ensure_budget(projected: int, provider: str, override: bool = False) -> None:
    """Refuse to start a run that cannot finish within today's budget.

    Only meaningful for the metered hosted providers; a local Ollama run has no
    quota and is never blocked.
    """
    if projected <= 0 or provider == "ollama":
        return

    usage = snapshot()
    if projected <= usage.spendable:
        print(f"budget: {usage.used:,} used today, {usage.remaining:,} left; "
              f"this run needs ~{projected:,} (margin {SAFETY_MARGIN:,} kept)")
        return

    message = (
        f"\nRefusing to start: this run would not finish.\n\n"
        f"  projected cost   ~{projected:,} tokens\n"
        f"  used today        {usage.used:,} of {DAILY_LIMIT:,}\n"
        f"  remaining         {usage.remaining:,}\n"
        f"  spendable         {usage.spendable:,}  (keeping a {SAFETY_MARGIN:,} margin)\n\n"
        f"Groq's daily cap is invisible to the API, so starting anyway would\n"
        f"abort partway and produce scores that cannot be compared against a\n"
        f"clean run. The counter rolls over at midnight local time.\n\n"
        f"Options: wait for the reset, run fewer configurations with --only,\n"
        f"drop --ragas, use --llm-provider ollama, or override with\n"
        f"--ignore-budget if you know the counter is stale."
    )
    if override:
        print(message.replace("Refusing to start", "WARNING (overridden)"))
        return
    raise SystemExit(message)


def describe() -> str:
    u = snapshot()
    return (f"token budget {u.date}: {u.used:,}/{DAILY_LIMIT:,} used "
            f"({u.calls} calls), {u.remaining:,} left, {u.spendable:,} spendable")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the local token counter.")
    parser.add_argument("--reset", action="store_true",
                        help="zero today's counter (use only if it is known stale)")
    args = parser.parse_args(argv)

    config.enable_utf8_stdout()
    if args.reset:
        data = _read()
        data.update({"date": _today(), "used": 0, "calls": 0})
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print("counter reset")

    print(describe())
    hist = _read().get("history", {})
    if hist:
        print("\nprevious days:")
        for day, used in sorted(hist.items())[-7:]:
            print(f"  {day}  {used:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
