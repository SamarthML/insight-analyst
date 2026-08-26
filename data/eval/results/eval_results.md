# Retrieval evaluation — results and analysis

Runs 2026-08-23, 2026-08-25 and 2026-08-26. Corpus 16,917 chunks | embeddings
`BAAI/bge-small-en-v1.5` (local) | judge `groq:openai/gpt-oss-120b` | k=5 |
28 questions × 5 configs.

> This file is the curated write-up. The harness regenerates raw tables into
> `eval_tables.md` on every run; it does not overwrite this document.
>
> **STATUS (2026-08-26): complete, with one gap.** Retrieval metrics are final
> for all five configurations, `routed` included. LLM-judged metrics exist for
> all five, but `routed`'s faithfulness (0.969) and answer relevancy (0.699)
> are prior-run values from 2026-08-25 that **cannot be reproduced from the
> current repository state** -- see "The lost routed scores" below. A
> 2026-08-26 replicate of `hybrid+filter` establishes a noise floor for every
> judged number in this report -- see "Run-to-run variance", and read the
> generation-quality comparisons against it.
>
> The original 2026-08-23 status -- the two filter-on configurations pending on
> Groq's 200,000 tokens/day cap -- was resolved by the 2026-08-25 run.

---

## Headline

**Edition filtering and reranking fix different problems, and neither is a
general win.** Filtering carries exact-fact questions (hit 0.538 -> 0.692) and
does nothing for conceptual ones; reranking carries conceptual questions
(0.333 -> 0.444) and does nothing for exact-fact. Running both is *worse* on
exact-fact than filtering alone.

**Reranking improves retrieval metrics without improving answers.** On the two
questions where the filter-off configurations diverged, reranking lost one
correct answer and produced one confidently wrong one -- the entire
faithfulness drop from 1.000 to 0.843.

**Edition filtering fixed less than the diagnosis predicted.** 10 of 22 misses
were attributed to year discrimination; filtering recovers 2 of them outright
(ef-06, ef-07) and improves ranks on others. The rest turned out to be a
*different* failure hiding behind the same symptom -- see "Known limitation".

---

## Track A — answerable questions (exact-fact + conceptual, n=22)

Retrieval scored by set comparison against the ground-truth chunk citations in
`questions.yaml`. No LLM judge, no estimation.

| config | n | hit rate | precision | recall | MRR | F1 |
|---|---|---|---|---|---|---|
| hybrid only | 22 | 0.455 | 0.100 | 0.409 | 0.311 | 0.158 |
| hybrid+rerank | 22 | 0.500 | 0.100 | 0.432 | 0.345 | 0.160 |
| hybrid+filter | 22 | **0.545** | 0.118 | **0.500** | 0.350 | **0.188** |
| hybrid+filter+rerank | 22 | **0.545** | 0.109 | 0.477 | **0.356** | 0.175 |

Edition filtering beats reranking on every retrieval measure, and the two are
not additive: adding the reranker on top of the filter *lowers* recall
(0.500 -> 0.477).

### The two techniques fix different problems

| category | hybrid only | +rerank | +filter | +filter+rerank |
|---|---|---|---|---|
| **exact_fact** hit | 0.538 | 0.538 | **0.692** | 0.615 |
| **exact_fact** MRR | 0.385 | 0.372 | **0.451** | 0.391 |
| **conceptual** hit | 0.333 | **0.444** | 0.333 | **0.444** |
| **conceptual** MRR | 0.204 | **0.306** | 0.204 | **0.306** |

This is a clean dissociation, and it is the most useful result in the report.
Filtering moves exact-fact questions and does nothing at all for conceptual
ones; reranking moves conceptual questions and does nothing for exact-fact.
Each addresses a different failure, and the numbers for the other category are
identical to the baseline in both cases.

The combination is *worse* than filtering alone on exact-fact (0.692 -> 0.615
hit, 0.451 -> 0.391 MRR). The reranker demotes correct chunks that filtering
had already surfaced, because a wrong-edition chunk from a near-identical
filing scores as highly topical. On this evidence the reranker is worth running
only for conceptual queries, not as an unconditional stage.

**Read `precision` with care.** It is capped at `n_truth / k`. Most questions
have one ground-truth chunk and k=5, so the ceiling is 0.200, not 1.0. The
observed 0.100 means "about half of achievable", and it is identical across
configs because both retrieve the same number of chunks. Hit rate and MRR are
the interpretable columns.

### Generation quality (RAGAS, LLM-judged)

Covers 2 of the 4 configurations (see STATUS above).

| config | faithfulness | answer relevancy |
|---|---|---|
| hybrid only | **1.000** | 0.640 |
| hybrid+rerank | **0.843** | 0.626 |
| hybrid+filter | _pending_ | _pending_ |
| hybrid+filter+rerank | _pending_ | _pending_ |

Reranking scores *worse* on both. The next section explains why, because the
naive reading — "reranking makes the generator hallucinate" — is not quite
right.

### Per question (rank of first ground-truth chunk; lower is better)

| id | category | hybrid only | hybrid+rerank | hybrid+filter | hybrid+filter+rerank |
|---|---|---|---|---|---|
| ef-01 | exact_fact | 1 | 1 | 1 | 1 |
| ef-02 | exact_fact | 1 | 2 | 1 | 2 |
| ef-03 | exact_fact | 3 | miss | 2 | miss |
| ef-04 | exact_fact | 3 | 2 | 3 | 2 |
| ef-05 | exact_fact | miss | miss | miss | miss |
| ef-06 | exact_fact | miss | miss | 5 | 3 |
| ef-07 | exact_fact | miss | miss | 3 | 4 |
| ef-08 | exact_fact | 1 | 1 | 1 | 1 |
| ef-09 | exact_fact | 3 | 1 | 2 | 1 |
| ef-10 | exact_fact | miss | miss | miss | miss |
| ef-11 | exact_fact | miss | 3 | miss | miss |
| ef-12 | exact_fact | miss | miss | miss | miss |
| ef-13 | exact_fact | 1 | 2 | 1 | 2 |
| co-01 | conceptual | miss | miss | miss | miss |
| co-02 | conceptual | miss | miss | miss | miss |
| co-03 | conceptual | miss | miss | miss | miss |
| co-04 | conceptual | miss | 2 | miss | 2 |
| co-05 | conceptual | miss | miss | miss | miss |
| co-06 | conceptual | 2 | 1 | 2 | 1 |
| co-07 | conceptual | 1 | 1 | 1 | 1 |
| co-08 | conceptual | 3 | 4 | 3 | 4 |
| co-09 | conceptual | miss | miss | miss | miss |

---

## Why faithfulness fell when retrieval improved

Both configurations refused exactly 8 of 22 Track A questions, so this is not
the usual artifact where one config refuses more and refusals score perfect
faithfulness. The refusal *sets* differ by exactly one question each way, and
those two questions explain the whole gap.

**ef-05 — "How much did Microsoft's revenue increase in fiscal year 2024?"**
True answer: $33.2 billion, or 16%. Neither config retrieved the ground-truth
chunk.

| config | response |
|---|---|
| hybrid only | `NOT_IN_CONTEXT` — correct behaviour |
| hybrid+rerank | **"Microsoft's revenue grew by $8.5 billion in fiscal year 2024" — wrong** |

Reranking promoted Microsoft chunks from the FY2026 and FY2025 filings. They
are topically perfect — same company, same metric, same phrasing — so the
cross-encoder scored them highly, and the generator found them authoritative
enough to answer from. Plain hybrid retrieval left the context visibly
incoherent, and the model correctly declined.

**ef-03 — "What was NVIDIA's revenue in fiscal year 2026?"**
True answer: $215.9 billion, up 65%.

| config | response |
|---|---|
| hybrid only | ground truth at rank 3 → **"$215.9 billion" — correct** |
| hybrid+rerank | ground truth demoted out of top-5 → `NOT_IN_CONTEXT` |

So on the only two questions where the configurations diverged, reranking
**lost one correct answer and produced one confidently wrong one**.

The mechanism is worth stating plainly, because it is the opposite of the
intuition that better retrieval is always safer: a reranker optimises for
topical relevance, and wrong-year chunks from a near-identical filing are
maximally topically relevant. Sharpening relevance made a wrong answer look
well-supported. The faithfulness drop from 1.000 to 0.843 is that single
fabricated figure.

---

## Track B — adversarial (n=6)

Pass = the system declined, or asked for clarification, instead of answering
confidently. **Reported separately from Track A and never averaged with it.**

| config | n | passed | pass rate |
|---|---|---|---|
| hybrid+rerank | 6 | 6 | **1.000** |
| hybrid only | 6 | 6 | **1.000** |

| id | subtype | response (both configs) |
|---|---|---|
| adv-01 | unanswerable (Toyota) | `NOT_IN_CONTEXT` |
| adv-02 | unanswerable (Boeing) | `NOT_IN_CONTEXT` |
| adv-03 | unanswerable (Pfizer) | `NOT_IN_CONTEXT` |
| adv-04 | ambiguous (no company/year) | `NEEDS_CLARIFICATION: which company and fiscal year?` |
| adv-05 | ambiguous (no company) | `NEEDS_CLARIFICATION: which company's risk factors?` |
| adv-06 | ambiguous (year unspecified) | `NEEDS_CLARIFICATION: which year or reporting period?` |

### Validity check: is 100% real, or over-refusal?

A perfect adversarial score is exactly what a system that refuses *everything*
would produce, so the number is meaningless without this check. Three pieces of
evidence say it is genuine:

1. The system produced **real answers on 14 of 22** Track A questions. It is
   plainly not refusing by default.
2. It wrongly refused **zero** Track A questions where the ground-truth chunk
   *was* retrieved. Every Track A refusal followed an actual retrieval failure.
3. It **discriminated within** Track B, using `NOT_IN_CONTEXT` for the three
   unanswerable questions and `NEEDS_CLARIFICATION` for the three ambiguous
   ones. On adv-06 it correctly inferred that only JPMorgan reports CET1 and
   asked only for the *year* — the partially-resolvable case the question was
   written to test.

Caveat: n=6 with a hand-written prompt that names both sentinel strings. This
measures that the refusal path works, not that it is robust to adversarial
phrasing at scale.

---

## Run-to-run variance (the noise floor)

`hybrid+filter` was measured twice under identical conditions: 2026-08-25 and
again 2026-08-26, same judge, same `questions.yaml`, same `k=5`, same
`candidates=10`, no code change in between. Nothing differed but the date.

| `hybrid+filter` | faithfulness | answer relevancy | Track B |
|---|---:|---:|---:|
| 2026-08-25 | 0.948 | 0.740 | 6/6 |
| 2026-08-26 | 0.917 | 0.719 | 5/6 |
| **drift** | **0.031** | **0.021** | **1 question** |

Retrieval was identical across both runs (hit 0.545, recall 0.500, MRR 0.350),
exactly as it must be -- no LLM participates in it. All of the drift is in the
judged metrics.

This is the most useful methodological result in the report, because it puts a
number on something previously handled by intuition. **0.031 on faithfulness and
0.021 on answer relevancy is the noise floor.** Any judged gap smaller than that
is not evidence of anything. The earlier note that "differences of one or two
questions move the aggregates by ~0.045" was a bound on the *retrieval* metrics;
this is the measured equivalent for the *judged* ones.

It also retroactively qualifies one earlier claim. The filtering-lifts-relevancy
result (0.740 vs 0.626-0.640, roughly 0.10) survives comfortably -- about five
times the noise floor. The `hybrid+filter+rerank` faithfulness gap of 0.048
against `hybrid+filter`, already flagged as partial because of skipped 429 jobs,
is only 1.5x the floor and should be treated as weak.

---

## Query-type routing

`src/retrieval/router.py` classifies each query and dispatches: exact-fact
queries take the temporal filter, conceptual queries take the reranker,
ambiguous queries fall back to plain hybrid search. Over the 28 evaluation
questions the split was 15 exact-fact / 10 conceptual / 3 ambiguous.

The design follows directly from the dissociation documented above -- filtering
fixes exact-fact and does nothing for conceptual, reranking the reverse -- so
the question routing answers is whether a per-query choice captures both wins
without inheriting the combined configuration's interference.

### Retrieval: routing wins, decisively and without a judge

| config | exact-fact hit | conceptual hit | overall hit | MRR |
|---|---:|---:|---:|---:|
| hybrid only | 0.538 | 0.333 | 0.455 | 0.311 |
| hybrid+rerank | 0.538 | **0.444** | 0.500 | 0.345 |
| hybrid+filter | **0.692** | 0.333 | 0.545 | 0.350 |
| hybrid+filter+rerank | 0.615 | **0.444** | 0.545 | 0.356 |
| **routed** | **0.692** | **0.444** | **0.591** | **0.392** |

Routing reaches filtering's exact-fact number *and* reranking's conceptual
number simultaneously. No fixed configuration does: stacking both stages, the
obvious way to try, drops exact-fact to 0.615 because the reranker demotes
chunks the filter had already surfaced. Selecting between the strategies avoids
the interference that combining them creates.

Overall hit rate 0.545 -> 0.591 and MRR 0.356 -> 0.392. Both are deterministic
set comparisons against ground-truth chunk labels -- no judge, no sampling, and
verifiably identical across runs -- so the noise floor above does not apply.
**This is the only decisive result in the routing comparison, and the whole case
for routing rests on it.**

### Generation quality: a wash

Compared against the *same-day* 2026-08-25 `hybrid+filter` measurement:

| metric | routed | hybrid+filter | delta | noise floor | verdict |
|---|---:|---:|---:|---:|---|
| faithfulness | 0.969 | 0.948 | +0.021 | 0.031 | within noise |
| answer relevancy | 0.699 | 0.740 | -0.041 | 0.021 | exceeds floor, against routing |

Routing shows no measurable faithfulness benefit: +0.021 against a 0.031 floor
is not an effect. The relevancy difference does clear its floor, which means the
evidence leans mildly *against* routing on that metric -- but two caveats keep
it from being decisive. Both routed figures are unreproducible (see below), and
`answer_relevancy` runs at `strictness=1`, noisier per sample than the default
and possibly not fully captured by a floor derived from the same setting.

The defensible summary is that **routing does not improve generation quality and
may cost a little answer relevancy**, and that its case rests on retrieval.

A comparison to avoid: routing's 0.969 against the *2026-08-26* `hybrid+filter`
faithfulness of 0.917 gives a much more flattering +0.052. That crosses runs, so
most of the gap is the 0.031 day-to-day drift rather than the configuration.
Mixing the day effect into the config effect is precisely the error the variance
measurement exists to prevent.

### Track B: tied at 5/6, both failing adv-05

Both configurations pass 5 of 6 and fail the same question. The tempting reading
-- that routing changed something about `adv-05` -- is wrong in both directions.
It fixed nothing, and it broke nothing.

The 2026-08-26 replicate settles it: the *identical* `hybrid+filter`
configuration passed `adv-05` on 2026-08-25 and failed it on 2026-08-26, with no
change to code, prompts, questions or settings. `adv-05` ("What are the main
risk factors?", ambiguous as to company) sits on the judge's decision boundary
and flips between a confident answer and `NEEDS_CLARIFICATION` independently of
what retrieval feeds it.

This also revises an earlier finding. The 2026-08-23 report attributed the sole
`adv-05` failure to `hybrid+filter+rerank` and read it as a third instance of
the reranker assembling a falsely coherent context. That mechanism remains well
evidenced by ef-05 and ef-03, but `adv-05` can no longer be counted as an
instance of it: the question fails intermittently regardless of configuration.
Two clean instances, not three.

### Latency

Measured retrieval-only, with no generation in the loop:

| config | median | mean (excl. cold start) | raw mean |
|---|---:|---:|---:|
| hybrid+filter | **0.96s** | 0.93s | 6.81s |
| routed | 1.33s | 2.05s | 2.05s |

Routing is slightly **slower**, not faster. Classification adds a step, and the
conceptual branch still invokes the cross-encoder, so routed queries pay the
rerank cost on roughly a third of the set. The trade is ~0.4s of median latency
for +0.046 hit rate and +0.042 MRR.

`hybrid+filter`'s raw mean of 6.81s is an artifact and should not be quoted: a
single query (`ef-01`) took 165.73s on a cold start while the embedding model
loaded from disk. Its median over the same run is 0.96s and its mean excluding
that one query is 0.93s. The outlier is reported separately rather than folded
into an average, because including it overstates steady-state latency by about
seven times. Cold-start cost is real, but it is a startup property, not a
per-query one.

### Verdict

Routing is the better default, on retrieval and ranking quality alone.
Generation quality is a wash within measured noise, Track B is tied on a
question that flips on its own, and latency is marginally worse. The retrieval
gain is real, deterministic and reproducible; that is enough to justify routing
as the default, and it is the only part of this comparison that is.

**Routing resolves none of the open limitations.** Within-document ranking
failures (co-01/co-02 front matter, ef-05 sibling competition), the
corpus-specificity of the institution signal the classifier keys on, and the
filter/rerank coupling on edition-targeted conceptual questions all stand
unchanged. Routing selects between two strategies; it does not improve either.

---

## The lost routed scores

`routed`'s faithfulness (0.969) and answer relevancy (0.699) are quoted
throughout this report as prior-run values from 2026-08-25. They cannot be
reproduced from the current repository, and the reason is a harness bug worth
recording.

`checkpoint()` in `src/evaluation/run_eval.py` merged per-configuration
retrieval results into the existing raw file, but assigned the RAGAS block
wholesale:

    for name, results in by_config.items():
        payload[name] = _serialise(results)     # merges per config
    if ragas:
        payload["_ragas"] = ragas               # replaced everything

So the 2026-08-26 run with `--only "hybrid+filter"` rewrote `_ragas` to contain
only `hybrid+filter`, discarding every other configuration's judged scores. The
failure is quiet in the worst way: the per-config retrieval rows all survive, so
the file looks intact, and `eval_tables.md` is regenerated on every run and
reflects only the current one. `data/eval/` has never been tracked in git, so
there is no commit to recover from either.

The bug is fixed -- `_ragas` now merges by configuration key like everything
else. The lost values are not recoverable. To restore them to first-hand status,
re-run the routed configuration:

    python -m src.evaluation.run_eval --only "routed" \
        --candidates 10 --ragas --llm-provider groq

That costs roughly 105,000 judge tokens, about half of one day's Groq quota.
Until then, treat 0.969 and 0.699 as reported-but-unverified. Every other number
in this report is reproducible from `data/eval/results/`.

The wider lesson matches the one that produced the token guard: results that
cost a day of quota to produce must survive the next run that touches the same
file. Per-config checkpointing was added after a quota crash destroyed finished
work; this was the same class of failure, in the one part of the payload that
had been left out of it.

## Known limitation: no year / document-version discrimination

**Partially addressed.** Edition filtering (`src/retrieval/metadata.py`) now
parses the date out of each chunk's source filename and restricts the candidate
pool before ranking. It recovers ef-06 and ef-07 outright and improves ef-03,
ef-09 and ef-11's ranks.

It fixed fewer than the 10 misses attributed to it, and diagnosing why produced
the more useful result: **two distinct failures were being conflated**. Only
some misses were caused by retrieving the wrong *edition*. The remainder
survive filtering because the ground-truth chunk ranks poorly *within* the
correct document:

* **Front matter outranks content** (co-01, co-02). With the filter on, the
  pool is correctly dominated by the right Beige Book -- but the chunks
  returned are "About This Publication" and "What is the Beige Book?", which
  match the words "Federal Reserve", "economic conditions" and "Districts"
  better than the actual national summary does. Verified: the answer text is
  not present in those chunks, so this is not a labelling artifact.
* **Sibling-chunk competition** (ef-05): the correct chunk loses to four other
  chunks of the same filing.
* **Subject year is not edition year** (co-04, co-05): "growth in 2025" is
  answered by a report published in 2026. Deliberately left unfiltered.

Raising `fetch_k` to 40/60/100 does not help and breaks co-04.

### Symptom

Retrieval reliably finds the right company and the right document *type*, then
returns the wrong *edition*:

```
ef-05  "Microsoft revenue increase in fiscal 2024"
       got  msft_2026 c223, msft_2025 c219, msft_2024 c248, ...
       want msft_2024 c245

co-01  "January 2025 Beige Book"
       got  BB_20251126, BB_20260114, BB_20250604, BB_20250716, BB_20251015
       want BB_20250115          (all five wrong dates)
```

Affected: ef-05, ef-06, ef-07 (Microsoft FY2024/25/26), ef-10, ef-12,
co-01, co-02, co-03, co-05, co-09.

### Diagnosis

The corpus holds three 10-K filings per company and thirteen near-identical
Beige Books. Within one company, filings differ almost entirely in their
*figures and dates*, not their language, and neither retriever can use that:

- **Dense.** The distinguishing token is a year. In a 384-dimensional
  sentence embedding, "fiscal 2024" and "fiscal 2025" are nearly collinear;
  the year moves the vector far less than the surrounding boilerplate holds it
  in place.
- **BM25.** `2024` appears across most of the corpus, so its IDF is close to
  zero. It contributes almost nothing to the score, and a chunk dense in
  `revenue` and `fiscal` outranks the chunk that actually carries the year.
- **Reranking cannot fix it,** and as ef-05 shows it can make matters worse:
  a wrong-year chunk is topically ideal, so the cross-encoder promotes it.

Both retrievers are blind to the one attribute that distinguishes the
documents, so fusing them cannot recover it.

### Candidate fix (not implemented)

Metadata filtering rather than better ranking. Every chunk already carries its
`source` filename, which encodes the filing date
(`sec_10k_msft_2024-07-30.txt`, `fed_beige_book_20250115.pdf`). Parsing a year
or period from the query and applying it as a pre-filter would constrain the
candidate pool before scoring. This is a retrieval-architecture change, not a
model change, and should be scoped as its own phase with a re-run of this
harness to quantify the gain.

---

## Method notes and caveats

- **n=22 / n=6.** Differences of one or two questions move the aggregates by
  ~0.045. Treat small gaps as noise; the ef-05 / ef-03 divergence is reported
  as two specific cases, not as a rate.
- **Context precision/recall are not RAGAS numbers.** This project has real
  ground-truth chunk labels, so they are computed exactly rather than
  estimated by a judge. RAGAS is used only for faithfulness and answer
  relevancy, which genuinely need one.
- **`answer_relevancy` runs at `strictness=1`.** RAGAS defaults to 3, which it
  implements by requesting `n=3` completions in a single call; Groq caps `n`
  at 1 and rejects it. With `raise_exceptions=False` this surfaced only as a
  silent `NaN`. Strictness 1 averages one generated question instead of three,
  so the metric is noisier per sample.
- **Only generation and judging leave the machine.** Retrieval, embeddings and
  reranking are local, including RAGAS's own embedding needs, which are wired
  to the same bge-small model the index was built with.
- **Groq free tier is 8,000 TPM** account-wide (not per model). The run is
  paced to ~4 requests/minute with retries; the full four passes took 1h 45m
  for ~200k tokens.

## Reproduce

```bash
python -m src.evaluation.run_eval                                  # retrieval only, ~6s
python -m src.evaluation.run_eval --ragas --llm-provider groq      # full, ~1h45m
python -m src.evaluation.run_eval --no-rerank --ragas --llm-provider groq
```

Raw per-question output, including every generated answer and the chunks
retrieved for it, is in `eval_raw.json`.


---

## Quota note (why the runs were split across days)

Groq's free tier enforces **200,000 tokens per day per model** in addition to
the 8,000 tokens/minute limit. The daily cap appears in no response header --
only per-minute buckets are exposed -- so it is invisible until a request 429s
with `tokens per day (TPD): Limit 200000`.

One configuration with RAGAS costs roughly 100,000 tokens, so a day's quota
buys two. This is why the matrix could not be measured in one sitting, and why
every LLM-judged comparison in this report has to state its date.

The 2026-08-23 run measured the two *filter-off* configurations and then failed
partway through its second pass. The 2026-08-25 run completed `hybrid+filter`,
`hybrid+filter+rerank` and `routed`. The 2026-08-26 run re-measured
`hybrid+filter` alone, which produced the variance baseline above. The
filter-off rows are carried from 2026-08-23: that run used the same judge
(`openai/gpt-oss-120b`), the same finalised `questions.yaml`, the same `k=5`
and `candidates=10`, and predates edition filtering, so it is exactly configs 1
and 2 of the matrix.

Results merge into `eval_raw.json` rather than replacing it. Three hardening
changes came out of these failures: every configuration is checkpointed to disk
as soon as it completes, so a quota error can no longer discard hours of
finished work; retrieval-only runs write to `eval_raw_retrieval.json` so a
cheap sanity check cannot overwrite an expensive run's generated answers; and a
pre-run token guard (`src/evaluation/budget.py`) refuses to start a run that
cannot finish inside what remains of the day's cap, since the daily limit is
invisible to the API until it 429s.

The `_ragas` overwrite described in "The lost routed scores" was the one gap
this hardening had missed, and it is now closed too.

### Resolved: ef-05 at `candidates=10`

**ef-05** ("How much did Microsoft's revenue increase in fiscal year 2024?",
true answer $33.2 billion / 16%). Targeted testing at `candidates=20` had shown
filtering changing the fabricated answer from "$8.5 billion" to "$20.9 billion"
-- narrower, better-sourced context, still wrong, and arguably more convincing
because it now shows a derivation. That needed confirming at `candidates=10`
rather than assuming.

**Confirmed.** At `candidates=10` every filter-on configuration reproduces the
same fabrication:

| config | answer |
|---|---|
| hybrid+filter | "grew by about **$20.9 billion**, rising from $88.5 billion in FY 2023 to $109.4 billion in FY 2024" |
| hybrid+filter+rerank | "grew by $20.9 billion ... a 24% increase in fiscal year 2024" |
| routed | "grew by about $20.9 billion ... a 24% increase" |

Retrieval misses the ground-truth chunk in all of them (`first_rank` is null),
and routing does not help -- it classifies ef-05 as exact-fact and takes the
filter branch, so it inherits the filtered configuration's behaviour exactly.
The candidate-pool size is not the variable that matters here; the ground-truth
chunk loses to its own siblings within the correct document, which is the
sibling-chunk competition failure documented above. The $20.9 billion figure is
a derivation from two real numbers in the wrong chunks, which is why it reads as
well-sourced.
