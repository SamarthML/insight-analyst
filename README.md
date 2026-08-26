# Insight Analyst

An advanced RAG system for business research and reporting — hybrid search,
reranking, and RAG evaluation. Runs entirely locally on Ollama.

> **Status: Phases 1-2 and 4 complete.** Ingestion, hybrid retrieval and
> cross-encoder reranking run end to end in ~1.2s per query, and a two-track
> evaluation harness scores retrieval against labelled ground truth. Report
> generation and the served API are not implemented yet.

## Layout

```
data/documents/     source corpus (PDF, Markdown, text)
data/tabular/       CSVs (Phase 3)
src/config.py       paths, model names, chunk sizes
src/ingestion/      corpus fetching, loading, chunking, index building
src/retrieval/      hybrid retrieval + reranking (Phase 2)
notebooks/          experimentation
storage/            built indexes (gitignored, rebuildable)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # optional; defaults work as-is
ollama pull nomic-embed-text
```

On a machine without a GPU, embed with `sentence-transformers` instead — the
Ollama path takes ~6 hours on this corpus versus ~1.3. It needs a matching
chunk size, since that model's context is smaller; both settings and the
reasoning are in [Engineering notes](#engineering-notes).

```bash
EMBEDDING_BACKEND=sentence-transformers
CHUNK_SIZE_TOKENS=250
CHUNK_OVERLAP_TOKENS=40
```

## Get a corpus

`data/documents/` ships with three synthetic Northwind Analytics documents whose
answers are known, which is enough to exercise the pipeline but far too small to
tell dense, keyword and hybrid retrieval apart. To pull a real one:

```bash
$env:SEC_CONTACT_EMAIL = "you@example.com"   # SEC fair-access policy
python -m src.ingestion.fetch_corpus --list  # what it would fetch
python -m src.ingestion.fetch_corpus
```

That collects 33 public documents — 15 SEC 10-K filings across five companies in
different sectors, plus World Bank, BIS and Federal Reserve research PDFs — and
leaves the Northwind samples in place. Filing URLs are resolved from EDGAR's
official API at run time rather than hardcoded, since accession numbers change
with every filing.

## Build the indexes

```bash
python -m src.ingestion.build_index
```

Useful flags: `--preview N` (sample chunks), `--skip-vectors` (chunking/BM25
only, for tuning), `--backend sentence-transformers`.

Chunking alone, for a quick sanity check:

```bash
python -m src.ingestion.loader
```

## Retrieve

```bash
python -m src.retrieval.hybrid "common equity tier 1 capital ratio"
python -m src.retrieval.rerank "risks from supply chain disruption"
python -m src.retrieval.compare_retrieval          # all four methods, side by side
```

`compare_retrieval` prints vector-only, BM25-only, hybrid and reranked results
in four columns for the same query, which is how the retrieval stack was
validated qualitatively before any evaluation harness existed. It reports where
each fused hit came from, so it is visible whether fusion is doing real work or
just echoing one retriever.

The pipeline is dense + BM25 -> Reciprocal Rank Fusion -> cross-encoder rerank:

| stage | cost | notes |
|---|---|---|
| hybrid retrieval | 0.53s | 20 candidates from each retriever, fused |
| cross-encoder rerank | ~0.6s | top-10 rescored down to top-5 |
| **end to end** | **~1.2s** | measured warm, CPU-only |

Reranking with `BAAI/bge-reranker-base` instead is ~9x slower (~24s for 20
candidates) and is exposed via `--rerank-model` for offline evaluation rather
than interactive use.

## Evaluate

```bash
python -m src.evaluation.run_eval                                 # retrieval only, ~6s
python -m src.evaluation.run_eval --ragas --llm-provider groq     # + generation, ~1h45m
```

Two tracks, reported separately and never averaged, because they measure
different things — a system that refuses everything would look excellent on one
and terrible on the other.

**Track A** (22 answerable questions) scores retrieval against the ground-truth
chunk citations in `data/eval/questions.yaml`. Because those labels exist,
context precision and recall are computed by exact set comparison rather than
estimated by an LLM judge; RAGAS is used only for faithfulness and answer
relevancy, which genuinely need one.

**Track B** (6 adversarial questions) is a pass/fail check on whether the system
declines an unanswerable question or asks for clarification on an ambiguous one.

| Config | Hit rate | Recall | MRR | Exact-fact hit | Conceptual hit | Faithfulness | Answer relevancy | Track B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hybrid only | 0.455 | 0.409 | 0.311 | 0.538 | 0.333 | **1.000** | 0.640 | 6/6 |
| Hybrid + rerank | 0.500 | 0.432 | 0.345 | 0.538 | **0.444** | 0.843 | 0.626 | 6/6 |
| Hybrid + filter | 0.545 | **0.500** | 0.350 | **0.692** | 0.333 | 0.948 | **0.740** | 6/6 |
| Hybrid + filter + rerank | 0.545 | 0.477 | 0.356 | 0.615 | **0.444** | 0.900\* | 0.714\* | 5/6 |
| **Routed** (Phase 4b) | **0.591** | **0.500** | **0.392** | **0.692** | **0.444** | 0.969† | 0.699† | 5/6 |

\* The reranked configuration's RAGAS pass exhausted the judge model's daily
token budget: 3 of 44 scoring jobs returned 429 and were skipped, so these two
figures are means over ~41 jobs rather than 44. Every other row is complete.
The direction agrees with the rest of the evidence, but treat that specific gap
as suggestive rather than decisive.

† `Routed`'s faithfulness and answer relevancy are from the 2026-08-25 run and
**cannot currently be reproduced from this repository.** The harness replaced the
stored `_ragas` block wholesale on every `--only` run instead of merging it per
configuration, so the 2026-08-26 `hybrid + filter` re-run overwrote them. The bug
is fixed in `src/evaluation/run_eval.py`, but the values themselves are gone --
`data/eval/` has never been tracked in git, so there is no commit to recover them
from. Treat these two figures as prior-run values pending a re-measurement of the
routed configuration (~105k judge tokens, one full day of the Groq quota). Every
other number in this table is reproducible from `data/eval/results/`.

**`hybrid + filter` is the strongest of the four fixed configurations** — best
answer relevancy (0.740 vs 0.626–0.640 unfiltered), best exact-fact retrieval,
near-top faithfulness, and a perfect adversarial score. Filtering lifts answer
relevancy by roughly 0.10 across the board, comfortably clear of the 0.021 noise
floor established below, and a larger effect than anything reranking produced.

**`Routed` is the better default overall**, but on narrower grounds than the
table suggests: it wins decisively on retrieval and ranking, and is a wash on
generation quality. See [Query-type routing](#query-type-routing-phase-4b).

The headline is not the aggregate. Reranking improves every retrieval measure
yet scores *worse* on faithfulness, and the reason is specific: on the only two
questions where the filter-off configurations diverged, reranking lost one
correct answer and produced one confidently wrong one. It is also the only
configuration to fail Track B — `hybrid + filter + rerank` answered the
deliberately ambiguous "What are the main risk factors?" with a confident list
for an unnamed company, where every other configuration asked which company was
meant. Filtering alone passed 6/6 with no adversarial regressions; reranking
introduced the only failure. That is the same mechanism a third time — the
reranker assembles a topically coherent context out of chunks that should not
cohere, and the model stops noticing what it should have questioned. The three
instances span both question categories and three distinct failure shapes:
ef-05, where a correct refusal became a fabrication; ef-03, where a correct
answer was demoted out of the context entirely; and adv-05, where a request
for clarification became a confident answer. Full analysis, including the
mechanism, is in
[data/eval/results/eval_results.md](data/eval/results/eval_results.md).

### Measurement dates and run-to-run variance

The rows above are not all from one sitting — Groq's 200,000 tokens/day cap makes
a single five-configuration run impossible. `Hybrid only` and `Hybrid + rerank`
were measured 2026-08-23; `Hybrid + filter`, `Hybrid + filter + rerank` and
`Routed` on 2026-08-25.

Re-running `hybrid + filter` unchanged on 2026-08-26 — same judge, same questions,
same `k=5`, same `candidates=10`, nothing altered but the date — gives a direct
read on how much of any gap is measurement noise rather than effect:

| `hybrid + filter` | Faithfulness | Answer relevancy | Track B |
|---|---:|---:|---:|
| 2026-08-25 | 0.948 | 0.740 | 6/6 |
| 2026-08-26 | 0.917 | 0.719 | 5/6 |
| **drift** | **0.031** | **0.021** | **1 question** |

Retrieval was identical across both runs (hit 0.545, recall 0.500, MRR 0.350), as
it must be — no LLM is involved. The entire drift sits in the judged metrics.
**Treat 0.031 on faithfulness and 0.021 on answer relevancy as the noise floor
for every LLM-judged number in this report**; no gap smaller than that should be
read as a real difference.

### Filtering vs. reranking: two fixes for two different problems

After diagnosing that ~10 of 22 Track A misses traced to a single root cause —
the system failing to distinguish between near-identical filings that differ
only by year — I built temporal metadata filtering as a targeted fix and
re-ran the full retrieval matrix to see how it interacted with the existing
reranker.

The numbers are in the comparison table above. The result is a clean
dissociation, not a simple "which is better" answer:

- **Filtering fixes exact-fact questions** (0.538 → 0.692 hit rate) and does
  nothing for conceptual ones.
- **Reranking fixes conceptual questions** (0.333 → 0.444 hit rate) and does
  nothing for exact-fact ones.
- **Combining them is worse than filtering alone on exact-fact** (0.615 vs.
  0.692) — reranking actively demotes correct chunks once filtering has
  already removed the wrong-year noise.

The reason is that filtering and reranking address different failure modes,
not the same one. Filtering is a **retrieval-time** fix: it stops the wrong
document from entering the candidate pool at all. Reranking is a
**ranking-time** fix: it re-scores documents that are already in the pool by
topical relevance. Exact-fact questions in this corpus fail mostly because
near-duplicate filings (e.g. Microsoft's FY2024/2025/2026 10-Ks) are *all*
topically relevant to the same query — the only thing that distinguishes the
right one is the year, a token that barely moves a 384-dim embedding and
carries almost no IDF weight in BM25. Filtering removes the wrong-year
candidates before ranking ever happens, which directly fixes this. Reranking,
by contrast, is still doing its normal job — scoring by topical relevance —
on a candidate pool where topical relevance is no longer the thing that needs
discriminating. Its re-scoring adds noise rather than signal once filtering
has already solved the actual problem.

**Practical implication:** rather than always stacking reranking on top of
hybrid search, the stronger architecture routes by query type — apply
temporal/entity filtering when the query names a specific year or period, and
reserve reranking for conceptual/thematic queries where paraphrase and
semantic drift, not near-duplicate documents, are the dominant failure mode.

#### Case study: ef-05 (Microsoft FY2024 revenue)

This question surfaced the clearest example of the interaction above. Across
candidate-pool sizes, retrieval failed to surface the ground-truth chunk in
every configuration tested, and the model fabricated an answer instead of
declining:

| Config                        | Candidates | Fabricated answer | Ground truth   |
|--------------------------------|-----------:|--------------------|------------------|
| Hybrid + rerank                | 10         | "$8.5 billion"     | $33.2B / 16%     |
| Hybrid + filter + rerank       | 20         | "$20.9 billion" (with a shown derivation) | $33.2B / 16% |
| Hybrid + filter                | 10         | "$20.9 billion ... rising from $88.5 billion to $109.4 billion – a 24% increase" | $33.2B / 16% |
| Hybrid + filter + rerank       | 10         | "$20.9 billion (from $88.5 billion to $109.4 billion), a 24% increase" | $33.2B / 16% |

At `candidates=10` the answer matches the **$20.9B** fabrication, not the
$8.5B one, and both filtered configurations converge on the identical figure
with the identical derivation — filtering made the fabrication *reproducible*.
The model is performing a real subtraction over a real segment table in the
correct filing; it is just the wrong table, because the ground-truth chunk
never reaches the context in any configuration tested. That makes $20.9B
arguably the more dangerous failure even though both are wrong: the earlier
answer offered no reasoning and would likely draw scrutiny on inspection,
while this one presents a derivation that lends it unearned credibility. A
system that explains its wrong answer can mislead more than one that simply
states it, since the explanation borrows credibility from reasoning the
underlying retrieval never supported. This is a generation-quality risk that
hit-rate and faithfulness scores alone don't fully capture, and it argues for
the system correctly declining to answer rather than for improving its
explanations when it is wrong.

### Query-type routing (Phase 4b)

The implication above is implemented in `src/retrieval/router.py`: the query is
classified, exact-fact queries take the temporal filter, conceptual ones take the
reranker, and ambiguous ones fall back to plain hybrid search. Across the 28
evaluation questions the classifier split them 15 exact-fact / 10 conceptual /
3 ambiguous.

**Retrieval is where routing wins, and it is the one decisive result here.**

| | Exact-fact hit | Conceptual hit | Overall hit | MRR |
|---|---:|---:|---:|---:|
| Hybrid + filter | **0.692** | 0.333 | 0.545 | 0.350 |
| Hybrid + rerank | 0.538 | **0.444** | 0.500 | 0.345 |
| **Routed** | **0.692** | **0.444** | **0.591** | **0.392** |

Routing captures both halves of the dissociation at once — it matches filtering's
exact-fact number and reranking's conceptual number in a single configuration,
which no fixed strategy does. Overall hit rate rises to 0.591 against 0.545 for
the best fixed config, and MRR to 0.392 against 0.356. These are deterministic
set comparisons against ground-truth chunk labels: no judge, no sampling, no
run-to-run variance. Unlike the generation metrics, they are not subject to the
noise floor above, which is what makes this the load-bearing result.

**Generation quality is a wash.** Measured against the same-day (2026-08-25)
`hybrid + filter` figures, routing scores +0.021 on faithfulness (0.969 vs 0.948)
and −0.041 on answer relevancy (0.699 vs 0.740). The faithfulness gain is
*smaller* than the 0.031 faithfulness noise floor and cannot be called an effect.
The relevancy loss does exceed the 0.021 relevancy noise floor, so if anything
the evidence leans mildly *against* routing there — though it rests on figures
that are themselves unreproducible (see † above) and on a metric running at
`strictness=1`. The honest reading: routing does not measurably improve
generation quality, and may cost a little answer relevancy.

Comparing routing's 0.969 against the *2026-08-26* `hybrid + filter` faithfulness
of 0.917 would show a far more flattering +0.052 — but those are different days,
and most of that gap is the 0.031 day-to-day drift rather than the configuration.
Only same-day comparisons are used above.

**Track B is 5/6 for both, and both fail the same question, `adv-05`.** This is
not routing fixing anything, nor routing breaking anything. The identical
`hybrid + filter` configuration passed `adv-05` on 2026-08-25 and failed it on
2026-08-26 with nothing changed but the date. `adv-05` ("What are the main risk
factors?") is inherently borderline for this judge and model, flipping between a
confident answer and a request for clarification independently of which retrieval
strategy feeds it.

**Latency**, measured retrieval-only with no generation in the loop:

| | Median | Mean (excl. cold start) |
|---|---:|---:|
| Hybrid + filter | **0.96s** | 0.93s |
| Routed | 1.33s | 2.05s |

Routing is slightly *slower*, not faster: classification adds a step, and the
conceptual branch still invokes the cross-encoder. The trade is roughly 0.4s of
median latency for +0.046 hit rate and +0.042 MRR.

> `hybrid + filter`'s raw mean over that run is 6.81s, which is not a real
> result: one query (`ef-01`) took 165.73s on a cold start while the embedding
> model loaded. Excluding it gives 0.93s, in line with the median. The outlier is
> called out separately rather than folded into a mean, since averaging it in
> overstates steady-state latency by a factor of seven.

**Verdict: routing is the better default**, on retrieval and ranking quality
alone. Generation quality is a wash within measured noise, and it costs a little
latency. That is a narrower claim than the headline hit-rate gap suggests, and it
is the one the evidence actually supports.

**Not resolved by routing.** The open items from the filtering work stand
unchanged: within-document ranking failures (front matter outranking content on
co-01/co-02, sibling-chunk competition on ef-05), the corpus-specificity of the
institution signal the router keys on, and the filter/rerank coupling on
edition-targeted conceptual questions. Routing selects between strategies; it
does not improve either one.

## Run the UI

```bash
streamlit run src/ui/app.py
```

Local, single-user, no auth and no persistence. Two modes share one page:

- **Quick answer** — one question through the routed pipeline, with its
  citations and a collapsible "how this was answered" panel showing the route
  taken, whether edition filtering and reranking were applied, and the routing
  signals that decided it.
- **Generate report** — a topic is decomposed into sub-questions, each routed
  and answered separately, then synthesised into one markdown document
  (`src/reporting/generate_report.py`). Expect minutes rather than seconds: a
  report is 6+ model calls against a rate-limited hosted judge, so the two
  modes set different expectations while they work.

A declined or ambiguous question renders as an informational notice, never as
an error — the system declining to answer is Track B behaviour working, not a
failure. The sidebar carries the provider, chunk count, sub-question count, and
today's remaining token budget.

The report generator also runs headless:

```bash
python -m src.reporting.generate_report "NVIDIA's revenue growth and risk factors" \
    --llm-provider groq
```

Both paths call `budget.ensure_budget` before spending anything, so a report
that could not finish inside the day's remaining quota is refused rather than
aborted halfway.

## Engineering notes

Decisions and bugs from across the project worth keeping visible, because
each looked fine until a targeted query exposed it.

### Embedding model: bge-small over nomic-embed-text on CPU

The documented default is `nomic-embed-text` through Ollama, and on a GPU box
that is the better model — 768 dimensions and a 2048-token context, which fits
this project's chunks with room to spare. On a CPU-only machine it is not
viable for a corpus this size:

| | nomic-embed-text (Ollama) | bge-small-en-v1.5 (sentence-transformers) |
|---|---|---|
| params / dims | 137M / 768 | 33M / 384 |
| context | 2048 tokens | 512 tokens |
| measured rate | 0.23 chunks/s | 3.8 chunks/s |
| this corpus | ~6.2 h (projected) | 77 min (measured) |

The gap is bigger than the parameter counts suggest because Ollama processes
embed requests serially through one slot: even though `langchain-ollama` sends
the whole batch in a single HTTP call, the server walks it one prompt at a time
at ~4.3s each. `sentence-transformers` batches properly across cores.

Both are free and fully local, so this is purely a speed/quality trade, not a
cost one. Switch back by reverting `EMBEDDING_BACKEND` and the chunk sizes in
`.env` — but see the tokenizer note below, since the two must change together.

### Measure throughput on realistic data, not on a smoke test

The pre-flight check embeds one short string to confirm Ollama is up and the
model is pulled. That check is worth keeping — it caught a 7.2s cold load
versus 0.28s warm, i.e. the model genuinely was not resident until first use.

What it cannot tell you is how long the real run takes. Extrapolating from it
predicted 7–12 minutes; the actual pass was on track for 6.2 hours, a ~30x
error. The smoke test embedded 3-token strings while real chunks average 641
tokens — roughly 200x the work per call, and transformer cost grows with
sequence length rather than staying flat.

The practical rule: a liveness check and a throughput measurement are two
different tests. Liveness can use a toy input. Throughput must use inputs
drawn from the actual corpus, at the actual chunk size, or the number it gives
is not merely imprecise but wrong by an order of magnitude.

### The tokenizer mismatch that would have truncated 60% of every chunk

This is the one that would have shipped silently. The chunker sizes chunks with
**tiktoken** (`cl100k`), but `bge-small` tokenizes with **WordPiece** and hard
truncates at 512 tokens. The two disagree sharply on this corpus:

```
one real 2735-char chunk
  tiktoken  (chunker's view) :  641 tokens  -> comfortably under 800
  WordPiece (model's view)   : 1257 tokens  -> truncated at 512
```

The ratio measures ~1.96x. Dense financial filings are full of numerals, ticker
symbols and currency strings, which WordPiece shatters into many subword pieces
where a BPE vocabulary keeps them whole.

At the original 800-token chunk size, roughly 60% of every chunk's tail would
never have reached the embedding. Nothing errors — the tail is dropped in
silence. BM25 would still index the full text, so hybrid search would partially
mask it, and the failure would surface only as mediocre dense recall that is
very hard to attribute.

The fix is to size chunks in the *embedding model's* tokenizer, not the
chunker's: 512 / 1.96 ≈ 260, so `CHUNK_SIZE_TOKENS=250` with overlap 40. That
took the corpus from 5,227 to 16,917 chunks and the build to 77 minutes.

It does not fully eliminate truncation, and the residual is instructive. After
re-chunking, 54 chunks (0.32%) still exceed 512 WordPiece tokens, one reaching
2,382. The 1.96x average hides a long tail: chunks that are mostly numeric
tables tokenize at up to ~9.5x, because every figure fragments into subwords.
An average ratio is the wrong tool for sizing against a hard limit — the
correct fix for the remainder is to measure each chunk in the model's
tokenizer and split on that, rather than to pick a global token budget and
hope. Left as-is here because 0.32% of chunks losing their tail is a very
different failure from 100% of chunks losing 60%.

Generalised: whenever the chunker and the embedding model use different
tokenizers, the chunk budget must be expressed in the model's units. Verify it
rather than assuming — `SentenceTransformer(...).max_seq_length` and a
`tokenizer.encode()` on one real chunk settle it in seconds.

### The possessive that made BM25 answer the wrong company

Phase 2 surfaced a Phase 1 bug. Asked "what was Apple's revenue in fiscal 2023",
BM25 returned five NVIDIA chunks, none of which contained the word "Apple" even
once.

The tokenizer keeps intra-word apostrophes so `O'Brien` survives as one token.
The side effect is that `Apple's` becomes the token `apple's` -- which appears
nowhere in the corpus, because the filings themselves use a curly apostrophe
(U+2019) that the token pattern does not match at all:

```
idf["apple"]   = 3.97      (483 occurrences)
idf["apple's"] = ABSENT FROM INDEX
```

So the single most discriminative word in the query scored exactly zero, and
BM25 ranked on `revenue`, `fiscal` and `2023` alone. A dense NVIDIA revenue
table wins that contest on raw term frequency. BM25 has no requirement that all
query terms appear, so nothing flags it.

The fix emits both forms -- `Apple's` yields `["apple's", "apple"]` -- rather
than stripping apostrophes globally, which would merge `o'brien` into `obrien`.
Cost is a slight inflation of document length for BM25's normalisation.

Two things worth keeping from this. First, the bug lived in a *comment-justified*
design decision: preserving `O'Brien` was deliberate and reasonable, and the
possessive case simply was not considered. Second, it was invisible from inside
Phase 1 -- the index built cleanly and every self-consistency check passed.
Only a query whose expected answer was known revealed it, which is the argument
for building the comparison harness before trusting retrieval.

### Resolved: no year or document-version discrimination

Traced by the evaluation harness to **10 of 22 retrieval misses**, the single
largest quality issue in the stack.

Retrieval reliably finds the right company and the right document type, then
returns the wrong edition:

```
"Microsoft revenue increase in fiscal 2024"
   got  msft_2026, msft_2025, msft_2024 (wrong chunk) ...
"January 2025 Beige Book"
   got  five Beige Books, none of them January 2025
```

The corpus holds three 10-K filings per company and thirteen near-identical
Beige Books. Within one company the filings differ almost entirely in their
figures and dates rather than their language, and neither retriever can use
that. In a 384-dimensional embedding "fiscal 2024" and "fiscal 2025" are
nearly collinear -- the year moves the vector far less than the surrounding
boilerplate holds it in place. For BM25, `2024` appears across most of the
corpus, so its IDF is near zero and it barely scores. Both retrievers are blind
to the one attribute that separates the documents, so fusing them cannot
recover it.

Reranking does not help here and can actively hurt. A wrong-year chunk from a
near-identical filing is *topically ideal*, so a cross-encoder promotes it
confidently. On one evaluation question that turned a correct refusal into a
fabricated figure: plain hybrid retrieval returned visibly incoherent context
and the model declined, while the reranked context looked authoritative enough
that it answered, wrongly. Sharpening relevance made a wrong answer look
well-supported.

This was fixed with temporal metadata filtering: every chunk already carries
a `source` filename encoding the date (`sec_10k_msft_2024-07-30.txt`,
`fed_beige_book_20250115.pdf`), so a period parsed from the query is applied
as a pre-filter, constraining the candidate pool before scoring happens
rather than hoping ranking sorts it out after the fact. Results are in
[Filtering vs. reranking](#filtering-vs-reranking-two-fixes-for-two-different-problems)
above -- filtering raised exact-fact hit rate from 0.538 to 0.692, and,
notably, combining it with reranking performs *worse* than filtering alone
on exact-fact questions, since reranking has nothing useful left to sharpen
once the wrong-year noise is already removed. Filtering resolved the
edition-selection defect specifically -- two outright recoveries plus improved
ranks on three more -- while the remaining misses turned out to be distinct
failure modes: front-matter chunks outranking content, sibling-chunk
competition, and subject-year versus edition-year confusion, documented in
[eval_results.md](data/eval/results/eval_results.md) as a separate finding
rather than folded into this fix.

### Known limitation: per-filer fiscal calendar not modelled in the edition filter

`YEAR_WINDOW` in `src/retrieval/metadata.py` is keyed on **document type**, not
on the individual filer. Every SEC 10-K therefore shares one `(0, 1)` window.
That `+1` exists for JPMorgan and Coca-Cola, calendar-year filers whose 10-K
lands the following February, so a question about their fiscal 2023 has to admit
a document dated 2024. For Apple, Microsoft and NVIDIA the fiscal year equals
the filing year, and the same slack is pure noise — it admits an edition that
can never be the one asked for.

Traced live on `"What was Apple's revenue in fiscal 2023?"`:

```
parse_query  ->  QuerySpec(years=(2023,), month=0, entity='aapl')     correct
routing      ->  EXACT-FACT [filter=on, rerank=off]  margin=6         correct
filter       ->  doc_year $in [2023, 2024]        <- admits FY2024 by design
final pool   ->  [1] aapl_2024-11-01 chunk 177    year=2024
                 [2] aapl_2023-11-03 chunk 145    year=2023
                 [3] aapl_2024-11-01 chunk 135    year=2024
                 [4] aapl_2024-11-01 chunk 176    year=2024
                 [5] aapl_2023-11-03 chunk 176    year=2023
```

Detection and routing are both right. Three of the five chunks reaching the
answer prompt are nonetheless from the FY2024 filing, on a question that named
fiscal 2023 explicitly.

**The answer came back correct anyway, and the reason matters.** The model
returned Apple's fiscal 2023 total net sales of $383,285 million, citing chunk
176 of the *FY2024* filing — because a 10-K reports the prior year as a
comparative, so the wrong edition still carried the right figure. That is a
property of how 10-Ks are written, not of the filter working. The same slack is
unsafe wherever a figure appears only in its own edition, where the prior-year
column is absent, or where the model reads the current-year column instead; and
it spends three of five context slots on a document that, at best, duplicates
information already available in the correct one. Precision pays for it either
way.

**Distinct from the ef-05 sibling-chunk finding.** There the filter worked and
downstream ranking still picked the wrong chunk *within the correct document*.
Here the filter's own window admits the *wrong document* deliberately, before
ranking is consulted at all. Different stage, different fix.

The documented rationale for erring toward admission holds in general — a false
negative silently deletes the right answer, while a false positive merely leaves
a candidate for ranking to sort out. The tension is that Phase 4 established
ranking *cannot* reliably discriminate near-duplicate editions: that is the
finding edition filtering was built to act on. So wherever the window's slack is
unnecessary for a given filer, it hands the problem back to the stage already
shown to be unable to solve it.

**Candidate fix, not applied.** A per-ticker offset table — `AAPL`/`MSFT`/`NVDA`
to `(0, 0)`, `JPM`/`KO` to `(0, 1)` — instead of one window per document type.
This changes Phase 4-evaluated retrieval logic, so it needs a full harness re-run
to confirm it is a net improvement rather than a regression traded for a fix, and
is scoped as a follow-up.

## Roadmap

- [x] **Phase 1** — ingestion: loading, semantic-aware chunking, Chroma + BM25
- [x] **Phase 2** — hybrid retrieval (dense + BM25 fusion), cross-encoder reranking
- [ ] **Phase 3** — tabular/CSV analysis
- [x] **Phase 4** — RAG evaluation harness (two-track: labelled retrieval + adversarial)
- [x] **Phase 4b** — query-type routing (filter for exact-fact, rerank for conceptual)
- [x] **Phase 5a** — report generation (multi-section synthesis, edition-aware retrieval)
- [ ] **Phase 5b** — served via LangServe API
- [x] **Phase 5c** — local Streamlit UI (quick answer + report modes)
