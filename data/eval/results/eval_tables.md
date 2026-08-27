# Retrieval evaluation

Corpus: 16,917 chunks | embeddings: `BAAI/bge-small-en-v1.5`
| LLM: `groq:openai/gpt-oss-120b`

## Track A — answerable questions (exact-fact + conceptual)

Retrieval scored against the ground-truth chunk citations in `questions.yaml`. Deterministic set comparison, no LLM judge.

| config | n | hit rate | precision | recall | MRR | F1 |
|---|---|---|---|---|---|---|
| routed | 22 | 0.591 | 0.118 | 0.500 | 0.392 | 0.188 |

### Generation quality (RAGAS, LLM-judged)

Judged metrics over the generated answer. Context precision/recall are NOT taken from RAGAS -- they are measured exactly against ground-truth chunk labels in the table above.

| config | faithfulness | answer relevancy |
|---|---|---|
| routed | 0.942 | 0.744 |

### By question category

| category | config | n | hit rate | recall | MRR |
|---|---|---|---|---|---|
| exact_fact | routed | 13 | 0.692 | 0.615 | 0.451 |
| conceptual | routed | 9 | 0.444 | 0.333 | 0.306 |

### Per question (rank of first ground-truth chunk; lower is better)

| id | category | routed |
|---|---|---|
| ef-01 | exact_fact | 1 |
| ef-02 | exact_fact | 1 |
| ef-03 | exact_fact | 2 |
| ef-04 | exact_fact | 3 |
| ef-05 | exact_fact | miss |
| ef-06 | exact_fact | 5 |
| ef-07 | exact_fact | 3 |
| ef-08 | exact_fact | 1 |
| ef-09 | exact_fact | 2 |
| ef-10 | exact_fact | miss |
| ef-11 | exact_fact | miss |
| ef-12 | exact_fact | miss |
| ef-13 | exact_fact | 1 |
| co-01 | conceptual | miss |
| co-02 | conceptual | miss |
| co-03 | conceptual | miss |
| co-04 | conceptual | 2 |
| co-05 | conceptual | miss |
| co-06 | conceptual | 1 |
| co-07 | conceptual | 1 |
| co-08 | conceptual | 4 |
| co-09 | conceptual | miss |

## Track B — adversarial (unanswerable + ambiguous)

Pass = the system declined or asked for clarification instead of answering confidently. **Reported separately from Track A and never averaged with it.**

| config | n | passed | pass rate |
|---|---|---|---|
| routed | 6 | 6 | 1.000 |

| id | subtype | routed |
|---|---|---|
| adv-01 | | PASS |
| adv-02 | | PASS |
| adv-03 | | PASS |
| adv-04 | | PASS |
| adv-05 | | PASS |
| adv-06 | | PASS |
