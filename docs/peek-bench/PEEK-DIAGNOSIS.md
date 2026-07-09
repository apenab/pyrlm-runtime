# PEEK Diagnosis — Phase B (2026-05-21)

Diagnostic analysis of PEEK behaviour using the rich per-update traces
captured in Phase A. Each q{NN}.json holds the map before / after, the
parsed Distiller / Cartographer outputs, and the diff of items
added / evicted / modified.

Source: `docs/peek-bench/runs/run_20260521_221901_compare_gpt-5.4-mini_n5/traces/`.
Reproduce with `examples/peek_bench/analyze_peek_trace.py`.

---

## Setup

- 5 largest counting contexts of `oolongbench/oolong-synth` (one per sub-dataset).
- 25 queries each, fully online (`evolve_steps=-1`).
- gpt-5.4-mini (Azure), B=1024 tokens, peek-ai vendored @ 57de91ac, no patches.

The aggregate score result (Δ=−2.5pp) is within noise of Phase 3 (Δ=−1.8pp);
per-sub-dataset numbers swing several points run-to-run due to LLM
non-determinism. The traces below are about behaviour, not score.

## Aggregate behaviour per sub-dataset

| Sub-dataset | Neutral tag rate | Ops/query | Avg map size | Cart emit_repl | Silent fail rate | Max dup pairs |
|---|---|---|---|---|---|---|
| imdb (20037) | 16% | 2.20 | 4.16 | 27 | 0% | 1 |
| agnews (30036) | 24% | 2.16 | 5.52 | 20 | 0% | 3 |
| negation (40033) | 24% | 2.32 | 5.40 | 29 | 0% | 1 |
| multinli (70033) | 13% | 2.32 | 4.76 | 37 | 0% | 1 |
| **metaphors (80021)** | **33%** | **3.56** | **7.96** | **62** | 0% | 1 |

## Audit hypotheses, tested

### H1 — Silent operation failures: **REFUTED**

The audit predicted that Cartographer would emit REPLACE / DELETE ops
against non-existent item IDs and they would silently no-op. Tested across
all 5 contexts × 25 queries: **0 silent failures out of 327 emitted ops.**

The Cartographer prompt is good enough that it never references items
that have just been evicted or that never existed. Dropping patch C1
(operation validation) from Phase C — no upside.

### H2 — Duplicate detection: **WEAKENED**

The original by-hand inspection of the Phase 3 metaphors map showed
duplicates (cu-00015 ≈ cu-00019). In this Phase A.5 rerun, duplicate
pairs (Jaccard ≥ 0.7) peaked at 1 for metaphors and 3 for agnews — not
the dominant signal. The duplicates we saw in Phase 3 appear to be a
particular run artifact, not systematic.

Keeping C2 (duplicate detection at ADD time) as a low-priority candidate.

### H3 — Distiller blindness / score-zero stickiness: **CONFIRMED**

Metaphors has **33% neutral tags** — 2× higher than other sub-datasets.
This means the Distiller spends a third of its tag budget saying "I don't
know if this item helped" on metaphors.

Items tagged neutral never accumulate score and live forever until age
eviction. Metaphors's map is **largest (7.96 items avg) and churns most
(3.56 ops/query, 62 REPLACEs)** because old neutral items remain pinned
while Cartographer keeps editing them.

Direct mechanism:
- `evictor.py:19-28`: `update_scores()` treats "neutral" as a no-op
  (`out.setdefault(item_id, 0)`).
- `evictor.py:30`: sort key is `(scores.get(bid, 0), _id_age(bid))` — at
  score 0, the *older* item evicts first, which inverts value.
- Net effect: a freshly-added "neutral" item is more protected than an
  older neutral item that has demonstrated mild utility, *unless* the
  Distiller explicitly retags the older item as helpful in a subsequent
  pass.

This is the dominant failure mode the data supports. **C3 (score decay)
moves to high priority.**

### H4 — Map content hallucination: **CONFIRMED**

q05 of metaphors (this run) added the item:

```
[cu-00010] Corpus-level label balance is exact: correct = 39,060 and
            incorrect = 39,060.
```

The actual `oolongbench/oolong-synth` metaphors split (full corpus) is
heavily skewed (~3:1 incorrect>correct in the original Phase 3 spot
check). The Cartographer added a fabricated "exact balance" claim
derived from the agent's intermediate REPL exploration, then propagated
it through subsequent queries.

This is not addressed by any of the original C1–C5 candidates. A new
candidate emerges: **C4' (cache-candidate provenance check)** — only
accept Cartographer ADDs whose content is corroborated by a
`cache_candidate` in the Distiller output for the *current* query
(rather than the Cartographer freely inventing items).

## Per-item lifetime patterns

Across the 5 contexts, `used_after` (proxy: literal `[id]` substring in
later trajectories) is **0 for every item**. This is not surprising —
the agent uses the map's *content*, not its IDs, so the proxy is wrong.
A better proxy would be content-level token overlap between map items
and later root_call outputs. Adding to follow-up.

Top-scoring items in agnews are concrete reusable facts
(`rr-00004: Full-corpus label counts: Sci/Tech 9706, Business 13361, …`).
Top-scoring in metaphors are vaguer structural notes
(`cu-00002: Each example pairs a metaphorical statement with a candidate
literal interpretation`). The qualitative difference is consistent
with the "PEEK helps when the map can hold concrete answers"
observation from Phase 3.

## Implications for Phase C

Priority order based on evidence:

- **C3 (score decay) — high priority, strongest evidence.**
  Specifically: multiply existing scores by `decay = 0.85` at each update,
  and tag "neutral" as `-0.5` (down from 0) so undecorated items drift
  toward eviction. Decision rule: improve neutral_rate distribution
  symmetry across sub-datasets, and improve score on metaphors / multinli.

- **C4' (Cartographer provenance check) — medium-high priority,
  motivated by observed hallucination.**
  Require every ADD operation's content to overlap (token-level) with at
  least one of the Distiller's `cache_candidates` for that query. Block
  ADDs that the LLM invented out of nothing.

- **C2 (duplicate detection) — low priority.** Keep as a small
  defensive patch; the trace evidence does not support it as dominant.

- **C1 (operation validation) — dropped.** No silent failures observed.

- **C5 (section-aware eviction) — keep as candidate.** Maps shrink to
  4–8 items, so eviction does happen; protecting at least one item per
  critical section may matter. Defer until C3 is in.

Phase C will implement C3 first, measure, then layer C4' if C3 alone
does not move the needle.

---

## Post-C3 trace findings (2026-05-22)

After C3 was applied and validated with the clean A/B (+5.7pp aggregate
improvement), inspection of the per-query map evolution under C3 surfaced
two distinct new failure modes that the original Phase B audit did not
anticipate:

### Finding 1 — `rr-*` items are evicted prematurely under C3

`reusable_results` (rr-*) items are concrete facts the agent computed
(exact counts, frequency rankings, distinct keys). They take real REPL
work to derive. C3 treats every item identically: decay 0.85 / queue
toward eviction unless re-tagged helpful by the Distiller.

The problem: the Distiller can only mark an item helpful when it is
*used* in the current trajectory. An rr-* item like "Exact label counts:
Sci/Tech 9706; Business 13361; ..." is highly valuable but only relevant
to questions about label frequency. Across a 25-query mix it may be
helpful on, say, 4 queries and neutral on the rest. Under C3 it decays
fast enough to be evicted before its next relevant query arrives.

**Concrete instance (agnews, C3 run):**
- q05 map contained `[rr-00004] Exact label counts: Sci/Tech 9706;
  Business 13361; Sports 12322; World 8879.` with score 3.
- By q12 the item had been evicted.
- Agnews score Δ in this run was still +11.6pp (best in the cohort) —
  C3 helped overall — but the eviction of useful concrete results is
  visible cost, not a gain.

### Finding 2 — newly-added items are fragile under C3

A new ADD enters `scores` with score 0. C3's neutral penalty turns the
first unrelated query into −0.5; one more makes it −1.0. Items can be
evicted before they have a chance to prove useful, especially
expensive-to-derive `rr-*` items added at the moment of computation.

**Concrete instance (metaphors, C3 run q24):** `[rr-00016] Corpus-wide
processing can use one regex/pass...` born at score 0. Already at
neutral-penalty risk on the next update.

### Finding 3 — hallucinated content was absent in the C3 run (cannot attribute to patch)

In the Phase A.5 (no-patch) metaphors run, q05 added a fabricated
balance claim "correct = incorrect = 39,060". The C3 run did not
reproduce this — the q05 map instead held a correct "binary
correct/incorrect" statement. Whether C3 *prevents* such hallucinations
or whether the Distiller happened not to feed an invented claim is not
distinguishable from a single A/B run; the architectural loophole (the
Cartographer can ADD items without grounding in any Distiller
`cache_candidate`) remains.

---

## Implications for Phase C.2

The original Phase C plan listed C5 as "section-aware eviction"
(low-priority placeholder). The post-C3 traces motivate a different,
more specific patch:

**C5' — Section-weighted decay and birth bonus.** Replaces the original
C5 with a mechanism-specific change in `vendor/peek/core/evictor.py`:

- Per-section `DECAY` and `NEUTRAL_PENALTY` instead of global constants.
- Suggested defaults (subject to A/B):
  - `reusable_results`: decay 0.95, neutral_penalty 0.25 (slow drift —
    these are computed facts)
  - `parsing_schema`, `domain_constants`: decay 0.90, neutral_penalty 0.4
  - `context_understanding`, `context_roadmap`: decay 0.80,
    neutral_penalty 0.6 (faster drift — abstract claims should evolve)
  - `error_patterns`: decay 0.80, neutral_penalty 0.5
- Birth bonus: newly added `reusable_results` items start at +1.0
  instead of 0 (grace period to demonstrate utility).

**C4' — Cartographer ADD provenance check** is still relevant
independently. It closes the hallucination loophole structurally.
Recommend running C5' first (stronger trace evidence), then layering
C4' if hallucination is observed in the C5' traces.

The original C2 (duplicate detection) and the dropped C1 (operation
validation) remain off the table.
