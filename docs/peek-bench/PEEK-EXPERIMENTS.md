# PEEK Experiments — pyrlm-runtime

Reference: arXiv:2605.19932 (PEEK: Context Map as an Orientation Cache for Long-Context LLM Agents)

## North Star

Replicate the PEEK result from Table 1 of the paper on our own `pyrlm_runtime.RLM` stack.
Threshold to promote to `main`: **Δ score ≥ +5pp** AND **steps ≤ baseline** AND **cost ≤ 1.5× baseline**.

---

## Hypothesis (pre-committed 2026-05-21)

> Integrating PEEK as an orientation-cache over `pyrlm_runtime.RLM` on a
> "same Context × N queries" workload (oolong-synth) will produce ≥ +5 absolute
> score points over our anchored RLM baseline, with equal or fewer iterations per
> query and total cost ≤ 1.5× the baseline cost.
>
> Rationale: the paper reports +27.8pp over RLM-base on TREC-Q-coarse; +5pp is
> ~1/5 of that signal, allowing for differences between their RLM and ours.

## Decision Rule (pre-committed 2026-05-21)

After Phase 2 pilot (N=30 contexts, oolong-synth, cache LLM OFF, seed=42):

- **PROMOTE** → merge to main: Δ score ≥ +5pp AND steps/q ≤ baseline AND tokens/q ≤ 1.5× baseline
- **HOLD** → scale to N=150 across splits: +2pp ≤ Δ score < +5pp AND other KPIs OK
- **REJECT** → document below, do not merge: Δ score < +2pp OR steps/q > 1.2× baseline OR tokens/q > 2× baseline

Apply mechanically. Do not promote "close enough".

---

## Scorer validation (2026-05-21)

Our `score_example()` in `examples/peek_bench/run_oolong_rlm_vs_peek.py` was
verified against the official scorer in
[`abertsch72/oolong/src/eval/eval_helpers.py`](https://github.com/abertsch72/oolong/blob/main/src/eval/eval_helpers.py)
(`synth_process_response` + `synth_attempt_answer_parse`).

| Aspect | Official | Ours | Match |
|---|---|---|---|
| Parse: split on `:`, strip `*[]` | yes | yes | ✓ |
| Exact match → 1.0 | yes | yes | ✓ |
| Comparison ("more common"/etc) | substring match in gold | substring match in gold | ✓ |
| NUMERIC | `0.75 ** abs(diff)` | `0.75 ** abs(diff)` | ✓ |
| DATE | `dateutil.parser.parse` flexible match | (originally missing — now fixed) | ✓ after fix |
| MONTH_YEAR (0.8%) | not handled | not handled | ✓ tied |

**Type distribution in oolong-synth:** COMPARISON 31.5%, NUMERIC 27.4%, LABEL 24.0%, USER 14.4%, DATE 1.9%, MONTH_YEAR 0.8% (5200 total rows).

The first N=30 run (Phase 2 below) used the pre-fix scorer. DATE answers are
1.9% of queries; re-scoring Phase 2 offline with the fixed scorer changed
both baseline and PEEK by +2.1pp symmetrically — net Δ unchanged.

---

## Runs

### Pilot — N=5 contexts, compare mode (2026-05-21)

**Status:** COMPLETED — pilot only; plan specifies N=30, decision rule not applied here.

**Command:**
```bash
uv run python examples/peek_bench/run_oolong_rlm_vs_peek.py \
  --model gpt-5.4-mini --n-contexts 5 --mode compare \
  --seed 42 --env-tips --evolve-steps 4
```

**Results:**

| Method | N contexts | avg_score | avg_steps/q | avg_tokens/q |
|---|---|---|---|---|
| RLM baseline | 5 | 0.8398 | 9.3 | 16,691 |
| RLM + PEEK | 5 | 0.8349 | 10.3 | 17,096 |
| Δ | — | **−0.0048 (−0.6%)** | +1.0 | +405 |

Run artifacts: `docs/peek-bench/runs/run_20260521_144509_compare_gpt-5.4-mini_n5/`

N=5 surfaced a baseline of 84% (ceiling effect) and PEEK adding ~+1 step and
~+400 tokens per query on small contexts. Scaled to N=30 (Phase 2) before
applying the decision rule.

---

### Phase 2 — N=27 contexts (of 30 planned), compare mode (2026-05-21)

**Status:** COMPLETED (27/30 contexts — Mac crashed during contexts 28–30; remaining cannot reverse the result, see below) → **REJECT**

**Command:**
```bash
uv run python examples/peek_bench/run_oolong_rlm_vs_peek.py \
  --model gpt-5.4-mini --n-contexts 30 --mode compare \
  --seed 42 --env-tips --evolve-steps 4
```

**Results (re-scored offline with fixed DATE branch):**

| Method | N contexts | avg_score | avg_steps/q | avg_tokens/q |
|---|---|---|---|---|
| RLM baseline | 27 | 0.8308 | 10.11 | 16,676 |
| RLM + PEEK | 27 | 0.8148 | 9.75 | 16,994 |
| Δ | — | **−0.0161 (−1.93%)** | −0.35 | +318 (1.02×) |

Run artifacts: `docs/peek-bench/runs/run_20260521_155200_compare_gpt-5.4-mini_n30/`

**Decision rule outcome: REJECT**
- score ≥ +5pp: ✗ (actual: −1.6pp)
- steps ≤ baseline: ✓ (−0.35 steps/q)
- tokens ≤ 1.5×: ✓ (1.02×)

**Why 3 missing contexts cannot reverse the result.** To move from −1.6pp to
HOLD threshold (+2pp) would require each remaining context to show Δ ≥ +30pp,
which is more than the paper's best split (+27.8pp). Not realistic.

**Scorer cross-check.** Aggregating with the original scorer (no DATE branch)
gives baseline=0.8098, peek=0.7938, Δ=−1.61pp — identical delta. The DATE fix
raises both methods by ~+2.1pp symmetrically. Scorer choice does not affect the
verdict. 38 DATE queries re-scored, 24 changed; net delta change: 0.

**Observation:** PEEK reduces steps (−0.35/q) but degrades score (−1.6pp).
On structured, labeled, easy tasks the orientation cache adds no usable
guidance and biases the model toward earlier termination with less-verified
answers.

---

### Phase 3 — N=5 large contexts, fully-online (pre-committed 2026-05-21)

**Motivation:** Two findings from the N=27 audit point to a follow-up before
publishing the negative result:
1. Large contexts (>6M chars) in N=27 partial data showed PEEK genuinely
   reducing steps (e.g. ctx 20024 baseline=65 → PEEK=12).
2. Paper's Algorithm 1 takes `m ≤ n` and does not endorse `m=4` as universal
   default; peek-ai's `m=4` freezes the map after ~10% of queries on a 25–50
   query context.

**Setup:**
- N=5 hand-picked largest counting contexts (one per dataset for diversity):
  - 20037 (14.7M chars, imdb)
  - 30036 (13.4M chars, **agnews** — closest to paper's hardest split)
  - 80021 (12.8M chars, metaphors)
  - 40033 (6.8M chars, negation)
  - 70033 (7.0M chars, multinli)
- `evolve_steps=-1` → PeekSession with `evolve_steps=None` (fully online; map
  never freezes)
- Model: gpt-5.4-mini (Azure), cache OFF
- Same scorer (with DATE fix), same env-tips

**Hypothesis (pre-committed):**
> On the 5 largest counting contexts of oolong-synth, with the map fully
> online, RLM+PEEK will recover the +5pp threshold over the RLM baseline that
> the paper's TREC-Q-coarse result suggests. Specifically: Δ score ≥ +5pp,
> Δ steps ≤ −2 (large reduction expected from orientation cache amortizing
> over many queries), tokens ≤ 1.5× baseline.
>
> Rationale: large contexts give PEEK room to amortize the cost of building a
> map; fully-online evolution keeps the map adapting as queries diversify;
> counting tasks on multi-million-char contexts have low baseline scores
> (visible in partial N=27 data, e.g. ctx 60022 with 4 baseline failures).

**Decision rule (pre-committed):**
- **PROMOTE** → integrate PEEK as optional feature: Δ score ≥ +5pp AND steps ≤ baseline AND tokens ≤ 1.5×
- **HOLD** → run N=15 across all large contexts: +2pp ≤ Δ < +5pp AND other KPIs OK
- **REJECT (final)** → document, do not integrate, post negative result on X: Δ < +2pp

Apply mechanically.

**Command:**
```bash
uv run python examples/peek_bench/run_oolong_rlm_vs_peek.py \
  --model gpt-5.4-mini --mode compare \
  --context-ids 20037,30036,80021,40033,70033 \
  --evolve-steps -1 --env-tips
```

**Status:** COMPLETED 2026-05-21 → **REJECT in aggregate**

**Aggregate (N=5):**

| Method | avg_score | avg_steps/q | avg_tokens/q |
|---|---|---|---|
| RLM baseline | 0.9337 | 13.0 | 17,867 |
| RLM + PEEK | 0.9156 | 8.5 | 14,909 |
| Δ | **−0.0181 (−1.9%)** | **−4.5** | **−2,959 (0.83×)** |

Decision rule outcome: **REJECT** (score < +2pp threshold).
- score ≥ +5pp: ✗ (actual: −1.8pp)
- steps ≤ baseline: ✓ (−4.5 steps/q)
- tokens ≤ 1.5×: ✓ (0.83×, far below threshold)

**Per-dataset breakdown:**

| Dataset | M chars | base_s | peek_s | Δ score | Δ steps | Δ tokens | W/T/L |
|---|---|---|---|---|---|---|---|
| agnews | 13.4 | 0.924 | 1.000 | **+7.6pp** | −1.3 | −1,822 | 2/23/0 |
| negation | 6.8 | 0.960 | 1.000 | +4.0pp | −0.7 | −21 | 1/24/0 |
| imdb | 14.7 | 0.922 | 0.880 | −4.2pp | **−18.1** | **−9,609** | 2/20/3 |
| multinli | 7.0 | 0.943 | 0.885 | −5.7pp | −1.8 | −2,454 | 0/23/2 |
| metaphors | 12.8 | 0.920 | 0.813 | −10.7pp | −0.3 | −887 | 0/22/3 |

Run artifacts: `docs/peek-bench/runs/run_20260521_195528_compare_gpt-5.4-mini_n5/`

**Observations:**

- agnews (+7.6pp, perfect 1.000 score) is one of the three splits in Table 1
  of the PEEK paper. PEEK replicates on it within oolong-synth.
- Per-sub-dataset Δ scores span 18 points (+7.6pp on agnews to −10.7pp on
  metaphors) on contexts of similar size and same task type (counting).
  Difficulty alone does not explain the variance — baseline scores are
  0.92–0.96 across all five.
- imdb: −4.2pp score with −18.1 steps/q and −9,609 tokens/q. The map drives
  early termination; on three queries this terminates before reaching the
  correct answer.
- metaphors: on the three score regressions PEEK uses more steps (13 vs 8.3),
  not fewer. Failure mode here is different from imdb — the map content
  itself misleads.
- Token ratio 0.83× across all five sub-datasets. PEEK is strictly more
  token-efficient than baseline in this setup; the cost is purely accuracy.
- `evolve_steps=-1` (fully online) verified active: max map_step = 25,
  `evolving=True` on all queries.

---

## Diagnosis across all three runs

**1. Ceiling effect (Pilot and Phase 2 only).**
Baseline at 84% on small/medium contexts gave PEEK <16pp of headroom vs the 70pp the paper
had on TREC-Q-coarse. This explains the Phase 2 aggregate (Δ=−1.6pp at N=27).

**2. Effect is dataset-dependent, not just difficulty-dependent (Phase 3).**
On the 5 largest counting contexts in Phase 3, baseline scores are 0.92–0.96 (still
high), yet PEEK shows +7.6pp on **agnews** and +4.0pp on **negation**, while
losing −4 to −10pp on imdb/multinli/metaphors. Same baseline level, same
task type, same model — different sub-dataset, opposite sign. This rules out
"pure ceiling effect" as the only mechanism.

**3. `evolve_steps=4` (peek-ai default) starves evolution.**
With ~25 queries per context, the map only evolved during the first 4 queries.
Phase 3 used `evolve_steps=-1` (fully online) — the map evolved through all 25,
which is closer to what Algorithm 1 in the paper supports.

**4. Two distinct failure modes on losing sub-datasets (Phase 3).**
- imdb: PEEK uses fewer steps on regressions — premature termination.
- metaphors: PEEK uses *more* steps on regressions (13 vs 8.3) — the map
  content itself misleads, not just the stopping rule.

**5. Model mismatch.**
gpt-5.4-mini ≠ gpt-5-mini-2025-08-07 from the paper. Probably a minor factor
given the dataset-specific results in Phase 3.

---

### Open follow-ups

- Official OOLONG (Bertsch et al., arXiv:2511.02817) — trec_coarse/agnews/yahoo
  splits where baseline scores are ~30%. Not on HuggingFace as of 2026-05-21.
- Scale Phase 3 to N=10–15 contexts per sub-dataset to confirm the per-dataset
  pattern (Phase 3 has only 1 context per sub-dataset).
- The PEEK blog post ([@astrogu_](https://zhuohangu.github.io/blog-post-peek/))
  notes: *"the usefulness of the context map depends on how the agent interacts
  with the context. If that interaction reveals little reusable knowledge, the
  map has little to cache."* Phase 3 is an empirical instance.

---

## Integration & paper audits (2026-05-21)

### Integration audit vs peek-ai upstream (`zhuohangu/peek` commit 57de91ac)

- `CachePolicy.update(*, trajectory: str, question: str = "")` — kwargs match ✓
- `LMClient.completion(messages) -> str`, `last_usage() -> Usage` — match ✓
- `evolving = steps < evolve_steps`, counter increments on every `update()` call ✓
- Token budget enforced by Evictor at end of every evolving step (default tokenizer `tiktoken o200k_base`) ✓
- Map persists per `context_window_id`, resets between contexts — matches paper's Algorithm 1 ✓
- Trajectory format: peek's Distiller takes free-form `{trace_history}` string; no special markers required ✓

### Paper audit — risks affecting our setup

- Paper's Appendix D notes that dataset retrofits like BrowseComp-Plus /
  FanOutQA / QuALITY "did not exercise what PEEK was designed for" — PEEK
  requires shared orientation across queries on a persistent context.
  oolong-synth may fall in the same category.
- peek-ai's `evolve_steps=4` default may starve evolution on workloads with
  many queries per context (mitigated in Phase 3 with `evolve_steps=-1`).
- Model is gpt-5.4-mini vs paper's gpt-5-mini-2025-08-07.

---

## Notes

- **Dataset:** `oolongbench/oolong-synth` (not the Bertsch et al. OOLONG from the paper — that dataset is not yet on HuggingFace as of 2026-05-21).
  This means absolute scores are **not directly comparable** to Table 1 of the paper. However, the **relative delta (Δ)** between RLM and RLM+PEEK should be comparable since the task structure (same context × N aggregation queries) is identical.
- **Model:** Fix the exact deployment name and API version per run; report in the Run ID.
- **Cache OFF:** Use adapters with no `cache_dir`; never compare against cached runs.
- **peek-ai version:** commit 57de91ac (git+https://github.com/zhuohangu/peek.git, 2026-05-20).
  Update if a new version ships before experiments complete.
- **Scorer:** validated against `abertsch72/oolong` official implementation 2026-05-21; DATE branch added to match official behavior.
- **Reference reading:** PEEK paper (arXiv:2605.19932), official OOLONG paper (Bertsch et al., arXiv:2511.02817), [PEEK blog post by @astrogu_](https://zhuohangu.github.io/blog-post-peek/), peek-ai upstream ([zhuohangu/peek](https://github.com/zhuohangu/peek)).

---

## Phase 4 — Targeted improvements on a vendored fork

Started 2026-05-21. Goal: reduce the per-sub-dataset variance from Phase 3
by patching peek-ai internals based on trace-derived evidence. peek-ai is
vendored at `vendor/peek/` (commit 57de91ac) so we can patch single files
without forking upstream.

### Phase 4.A — Vendoring + observability

- `vendor/peek/` mirrors `zhuohangu/peek` at 57de91ac (Apache-2.0).
- `PeekSession.create(trace_dir=…)` writes a JSON snapshot per update:
  query, trajectory, map before/after with per-item scores, parsed
  Distiller / Cartographer outputs, and the diff.
- Benchmark harness gains `--trace` flag.
- Phase 3 re-run with `--trace` (run dir
  `run_20260521_221901_compare_gpt-5.4-mini_n5`): aggregate Δ=−2.5pp vs
  Phase 3's Δ=−1.8pp; per-sub-dataset deltas swing ±5–7pp due to LLM
  non-determinism. Trace files written for all 125 queries.

### Phase 4.B — Diagnosis (PEEK-DIAGNOSIS.md)

`examples/peek_bench/analyze_peek_trace.py` aggregates the traces.
Findings:

- **Silent failures (audit H1): REFUTED.** 0/327 emitted REPLACE/DELETE
  ops referenced non-existent items. Patch C1 dropped.
- **Distiller blindness / score-zero stickiness (audit H3): CONFIRMED.**
  metaphors has 33% neutral tags vs 13–24% elsewhere, 2× ops/query,
  largest map.
- **Map content hallucination: CONFIRMED** by a concrete metaphors q05
  trace.
- **Duplicates: low** (≤3 pairs/ctx) in this run; not the dominant mode.

### Phase 4.C.1 — Patch C3 (score decay)

**Patch:** `vendor/peek/core/evictor.py` — `update_scores` multiplies
existing scores by `SCORE_DECAY=0.85` before applying new tags, and
treats `neutral` as `-NEUTRAL_PENALTY=0.5` (was 0). `scores` value type
widens from int to float. Constants `1.0 / 0.0` recover upstream.

**Control:** identical-arm in-session run with C3 disabled (constants
reset to 1.0 / 0.0). Saved to `runs/c3_control_no_patch/`.

**Hypothesis (pre-committed 2026-05-21):**
> Score decay reduces metaphors' anomalously large map size (Phase A
> rerun: 7.96 avg items vs 4.16–5.52 elsewhere) and high Cartographer
> churn (3.56 ops/q vs 2.16–2.32 elsewhere) by allowing untagged or
> neutral items to drift toward eviction. On aggregate, Δ score moves
> ≥ +2pp closer to zero or positive vs the in-session control.

**Decision rule (pre-committed):** KEEP if aggregate improvement ≥ +2pp
AND no sub-dataset regresses > −5pp vs the in-session control.

**Results (clean in-session A/B):**

| Setup | baseline | peek | Δ |
|---|---|---|---|
| Control (decay=1.0, neutral=0) | 0.8771 | 0.8370 | −4.0pp |
| C3 (decay=0.85, neutral=−0.5) | 0.8651 | 0.8810 | **+1.6pp** |
| **Patch effect on Δ** | | | **+5.7pp** |

Per-sub-dataset (improvement = C3 Δ − control Δ):

| Sub-dataset | Ctrl Δ | C3 Δ | Improvement |
|---|---|---|---|
| agnews | −6.7 | +11.6 | +18.3 |
| multinli | −9.7 | −2.3 | +7.4 |
| metaphors | −8.0 | −4.0 | +4.0 |
| imdb | +4.3 | +3.4 | −1.0 |
| negation | 0.0 | −0.7 | −0.7 |

Trace-level evidence (neutral_rate / avg items, control → C3):
metaphors 29% → 4% and 4.40 → 5.60 items; multinli 25% → 9% and 6.36 →
4.92 items.

**Decision rule outcome: KEEP.**
- aggregate improvement ≥ +2pp: ✓ (+5.7pp)
- no dataset regression > −3pp: ✓ (max −1.0pp, well inside the LLM
  noise floor of ~±5pp on the baseline arm)
- trace-level mechanism confirmed: ✓

### Phase 4.C.2 — Patch C5' (section-weighted decay + rr-* birth bonus) → **REJECTED**

**Status:** Implemented, tested across 3 paired A/B trials, **reverted**.
Code removed from the working tree. The vendored library now exposes
only C3 (uniform decay) as the validated improvement. This section
documents the experiment as a negative ablation, per the empirical
methodology rule "document the result regardless of outcome".

**Motivation (the hypothesis that prompted C5'):** Post-C3 trace
inspection (`PEEK-DIAGNOSIS.md` → "Post-C3 trace findings") surfaced
two failure modes that uniform C3 does not address:
1. Concrete `rr-*` items (e.g. agnews q05's `rr-00004` with exact label
   counts) decay and get evicted before their next relevant query.
2. Newly-added items start at score 0 and become eviction targets after
   one neutral-tagged query.

**Implementation tested:** per-section decay rates and per-section
neutral penalties, plus a birth-bonus of +1.0 for fresh `reusable_results`
items. Mechanism in the vendored library (generic per-section dicts as
optional kwargs); workload-specific values lived in the benchmark harness
(`OOLONG_DECAY_BY_SECTION` etc.) — never inside `vendor/peek/`. Tested
config:

| Section | decay | neutral_penalty | birth_bonus |
|---|---|---|---|
| reusable_results | 0.95 | 0.25 | +1.0 |
| parsing_schema | 0.90 | 0.40 | 0 |
| domain_constants | 0.90 | 0.40 | 0 |
| context_understanding | 0.80 | 0.60 | 0 |
| context_roadmap | 0.80 | 0.60 | 0 |
| error_patterns | 0.80 | 0.50 | 0 |

**Pre-committed hypothesis (2026-05-22):**
> Aggregate Δ improves by ≥ +2pp vs the C3 in-session control. `rr-*`
> items survive ≥ 10 queries on average (vs ≤ 7 under C3).

**Pre-committed decision rule:**
- KEEP if aggregate improvement ≥ +2pp AND no sub-dataset regresses by
  more than −3pp (later relaxed to −7pp once we observed the actual
  ±5–20pp per-sub-dataset noise floor on N=1 runs).
- REVERT otherwise.

**Three paired A/B trials (c5p vs c3, both arms in same session):**

| Trial | Setup | c5p Δ | c3 Δ | Patch effect | Baseline shift |
|---|---|---|---|---|---|
| 1 | 95-min gap (contaminated) | −3.8pp | +5.5pp | **−9.3pp** | +8.1pp |
| 2 | True parallel | +2.2pp | −3.4pp | **+5.6pp** | −3.1pp |
| 3 | True parallel | −6.0pp | −0.0pp | **−6.0pp** | +6.6pp |
| **Mean** | | **−2.5pp** | **+0.7pp** | **−3.2pp** | +3.9pp |

Direct PEEK-score comparison (less noisy than the differential):

| Trial | c5p peek | c3 peek | c5p − c3 |
|---|---|---|---|
| 1 | 0.903 | 0.915 | −1.2pp |
| 2 | 0.891 | 0.866 | +2.5pp |
| 3 | 0.878 | 0.872 | +0.6pp |
| **Mean** | **0.891** | **0.884** | **+0.7pp** |

**Decision: REVERT.**
- Trial 3 patch effect: −6.0pp (well below the +2pp KEEP threshold).
- 2 of 3 trials negative on the differential; mean −3.2pp.
- Direct peek-score mean: c5p +0.7pp, statistically indistinguishable
  from C3 at this N.
- Mechanism evidence (rr-* survival) was mixed across trials — not the
  clean uplift seen in the C3 vs upstream comparison.

**Methodological lessons recorded:**
- Per-sub-dataset noise floor on N=1 oolong-synth contexts is ±5–20pp
  (observed via baseline shifts between paired arms). The original
  +5pp aggregate decision rule for C3 worked because C3 produced a
  large effect; C5' fell inside the noise.
- Parallel arms are mandatory for any future A/B at this N; the 95-min
  gap of trial 1 contaminated the baseline by 8pp.
- A regex bug surfaced in `vendor/peek/_io.py` during trial 3 (one arm
  hung in catastrophic backtracking on a malformed LLM fence). Fixed
  with an O(n) state machine; regression test added. Independent of
  C5' but documented here so the next experimenter knows the trap.

**Working tree after revert:** C5' code and `OOLONG_*` constants
removed; `vendor/peek/core/evictor.py` exposes only the C3 globals
(`SCORE_DECAY=0.85`, `NEUTRAL_PENALTY=0.5`). The `--peek-policy` flag
removed from the benchmark harness. Run artifacts retained under
`docs/peek-bench/runs/c5p_*` for future re-analysis.

### Phase 4.D — Bug fix in `vendor/peek/_io.py`

Discovered during Phase 4.C.2 trial 3: `extract_json` used
`re.findall(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)`,
which catastrophically backtracks when the LLM emits an opening fence
without a closing one. Reproduced in production: a benchmark arm hung
at 100% CPU for 45 minutes on a single LLM response.

**Fix:** replace the regex with an O(n) state-machine scan
(`_scan_fenced_blocks`) that bails out gracefully on unclosed fences.
Same fix mirrored in `examples/peek_bench/analyze_peek_trace.py:_extract_json`.

**Regression tests** in `tests/test_peek_integration.py`:
- `test_extract_json_handles_unclosed_fence_quickly` — 100KB unclosed
  fence completes in < 1s (pre-fix: hangs).
- `test_extract_json_still_finds_valid_fenced_json` — happy path intact.
- `test_extract_json_finds_second_block_when_first_invalid` — multi-block
  extraction intact.

This fix is upstream-contributable to `zhuohangu/peek`.
