"""REPL extensions for the OBLIQ-Bench RLM rerank experiment.

Exposes two domain-specific functions to the RLM root model:

* ``verify_relevance_batch(query, doc_ids)`` — for each ``doc_id``, fetch
  the document text and ask a sub-LLM (via ``llm_batch`` in the REPL scope)
  whether the passage is relevant to the query. Returns a list of
  ``{doc_id, relevant: bool, reason: str}`` dicts. All subcalls execute in
  parallel via :class:`pyrlm_runtime.RLM`'s ``parallel_subcalls``.

* ``read_doc(doc_id)`` — synchronous corpus lookup, no LLM involved.

Also injects two REPL variables for convenience: ``query`` (the search
query string) and ``bm25_pool`` (the pre-fetched top-N candidates).
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable


_VERIFY_PROMPT_TEMPLATE = (
    "You are a relevance judge for an information-retrieval task.\n\n"
    "Search query:\n{query}\n\n"
    "Candidate passage:\n{passage}\n\n"
    "Rate how similar this passage is to the query in terms of underlying "
    "technique, structure, or approach (NOT surface topic). Use this 1-5 "
    "scale and be willing to assign middle scores when uncertain:\n"
    "  1 = clearly unrelated, no meaningful overlap\n"
    "  2 = vague topical overlap, no shared approach\n"
    "  3 = plausible structural or methodological similarity, uncertain\n"
    "  4 = clear similarity in technique or reasoning structure\n"
    "  5 = strong match in proof / solution / reasoning technique\n\n"
    "Important: when the passage might share the query's reasoning approach "
    "but you cannot be sure, assign 3 or 4 — do NOT default to 1 or 2. "
    "Use the full range of the scale.\n\n"
    "Reply with exactly one JSON object on a single line, nothing else:\n"
    '{{"score": <integer 1-5>, "reason": "<one short sentence>"}}'
)


_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)
_SCORE_RE = re.compile(r'"score"\s*:\s*([1-5])')
_FALLBACK_SCORE_RE = re.compile(r"\b([1-5])\b")


def _parse_verify_response(text: str) -> tuple[int, str]:
    """Best-effort parse of the verifier response.

    Returns ``(score, reason)`` where ``score`` is an integer 1-5. Falls
    back to score 1 when no parseable integer is found.
    """
    match = _JSON_OBJECT_RE.search(text)
    if match:
        try:
            data = json.loads(match.group(0))
            raw_score = data.get("score")
            score: int | None = None
            if isinstance(raw_score, int) and 1 <= raw_score <= 5:
                score = raw_score
            elif isinstance(raw_score, str):
                stripped = raw_score.strip()
                if stripped.isdigit() and 1 <= int(stripped) <= 5:
                    score = int(stripped)
            if score is not None:
                return score, str(data.get("reason", "")).strip()
        except Exception:
            pass
    # Regex fallback over the raw text.
    score_match = _SCORE_RE.search(text)
    if score_match:
        return int(score_match.group(1)), text.strip()[:200]
    fallback = _FALLBACK_SCORE_RE.search(text)
    if fallback:
        return int(fallback.group(1)), text.strip()[:200]
    return 1, text.strip()[:200]


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def build_rlm_rerank_extensions(
    *,
    corpus: dict[str, str],
    query_text: str,
    bm25_pool: list[dict[str, Any]],
    max_passage_chars: int = 600,
    subcall_max_tokens: int = 256,
    verify_log: list[dict[str, Any]] | None = None,
) -> Callable[..., dict[str, Any]]:
    """Build a callable suitable for :attr:`RLM.repl_extensions`.

    The returned callable, when invoked by the RLM, sets ``query`` and
    ``bm25_pool`` in the REPL globals and returns the dict
    ``{"verify_relevance_batch": fn, "read_doc": fn}`` so the RLM can call
    them from generated code.

    If ``verify_log`` is provided, each ``verify_relevance_batch`` invocation
    appends an entry like
    ``{"input_count": N, "relevant_count": K, "verdicts": [...]}``
    so the bench can inspect what the sub-LLM said after the run finishes.
    """

    def extensions(*, rlm: Any, repl: Any, retriever: Any, log_diag: Any) -> dict[str, Any]:
        # Inject context variables the model can use directly.
        repl.set("query", query_text)
        repl.set("bm25_pool", bm25_pool)

        def read_doc(doc_id: str) -> str:
            return corpus.get(str(doc_id), "")

        def verify_relevance_batch(query: str, doc_ids: list[str]) -> list[dict[str, Any]]:
            if not doc_ids:
                return []
            llm_batch = repl.get("llm_batch")
            if llm_batch is None:
                raise RuntimeError(
                    "llm_batch is not registered in the REPL — verify_relevance_batch "
                    "must be invoked from within an RLM run."
                )
            prompts: list[str] = []
            for doc_id in doc_ids:
                passage = _truncate(read_doc(doc_id), max_passage_chars)
                prompts.append(
                    _VERIFY_PROMPT_TEMPLATE.format(query=query, passage=passage)
                )
            raw_results = llm_batch(prompts, max_tokens=subcall_max_tokens)
            out: list[dict[str, Any]] = []
            for doc_id, raw in zip(doc_ids, raw_results, strict=False):
                score, reason = _parse_verify_response(str(raw))
                out.append(
                    {
                        "doc_id": str(doc_id),
                        "score": score,
                        # Convenience flag (back-compat): score >= 3 → relevant.
                        "relevant": score >= 3,
                        "reason": reason,
                    }
                )
            if verify_log is not None:
                verify_log.append(
                    {
                        "input_count": len(doc_ids),
                        "relevant_count": sum(1 for v in out if v["relevant"]),
                        "score_distribution": {
                            str(s): sum(1 for v in out if v["score"] == s)
                            for s in (1, 2, 3, 4, 5)
                        },
                        "verdicts": [
                            {
                                "doc_id": v["doc_id"],
                                "score": v["score"],
                                "relevant": v["relevant"],
                                "reason": v["reason"][:200],
                            }
                            for v in out
                        ],
                    }
                )
            return out

        return {
            "verify_relevance_batch": verify_relevance_batch,
            "read_doc": read_doc,
        }

    return extensions


RLM_RERANK_SYSTEM_PROMPT = """\
You are a retrieval reranker working inside a Python REPL. You will be
given a SEARCH QUERY and a pool of 50 candidate documents that a
first-stage BM25 retriever already pulled from a corpus of mathematics
problems. Your only job is to choose the 10 best candidates from the
pool and assign them to a Python variable called ``top10_ids``. The
runtime will detect that variable and auto-finalize your answer — you do
not need to write any natural-language summary.

The REPL has these variables and helpers in scope (already defined; do
NOT redefine them):

* ``query`` (str): the search query string. Treat it as the user request.
* ``bm25_pool`` (list[dict]): exactly 50 candidates from BM25. Each item
  has keys ``doc_id`` (str), ``content`` (str), ``preview`` (str),
  ``score`` (float), ``metadata`` (dict).
* ``verify_relevance_batch(query, doc_ids) -> list[dict]``: your most
  powerful primitive. Sends each doc_id to a parallel sub-LLM that rates
  the passage's similarity to the query in technique / structure / approach
  on a 1-5 scale (5 = strong technique match, 1 = clearly unrelated).
  Returns
  ``[{"doc_id": str, "score": int, "relevant": bool, "reason": str}, ...]``
  where ``relevant`` is the convenience flag ``score >= 3``. ALL of the
  subcalls run in parallel — calling it once with 50 doc_ids is the
  cheapest way to score the entire pool.
* ``read_doc(doc_id) -> str``: full text of a corpus document.
* Standard pyrlm-runtime REPL helpers (``llm_query``, ``llm_batch``,
  ``llm_query_json``, ``ctx``, ``P``, etc.) are also available.

REQUIRED procedure — do not skip any step:

1. Run ``pool_ids = [d["doc_id"] for d in bm25_pool]`` so you have the
   list of 50 candidate IDs.
2. Run ``verdicts = verify_relevance_batch(query, pool_ids)`` exactly
   once. This is mandatory — without this call you cannot finalize. Each
   verdict carries an integer ``score`` from 1 (clearly unrelated) to 5
   (strong technique match).
3. Build ``top10_ids`` (a Python list of length 10) by ranking the entire
   pool primarily by the verifier ``score`` (descending), with the
   first-stage BM25 score as a tie-breaker among docs sharing the same
   verifier score:

       score_by_id = {v["doc_id"]: v["score"] for v in verdicts}
       bm25_by_id  = {d["doc_id"]: d["score"] for d in bm25_pool}
       ranked = sorted(
           bm25_pool,
           key=lambda d: (
               -score_by_id.get(d["doc_id"], 0),
               -bm25_by_id.get(d["doc_id"], 0.0),
           ),
       )
       top10_ids = [d["doc_id"] for d in ranked[:10]]

   Do NOT discard low-scoring docs — every position in the top-10 must be
   filled, and a low-scoring doc is still better than no doc at all.
4. Assign the final list to ``top10_ids`` in the REPL and print it for
   verification:

       top10_ids = [...]   # list of 10 real doc_id strings from bm25_pool
       print(top10_ids)

   As soon as ``top10_ids`` is a Python list whose 10 elements are real
   doc_id strings from the pool, the runtime auto-finalizes and returns
   it as your answer. Do not write any further commentary.

CRITICAL: the elements of ``top10_ids`` must be the actual doc_id values
that appear in ``bm25_pool`` — strings like
``"american-math-monthly___2015___11716"``, not placeholders like
``"doc_id_1"`` or ``"doc_1"``. If your final list does not consist of
real doc_ids from the pool the run is wasted.
"""
