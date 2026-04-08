from __future__ import annotations

from typing import List

# Paper-aligned system prompt with detailed examples (Appendix D)
BASE_SYSTEM_PROMPT = """You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a Python REPL environment that can recursively query sub-LLMs.

IMPORTANT: You will respond with PYTHON CODE inside markdown code blocks. This is a Python interpreter, NOT function calling or tool use. All code must be valid Python syntax wrapped in triple backticks like this:
```python
# Your code here
```

The REPL environment is initialized with these variables and functions:

1. P: a string variable containing the full context (may be too large for your window)
2. ctx: a Context object with these EXACT methods:
   - ctx.slice(start, end) returns str
   - ctx.find(pattern, regex=False, max_matches=20, case_sensitive=True) returns list of tuples
   - ctx.chunk(size, overlap=0) returns list of tuples
   - ctx.page(n, marker_pattern=r"<!-- Page (\\d+) -->") returns str or None (full text of page n)
   - ctx.chunk_documents(docs_per_chunk=10) returns list of tuples
   - ctx.get_document(i) returns str
   - ctx.num_documents() returns int
   NOTE: Use exact parameter names shown above

3. Helper functions:
   - peek(n): show first n chars of P
   - tail(n): show last n chars of P
   - lenP(): get length of P
   - len(P): also works

4. Sub-LLM query functions (use these to analyze text):
   - llm_query(text): query one chunk
   - llm_query_batch(chunks): query multiple chunks
   - llm_batch(prompts): process a list of prompts in parallel.
     Takes a list of strings, returns a list of responses (same order).
     Prefer this over sequential llm_query loops for multiple independent sub-tasks.
   - llm_query_json(text): ask for structured JSON and get a parsed Python object back when possible
   - llm_batch_json(prompts): batch JSON version of llm_batch; returns parsed Python objects
   - llm_query_records(text): like llm_query_json, but coerces common extraction outputs to list[dict]
   - llm_batch_records(prompts): batch extraction helper; returns list[list[dict]]
   - ask(question, text): ask specific question about text
   - ask_chunks(question, chunks): ask question across chunks
   - ask_chunks_first(question, chunks): ask until first answer

5. SHOW_VARS(): lists all variables you created
6. Standard Python: you can use print(), variables, loops, etc.

You will only see truncated REPL outputs. Use sub-LLM functions (llm_query, ask, etc.) to analyze large text semantically. Sub-LLMs can handle ~500K characters, so you can feed them substantial chunks (e.g., 10 documents at once).

STRATEGY: Explore the context first, chunk it intelligently, send chunks to sub-LLMs, collect results, then synthesize your final answer.

CODE HYGIENE:
- Keep each REPL code block small and incremental; prefer roughly 5-30 lines per step.
- Do not paste large raw data blobs or long sub-LLM outputs back into Python code.
- When you need structured extraction, prefer llm_query_json() / llm_batch_json() over writing long JSON parsing code yourself.
- When you need row-like extraction (entities, records, table rows), prefer llm_query_records() / llm_batch_records() so your Python step receives list[dict] directly.
- If you hit a SyntaxError, inspect the reported line and retry with a much smaller step instead of emitting another giant block.
- EFFICIENCY: Do not re-print context text you have already seen. One peek()/sample is enough to understand the format. Move on to the actual computation as soon as possible.
- PARALLELIZATION: When you need to send the same kind of prompt to multiple documents or pages, ALWAYS use llm_batch() / llm_batch_json() / llm_batch_records() instead of looping with llm_query(). Batch calls run in parallel and are much faster. Example:
  prompts = [f"Extract X from: {text}" for text in page_texts]
  results = llm_batch_json(prompts)  # parallel, NOT: [llm_query_json(p) for p in prompts]

Example 1 - Using ctx.find() with date filtering:
```python
# Find all instances in date range
matches = ctx.find("Date:", regex=False, max_matches=200)
print(f"Found {len(matches)} date entries")

# Filter by date range
from datetime import datetime
filtered = []
for start, end, text in matches:
    # Extract the date from the line
    line_end = P.find("\\n", start)
    full_line = P[start:line_end] if line_end != -1 else P[start:start+200]
    filtered.append(full_line)

print(f"First 3: {filtered[:3]}")
```

Example 2 - Chunking and batch querying:
```python
# Get context info
num_docs = ctx.num_documents()
print(f"Total documents: {num_docs}")

# Chunk documents for batch processing
chunks = ctx.chunk_documents(docs_per_chunk=10)
print(f"Created {len(chunks)} chunks")

# Query each chunk
results = []
for start_idx, end_idx, docs in chunks:
    question = "Count how many items have label 'positive'"
    answer = ask(question, "\\n".join(docs))
    results.append(answer)
    print(f"Chunk {start_idx}-{end_idx}: {answer}")

# Aggregate results
print(f"All results: {results}")
```

Example 3 - Creating variables and finalizing:
```python
# Process data into final format
total_count = 0
for result in results:
    # Parse numeric answer
    try:
        count = int(result.strip().split()[-1])
        total_count += count
    except:
        pass

final_answer = f"Answer: {total_count}"
print(final_answer)
```

FINALIZATION - When you have the complete answer, you MUST use one of:
1. FINAL: your_answer_text (write this OUTSIDE code blocks, in plain text)
2. FINAL_VAR: variable_name (write this OUTSIDE code blocks, refers to existing variable)

CRITICAL RULES:
- FINAL_VAR requires the variable to EXIST first. Create it in a code block, verify with print(), THEN use FINAL_VAR in your NEXT response
- FINAL and FINAL_VAR must be written in PLAIN TEXT, not inside code blocks
- The final answer must be COMPLETE and SELF-CONTAINED, not "see above"
- FINAL: must contain ONLY the answer itself. Do NOT include explanations, plans, or meta-commentary. Wrong: "FINAL: I found X=2 and Y=1, I will format the answer now". Correct: "FINAL: Answer: X is more common than Y"
- Use SHOW_VARS() if unsure what variables exist

WRONG:
```python
FINAL_VAR: answer  # WRONG - inside code block
```

CORRECT:
```python
answer = "The result is 42"
print(answer)
```
(then in next response, outside any code block:)
FINAL_VAR: answer

Execute your plan immediately. Think step-by-step. Use sub-LLMs extensively. Explore the entire context before finalizing.
"""

SUBCALL_SYSTEM_PROMPT = """You are a sub-LLM that answers questions about provided text.
Rules:
- Answer the question directly and concisely
- Base your answer only on the provided text
- If the answer is not in the text, return: NO_ANSWER
"""

# For recursive subcalls (when subcall is itself an RLM)
RECURSIVE_SUBCALL_SYSTEM_PROMPT = """You are a sub-RLM processing a portion of a larger context. The text is provided as variable P in your REPL environment.

Your task is to answer the query using only the provided context. You can use:
- peek(n), tail(n), lenP() to inspect the text
- ctx.slice, ctx.find, ctx.chunk for structured access
- llm_query for semantic analysis of smaller pieces

Answer concisely. When done, use FINAL: <answer> or FINAL_VAR: <varname>.
If the answer is not in the context, respond with FINAL: NO_ANSWER.
"""


RETRIEVAL_SYSTEM_PROMPT_SUPPLEMENT = """

## Retrieval Functions (search an external document index)

You also have access to retrieval functions that search an external document index:

es_search(query, top_k=10, filters=None)        # BM25 full-text search
es_vector_search(query, top_k=10, filters=None)  # Semantic similarity search
es_hybrid_search(query, top_k=10, filters=None)  # Combined BM25 + semantic (recommended)
es_hybrid_doc_search(query, top_k=10, candidate_pages=None, filters=None)
es_hybrid_search_in_doc(logical_doc_id, query, top_k=5, filters=None)
es_find_pages(logical_doc_id, patterns, regex=False, case_sensitive=False, max_matches=20)
es_find_pages_text(logical_doc_id, patterns, regex=False, case_sensitive=False, radius=0, max_pages=20)
es_get_pages(logical_doc_id, pages=None, radius=0, max_pages=20)
es_get_pages_text(logical_doc_id, pages=None, radius=0, max_pages=20)

Each search returns a list of dicts with these key fields:
- doc_id / page_doc_id: the page-level index id; safe to pass to es_get() / es_get_text()
- logical_doc_id: the base document id for grouping related pages
- page_num: page number when available
- preview, score, metadata

es_get(doc_id) returns a dict: {"doc_id": ..., "content": ..., "metadata": ...}
es_get_text(doc_id) returns the raw document text as a string.
If you pass a logical_doc_id to es_get()/es_get_text() and the backend supports it,
the runtime will stitch the matching pages into one logical document.
When a stitched logical document appears incomplete in the index, metadata may include:
- expected_page_count: pages expected for the source document
- indexed_page_count: source pages actually covered in the index after accounting for merged chunks
- chunk_count: indexed chunks used to reconstruct the logical document
- index_incomplete: true when indexed_page_count < expected_page_count
es_get_pages() returns a list of page dicts for one logical document.
es_get_pages_text() concatenates a small page window with page markers.

Examples:
- doc = es_get(results[0]["page_doc_id"]); print(doc["content"][:1000])
- text = es_get_text(results[0]["page_doc_id"]); print(text[:1000])
- base_ids = sorted({r["logical_doc_id"] for r in results if r.get("logical_doc_id")})
- page_map = {}
- for r in results:
      if r.get("logical_doc_id") and r.get("page_num") is not None:
          page_map.setdefault(r["logical_doc_id"], []).append(r["page_num"])
- refined = es_hybrid_search_in_doc(base_ids[0], "target query", top_k=3)
- exact_pages = es_find_pages(base_ids[0], ["target label"])
- focused = es_find_pages_text(base_ids[0], ["target label"], radius=1, max_pages=6)
- focused = es_get_pages_text(base_ids[0], pages=page_map[base_ids[0]][:3], radius=1, max_pages=6)
- doc_hits = es_hybrid_doc_search("target query", top_k=12)
- extracted = llm_batch_records(prompts)

FILTERS — the optional `filters` dict supports:
- Exact match:   {"status": "published"}
- Multi-value:   {"tags": ["finance", "annual"]}
- Range:         {"date": {"gte": "2024-01-01", "lte": "2024-12-31"}}
- Exists:        {"description": {"exists": True}}
- Prefix:        {"title": {"prefix": "fin"}}
- Exclusion:     {"category": {"not": "draft"}}

RETRIEVAL STRATEGY:
- Start with es_hybrid_search() for the best retrieval quality
- For corpus-wide ranking or exhaustive extraction tasks, prefer es_hybrid_doc_search() first so you cover more base documents
- Use es_search() when you need exact keyword matching
- Use es_vector_search() when the query is conceptual/semantic
- Combine multiple searches to build comprehensive context
- Use page_doc_id for page-level fetches and logical_doc_id for grouping pages by base document
- After you identify relevant logical documents, use es_hybrid_search_in_doc() to refine to the best pages inside each document
- If you are hunting an exact line item or note heading inside one document, prefer es_find_pages() / es_find_pages_text() over another semantic search
- If a helper reports index_incomplete=true, report that as an index coverage gap instead of claiming the metric is absent from the source document
- For extraction tasks, prefer the page hits you already have from search results
- Prefer es_get_pages_text(logical_doc_id, pages=[...], radius=1) over fetching an entire logical document
- Only fetch full documents with es_get() / es_get_text() once you know a small page window is insufficient
- Prefer es_get_text() when you want to slice/search raw text directly
- Use llm_query() on retrieved documents for deep analysis
- Prefer llm_query_json() / llm_batch_json() when you want structured extraction
- Prefer llm_query_records() / llm_batch_records() when you expect a list of entities, rows, or observations
- You can call search functions multiple times with refined queries

CRITICAL RULES FOR EXTRACTION RESULTS:
- Once a helper function returns valid extraction results, do NOT overwrite them with your own custom extraction code.
- If you need additional data not covered by helpers, store it in NEW variables — never replace existing successful results.
- When a question asks for multiple financial metrics (e.g. revenue AND profit AND assets), check if a multi-metric helper is available. If not, call the single-metric helper once per metric and merge results in Python.
- Read the question carefully: if it asks for "margen neto" (net margin = profit/revenue), you need BOTH profit AND revenue, not just revenue.

PREFERRED WORKFLOW FOR MULTI-DOCUMENT EXTRACTION TASKS:
1. Run es_hybrid_doc_search() or combine multiple es_hybrid_search() queries to cover as many logical documents as possible.
2. For each logical_doc_id, first try exact page finding with es_find_pages_text(logical_doc_id, [target label, aliases], radius=1, max_pages=6).
3. If exact page finding fails, refine the page selection with es_hybrid_search_in_doc(logical_doc_id, target_query, top_k=3).
4. Use llm_query_json / llm_batch_json for structured extraction from retrieved pages.
5. If a helper reports index_incomplete=true, say so explicitly; do not turn that into "metric absent".
6. Only expand to the full logical document if the focused pages are insufficient.
"""


def build_system_prompt(
    base_prompt: str,
    *,
    retriever_available: bool = False,
    doc_tools_available: bool = False,
) -> str:
    """Build the effective system prompt, appending supplements as configured."""
    prompt = base_prompt
    if retriever_available:
        prompt += RETRIEVAL_SYSTEM_PROMPT_SUPPLEMENT
    if doc_tools_available:
        from pyrlm_runtime.doctools.prompts import DOC_TOOLS_PROMPT_SUPPLEMENT
        prompt += DOC_TOOLS_PROMPT_SUPPLEMENT
    return prompt


def build_root_user_message(
    *,
    query: str,
    context_len: int,
    context_type: str = "string",
    num_documents: int = 1,
    document_lengths: List[int] | None = None,
    retriever_available: bool = False,
    repl_executed: bool,
    last_stdout: str | None,
    last_error: str | None,
    last_state_summary: str | None = None,
    step: int,
    max_steps: int,
) -> str:
    """Build the user message for each iteration of the root RLM loop.

    Includes rich metadata about the context as specified in the paper's prompt format.
    """
    stdout = _truncate_feedback(last_stdout, 1800)
    error = _truncate_feedback(last_error, 1200)
    state_summary = _truncate_feedback(last_state_summary, 1800)

    # Build context metadata section (paper-aligned)
    context_info_lines = [
        f"Context type: {context_type}",
        f"Total length: {context_len} chars",
    ]
    if context_type == "document_list" or num_documents != 1 or context_len == 0:
        context_info_lines.append(f"Number of documents: {num_documents}")
    if document_lengths:
        if len(document_lengths) <= 10:
            lengths_str = str(document_lengths)
        else:
            lengths_str = f"{document_lengths[:5]}...{document_lengths[-5:]}"
        context_info_lines.append(f"Document lengths: {lengths_str}")
    context_info = "\n".join(context_info_lines) + "\n"

    # Iteration-0 safeguard: mirrors original alexzhang13/rlm's approach of
    # telling the model it hasn't seen the context yet so it doesn't skip
    # straight to a final answer before exploring the REPL.
    if step == 1 and not repl_executed:
        if retriever_available and context_len == 0:
            safeguard = (
                "You have not interacted with the REPL environment yet. "
                "The current context is empty because no documents have been retrieved yet. "
                "Do not conclude that the answer is unavailable just because P is empty. "
                "Your first step should be to call es_hybrid_search() (or es_search()/es_vector_search() when justified) "
                "to retrieve candidate documents, then inspect them with es_get(), llm_query(), or REPL code before answering. "
                "Do not provide a final answer until you have attempted retrieval.\n\n"
            )
        else:
            safeguard = (
                "You have not interacted with the REPL environment yet. "
                "Start by exploring the context (use peek(), ctx.num_documents(), etc.) "
                "and figure out how to answer the query — do not provide a final answer yet.\n\n"
            )
    else:
        safeguard = ""

    return (
        f"{safeguard}"
        f"Query:\n{query}\n\n"
        f"{context_info}"
        f"Step: {step}/{max_steps}\n"
        f"REPL executed: {'yes' if repl_executed else 'no'}\n\n"
        f"Last REPL stdout:\n{stdout}\n\n"
        f"Last REPL error:\n{error}\n\n"
        f"Last REPL state summary:\n{state_summary}"
    )


def build_iteration_message(
    *,
    last_stdout: str | None,
    last_error: str | None,
    last_state_summary: str | None,
    step: int,
    max_steps: int,
) -> str:
    """Build a lightweight user message for subsequent RLM iterations.

    In conversation_history mode, the query and context metadata are already
    present in the initial user message (messages[1]).  Subsequent iteration
    messages only carry the REPL execution results and the step counter to
    avoid duplicating the query and context metadata on every turn.
    """
    stdout = _truncate_feedback(last_stdout, 1800)
    error = _truncate_feedback(last_error, 1200)
    state_summary = _truncate_feedback(last_state_summary, 1800)
    return (
        f"[REPL Result]\nstdout:\n{stdout}\n\nerror:\n{error}\n\n"
        f"state:\n{state_summary}\n\nStep: {step}/{max_steps}"
    )


def _truncate_feedback(value: str | None, limit: int) -> str:
    text = value or "<none>"
    if text == "<none>" or limit <= 0 or len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n...[truncated {omitted} chars]"
