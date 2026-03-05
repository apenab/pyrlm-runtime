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
   - ask(question, text): ask specific question about text
   - ask_chunks(question, chunks): ask question across chunks
   - ask_chunks_first(question, chunks): ask until first answer

5. SHOW_VARS(): lists all variables you created
6. Standard Python: you can use print(), variables, loops, etc.

You will only see truncated REPL outputs. Use sub-LLM functions (llm_query, ask, etc.) to analyze large text semantically. Sub-LLMs can handle ~500K characters, so you can feed them substantial chunks (e.g., 10 documents at once).

STRATEGY: Explore the context first, chunk it intelligently, send chunks to sub-LLMs, collect results, then synthesize your final answer.

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
    line_end = P.find("\n", start)
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
    answer = ask(question, "\n".join(docs))
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


def build_root_user_message(
    *,
    query: str,
    context_len: int,
    context_type: str = "string",
    num_documents: int = 1,
    document_lengths: List[int] | None = None,
    repl_executed: bool,
    last_stdout: str | None,
    last_error: str | None,
    step: int,
    max_steps: int,
) -> str:
    """Build the user message for each iteration of the root RLM loop.

    Includes rich metadata about the context as specified in the paper's prompt format.
    """
    stdout = last_stdout or "<none>"
    error = last_error or "<none>"

    # Build context metadata section (paper-aligned)
    if document_lengths and len(document_lengths) > 1:
        if len(document_lengths) <= 10:
            lengths_str = str(document_lengths)
        else:
            lengths_str = f"{document_lengths[:5]}...{document_lengths[-5:]}"
        context_info = (
            f"Context type: {context_type}\n"
            f"Total length: {context_len} chars\n"
            f"Number of documents: {num_documents}\n"
            f"Document lengths: {lengths_str}\n"
        )
    else:
        context_info = f"Context type: {context_type}\nTotal length: {context_len} chars\n"

    # Iteration-0 safeguard: mirrors original alexzhang13/rlm's approach of
    # telling the model it hasn't seen the context yet so it doesn't skip
    # straight to a final answer before exploring the REPL.
    if step == 1 and not repl_executed:
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
        f"Last REPL error:\n{error}"
    )


def build_iteration_message(
    *,
    last_stdout: str | None,
    last_error: str | None,
    step: int,
    max_steps: int,
) -> str:
    """Build a lightweight user message for subsequent RLM iterations.

    In conversation_history mode, the query and context metadata are already
    present in the initial user message (messages[1]).  Subsequent iteration
    messages only carry the REPL execution results and the step counter to
    avoid duplicating the query and context metadata on every turn.
    """
    stdout = last_stdout or "<none>"
    error = last_error or "<none>"
    return f"[REPL Result]\nstdout:\n{stdout}\n\nerror:\n{error}\n\nStep: {step}/{max_steps}"
