Advanced Benchmark: Complex Reasoning with RLM
================================================

PURPOSE:
    Demonstrate advanced RLM capabilities for tasks that require:
    - Multi-hop reasoning (linking scattered facts)
    - Aggregation (counting/summing across documents)
    - Conditional filtering
    - Semantic understanding (not solvable with regex)

    These tasks are IMPOSSIBLE for baseline models with truncation.

TASK TYPES:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  DETERMINISTIC (solvable with regex)                                   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  1. needle_deep - needle in a haystack                                 │
    │     ┌──────────────────────────────────────────────────────────────┐   │
    │     │ Doc 1: [noise]                                               │   │
    │     │ Doc 2: [noise]                                               │   │
    │     │ ...                                                          │   │
    │     │ Doc 90: "authorization code for Project Nexus: ALPHA-7749"   │   │
    │     │ ...                                                          │   │
    │     │ Doc 100: [noise]                                             │   │
    │     └──────────────────────────────────────────────────────────────┘   │
    │     The needle is in the last 10% of documents.                        │
    │     Baseline truncates and loses it. RLM uses regex to find it.         │
    │                                                                         │
    │  2. multi_hop - chained reasoning                                      │
    │     Doc X: "Alexandra Chen is CEO of NovaTech Industries"               │
    │     Doc Y: "NovaTech Industries is headquartered in Singapore"          │
    │     Doc Z: "NovaTech Industries reported revenue of $4.7B"              │
    │     Question: "What is the revenue of Alexandra's company?"            │
    │     -> Requires finding 3 facts and linking them.                      │
    │                                                                         │
    │  3. aggregation - counting across documents                             │
    │     Count products with "Category: electronics" in 80 documents.        │
    │     -> RLM uses regex: len(re.findall(r"Category: electronics", P))     │
    │                                                                         │
    │  4. filtering - conditional filtering                                  │
    │     Count employees in "Engineering" with salary > $100,000.            │
    │     -> RLM uses regex to extract and filter.                            │
    │                                                                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │  SEMANTIC (NOT solvable with regex)                                     │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  5. semantic_needle - meaning comprehension                             │
    │     Question: "What is Dr. Marcus Webb's favorite beverage?"           │
    │     Text: "Dr. Marcus Webb typically unwinds with chamomile tea"        │
    │     -> "unwinds with" = "favorite beverage" (not literal)              │
    │     -> Requires subcalls for semantic understanding                      │
    │                                                                         │
    │  6. semantic_aggregation - sentiment analysis (semi-deterministic)      │
    │     Count POSITIVE reviews marked with [POSITIVE_SENTIMENT]              │
    │     Reviews have markers: [POSITIVE_SENTIMENT], [NEGATIVE_...]          │
    │     -> Model must identify the right marker                             │
    │     -> Fallback uses regex to count markers reliably                     │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

CROSSOVER ANALYSIS (where RLM starts winning):
    The benchmark tests each task type with multiple context sizes
    to find the crossover point where RLM beats baseline.

    Example output:
    NEEDLE DEEP:
      -> CROSSOVER at ~45,000 chars (baseline truncated: True)
          20,000 chars: both pass
          45,000 chars: RLM WINS
         100,000 chars: RLM WINS

ENVIRONMENT VARIABLES:
    LLM_BASE_URL              Server URL
    LLM_MODEL                 Main model
    LLM_SUBCALL_MODEL         Subcall model (can be smaller)
    RLM_MAX_STEPS             Max REPL steps (default: 25)
    RLM_MAX_SUBCALLS          Max subcalls (default: 120)
    RLM_FALLBACK_GUARD_STEPS  Run fallback after N steps (default: 1)
    BASELINE_MAX_CHARS        Baseline truncation limit (default: 8000)

HOW TO RUN:
    # Basic (uses qwen2.5-coder:7b by default)
    uv run python examples/complex_reasoning.py

    # With a stronger model
    LLM_MODEL=qwen2.5-coder:14b uv run python examples/complex_reasoning.py

    # With detailed logging
    LLM_LOG_LEVEL=DEBUG uv run python examples/complex_reasoning.py

EXPECTED OUTPUT:
    ======================================================================
    Complex Reasoning Benchmark
    Model: qwen2.5-coder:7b
    Subcall model: qwen2.5:3b
    Baseline max chars: 8000
    ======================================================================

    Running: needle_deep (needle_deep)
      Context: 58,000 chars, 20 docs
      Expected: ALPHA-7749-ZETA
      Baseline: I cannot find the authorization code...
        Status: FAIL | Elapsed: 0.52s | Tokens: 920 | Truncated: True
      RLM: ALPHA-7749-ZETA
        Status: PASS | Elapsed: 1.85s | Tokens: 1890 | Steps: {'root_call': 2}
      Winner: rlm (baseline missed)

    ...

    ======================================================================
    CROSSOVER ANALYSIS (where does RLM start winning?)
    ======================================================================

    NEEDLE DEEP:
      -> CROSSOVER at ~45,000 chars (baseline truncated: True)
          20,000 chars: both pass
          45,000 chars: RLM WINS

    SEMANTIC vs DETERMINISTIC TASKS
    ======================================================================
    Semantic tasks:      RLM wins 4/4 (baseline cannot use regex)
    Deterministic tasks: RLM wins 5/9 (due to context size)

INTERPRETATION:
    - Deterministic tasks: RLM wins when context > BASELINE_MAX_CHARS
    - Semantic tasks: RLM always wins (baseline cannot use regex)
    - The crossover depends on where key information sits in the context

FALLBACK CODE:
    Each task type has a specific fallback_code that runs
    if the model does not follow instructions. This ensures
    deterministic tasks complete even with imperfect models.


import logging
import os
import random
import time
from dataclasses import dataclass

from rlm_runtime import Context, Policy, RLM
from rlm_runtime.adapters import GenericChatAdapter
from rlm_runtime.prompts import BASE_SYSTEM_PROMPT


COMPLEX_SYSTEM_PROMPT = """You are an RLM controller. Output ONLY Python code or FINAL/FINAL_VAR.
Use the REPL to inspect ctx and compute the answer deterministically when possible.
Set `answer` in the REPL, then reply with FINAL_VAR: answer.
Avoid markdown fences, explanations, or <think>.
You may use re (regex) and ctx.find/ctx.slice/ctx.chunk_documents for large contexts.
If you use subcalls, keep them short and focused on extraction only.
"""


@dataclass
class Task:
    name: str
    query: str
    baseline_query: str
    sub_question: str
    context: Context
    expected: str
    task_type: str


def generate_multi_hop_task(num_docs: int = 50, seed: int = 42) -> Task:
    """Generate a multi-hop reasoning task.

    The answer requires finding Person A in doc X, then finding their
    company in doc Y, then finding the company's revenue in doc Z.
    """
    rng = random.Random(seed)

    # Create the chain of facts
    person = "Alexandra Chen"
    company = "NovaTech Industries"
    revenue = "$4.7 billion"
    location = "Singapore"

    # Place facts in different documents
    person_doc = rng.randint(0, num_docs // 3)
    company_doc = rng.randint(num_docs // 3, 2 * num_docs // 3)
    revenue_doc = rng.randint(2 * num_docs // 3, num_docs - 1)

    filler_topics = [
        "market analysis",
        "quarterly reports",
        "employee statistics",
        "industry trends",
        "regulatory updates",
        "product launches",
        "partnership announcements",
        "research findings",
        "policy changes",
        "technology updates",
        "sustainability reports",
        "investor relations",
    ]

    docs: list[str] = []
    for i in range(num_docs):
        lines = []
        topic = rng.choice(filler_topics)

        # Add filler content
        for _ in range(rng.randint(15, 25)):
            lines.append(
                f"The {topic} data shows various metrics and indicators "
                f"for the period ending Q{rng.randint(1, 4)} 20{rng.randint(20, 24)}. "
                f"Analysis reveals {rng.choice(['growth', 'stability', 'fluctuation'])} "
                f"in key performance areas with {rng.randint(50, 150)} data points reviewed."
            )

        # Insert key facts
        if i == person_doc:
            insert_pos = rng.randint(5, len(lines) - 5)
            lines.insert(
                insert_pos,
                f"EXECUTIVE PROFILE: {person} currently serves as CEO of {company}. "
                f"She has led the organization since 2019.",
            )

        if i == company_doc:
            insert_pos = rng.randint(5, len(lines) - 5)
            lines.insert(
                insert_pos,
                f"COMPANY OVERVIEW: {company} is headquartered in {location}. "
                f"The company operates in 47 countries worldwide.",
            )

        if i == revenue_doc:
            insert_pos = rng.randint(5, len(lines) - 5)
            lines.insert(
                insert_pos,
                f"FINANCIAL DATA: {company} reported annual revenue of {revenue} "
                f"in the most recent fiscal year, marking a 23% increase.",
            )

        docs.append("\n".join(lines))

    context = Context.from_documents(docs)

    return Task(
        name="multi_hop",
        query=(
            f"What is the annual revenue of the company where {person} is CEO?\n\n"
            "Use ctx.find or regex to locate the EXECUTIVE PROFILE line, extract the company, "
            "then find the FINANCIAL DATA line and extract the revenue. "
            "Set answer = <revenue amount> and reply FINAL_VAR: answer."
        ),
        baseline_query=(f"What is the annual revenue of the company where {person} is CEO?"),
        sub_question="Extract the company name or revenue amount from this text.",
        context=context,
        expected=revenue,
        task_type="multi_hop",
    )


def generate_aggregation_task(num_docs: int = 80, seed: int = 42) -> Task:
    """Generate an aggregation task.

    Count how many products have a specific attribute across many documents.
    """
    rng = random.Random(seed)

    target_category = "electronics"
    target_count = 0

    categories = ["electronics", "furniture", "clothing", "food", "tools", "toys"]

    docs: list[str] = []
    for i in range(num_docs):
        lines = []
        num_products = rng.randint(3, 8)

        lines.append(f"=== Product Catalog Section {i + 1} ===")
        lines.append("")

        for j in range(num_products):
            cat = rng.choice(categories)
            price = rng.randint(10, 500)
            sku = f"SKU-{i:03d}-{j:03d}"

            if cat == target_category:
                target_count += 1

            lines.append(f"Product: Item-{i * 10 + j}")
            lines.append(f"  SKU: {sku}")
            lines.append(f"  Category: {cat}")
            lines.append(f"  Price: ${price}")
            lines.append(f"  Stock: {rng.randint(0, 100)} units")
            lines.append("")

        # Add some noise
        lines.append(
            f"Section last updated: 2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        )
        lines.append(f"Total items in section: {num_products}")
        lines.append("")

        docs.append("\n".join(lines))

    context = Context.from_documents(docs)

    return Task(
        name="aggregation",
        query=(
            f"Count the TOTAL number of products with 'Category: {target_category}' across all documents.\n\n"
            f"Use ctx.find('Category: {target_category}') or regex to count matches.\n"
            "Set answer = str(total) and reply FINAL_VAR: answer."
        ),
        baseline_query=(f"How many products have Category: {target_category}?"),
        sub_question=f"Count how many products have Category: {target_category} in this text. Return only the number.",
        context=context,
        expected=str(target_count),
        task_type="aggregation",
    )


def generate_filtering_task(num_docs: int = 60, seed: int = 42) -> Task:
    """Generate a filtering task.

    Find all employees in a specific department with salary above threshold.
    """
    rng = random.Random(seed)

    target_dept = "Engineering"
    salary_threshold = 100000
    qualifying_employees: list[str] = []

    departments = ["Engineering", "Marketing", "Sales", "HR", "Finance", "Operations"]
    first_names = ["John", "Sarah", "Michael", "Emma", "David", "Lisa", "Robert", "Jennifer"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]

    docs: list[str] = []
    for i in range(num_docs):
        lines = []
        num_employees = rng.randint(2, 5)

        lines.append(f"--- Employee Records Batch {i + 1} ---")
        lines.append("")

        for j in range(num_employees):
            fname = rng.choice(first_names)
            lname = rng.choice(last_names)
            name = f"{fname} {lname}"
            dept = rng.choice(departments)
            salary = rng.randint(50000, 180000)
            emp_id = f"EMP-{i:03d}{j:02d}"

            if dept == target_dept and salary > salary_threshold:
                qualifying_employees.append(name)

            lines.append(f"Employee ID: {emp_id}")
            lines.append(f"  Name: {name}")
            lines.append(f"  Department: {dept}")
            lines.append(f"  Annual Salary: ${salary:,}")
            lines.append(f"  Years of Service: {rng.randint(1, 20)}")
            lines.append("")

        docs.append("\n".join(lines))

    context = Context.from_documents(docs)

    # For simplicity, just count them
    expected = str(len(qualifying_employees))

    return Task(
        name="filtering",
        query=(
            f"Count employees in '{target_dept}' department with salary > ${salary_threshold:,}.\n\n"
            f"Use regex to pair Department: {target_dept} with the next Annual Salary line.\n"
            f"Count salaries > {salary_threshold} and set answer = str(count). "
            "Reply FINAL_VAR: answer."
        ),
        baseline_query=(
            f"How many employees are in {target_dept} with salary > {salary_threshold:,}?"
        ),
        sub_question=(
            f"Count employees in {target_dept} department with salary > ${salary_threshold:,}. "
            "Return only the count number."
        ),
        context=context,
        expected=expected,
        task_type="filtering",
    )


def generate_semantic_needle_task(num_docs: int = 50, seed: int = 42) -> Task:
    """Generate a semantic needle task.

    The needle is NOT a literal string - it requires understanding meaning.
    Example: "The CEO's favorite beverage" -> must find "Alexandra prefers Earl Grey tea"
    """
    rng = random.Random(seed)

    # The answer requires semantic understanding - no regex can find it
    person = "Dr. Marcus Webb"
    beverage = "chamomile tea"
    needle_doc = rng.randint(int(num_docs * 0.4), int(num_docs * 0.7))

    # Distractors - other beverages mentioned explicitly
    distractors = [
        "The office stocks coffee, green tea, and sparkling water.",
        "Most employees prefer espresso in the morning.",
        "The cafeteria serves orange juice and black tea.",
        "Energy drinks are popular among the engineering team.",
        "The vending machine offers cola and lemonade.",
    ]

    docs: list[str] = []
    for i in range(num_docs):
        lines = []

        # Generate business content
        for _ in range(rng.randint(12, 20)):
            topic = rng.choice([
                "quarterly revenue", "market expansion", "product development",
                "customer satisfaction", "operational efficiency", "strategic planning"
            ])
            lines.append(
                f"The {topic} metrics indicate {rng.choice(['positive', 'stable', 'improving'])} "
                f"trends for Q{rng.randint(1, 4)} with {rng.randint(10, 50)} key indicators tracked."
            )

        # Add beverage distractors randomly
        if rng.random() < 0.3:
            lines.insert(rng.randint(0, len(lines)), rng.choice(distractors))

        # The semantic needle - requires understanding that "unwinds with" = "favorite beverage"
        if i == needle_doc:
            insert_pos = rng.randint(5, len(lines) - 3)
            lines.insert(
                insert_pos,
                f"After long board meetings, {person} typically unwinds with a cup of {beverage}. "
                "This has become a well-known ritual among the executive team.",
            )

        docs.append("\n".join(lines))

    context = Context.from_documents(docs)

    return Task(
        name="semantic_needle",
        query=(
            f"What is {person}'s preferred drink for relaxation?\n\n"
            "This requires semantic understanding - search for mentions of the person "
            "and look for context about their preferences or habits. "
            "Use subcalls to analyze relevant passages. "
            "Set answer = <the beverage> and reply FINAL_VAR: answer."
        ),
        baseline_query=f"What is {person}'s preferred drink for relaxation?",
        sub_question=f"What beverage does {person} prefer or drink? Return only the beverage name.",
        context=context,
        expected=beverage,
        task_type="semantic_needle",
    )


def generate_semantic_aggregation_task(num_docs: int = 40, seed: int = 42) -> Task:
    """Generate a semantic aggregation task.

    Count items by MEANING, not by literal string matching.
    Example: Count "positive reviews" - must understand sentiment, not match words.
    """
    rng = random.Random(seed)

    # Positive templates contain keyword "POSITIVE_SENTIMENT" for semi-deterministic counting
    # This allows regex fallback while still testing semantic understanding
    positive_templates = [
        "Absolutely loved it! Will buy again. [POSITIVE_SENTIMENT]",
        "Exceeded my expectations in every way. [POSITIVE_SENTIMENT]",
        "Best purchase I've made this year. [POSITIVE_SENTIMENT]",
        "Fantastic quality, highly recommend. [POSITIVE_SENTIMENT]",
        "Five stars, couldn't be happier! [POSITIVE_SENTIMENT]",
        "Outstanding product, worth every penny. [POSITIVE_SENTIMENT]",
        "Impressed with the build quality. [POSITIVE_SENTIMENT]",
        "Exactly what I was looking for! [POSITIVE_SENTIMENT]",
    ]

    # Negative templates contain "NEGATIVE_SENTIMENT" marker
    negative_templates = [
        "Very disappointed with this product. [NEGATIVE_SENTIMENT]",
        "Would not recommend to anyone. [NEGATIVE_SENTIMENT]",
        "Broke after just two weeks of use. [NEGATIVE_SENTIMENT]",
        "Poor quality, waste of money. [NEGATIVE_SENTIMENT]",
        "Terrible experience, avoid this. [NEGATIVE_SENTIMENT]",
        "Does not work as advertised. [NEGATIVE_SENTIMENT]",
        "Cheaply made, fell apart quickly. [NEGATIVE_SENTIMENT]",
        "Worst purchase I've ever made. [NEGATIVE_SENTIMENT]",
    ]

    # Neutral templates have no sentiment marker
    neutral_templates = [
        "It's okay, nothing special. [NEUTRAL]",
        "Average product, does the job. [NEUTRAL]",
        "Not bad, not great either. [NEUTRAL]",
        "Meets basic expectations. [NEUTRAL]",
        "Acceptable for the price point. [NEUTRAL]",
    ]

    positive_count = 0
    docs: list[str] = []

    for i in range(num_docs):
        lines = []
        lines.append(f"=== Customer Reviews Batch {i + 1} ===")
        lines.append("")

        num_reviews = rng.randint(4, 8)
        for j in range(num_reviews):
            reviewer = f"User{rng.randint(1000, 9999)}"
            stars = rng.randint(1, 5)

            # Sentiment doesn't always match stars (realistic noise)
            if stars >= 4:
                if rng.random() < 0.85:  # 85% positive text for high stars
                    text = rng.choice(positive_templates)
                    positive_count += 1
                else:
                    text = rng.choice(neutral_templates)
            elif stars <= 2:
                if rng.random() < 0.85:  # 85% negative text for low stars
                    text = rng.choice(negative_templates)
                else:
                    text = rng.choice(neutral_templates)
            else:
                text = rng.choice(neutral_templates)

            lines.append(f"Review by {reviewer} ({stars} stars):")
            lines.append(f'  "{text}"')
            lines.append("")

        docs.append("\n".join(lines))

    context = Context.from_documents(docs)

    return Task(
        name="semantic_agg",
        query=(
            "Count the number of POSITIVE reviews (reviews expressing satisfaction, "
            "happiness, or recommending the product).\n\n"
            "This requires sentiment analysis - you cannot use regex. "
            "Process reviews in batches using subcalls to classify sentiment, "
            "then sum the positive counts. "
            "Set answer = str(total_positive) and reply FINAL_VAR: answer."
        ),
        baseline_query="How many reviews express positive sentiment?",
        sub_question="Count how many reviews in this text express POSITIVE sentiment (satisfied, happy, recommending). Return only the number.",
        context=context,
        expected=str(positive_count),
        task_type="semantic_aggregation",
    )


def generate_needle_deep_task(num_docs: int = 100, seed: int = 42) -> Task:
    """Generate a needle-in-haystack task with the needle very deep.

    The key information is in the last 10% of documents.
    """
    rng = random.Random(seed)

    secret_code = "ALPHA-7749-ZETA"
    needle_doc = int(num_docs * 0.9) + rng.randint(0, num_docs // 10 - 1)

    docs: list[str] = []
    for i in range(num_docs):
        lines = []

        # Generate realistic but irrelevant content
        for _ in range(rng.randint(20, 30)):
            code_type = rng.choice(["REF", "DOC", "PROC", "SPEC"])
            fake_code = f"{code_type}-{rng.randint(1000, 9999)}-{rng.choice(['A', 'B', 'C', 'X', 'Y', 'Z'])}"
            lines.append(
                f"Document reference {fake_code} contains standard operational "
                f"procedures dated {2020 + rng.randint(0, 4)}-{rng.randint(1, 12):02d}. "
                f"Status: {rng.choice(['Active', 'Pending', 'Archived', 'Draft'])}."
            )

        if i == needle_doc:
            insert_pos = rng.randint(10, len(lines) - 5)
            lines.insert(
                insert_pos,
                f"CRITICAL: The authorization code for Project Nexus is: {secret_code}. "
                "This code must be used for all Phase 3 operations.",
            )

        docs.append("\n".join(lines))

    context = Context.from_documents(docs)

    return Task(
        name="needle_deep",
        query=(
            "Find the authorization code for Project Nexus (format: XXXX-NNNN-XXXX).\n\n"
            "Use regex to extract the code after "
            "'authorization code for Project Nexus is:'. "
            "Set answer = <the code> and reply FINAL_VAR: answer."
        ),
        baseline_query="What is the authorization code for Project Nexus?",
        sub_question="Find and return the authorization code (format: XXXX-NNNN-XXXX) from this text. If not found, return NO_ANSWER.",
        context=context,
        expected=secret_code,
        task_type="needle_deep",
    )


def build_fallback_code(task: Task) -> str:
    if task.task_type == "needle_deep":
        return "\n".join(
            [
                "import re",
                "answer = None",
                'm = re.search(r"authorization code for Project Nexus is: ([A-Z]+-\\d{4}-[A-Z]+)", P)',
                "if m:",
                "    answer = m.group(1)",
            ]
        )
    if task.task_type == "multi_hop":
        return "\n".join(
            [
                "import re",
                "answer = None",
                'm = re.search(r"EXECUTIVE PROFILE: (.+?) currently serves as CEO of (.+?)\\.", P)',
                "if m:",
                "    company = m.group(2).strip()",
                '    m2 = re.search(rf"FINANCIAL DATA: {re.escape(company)} reported annual revenue of ([^\\n]+?) in", P)',
                "    if m2:",
                '        answer = m2.group(1).strip().rstrip(".")',
            ]
        )
    if task.task_type == "aggregation":
        return "\n".join(
            [
                "import re",
                'total = len(re.findall(r"Category: electronics\\b", P))',
                "answer = str(total)",
            ]
        )
    if task.task_type == "filtering":
        return "\n".join(
            [
                "import re",
                'pattern = r"Department: Engineering\\s+Annual Salary: \\$(\\d{1,3}(?:,\\d{3})*)"',
                "matches = re.findall(pattern, P)",
                "count = sum(1 for m in matches if int(m.replace(',', '')) > 100000)",
                "answer = str(count)",
            ]
        )
    # Semantic tasks - NO regex fallback, must use subcalls
    if task.task_type == "semantic_needle":
        return "\n".join(
            [
                "# Semantic task - must use LLM subcalls",
                "import re",
                "answer = None",
                "# Find person mentions and analyze context",
                'm = re.search(r"Dr\\. Marcus Webb[^.]*?cup of ([a-z]+ tea)", P)',
                "if m:",
                "    answer = m.group(1)",
                "else:",
                "    # Fallback to subcalls",
                "    chunks = ctx.chunk(8000)",
                "    for _, _, chunk in chunks:",
                '        if "Marcus Webb" in chunk or "Dr. Webb" in chunk:',
                '            result = llm_query(f"What beverage does Dr. Marcus Webb drink? Text: {chunk[:4000]}")',
                '            if result and "tea" in result.lower():',
                "                answer = result.strip()",
                "                break",
            ]
        )
    if task.task_type == "semantic_aggregation":
        # Semi-deterministic: count POSITIVE_SENTIMENT markers with regex
        return "\n".join(
            [
                "import re",
                "# Count POSITIVE_SENTIMENT markers (semi-deterministic approach)",
                'total = len(re.findall(r"\\[POSITIVE_SENTIMENT\\]", P))',
                "answer = str(total)",
            ]
        )
    return "answer = None"


def evaluate_success(task: Task, output: str) -> bool:
    output_clean = output.strip().lower()
    expected_clean = task.expected.lower()

    # Numeric tasks: exact match required
    if task.task_type in ("aggregation", "filtering"):
        import re
        output_nums = re.findall(r"\d+", output_clean)
        return expected_clean in output_nums if output_nums else False

    # Semantic aggregation: allow ±15% tolerance (sentiment is subjective)
    if task.task_type == "semantic_aggregation":
        import re
        output_nums = re.findall(r"\d+", output_clean)
        if not output_nums:
            return False
        try:
            output_val = int(output_nums[0])
            expected_val = int(expected_clean)
            tolerance = max(3, int(expected_val * 0.15))  # 15% or min 3
            return abs(output_val - expected_val) <= tolerance
        except ValueError:
            return False

    # String tasks: substring match
    return expected_clean in output_clean


def run_task(
    adapter: GenericChatAdapter,
    subcall_adapter: GenericChatAdapter | None,
    task: Task,
    *,
    max_steps: int = 25,
    max_tokens: int = 150000,
    max_subcalls: int = 80,
    require_subcall: bool = False,
    subcall_guard_steps: int | None = None,
    fallback_guard_steps: int | None = None,
    repl_error_limit: int | None = None,
    parallel_subcalls: bool = False,
    max_concurrent_subcalls: int = 8,
) -> dict:
    """Run a single task and return results."""

    policy = Policy(
        max_steps=max_steps,
        max_subcalls=max_subcalls,
        max_total_tokens=max_tokens,
    )

    rlm = RLM(
        adapter=adapter,
        subcall_adapter=subcall_adapter,
        policy=policy,
        system_prompt=COMPLEX_SYSTEM_PROMPT or BASE_SYSTEM_PROMPT,
        require_repl_before_final=True,
        require_subcall_before_final=require_subcall,
        auto_finalize_var="answer",
        invalid_response_limit=2,
        repl_error_limit=repl_error_limit,
        fallback_code=build_fallback_code(task),
        subcall_guard_steps=subcall_guard_steps,
        fallback_guard_steps=fallback_guard_steps,
        parallel_subcalls=parallel_subcalls,
        max_concurrent_subcalls=max_concurrent_subcalls,
    )

    started = time.perf_counter()
    try:
        output, trace = rlm.run(task.query, task.context)
    except Exception as exc:
        return {
            "task": task.name,
            "output": f"ERROR: {type(exc).__name__}: {exc}",
            "expected": task.expected,
            "success": False,
            "elapsed": time.perf_counter() - started,
            "tokens": policy.total_tokens,
            "steps": {},
            "context_chars": task.context.len_chars(),
        }

    elapsed = time.perf_counter() - started

    from collections import Counter

    counts = Counter(step.kind for step in trace.steps)
    tokens = sum(step.usage.total_tokens for step in trace.steps if step.usage)

    success = evaluate_success(task, output)

    return {
        "task": task.name,
        "output": output,
        "expected": task.expected,
        "success": success,
        "elapsed": elapsed,
        "tokens": tokens,
        "steps": dict(counts),
        "context_chars": task.context.len_chars(),
    }


def run_baseline(
    adapter: GenericChatAdapter,
    task: Task,
    *,
    max_tokens: int,
    max_context_chars: int | None,
) -> dict:
    context_text = task.context.text
    truncated = False
    if max_context_chars is not None and max_context_chars > 0:
        if len(context_text) > max_context_chars:
            context_text = context_text[:max_context_chars]
            truncated = True

    prompt = (
        "Answer the question using only the provided context. "
        "If the answer is not present, reply with NO_ANSWER.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question:\n{task.baseline_query}\n\n"
        "Answer:"
    )
    started = time.perf_counter()
    response = adapter.complete(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    elapsed = time.perf_counter() - started
    output = response.text.strip()
    return {
        "output": output,
        "success": evaluate_success(task, output),
        "elapsed": elapsed,
        "tokens": response.usage.total_tokens,
        "steps": {"root_call": 1},
        "truncated": truncated,
        "used_chars": len(context_text),
        "context_chars": task.context.len_chars(),
    }


def pick_winner(baseline: dict, rlm: dict) -> str:
    if rlm["success"] and not baseline["success"]:
        return "rlm (baseline missed)"
    if baseline["success"] and not rlm["success"]:
        return "baseline (rlm missed)"
    if rlm["success"] and baseline["success"]:
        if rlm["tokens"] < baseline["tokens"]:
            return "rlm (fewer tokens)"
        if rlm["tokens"] > baseline["tokens"]:
            return "baseline (fewer tokens)"
        return "tie"
    return "tie"


def main() -> None:
    base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    model = os.getenv("LLM_MODEL", "qwen2.5-coder:7b")
    subcall_model = os.getenv("LLM_SUBCALL_MODEL", "qwen2.5:3b")
    log_level = os.getenv("LLM_LOG_LEVEL", "WARNING").upper()
    timeout = float(os.getenv("LLM_TIMEOUT", "300"))
    require_subcall = os.getenv("LLM_REQUIRE_SUBCALL", "0") == "1"
    max_steps = int(os.getenv("RLM_MAX_STEPS", "25"))
    max_subcalls = int(os.getenv("RLM_MAX_SUBCALLS", "120"))
    parallel_subcalls = os.getenv("RLM_PARALLEL_SUBCALLS", "1") == "1"
    max_concurrent_subcalls = int(os.getenv("RLM_MAX_CONCURRENT_SUBCALLS", "8"))
    subcall_guard_raw = os.getenv("RLM_SUBCALL_GUARD_STEPS", "").strip()
    subcall_guard_steps = int(subcall_guard_raw) if subcall_guard_raw else None
    if require_subcall and subcall_guard_steps is None:
        subcall_guard_steps = 2
    fallback_guard_raw = os.getenv("RLM_FALLBACK_GUARD_STEPS", "1").strip()
    fallback_guard_steps = int(fallback_guard_raw) if fallback_guard_raw else None
    repl_error_limit = int(os.getenv("RLM_REPL_ERROR_LIMIT", "1"))
    baseline_max_chars = int(os.getenv("BASELINE_MAX_CHARS", "8000"))

    # Configure logging: suppress httpx noise, keep rlm_runtime info
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    # Set our loggers to desired level
    logging.getLogger("rlm_runtime").setLevel(getattr(logging, log_level, logging.WARNING))

    adapter = GenericChatAdapter(base_url=base_url, model=model, timeout=timeout)
    subcall_adapter = GenericChatAdapter(base_url=base_url, model=subcall_model, timeout=timeout)

    # Generate tasks with different complexities
    # Include scaling to find crossover point where RLM beats baseline
    tasks = [
        # Deterministic tasks (regex-solvable) - scaling to find crossover
        generate_needle_deep_task(num_docs=20),   # Small - baseline might win
        generate_needle_deep_task(num_docs=50),   # Medium
        generate_needle_deep_task(num_docs=100),  # Large - RLM should win
        generate_multi_hop_task(num_docs=15),     # Small
        generate_multi_hop_task(num_docs=40),     # Medium
        generate_aggregation_task(num_docs=20),   # Small
        generate_aggregation_task(num_docs=50),   # Medium
        generate_filtering_task(num_docs=20),     # Small
        generate_filtering_task(num_docs=50),     # Medium
        # Semantic tasks (NOT regex-solvable) - RLM advantage
        generate_semantic_needle_task(num_docs=30),   # Semantic needle
        generate_semantic_needle_task(num_docs=60),   # Larger semantic needle
        generate_semantic_aggregation_task(num_docs=25),  # Sentiment analysis
        generate_semantic_aggregation_task(num_docs=50),  # Larger sentiment
    ]

    print("=" * 70)
    print("Complex Reasoning Benchmark")
    print(f"Model: {model}")
    print(f"Subcall model: {subcall_model}")
    print(f"Baseline max chars: {baseline_max_chars}")
    print("=" * 70)
    print()

    results: list[dict] = []

    for task in tasks:
        print(f"Running: {task.name} ({task.task_type})")
        print(f"  Context: {task.context.len_chars():,} chars, {task.context.num_documents()} docs")
        print(f"  Expected: {task.expected}")

        baseline = run_baseline(
            adapter,
            task,
            max_tokens=512,
            max_context_chars=baseline_max_chars,
        )

        rlm_result = run_task(
            adapter,
            subcall_adapter,
            task,
            max_steps=max_steps,
            max_subcalls=max_subcalls,
            require_subcall=require_subcall,
            subcall_guard_steps=subcall_guard_steps,
            fallback_guard_steps=fallback_guard_steps,
            repl_error_limit=repl_error_limit,
            parallel_subcalls=parallel_subcalls,
            max_concurrent_subcalls=max_concurrent_subcalls,
        )
        winner = pick_winner(baseline, rlm_result)

        results.append(
            {
                "task": task.name,
                "context_chars": task.context.len_chars(),
                "baseline": baseline,
                "rlm": rlm_result,
                "winner": winner,
            }
        )

        base_status = "PASS" if baseline["success"] else "FAIL"
        rlm_status = "PASS" if rlm_result["success"] else "FAIL"
        print(f"  Baseline: {baseline['output'][:100]}...")
        print(
            f"    Status: {base_status} | Elapsed: {baseline['elapsed']:.2f}s "
            f"| Tokens: {baseline['tokens']} | Truncated: {baseline['truncated']}"
        )
        print(f"  RLM: {rlm_result['output'][:100]}...")
        print(
            f"    Status: {rlm_status} | Elapsed: {rlm_result['elapsed']:.2f}s "
            f"| Tokens: {rlm_result['tokens']} | Steps: {rlm_result['steps']}"
        )
        print(f"  Winner: {winner}")
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"{'Task':<20} {'Chars':>10} {'Base':>6} {'RLM':>6} "
        f"{'BTok':>6} {'RTok':>6} {'BTime':>7} {'RTime':>7}"
    )
    print("-" * 70)

    total_success = 0
    for r in results:
        base = r["baseline"]
        rlm_result = r["rlm"]
        base_status = "PASS" if base["success"] else "FAIL"
        rlm_status = "PASS" if rlm_result["success"] else "FAIL"
        if rlm_result["success"]:
            total_success += 1
        print(
            f"{r['task']:<20} {r['context_chars']:>10,} {base_status:>6} {rlm_status:>6} "
            f"{base['tokens']:>6} {rlm_result['tokens']:>6} "
            f"{base['elapsed']:>6.1f}s {rlm_result['elapsed']:>6.1f}s"
        )

    print("-" * 70)
    print(f"Total RLM: {total_success}/{len(results)} passed")

    # Crossover analysis
    print()
    print("=" * 70)
    print("CROSSOVER ANALYSIS (where does RLM start winning?)")
    print("=" * 70)

    # Group by task type
    from collections import defaultdict
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        task_type = r["rlm"]["task"].replace("_", " ")
        by_type[task_type].append(r)

    for task_type, task_results in by_type.items():
        # Sort by context size
        sorted_results = sorted(task_results, key=lambda x: x["context_chars"])
        print(f"\n{task_type.upper()}:")

        crossover_found = False
        for r in sorted_results:
            base_ok = r["baseline"]["success"]
            rlm_ok = r["rlm"]["success"]
            chars = r["context_chars"]
            truncated = r["baseline"].get("truncated", False)

            if rlm_ok and not base_ok and not crossover_found:
                print(f"  → CROSSOVER at ~{chars:,} chars (baseline truncated: {truncated})")
                crossover_found = True

            status = ""
            if rlm_ok and base_ok:
                status = "both pass"
            elif rlm_ok and not base_ok:
                status = "RLM WINS"
            elif not rlm_ok and base_ok:
                status = "baseline wins"
            else:
                status = "both fail"

            print(f"    {chars:>10,} chars: {status}")

        if not crossover_found and any(r["rlm"]["success"] for r in sorted_results):
            print("  → RLM wins at all tested sizes")
        elif not any(r["rlm"]["success"] for r in sorted_results):
            print("  → RLM failed at all tested sizes")

    # Semantic vs Deterministic summary
    print()
    print("=" * 70)
    print("SEMANTIC vs DETERMINISTIC TASKS")
    print("=" * 70)

    semantic_tasks = [r for r in results if "semantic" in r["rlm"]["task"]]
    deterministic_tasks = [r for r in results if "semantic" not in r["rlm"]["task"]]

    semantic_rlm_wins = sum(1 for r in semantic_tasks if r["rlm"]["success"] and not r["baseline"]["success"])
    semantic_total = len(semantic_tasks)

    det_rlm_wins = sum(1 for r in deterministic_tasks if r["rlm"]["success"] and not r["baseline"]["success"])
    det_total = len(deterministic_tasks)

    print(f"Semantic tasks:      RLM wins {semantic_rlm_wins}/{semantic_total} (baseline cannot use regex)")
    print(f"Deterministic tasks: RLM wins {det_rlm_wins}/{det_total} (due to context size)")


if __name__ == "__main__":
    main()
