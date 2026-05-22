from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class BenchmarkPrompt:
    """A single benchmark item.

    Scoring fields:
      - ``expected`` + ``match_type`` enable rule-based grading for
        reasoning/multi-choice prompts. ``match_type`` ∈
        {"exact", "contains", "number", "choice"}.
      - ``expected_tool`` + ``expected_args_contains`` enable rule-based
        grading for tool-use prompts. ``expected_args_contains`` maps a tool
        argument name to a list of acceptable substrings (case-insensitive).
      - ``expected_behavior`` is the rubric the LLM-as-judge sees when scoring
        free-form responses (speed, quality_ua, code).
    """

    id: str
    prompt: str
    category: str
    expected: Optional[str] = None
    match_type: Optional[str] = None
    expected_tool: Optional[str] = None
    expected_args_contains: Optional[dict[str, list[str]]] = None
    expected_behavior: str = ""


# -----------------------------------------------------------------------------
# Speed — short prompts, scored only by latency/throughput, no quality grade
# -----------------------------------------------------------------------------

SPEED: list[BenchmarkPrompt] = [
    BenchmarkPrompt(
        id="speed_arith",
        prompt="What is 2 + 2? Reply with just the number.",
        category="speed",
        expected_behavior="minimal TTFT, 1-3 tokens output",
    ),
    BenchmarkPrompt(
        id="speed_three_words",
        prompt="Say 'hello' in exactly three words.",
        category="speed",
        expected_behavior="short warm-up completion",
    ),
    BenchmarkPrompt(
        id="speed_list_short",
        prompt="List five primary colors separated by commas.",
        category="speed",
        expected_behavior="~20 tokens, measure tokens/sec",
    ),
    BenchmarkPrompt(
        id="speed_one_paragraph",
        prompt="In one paragraph (about 60 words), describe what a database index is.",
        category="speed",
        expected_behavior="~60-90 tokens, steady throughput",
    ),
    BenchmarkPrompt(
        id="speed_haiku",
        prompt="Write a single English haiku about autumn rain. Output only the haiku.",
        category="speed",
        expected_behavior="3 lines, ~17 syllables, structured short output",
    ),
]


# -----------------------------------------------------------------------------
# Quality UA — 8 prompts, judged by local LLM (Qwen 32B fallback to largest)
# -----------------------------------------------------------------------------

QUALITY_UA: list[BenchmarkPrompt] = [
    BenchmarkPrompt(
        id="ua_translate_tech",
        prompt=(
            "Переклади українською зберігаючи технічну точність:\n"
            '"A B-tree index speeds up equality and range queries on indexed '
            'columns, but slows down INSERT and UPDATE operations."'
        ),
        category="quality_ua",
        expected_behavior=(
            "правильний переклад технічних термінів (B-tree, range query, "
            "INSERT/UPDATE), природна українська"
        ),
    ),
    BenchmarkPrompt(
        id="ua_translate_informal",
        prompt=(
            "Переклади українською у розмовному стилі:\n"
            '"Hey, I was thinking we could grab coffee tomorrow morning '
            "if you're free. Let me know what works for you."
        ),
        category="quality_ua",
        expected_behavior="розмовна українська без англіцизмів і кальок",
    ),
    BenchmarkPrompt(
        id="ua_business_letter",
        prompt=(
            "Склади українською офіційний лист (4-5 речень) до клієнта з "
            "вибаченням за затримку доставки на 3 дні та повідомленням про "
            "знижку 10% на наступне замовлення."
        ),
        category="quality_ua",
        expected_behavior=(
            "офіційно-діловий стиль, грамотна структура, конкретика про "
            "затримку і компенсацію"
        ),
    ),
    BenchmarkPrompt(
        id="ua_grammar_fix",
        prompt=(
            "Виправ помилки у реченні, поверни тільки виправлений варіант:\n"
            '"На протязі трьох років, я займаюся вивченням української мови '
            'і вважаю шо це дуже цікаве заняття."'
        ),
        category="quality_ua",
        expected_behavior=(
            "виправити 'на протязі' → 'протягом', 'шо' → 'що', "
            "правильна пунктуація"
        ),
    ),
    BenchmarkPrompt(
        id="ua_tech_explain",
        prompt=(
            "Поясни простою українською мовою (3-4 речення), що таке REST API "
            "і навіщо він потрібен. Приклад навести з реального життя."
        ),
        category="quality_ua",
        expected_behavior=(
            "технічна точність + простота, приклад зрозумілий нетехніку"
        ),
    ),
    BenchmarkPrompt(
        id="ua_qa_factual",
        prompt=(
            "Хто такий Тарас Шевченко і чим він відомий? Відповідай українською, "
            "2-3 речення."
        ),
        category="quality_ua",
        expected_behavior="історична точність, природна українська",
    ),
    BenchmarkPrompt(
        id="ua_summary",
        prompt=(
            "Стисло переказ українською (2 речення):\n"
            '"Python is a high-level, interpreted programming language known for '
            "its readability and broad standard library. Since its release in 1991 "
            "it has become one of the most popular languages for data science, "
            'web development, and automation."'
        ),
        category="quality_ua",
        expected_behavior="збереження ключових фактів, стиснення без втрати сенсу",
    ),
    BenchmarkPrompt(
        id="ua_creative",
        prompt=(
            "Напиши українською 3-4 речення опису осіннього ранку в Карпатах. "
            "Використовуй художні засоби."
        ),
        category="quality_ua",
        expected_behavior=(
            "художня українська, конкретні образи (тумани, ялини), без кальок"
        ),
    ),
]


# -----------------------------------------------------------------------------
# Code — judged by LLM (we deliberately skip subprocess execution)
# -----------------------------------------------------------------------------

CODE: list[BenchmarkPrompt] = [
    BenchmarkPrompt(
        id="code_reverse_words",
        prompt=(
            "Write a Python function `reverse_words(s: str) -> str` that "
            "reverses the order of words in a sentence while preserving the "
            "whitespace between them (multiple spaces should stay intact). "
            "Include one doctest example."
        ),
        category="code",
        expected_behavior=(
            "compilable Python, correct logic for multi-space input, "
            "valid doctest"
        ),
    ),
    BenchmarkPrompt(
        id="code_fix_bug",
        prompt=(
            "The following Python function is supposed to return the sum of "
            "even numbers in a list but has a bug:\n\n"
            "def sum_evens(nums):\n"
            "    total = 0\n"
            "    for n in nums:\n"
            "        if n % 2:\n"
            "            total += n\n"
            "    return total\n\n"
            "Explain the bug in one sentence and provide the fixed function."
        ),
        category="code",
        expected_behavior=(
            "identifies `n % 2` (truthy for odd) vs the needed `n % 2 == 0`; "
            "provides corrected function"
        ),
    ),
    BenchmarkPrompt(
        id="code_sql_top_n",
        prompt=(
            "Write a single SQL query that selects the top 3 customers by "
            "total order amount from tables `customers(id, name)` and "
            "`orders(id, customer_id, amount)`. Use standard SQL."
        ),
        category="code",
        expected_behavior=(
            "uses JOIN, SUM, GROUP BY customer, ORDER BY total DESC, LIMIT 3"
        ),
    ),
    BenchmarkPrompt(
        id="code_regex_email",
        prompt=(
            "Write a Python function `extract_emails(text: str) -> list[str]` "
            "that returns all email addresses found in the input text using "
            "a regular expression. Handle common email formats."
        ),
        category="code",
        expected_behavior=(
            "uses `re` module, reasonable email regex, returns list of strings"
        ),
    ),
    BenchmarkPrompt(
        id="code_async_fetch",
        prompt=(
            "Write a Python async function `fetch_all(urls: list[str]) -> list[str]` "
            "that fetches the response bodies of all given URLs concurrently "
            "using aiohttp and returns them in the same order. Include "
            "error handling for failed requests."
        ),
        category="code",
        expected_behavior=(
            "uses aiohttp.ClientSession, asyncio.gather, return_exceptions, "
            "preserves order"
        ),
    ),
]


# -----------------------------------------------------------------------------
# Reasoning — rule-based scoring with multiple match types
# -----------------------------------------------------------------------------

REASONING: list[BenchmarkPrompt] = [
    BenchmarkPrompt(
        id="reason_age",
        prompt=(
            "Alice is twice as old as Bob. In 5 years, Alice will be 1.5 times "
            "Bob's age. How old is Bob now? Reply with just the number."
        ),
        category="reasoning",
        expected="5",
        match_type="number",
    ),
    BenchmarkPrompt(
        id="reason_deduction",
        prompt=(
            "Three friends — Anna, Bohdan, Chris — each prefer a different "
            "drink (tea, coffee, juice). Anna does not drink coffee. Bohdan "
            "does not drink juice. Chris drinks tea. What does Anna drink? "
            "Answer with a single word."
        ),
        category="reasoning",
        expected="juice",
        match_type="contains",
    ),
    BenchmarkPrompt(
        id="reason_odd_one_out",
        prompt=(
            "Which one does not belong with the rest, and why? "
            "apple, banana, carrot, cherry, grape. "
            "Answer with one word for the odd one out."
        ),
        category="reasoning",
        expected="carrot",
        match_type="contains",
    ),
    BenchmarkPrompt(
        id="reason_train",
        prompt=(
            "A train leaves station A at 09:00 traveling at 60 km/h toward "
            "station B which is 180 km away. At what time does it arrive? "
            "Reply with just the time in 24h HH:MM format."
        ),
        category="reasoning",
        expected="12:00",
        match_type="contains",
    ),
    BenchmarkPrompt(
        id="reason_sequence",
        prompt=(
            "What is the next number in the sequence: 2, 6, 12, 20, 30, ...? "
            "Reply with just the number."
        ),
        category="reasoning",
        expected="42",
        match_type="number",
    ),
    BenchmarkPrompt(
        id="reason_choice",
        prompt=(
            "If all roses are flowers and some flowers fade quickly, which "
            "statement is necessarily true?\n"
            "A) All roses fade quickly.\n"
            "B) Some roses fade quickly.\n"
            "C) Some flowers are roses.\n"
            "D) No roses fade quickly.\n"
            "Reply with just the letter."
        ),
        category="reasoning",
        expected="C",
        match_type="choice",
    ),
]


# -----------------------------------------------------------------------------
# Tool Use — rule-based: correct tool + arguments contain expected substrings
# -----------------------------------------------------------------------------

TOOL_USE: list[BenchmarkPrompt] = [
    BenchmarkPrompt(
        id="tool_search_notes",
        prompt="Знайди мої нотатки про Python.",
        category="tool_use",
        expected_tool="notes_search",
        expected_args_contains={"query": ["python"]},
        expected_behavior="must call notes_search with query containing 'python'",
    ),
    BenchmarkPrompt(
        id="tool_create_note",
        prompt='Створи нотатку з назвою "shopping list" і вмістом "milk, bread, eggs".',
        category="tool_use",
        expected_tool="notes_create",
        expected_args_contains={"title": ["shopping"], "content": ["milk"]},
        expected_behavior="must call notes_create with title containing 'shopping' and content containing 'milk'",
    ),
    BenchmarkPrompt(
        id="tool_list_files",
        prompt="Покажи які файли є у моєму workspace.",
        category="tool_use",
        expected_tool="list_directory",
        expected_args_contains=None,
        expected_behavior="must call list_directory (path may be empty or '.')",
    ),
    BenchmarkPrompt(
        id="tool_read_file",
        prompt='Прочитай файл "report.md" з мого workspace.',
        category="tool_use",
        expected_tool="read_file",
        expected_args_contains={"path": ["report"]},
        expected_behavior="must call read_file with path containing 'report'",
    ),
    BenchmarkPrompt(
        id="tool_delete_note",
        prompt='Видали мою нотатку "old draft".',
        category="tool_use",
        expected_tool="notes_delete",
        expected_args_contains={"title": ["old", "draft"]},
        expected_behavior="must call notes_delete with title containing 'old' or 'draft'",
    ),
]


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

PROMPT_SETS: dict[str, list[BenchmarkPrompt]] = {
    "speed": SPEED,
    "quality_ua": QUALITY_UA,
    "code": CODE,
    "reasoning": REASONING,
    "tool_use": TOOL_USE,
}


def get_set(key: str) -> list[BenchmarkPrompt]:
    return list(PROMPT_SETS.get(key, []))


def set_keys() -> list[str]:
    return list(PROMPT_SETS.keys())


def prompts_by_category(category: str) -> list[BenchmarkPrompt]:
    return [p for prompts in PROMPT_SETS.values() for p in prompts if p.category == category]


def all_prompts() -> list[BenchmarkPrompt]:
    return [p for prompts in PROMPT_SETS.values() for p in prompts]


def make_tool_schemas_for_prompts(
    prompts: list[BenchmarkPrompt],
) -> Optional[list[dict]]:
    """Build a minimal OpenAI-shaped tool-schema list covering every tool
    referenced by ``tool_use`` prompts in the input.

    Returns ``None`` if no ``tool_use`` prompts are present. The schemas are
    intentionally bare — the benchmark only checks *which* tool the model
    decides to call and with what arguments, not the result of executing it.
    """
    tool_use = [p for p in prompts if p.expected_tool]
    if not tool_use:
        return None
    seen: dict[str, dict] = {}
    for p in tool_use:
        tool = p.expected_tool
        if not tool or tool in seen:
            continue
        seen[tool] = {
            "type": "function",
            "function": {
                "name": tool,
                "description": f"Bench stub for {tool}",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    return list(seen.values()) or None
