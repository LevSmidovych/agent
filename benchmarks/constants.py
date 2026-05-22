"""Named constants for the benchmarks package.

Centralised so changes (e.g. renaming a category, adding a new run type)
happen in one place rather than scattered string literals across UI,
runner, charts, and storage.
"""

from __future__ import annotations


# ---- benchmark_runs.run_type values --------------------------------------

RUN_TYPE_STANDARD = "standard"
RUN_TYPE_QUANTIZATION = "quantization"

RUN_TYPES = frozenset({RUN_TYPE_STANDARD, RUN_TYPE_QUANTIZATION})


# ---- benchmark_results.category values -----------------------------------

CATEGORY_SPEED = "speed"
CATEGORY_QUALITY_UA = "quality_ua"
CATEGORY_CODE = "code"
CATEGORY_REASONING = "reasoning"
CATEGORY_TOOL_USE = "tool_use"

# All categories in the canonical radar order.
RADAR_CATEGORIES: tuple[str, ...] = (
    CATEGORY_SPEED,
    CATEGORY_QUALITY_UA,
    CATEGORY_CODE,
    CATEGORY_REASONING,
    CATEGORY_TOOL_USE,
)

# Categories scored by rule-based graders (see benchmarks.scoring).
RULE_BASED_CATEGORIES: frozenset[str] = frozenset({
    CATEGORY_REASONING,
    CATEGORY_TOOL_USE,
})

# Categories scored by the LLM-as-judge (see benchmarks.judge).
JUDGE_CATEGORIES: frozenset[str] = frozenset({
    CATEGORY_SPEED,
    CATEGORY_QUALITY_UA,
    CATEGORY_CODE,
})


# ---- result match_type values --------------------------------------------

MATCH_EXACT = "exact"
MATCH_CONTAINS = "contains"
MATCH_NUMBER = "number"
MATCH_CHOICE = "choice"

MATCH_TYPES = frozenset({MATCH_EXACT, MATCH_CONTAINS, MATCH_NUMBER, MATCH_CHOICE})
