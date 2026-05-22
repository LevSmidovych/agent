"""Populate all data stores with mocked demo content for screenshots,
demonstration runs, and dipломна defense.

Generates:
  * 3 conversations (general / project_manager / cook) with realistic UA dialog
  * Long-term memory: 5 global facts + 3 profile-scoped facts
  * 7 markdown notes covering the three profiles
  * Knowledge base files for each profile (templates, recipes, docs)
  * Sample workspace files for the files-tool demo
  * 3 benchmark history runs with full v3 metrics (TTFT/TPS/VRAM/judge/pass)

Usage:
    python scripts/seed_demo_data.py            # add demo data alongside existing
    python scripts/seed_demo_data.py --reset    # wipe data/ first

After running, launch the app: the Memory window, KB window, notes search,
benchmark history etc. all have content out of the box.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmarks.constants import (  # noqa: E402
    CATEGORY_CODE,
    CATEGORY_QUALITY_UA,
    CATEGORY_REASONING,
    CATEGORY_SPEED,
    CATEGORY_TOOL_USE,
    RUN_TYPE_QUANTIZATION,
    RUN_TYPE_STANDARD,
)
from benchmarks.storage import BenchmarkStorage  # noqa: E402
from core.chroma_client import ChromaClientFactory  # noqa: E402
from core.exceptions import OllamaConnectionError  # noqa: E402
from memory.long_term import LongTermMemory  # noqa: E402
from memory.storage import Storage  # noqa: E402
from tools.notes import NotesStore  # noqa: E402


DATA_DIR = ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
NOTES_DIR = DATA_DIR / "notes"
KB_DIR = DATA_DIR / "knowledge_bases"
WS_DIR = DATA_DIR / "workspace"
DB_PATH = DATA_DIR / "app.db"


# ---------------------------------------------------------------------------
# Stub embedder for ChromaDB collections (so seeding works without Ollama).
# ---------------------------------------------------------------------------


class StubEmbedder:
    """Deterministic 768-dim embeddings derived from a hash of the input text.

    Vectors are NOT semantically meaningful — they only let ChromaDB index
    and retrieve consistently for demo purposes. Real semantic search will
    work after re-indexing with the real OllamaEmbedder once Ollama is up.
    """

    is_available = True

    def __call__(self, input):  # noqa: A002 — chromadb signature
        return [self._vec(t) for t in input]

    def embed_query(self, input):  # noqa: A002
        if isinstance(input, str):
            return self._vec(input)
        return [self._vec(t) for t in input]

    def embed_documents(self, input):  # noqa: A002
        return [self._vec(t) for t in input]

    def name(self) -> str:
        return "stub::demo"

    @staticmethod
    def _vec(text: str) -> list[float]:
        text = text or ""
        seed = sum(ord(c) for c in text) % 100_000
        rng = random.Random(seed)
        return [rng.uniform(-1.0, 1.0) for _ in range(768)]


def _try_real_embedder():
    """Return the real Ollama embedder if reachable, else a stub.

    The stub keeps the seed script offline-friendly; the real one means RAG
    search works immediately after seeding.
    """
    try:
        from core.embeddings import OllamaEmbedder

        emb = OllamaEmbedder("nomic-embed-text")
        emb.embed(["ping"])  # probe
        print("[ok] Ollama embedder reachable — using real embeddings.")
        return emb
    except (OllamaConnectionError, Exception) as exc:
        print(f"[!] Ollama embedder unavailable ({exc.__class__.__name__}); using stub.")
        return StubEmbedder()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def _hours_ago(n: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=n)


# ---------------------------------------------------------------------------
# Seeders
# ---------------------------------------------------------------------------


def seed_conversations(storage: Storage) -> None:
    """Three plausible conversations across the three profiles."""

    # ---- general profile ----
    conv_general = storage.create_conversation(
        profile_name="general", model_name="qwen2.5:14b",
    )
    storage.add_message(conv_general, role="user",
                        content="Привіт! Що ти за модель?")
    storage.add_message(conv_general, role="assistant",
                        content="Привіт! Я Qwen 2.5 — модель з відкритим вихідним кодом "
                                "від Alibaba. Чим можу допомогти?")
    storage.add_message(conv_general, role="user",
                        content="Поясни простими словами що таке градієнтний спуск.")
    storage.add_message(conv_general, role="assistant",
                        content="Уяви що ти стоїш на схилі гори у тумані і хочеш "
                                "дістатися найнижчої точки. Ти не бачиш далеко, "
                                "але можеш відчути нахил під ногами. Тому крокуєш "
                                "у бік найбільшого спуску — повторюєш так багато разів. "
                                "Градієнтний спуск працює так само: алгоритм маленькими "
                                "кроками рухає параметри моделі у напрямку зменшення "
                                "функції втрат, поки не досягне мінімуму.")
    storage.end_conversation(conv_general)

    # ---- project_manager profile (active conversation) ----
    conv_pm = storage.create_conversation(
        profile_name="project_manager", model_name="qwen2.5:14b",
    )
    storage.add_message(conv_pm, role="user",
                        content="Допоможи оцінити проект розробки мобільного додатку "
                                "для замовлення їжі. 3 розробники, 2 місяці.")
    storage.add_message(conv_pm, role="assistant",
                        content="Розіб'ю на роботу за фазами:\n\n"
                                "**Фаза 1 — Discovery (1 тиждень)**\n"
                                "- Інтервʼю зі стейкхолдерами\n"
                                "- Аналіз конкурентів\n"
                                "- User stories + acceptance criteria\n\n"
                                "**Фаза 2 — Design (2 тижні)**\n"
                                "- Wireframes, прототип у Figma\n"
                                "- Архітектура backend (API design)\n\n"
                                "**Фаза 3 — Development (5 тижнів)**\n"
                                "- Sprint 1-2: Auth + Catalog\n"
                                "- Sprint 3-4: Cart + Checkout\n"
                                "- Sprint 5: Push + History\n\n"
                                "**Фаза 4 — QA + Release (2 тижні)**\n"
                                "- Регресія, beta-тест, App Store submission\n\n"
                                "**Ризики:** App Store review (3-7 днів), інтеграція "
                                "з платіжкою, push на iOS. Закладай 15% буфера.")
    storage.add_message(conv_pm, role="user",
                        content="Які метрики треба збирати після релізу?")
    storage.add_message(conv_pm, role="assistant",
                        content="Розділимо на 3 групи:\n\n"
                                "**Activation:** install-to-first-order conversion, "
                                "time-to-first-order, D1/D7/D30 retention.\n\n"
                                "**Engagement:** orders per user/month, AOV, "
                                "session frequency, feature adoption (фільтри, обране).\n\n"
                                "**Quality:** crash-free rate (>99.5%), "
                                "checkout funnel drop-off, NPS, App Store rating.")
    # No end_conversation — leave it active so app shows it on restart.

    # ---- cook profile ----
    conv_cook = storage.create_conversation(
        profile_name="cook", model_name="qwen2.5:14b",
    )
    storage.add_message(conv_cook, role="user",
                        content="Маю курячу грудку, картоплю, цибулю. Що зварганити?")
    storage.add_message(conv_cook, role="assistant",
                        content="Класичний варіант — **запечена курка з картоплею**:\n\n"
                                "**Інгредієнти:**\n"
                                "- куряча грудка 500г\n"
                                "- картопля 4-5 шт\n"
                                "- цибуля 1 велика\n"
                                "- олія 3 ст. л., сіль, перець, паприка\n\n"
                                "**Приготування (45 хв):**\n"
                                "1. Розігрій духовку до 200°C.\n"
                                "2. Картоплю поріж на четвертинки, цибулю — пів-кільцями.\n"
                                "3. Грудку поріж шматочками, замаринуй із сіллю, "
                                "перцем, паприкою на 10 хв.\n"
                                "4. Викладай овочі і м'ясо у форму, полий олією.\n"
                                "5. Запікай 35-40 хв до золотистої скоринки.\n\n"
                                "Зверху можна посипати тертим сиром за 5 хв до кінця.")
    storage.end_conversation(conv_cook)

    print(f"[ok] Seeded 3 conversations (ids {conv_general}, {conv_pm}, {conv_cook}).")


def seed_long_term_memory(long_term: LongTermMemory) -> None:
    """Mix of global (cross-profile) facts and profile-specific ones."""

    # Global facts about the user — visible everywhere.
    global_facts = [
        "Користувач спілкується переважно українською мовою.",
        "Має алергію на горіхи — арахіс, фундук, мигдаль виключити з рецептів.",
        "Працює software engineer'ом з фокусом на Python і Go.",
        "Любить мінімалістичний стиль у відповідях, без зайвої води.",
        "Часовий пояс — Київ (UTC+2/+3).",
    ]
    for text in global_facts:
        long_term.add(text=text, scope="global", source="auto", profile=None)

    # PM-scoped facts.
    pm_facts = [
        "Поточний проект — мобільний додаток FoodHub, 3 розробники, дедлайн 2 місяці.",
        "Команда працює за Scrum зі спринтами по 2 тижні.",
        "Stakeholder — Catherine з product team, статус-репорти щопʼятниці.",
    ]
    for text in pm_facts:
        long_term.add(text=text, scope="profile", source="user",
                      profile="project_manager")

    # Cook-scoped facts.
    cook_facts = [
        "Не їсть свинину з релігійних міркувань.",
        "Любить італійську і японську кухні, не любить дуже гострі страви.",
    ]
    for text in cook_facts:
        long_term.add(text=text, scope="profile", source="user", profile="cook")

    print(f"[ok] Seeded long-term memory: {len(global_facts)} global, "
          f"{len(pm_facts)} project_manager, {len(cook_facts)} cook.")


def seed_notes(notes_store: NotesStore) -> None:
    """Mix of notes that the agent could surface via notes_search."""

    notes = [
        ("Python decorators",
         "Декоратор — це функція, яка приймає іншу функцію і повертає нову. "
         "Базовий шаблон:\n\n"
         "```python\n"
         "def my_decorator(fn):\n"
         "    def wrapper(*args, **kwargs):\n"
         "        # before\n"
         "        result = fn(*args, **kwargs)\n"
         "        # after\n"
         "        return result\n"
         "    return wrapper\n"
         "```\n\n"
         "Використання: `@my_decorator` над визначенням функції."),

        ("Scrum daily standup",
         "Три питання на дейлі:\n\n"
         "1. Що зробив учора?\n"
         "2. Що планую сьогодні?\n"
         "3. Які блокери / залежності від інших?\n\n"
         "Тривалість: 15 хвилин max. Деталі обговорюємо після стендапу 1-on-1."),

        ("Tomato sauce recipe",
         "Базовий томатний соус (на 4 порції пасти):\n\n"
         "- 800г консервованих помідорів (San Marzano)\n"
         "- 4 зубки часнику\n"
         "- 50г оливкової олії\n"
         "- свіжий базилік, сіль, перець\n\n"
         "Обсмажити часник до золотистого, додати помідори, тушити 20 хв "
         "на повільному вогні. Наприкінці — базилік і сіль. "
         "Бузий компонент для болоньєзе, маринари і піцци."),

        ("Project FoodHub backlog",
         "MVP скоп:\n"
         "- [ ] Auth (Email + Google OAuth)\n"
         "- [ ] Catalog with categories and search\n"
         "- [ ] Cart + Checkout (Stripe)\n"
         "- [ ] Order history + status notifications\n"
         "- [ ] Profile + address book\n\n"
         "Post-MVP:\n"
         "- [ ] Loyalty program\n"
         "- [ ] Restaurant ratings\n"
         "- [ ] Push for promotions"),

        ("Go error handling",
         "Завжди перевіряй помилки:\n\n"
         "```go\n"
         "result, err := doSomething()\n"
         "if err != nil {\n"
         "    return fmt.Errorf(\"doing X: %w\", err)\n"
         "}\n"
         "```\n\n"
         "`%w` обгортає помилку, дозволяє `errors.Is()` і `errors.As()` "
         "у викликача."),

        ("Stakeholder communication",
         "Шаблон статус-репорту:\n\n"
         "**Прогрес тижня:**\n"
         "- Що завершили\n\n"
         "**Наступний тиждень:**\n"
         "- Що в плані\n\n"
         "**Ризики:**\n"
         "- Технічні / організаційні\n\n"
         "**Потрібно від стейкхолдерів:**\n"
         "- Конкретні рішення / approval"),

        ("Список покупок",
         "На вихідні:\n"
         "- помідори чері 500г\n"
         "- моцарела 250г\n"
         "- базилік свіжий\n"
         "- оливкова олія 0.5л\n"
         "- хліб чабата\n"
         "- червоне сухе вино"),
    ]
    created = 0
    for title, content in notes:
        try:
            notes_store.create(title, content)
            created += 1
        except FileExistsError:
            pass  # already seeded earlier
    print(f"[ok] Seeded {created}/{len(notes)} notes "
          f"(skipped {len(notes) - created} existing).")


def seed_knowledge_bases() -> None:
    """Sample .md docs for each profile's KB."""

    pm = KB_DIR / "project_manager"
    cook = KB_DIR / "cook"
    general = KB_DIR / "general"
    for d in (pm, cook, general):
        d.mkdir(parents=True, exist_ok=True)

    _write_if_missing(pm / "agile_glossary.md", """\
# Agile glossary

**Sprint** — fixed time-box (typically 2 weeks) during which a usable, potentially shippable increment is created.

**User story** — short description of functionality from end user perspective:
"As a <role>, I want <feature> so that <benefit>."

**Acceptance criteria** — conditions a story must meet to be considered done. Written in Given/When/Then format.

**Definition of Done** — checklist agreed by the team: code reviewed, tests written, deployed to staging, documentation updated.

**Velocity** — sum of story points completed per sprint. Use historical average to predict capacity.

**Burndown chart** — visualizes remaining work over the sprint. Should trend downward.

**Retro action items** — concrete improvements agreed in retrospective. Track in following sprints.
""")

    _write_if_missing(pm / "estimation_techniques.md", """\
# Estimation techniques for software projects

## T-shirt sizing
XS / S / M / L / XL. Used early in planning when precision is impossible. Maps roughly to story points: XS=1, S=2, M=5, L=8, XL=13.

## Planning poker
Team members independently pick a Fibonacci-like number (1, 2, 3, 5, 8, 13, 21). Discuss outliers, re-vote until consensus.

## PERT (Three-point estimation)
For each task estimate Optimistic (O), Most Likely (M), Pessimistic (P). Weighted average: `(O + 4M + P) / 6`. Captures uncertainty.

## Reference-class forecasting
Find similar past projects, scale their duration to the current scope. Avoids overconfidence.

## Buffer rules of thumb
- Add 15-20% for medium-confidence estimates.
- Add 30-50% for high-uncertainty work (research, integrations with external systems).
- App Store/Google Play review: 3-7 days additional.
""")

    _write_if_missing(pm / "risk_management.md", """\
# Risk management basics

## Identification
Brainstorm in two passes:
1. Technical (third-party APIs, scalability unknowns, security)
2. Organizational (dependencies on other teams, key person availability)

## Assessment
For each risk score:
- **Probability** 1-5
- **Impact** 1-5
- **Risk score** = P × I

Risks with score > 10 need active mitigation plans.

## Mitigation strategies
- **Avoid** — change scope to remove the risk
- **Mitigate** — reduce probability or impact
- **Transfer** — buy insurance, outsource
- **Accept** — document and monitor

## Review cadence
Update risk register at every sprint planning. New risks emerge, old ones may close.
""")

    _write_if_missing(cook / "italian_basics.md", """\
# Італійська кухня — базові техніки

## Соус al dente для пасти

1. Закип'яти 4 л води, додай 40г солі (важливо — багато солі).
2. Закидай пасту, помішуй перші 30 секунд щоб не злиплася.
3. Час варіння на пакеті мінус 1 хвилина — пасту викидаєш сирувату.
4. Переклади у сковороду з соусом, додай 100мл крохмалевої води.
5. Тушкуй ще хвилину, помішуючи — крохмаль з води створює емульсію з олією у соусі.

## Pizza margherita тісто (24 год)

500г борошна 00, 7г сухих дріжджів, 10г солі, 325мл води. Замісити, ферментувати 24 год у холодильнику. Розкатати руками без качалки, випікати при 280°C.

## Risotto

Постійно помішувати рис (карнаролі або арборіо), додавати теплий бульйон поступово. Наприкінці — холодне масло і пармезан, без вогню. Має бути all'onda — текти як хвиля.
""")

    _write_if_missing(cook / "substitutes.md", """\
# Заміна інгредієнтів

| Що нема | Чим замінити | Пропорція |
|---|---|---|
| Олія соняшникова | Оливкова | 1:1 |
| Білок (1 шт) | Аквафаба (рідина з нуту) | 3 ст. л. |
| Сметана | Грецький йогурт + сік лимона | 1:1 |
| Шкварки | Смажені сухарі з олією | за смаком |
| Біле вино у соусі | Бульйон + 1 ст. л. оцту | 1:1 |
| Вершки 30% | 200мл молока + 50г розтопленого масла | для 250мл |

**Алергія на горіхи:** замінити горіхове масло на тахіні (з кунжуту) — схожа текстура, нейтральніший смак.
""")

    _write_if_missing(general / "ua_holidays.md", """\
# Українські державні свята

- **1 січня** — Новий рік
- **7 січня** — Різдво (старий стиль) — частково замінено на 25 грудня
- **25 грудня** — Різдво Христове (новий календар, з 2023)
- **8 березня** — Міжнародний жіночий день
- **1 травня** — День праці
- **9 травня** — День памʼяті та примирення
- **28 червня** — День Конституції
- **24 серпня** — День Незалежності
- **14 жовтня** — День захисників і захисниць України
""")

    print("[ok] Seeded knowledge bases (3 profiles, 6 files).")


def seed_workspace() -> None:
    WS_DIR.mkdir(parents=True, exist_ok=True)

    _write_if_missing(WS_DIR / "brain_dump.txt", """\
ідеї на тиждень
- переглянути scope FoodHub з командою
- зустріч з catherine у вівторок
- розібратися з push notifications на iOS
- купити подарунок мамі
- спробувати приготувати ризотто без вина
""")

    _write_if_missing(WS_DIR / "contacts.txt", """\
catherine product@example.com +380 50 111 22 33
backend lead andriy@example.com +380 67 222 33 44
designer kateryna design@example.com
qa lead serhii qa@example.com
""")

    _write_if_missing(WS_DIR / "retro_sprint24.md", """\
# Retro Sprint 24

## What went well
- Onboarding нового розробника пройшов плавно
- API design для Cart сервісу прийшов з першого ревʼю
- 0 critical bugs за спринт

## What went wrong
- Затримка на 2 дні через API rate limits у sandbox Stripe
- Тести впали через flaky integration test (відомий, але не виправили)
- Один дейлі пропустили через відсутність facilitator-а

## Action items
- [x] Виправити flaky test до кінця наступного спринта (Andriy)
- [ ] Налаштувати окремий Stripe key для CI (Serhii)
- [ ] Призначити back-up facilitator-а для дейлі (Кateryna)
""")

    _write_if_missing(WS_DIR / "shopping_list.txt", """\
- молоко 1л
- хліб чабата
- помідори чері 500г
- моцарела буффало
- оливкова олія 0.5л
- червоне сухе вино
- кава мелена 250г
""")

    print("[ok] Seeded workspace (4 files).")


def seed_benchmarks(bench: BenchmarkStorage) -> None:
    """Three historical runs with realistic v3 metrics."""

    rng = random.Random(42)

    # ---- Run 1: Quality UA across 3 models ----
    run1 = bench.create_run(
        prompt_set="quality_ua",
        notes="Демо-прогон якості UA на трьох моделях (з суддею)",
        run_type=RUN_TYPE_STANDARD,
    )
    # Override timestamp to look "yesterday".
    _backdate_run(bench, run1, hours_ago=26)
    models_q = ["qwen2.5:14b", "llama3.1:8b", "mistral-nemo:12b-instruct"]
    profile_speed = {"qwen2.5:14b": (28, 920),
                     "llama3.1:8b": (52, 540),
                     "mistral-nemo:12b-instruct": (38, 780)}
    profile_quality = {"qwen2.5:14b": (4.2, 4.5),
                       "llama3.1:8b": (3.4, 3.8),
                       "mistral-nemo:12b-instruct": (3.7, 4.1)}
    profile_vram = {"qwen2.5:14b": 16800,
                    "llama3.1:8b": 5400,
                    "mistral-nemo:12b-instruct": 7900}
    ua_prompts = [
        ("ua_translate_tech", "B-tree індекс прискорює запити рівності й діапазонів..."),
        ("ua_translate_informal", "Привіт! Я думав, ми могли б випити кави завтра вранці..."),
        ("ua_business_letter", "Шановний пане Іване, перепрошуємо за затримку доставки..."),
        ("ua_grammar_fix", "Протягом трьох років я займаюся вивченням української мови..."),
        ("ua_tech_explain", "REST API — це спосіб різним програмам обмінюватися даними..."),
        ("ua_qa_factual", "Тарас Шевченко — український поет і художник XIX століття..."),
        ("ua_summary", "Python — інтерпретована мова програмування, відома читабельністю..."),
        ("ua_creative", "Осінній ранок у Карпатах огортає густий туман..."),
    ]
    for model in models_q:
        tps_base, ttft_base = profile_speed[model]
        q_low, q_high = profile_quality[model]
        for pid, response in ua_prompts:
            tps = tps_base + rng.uniform(-2, 2)
            ttft = ttft_base + rng.uniform(-50, 50)
            tokens = rng.randint(50, 180)
            total = ttft + (tokens / tps) * 1000
            judge_score = round(rng.uniform(q_low, q_high), 1)
            bench.add_result(
                run_id=run1, model_name=model, prompt_id=pid,
                category=CATEGORY_QUALITY_UA,
                ttft_ms=round(ttft, 1), tokens_per_sec=round(tps, 2),
                total_time_ms=round(total, 1), output_tokens=tokens,
                output_text=response, vram_peak_mb=profile_vram[model],
            )
            # Now patch in judge score on the freshly-added result.
            result_id = _last_result_id(bench, run1)
            bench.update_judge_score(
                result_id, judge_score,
                _rationale_for_quality(judge_score, model),
            )
    bench.finish_run(run1)

    # ---- Run 2: Reasoning + Tool Use, two models ----
    run2 = bench.create_run(
        prompt_set="reasoning,tool_use",
        notes="Демо: логіка та виклик інструментів",
        run_type=RUN_TYPE_STANDARD,
    )
    _backdate_run(bench, run2, hours_ago=12)
    models_rt = ["qwen2.5:14b", "llama3.1:8b"]
    reasoning_truth = {
        "reason_age": ("5", "number", "5"),
        "reason_deduction": ("juice", "contains", "Anna drinks juice."),
        "reason_odd_one_out": ("carrot", "contains", "Carrot."),
        "reason_train": ("12:00", "contains", "Train arrives at 12:00."),
        "reason_sequence": ("42", "number", "42"),
        "reason_choice": ("C", "choice", "C"),
    }
    # Simulate model accuracy: qwen 5/6, llama 3/6
    model_correct = {"qwen2.5:14b": {"reason_age", "reason_deduction",
                                     "reason_odd_one_out", "reason_sequence",
                                     "reason_choice"},
                     "llama3.1:8b": {"reason_odd_one_out", "reason_sequence",
                                     "reason_choice"}}

    for model in models_rt:
        for pid, (expected, mtype, correct_answer) in reasoning_truth.items():
            is_correct = pid in model_correct[model]
            response = correct_answer if is_correct else _wrong_answer(pid)
            tps_base, ttft_base = profile_speed[model]
            tps = tps_base + rng.uniform(-3, 3)
            ttft = ttft_base + rng.uniform(-40, 40)
            tokens = rng.randint(5, 40)
            total = ttft + (tokens / tps) * 1000
            bench.add_result(
                run_id=run2, model_name=model, prompt_id=pid,
                category=CATEGORY_REASONING,
                ttft_ms=round(ttft, 1), tokens_per_sec=round(tps, 2),
                total_time_ms=round(total, 1), output_tokens=tokens,
                output_text=response,
                vram_peak_mb=profile_vram[model],
                pass_rate=1.0 if is_correct else 0.0,
                expected=expected, match_type=mtype,
            )

    tool_truth = {
        "tool_search_notes": ("notes_search", {"query": ["python"]},
                              [{"name": "notes_search",
                                "arguments": {"query": "python tips"}}]),
        "tool_create_note": ("notes_create",
                             {"title": ["shopping"], "content": ["milk"]},
                             [{"name": "notes_create",
                               "arguments": {"title": "shopping list",
                                             "content": "milk, bread, eggs"}}]),
        "tool_list_files": ("list_directory", None,
                            [{"name": "list_directory", "arguments": {"path": ""}}]),
        "tool_read_file": ("read_file", {"path": ["report"]},
                           [{"name": "read_file",
                             "arguments": {"path": "report.md"}}]),
        "tool_delete_note": ("notes_delete", {"title": ["old", "draft"]},
                             [{"name": "notes_delete",
                               "arguments": {"title": "old draft"}}]),
    }
    # qwen 5/5, llama 3/5 (wrong on create + delete)
    model_tool_correct = {
        "qwen2.5:14b": set(tool_truth),
        "llama3.1:8b": {"tool_search_notes", "tool_list_files", "tool_read_file"},
    }

    for model in models_rt:
        for pid, (etool, eargs, correct_calls) in tool_truth.items():
            is_correct = pid in model_tool_correct[model]
            if is_correct:
                tool_calls = correct_calls
                pass_rate = 1.0
            else:
                tool_calls = [{"name": "notes_read",
                               "arguments": {"title": "wrong tool"}}]
                pass_rate = 0.0
            tps_base, ttft_base = profile_speed[model]
            tps = tps_base + rng.uniform(-3, 3)
            ttft = ttft_base + rng.uniform(-40, 40)
            tokens = rng.randint(15, 50)
            total = ttft + (tokens / tps) * 1000
            bench.add_result(
                run_id=run2, model_name=model, prompt_id=pid,
                category=CATEGORY_TOOL_USE,
                ttft_ms=round(ttft, 1), tokens_per_sec=round(tps, 2),
                total_time_ms=round(total, 1), output_tokens=tokens,
                output_text="",
                vram_peak_mb=profile_vram[model],
                pass_rate=pass_rate,
                expected_tool=etool, expected_args=eargs,
                tool_calls=tool_calls,
            )
    bench.finish_run(run2)

    # ---- Run 3: Quantization sweep on qwen2.5:14b ----
    run3 = bench.create_run(
        prompt_set="reasoning",
        notes="Quantization sweep: qwen2.5:14b at Q4_K_M / Q8_0 / FP16",
        run_type=RUN_TYPE_QUANTIZATION,
    )
    _backdate_run(bench, run3, hours_ago=2)
    quant_models = [
        ("qwen2.5:14b-instruct-q4_K_M", 48.0, 980, 9100, 0.65),
        ("qwen2.5:14b-instruct-q8_0",    28.5, 920, 16800, 0.83),
        ("qwen2.5:14b-instruct-fp16",    14.2, 880, 28400, 0.95),
    ]
    for model, tps_base, ttft_base, vram, accuracy in quant_models:
        for pid, (expected, mtype, correct) in reasoning_truth.items():
            is_correct = rng.random() < accuracy
            response = correct if is_correct else _wrong_answer(pid)
            tps = tps_base + rng.uniform(-1, 1)
            ttft = ttft_base + rng.uniform(-30, 30)
            tokens = rng.randint(5, 35)
            total = ttft + (tokens / tps) * 1000
            bench.add_result(
                run_id=run3, model_name=model, prompt_id=pid,
                category=CATEGORY_REASONING,
                ttft_ms=round(ttft, 1), tokens_per_sec=round(tps, 2),
                total_time_ms=round(total, 1), output_tokens=tokens,
                output_text=response, vram_peak_mb=vram,
                pass_rate=1.0 if is_correct else 0.0,
                expected=expected, match_type=mtype,
            )
    bench.finish_run(run3)

    print(f"[ok] Seeded 3 benchmark runs (ids {run1}, {run2}, {run3}).")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _write_if_missing(path: Path, content: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _backdate_run(bench: BenchmarkStorage, run_id: int, hours_ago: float) -> None:
    """Move the run's started_at backwards so the history list looks realistic."""
    started = _iso(_hours_ago(hours_ago))
    bench._conn.execute(  # pragma: no cover — utility for demo
        "UPDATE benchmark_runs SET started_at = ? WHERE id = ?",
        (started, run_id),
    )
    bench._conn.commit()


def _last_result_id(bench: BenchmarkStorage, run_id: int) -> int:
    row = bench._conn.execute(
        "SELECT MAX(id) FROM benchmark_results WHERE run_id = ?", (run_id,),
    ).fetchone()
    return int(row[0])


def _wrong_answer(prompt_id: str) -> str:
    wrong = {
        "reason_age": "10",
        "reason_deduction": "Tea",
        "reason_odd_one_out": "banana",
        "reason_train": "11:30",
        "reason_sequence": "40",
        "reason_choice": "A",
    }
    return wrong.get(prompt_id, "I'm not sure.")


def _rationale_for_quality(score: float, model: str) -> str:
    if score >= 4.3:
        return f"Природна українська, точний переклад термінів."
    if score >= 3.8:
        return f"Загалом коректно, незначні стилістичні огріхи."
    return f"Помітні англіцизми чи кальки; зміст переданий частково."


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reset", action="store_true",
                        help="Wipe data/ directories before seeding.")
    parser.add_argument("--no-benchmarks", action="store_true",
                        help="Skip benchmark history seeding.")
    args = parser.parse_args()

    if args.reset:
        print("[!] --reset: wiping existing data...")
        for p in [DB_PATH, CHROMA_DIR]:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink()
        for d in [NOTES_DIR]:
            if d.exists():
                for f in d.glob("*.md"):
                    f.unlink()

    DATA_DIR.mkdir(exist_ok=True)
    embedder = _try_real_embedder()
    chroma_factory = ChromaClientFactory()

    storage = Storage(DB_PATH)
    seed_conversations(storage)

    long_term = LongTermMemory(
        chroma_path=CHROMA_DIR, embedder=embedder,
        chroma_factory=chroma_factory,
    )
    seed_long_term_memory(long_term)

    notes_store = NotesStore(
        notes_dir=NOTES_DIR, chroma_path=CHROMA_DIR,
        embedder=embedder, collection_name="notes",
        chroma_factory=chroma_factory,
    )
    seed_notes(notes_store)

    seed_knowledge_bases()
    seed_workspace()

    if not args.no_benchmarks:
        bench = BenchmarkStorage(storage.connection)
        seed_benchmarks(bench)

    storage.close()
    print("\n[OK] Demo data seeded successfully.")
    print(f"   App DB: {DB_PATH}")
    print(f"   Chroma: {CHROMA_DIR}")
    print(f"   Notes:  {NOTES_DIR}")
    print(f"   KB:     {KB_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
