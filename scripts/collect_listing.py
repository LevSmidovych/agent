"""Збирає всі основні файли коду в один текстовий файл для додатка диплому.

Запуск:
    .venv\\Scripts\\python.exe scripts\\collect_listing.py

Створює `thesis_listing.txt` у корені проекту з усіма файлами по розділах,
зі заголовками і номерами рядків — готово для вставки у Word/LaTeX додаток.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Розбито за розділами диплому. Кожен розділ — (заголовок, список файлів).
SECTIONS: list[tuple[str, list[str]]] = [
    ("Б.1 Точка входу і конфігурація", [
        "main.py",
        "pyproject.toml",
        "settings.yaml",
        "profiles/configs/project_manager.yaml",
        "profiles/configs/general.yaml",
        "profiles/configs/cook.yaml",
    ]),
    ("Б.2 Архітектурне ядро (core/)", [
        "core/agent.py",
        "core/llm_client.py",
        "core/ollama_parsing.py",
        "core/tool_executor.py",
        "core/prompt_builder.py",
        "core/embeddings.py",
        "core/chroma_client.py",
        "core/settings.py",
        "core/slash_commands.py",
        "core/exceptions.py",
    ]),
    ("Б.3 Система пам'яті (memory/)", [
        "memory/storage.py",
        "memory/migrations.py",
        "memory/short_term.py",
        "memory/long_term.py",
        "memory/classifier.py",
    ]),
    ("Б.4 Інструменти (tools/)", [
        "tools/base.py",
        "tools/notes.py",
        "tools/files.py",
    ]),
    ("Б.5 Профілі і RAG (profiles/)", [
        "profiles/loader.py",
        "profiles/rag.py",
    ]),
    ("Б.6 Система бенчмаркінгу (benchmarks/)", [
        "benchmarks/constants.py",
        "benchmarks/prompts.py",
        "benchmarks/scoring.py",
        "benchmarks/judge.py",
        "benchmarks/runner.py",
        "benchmarks/resources.py",
        "benchmarks/quantization.py",
        "benchmarks/charts.py",
        "benchmarks/exporter.py",
        "benchmarks/storage.py",
    ]),
    ("Б.7 UI (вибрані модулі)", [
        "ui/benchmark_window.py",
        "ui/benchmark_runner_worker.py",
        "ui/chart_widget.py",
        "ui/main_window.py",
    ]),
]


def collect(output: Path, include_line_numbers: bool = True) -> None:
    total_lines = 0
    files_written = 0
    files_missing = []

    with output.open("w", encoding="utf-8") as out:
        out.write("ДОДАТОК Б — Лістинг програмного коду\n")
        out.write("=" * 72 + "\n\n")
        out.write(
            "Проєкт: Персональний ШІ-агент v0.2 на локальних LLM\n"
            "Зібрано автоматично скриптом scripts/collect_listing.py\n\n"
        )

        for section_idx, (section_title, files) in enumerate(SECTIONS, start=1):
            out.write("\n" + "=" * 72 + "\n")
            out.write(f"{section_title}\n")
            out.write("=" * 72 + "\n")

            for file_idx, rel_path in enumerate(files, start=1):
                full = ROOT / rel_path
                out.write("\n")
                out.write("-" * 72 + "\n")
                out.write(f"Файл: {rel_path}\n")
                if not full.exists():
                    out.write("[файл відсутній]\n")
                    files_missing.append(rel_path)
                    continue
                try:
                    content = full.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    content = full.read_text(encoding="utf-8", errors="replace")

                lines = content.splitlines()
                out.write(f"Рядків: {len(lines)}\n")
                out.write("-" * 72 + "\n")

                if include_line_numbers:
                    width = len(str(len(lines)))
                    for ln, line in enumerate(lines, start=1):
                        out.write(f"{ln:>{width}}  {line}\n")
                else:
                    out.write(content)
                    if not content.endswith("\n"):
                        out.write("\n")

                total_lines += len(lines)
                files_written += 1

        out.write("\n\n" + "=" * 72 + "\n")
        out.write("Підсумок\n")
        out.write("=" * 72 + "\n")
        out.write(f"Файлів зібрано: {files_written}\n")
        out.write(f"Загалом рядків коду: {total_lines}\n")
        if files_missing:
            out.write(f"\nВідсутні файли ({len(files_missing)}):\n")
            for p in files_missing:
                out.write(f"  - {p}\n")

    print(f"[OK] Записано: {output}")
    print(f"     Файлів: {files_written}")
    print(f"     Рядків коду: {total_lines}")
    if files_missing:
        print(f"     Відсутніх файлів: {len(files_missing)}")
        for p in files_missing:
            print(f"       - {p}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-o", "--output", default="thesis_listing.txt",
        help="Куди записати (default: thesis_listing.txt у корені проекту)",
    )
    parser.add_argument(
        "--no-line-numbers", action="store_true",
        help="Без префіксу номерів рядків (чистий код)",
    )
    args = parser.parse_args()

    output_path = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    collect(output_path, include_line_numbers=not args.no_line_numbers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
