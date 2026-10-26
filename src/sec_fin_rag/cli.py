from __future__ import annotations

import argparse
import json

from .config import get_settings
from .evaluation import evaluate, evaluate_answers, evaluate_with_langsmith
from .pipeline import build_artifacts, build_structured_artifacts, download, index, parse
from .preflight import run_preflight


def _preflight() -> bool:
    checks = run_preflight(get_settings())
    for check in checks:
        marker = "PASS" if check.passed else ("FAIL" if check.required else "WARN")
        print(f"[{marker}] {check.name}: {check.detail}")
    return all(check.passed for check in checks if check.required)


def main() -> None:
    parser = argparse.ArgumentParser(prog="finrag", description="SEC Financial RAG pipeline")
    parser.add_argument(
        "command",
        choices=[
            "preflight",
            "download",
            "parse",
            "build-artifacts",
            "build-structured",
            "index",
            "evaluate",
            "evaluate-answers",
            "evaluate-langsmith",
            "all",
        ],
    )
    args = parser.parse_args()
    settings = get_settings()
    if args.command == "preflight":
        raise SystemExit(0 if _preflight() else 1)
    if args.command == "all" and not _preflight():
        raise SystemExit(1)
    if args.command in {"download", "all"}:
        download(settings)
    if args.command in {"parse", "all"}:
        parse(settings)
    if args.command in {"build-artifacts", "all"}:
        build_artifacts(settings)
    if args.command == "build-structured":
        build_structured_artifacts(settings)
    if args.command in {"index", "all"}:
        index(settings)
    if args.command in {"evaluate", "all"}:
        print(json.dumps(evaluate(settings), indent=2))
    if args.command == "evaluate-answers":
        print(json.dumps(evaluate_answers(settings), indent=2))
    if args.command == "evaluate-langsmith":
        print(f"LangSmith experiment complete: {evaluate_with_langsmith(settings)}")


if __name__ == "__main__":
    main()
