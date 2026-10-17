from __future__ import annotations

import argparse
import json

from .config import get_settings
from .evaluation import evaluate
from .pipeline import build_artifacts, download, index, parse
from .preflight import run_preflight


def _preflight() -> bool:
    checks = run_preflight(get_settings())
    for check in checks:
        marker = "PASS" if check.passed else "FAIL"
        print(f"[{marker}] {check.name}: {check.detail}")
    return all(check.passed for check in checks)


def main() -> None:
    parser = argparse.ArgumentParser(prog="finrag", description="SEC Financial RAG pipeline")
    parser.add_argument(
        "command",
        choices=["preflight", "download", "parse", "build-artifacts", "index", "evaluate", "all"],
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
    if args.command in {"index", "all"}:
        index(settings)
    if args.command in {"evaluate", "all"}:
        print(json.dumps(evaluate(settings), indent=2))


if __name__ == "__main__":
    main()
