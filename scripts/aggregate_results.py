#!/usr/bin/env python3
"""Aggregate sealed CADENCE result JSONs at the target-animal level.

Examples
--------
After all five locked Allen folds have completed::

    uv run python scripts/aggregate_results.py \
      --allen results/allen-vbo/locked-fold-* \
      --output results/report

Teacher and ICMS results can be added in the same invocation::

    uv run python scripts/aggregate_results.py \
      --allen results/allen-vbo \
      --teacher results/teacher-locked/full \
      --icms results/icms \
      --output results/report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from cadence.reporting import (
    DEFAULT_BOOTSTRAP_REPEATS,
    DEFAULT_SEED,
    AdaptedBatch,
    adapt_payload,
    aggregate_batches,
    write_report,
)


def _is_hidden_or_quarantine_path(path: Path) -> bool:
    """Reject interrupted-result trees, including visible aliases to them."""

    for component in path.parts:
        if component.startswith("."):
            return True
        tokens = component.casefold().replace(".", "-").replace("_", "-").split("-")
        if {"interrupted", "quarantine"} & set(tokens):
            return True
    return False


def _eligible_metric_file(path: Path) -> bool:
    return not (
        _is_hidden_or_quarantine_path(path) or _is_hidden_or_quarantine_path(path.resolve())
    )


def _metric_files(values: list[Path]) -> list[Path]:
    """Resolve visible committed-result files without ingesting quarantine trees."""

    files: list[Path] = []
    for value in values:
        if value.is_dir():
            files.extend(
                path for path in sorted(value.rglob("metrics.json")) if _eligible_metric_file(path)
            )
        elif value.is_file():
            if _eligible_metric_file(value):
                files.append(value)
        else:
            raise FileNotFoundError(value)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def _load(paths: list[Path], kind: Literal["allen", "teacher", "icms"]) -> list[AdaptedBatch]:
    batches: list[AdaptedBatch] = []
    for path in _metric_files(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path} does not contain a JSON object")
        batches.append(adapt_payload(payload, kind=kind, source_file=path))
    return batches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allen",
        type=Path,
        nargs="*",
        default=[],
        help="Allen metrics.json files or directories searched recursively",
    )
    parser.add_argument(
        "--teacher",
        type=Path,
        nargs="*",
        default=[],
        help="teacher metrics.json files or directories searched recursively",
    )
    parser.add_argument(
        "--icms",
        type=Path,
        nargs="*",
        default=[],
        help="ICMS metrics.json files or directories searched recursively",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPEATS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    batches = [
        *_load(args.allen, "allen"),
        *_load(args.teacher, "teacher"),
        *_load(args.icms, "icms"),
    ]
    if not batches:
        raise SystemExit("no metrics.json inputs were found")
    report = aggregate_batches(
        batches,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    paths = write_report(report, args.output)
    print(
        json.dumps(
            {
                "analyses": {
                    name: {
                        "n_independent_units": analysis["n_independent_units"],
                        "headline_status": analysis["conjunction"]["overall_status"],
                    }
                    for name, analysis in report["analyses"].items()
                },
                "artifacts": {name: str(path) for name, path in paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
