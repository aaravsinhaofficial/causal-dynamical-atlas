"""Command-line entry point."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cadence",
        description="Cross-animal perturbation-response experiments.",
    )
    parser.add_argument("--version", action="store_true", help="Print the package version.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.version:
        from cadence import __version__

        print(__version__)
