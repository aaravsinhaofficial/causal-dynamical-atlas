#!/usr/bin/env python
"""Create one leakage-audited, trimodal processed file per ICMS task mouse."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from cadence.data.dandi_icms import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_PROCESSED_ROOT,
    DEFAULT_RAW_ROOT,
    ICMSPreprocessConfig,
    preprocess_release,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--config", type=Path, default=Path("configs/dandi_icms.yaml"))
    parser.add_argument("--animal", action="append", dest="animals")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="process only locally present frozen assets (for explicit smoke tests)",
    )
    parser.add_argument("--skip-raw-verification", action="store_true")
    args = parser.parse_args()

    payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = ICMSPreprocessConfig(**payload["preprocessing"])
    result = preprocess_release(
        args.manifest,
        args.raw_root,
        args.output_root,
        config=config,
        animals=args.animals,
        allow_partial=args.allow_partial,
        verify_raw=not args.skip_raw_verification,
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
