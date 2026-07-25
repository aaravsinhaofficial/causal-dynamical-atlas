#!/usr/bin/env python
"""Download a checksum-locked scope of published DANDI:001868."""

from __future__ import annotations

import argparse
from pathlib import Path

from cadence.data.dandi_icms import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_RAW_ROOT,
    audit_frozen_manifest_against_api,
    download_release,
    load_frozen_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--scope",
        choices=("all", "task", "trimodal"),
        default="task",
        help="task downloads all 55 task-session assets; all downloads the 7.50 GB release",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--audit-api",
        action="store_true",
        help="first prove the frozen manifest still exactly matches the published API",
    )
    args = parser.parse_args()

    manifest = load_frozen_manifest(args.manifest)
    if args.audit_api:
        audit_frozen_manifest_against_api(manifest, workers=max(args.workers, 4))
    result = download_release(
        manifest,
        args.raw_root,
        scope=args.scope,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
