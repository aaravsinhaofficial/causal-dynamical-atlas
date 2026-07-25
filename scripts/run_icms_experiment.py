#!/usr/bin/env python
"""Run the sealed DANDI:001868 ICMS leave-one-animal-out workflow.

No stimulation response is opened by ``prepare``. Every biological stage
requires a clean checkout at the exact ``pre-outcome-v1.0.0`` tag.

Examples
--------
After committing and tagging the frozen protocol, prepare target-normal
supports and the target-independent physical lattice::

    uv run python scripts/run_icms_experiment.py prepare \
      --target ICMS92 \
      --output results/icms/loao-ICMS92 --optimization full

Open donor outcomes only::

    uv run python scripts/run_icms_experiment.py predict \
      --output results/icms/loao-ICMS92 --optimization full \
      --acknowledge-donor-outcomes

After the prediction hash exists, unseal the target exactly once::

    uv run python scripts/run_icms_experiment.py score \
      --output results/icms/loao-ICMS92 --acknowledge-target-outcomes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cadence.experiments.icms import (
    REPORT_METHODS,
    make_icms_config,
    predict_fold,
    prepare_fold,
    score_fold,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="read normal support only and write query files"
    )
    prepare.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed/dandi_001868"),
    )
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--target", required=True)
    prepare.add_argument("--protocol-commit")
    prepare.add_argument("--run-mode", choices=("biological", "synthetic"), default="biological")
    prepare.add_argument("--optimization", choices=("smoke", "full"), default="full")
    prepare.add_argument("--device")
    prepare.add_argument("--seed", type=int, default=20260725)
    prepare.add_argument("--overwrite", action="store_true")

    predict = subparsers.add_parser(
        "predict", help="after the freeze, fit donors and hash target predictions"
    )
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--optimization", choices=("smoke", "full"), default="full")
    predict.add_argument("--device")
    predict.add_argument("--seed", type=int, default=20260725)
    predict.add_argument(
        "--methods",
        nargs="+",
        choices=REPORT_METHODS,
        default=list(REPORT_METHODS),
    )
    predict.add_argument("--run-mode", choices=("biological", "synthetic"), default="biological")
    predict.add_argument(
        "--acknowledge-donor-outcomes",
        action="store_true",
        help="confirm HEAD is the frozen tag before opening donor stimulation responses",
    )
    predict.add_argument("--overwrite", action="store_true")

    score = subparsers.add_parser(
        "score", help="verify hashes, then open target stimulation outcomes"
    )
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--run-mode", choices=("biological", "synthetic"), default="biological")
    score.add_argument(
        "--acknowledge-target-outcomes",
        action="store_true",
        help="explicitly unseal target responses after prediction hashing",
    )
    score.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        config = make_icms_config(args.optimization, seed=args.seed, device=args.device)
        result = prepare_fold(
            processed_root=args.processed_root,
            output_directory=args.output,
            target_animal=args.target,
            config=config,
            protocol_commit=args.protocol_commit,
            run_mode=args.run_mode,
            overwrite=args.overwrite,
        )
        summary = {
            "command": "prepare",
            "output": result["output"],
            "target_animal": result["target_animal"],
            "normal_support_sessions": len(result["normal_supports"]),
            "target_query_sessions": len(result["target_queries"]),
            "prepare_manifest_sha256": result["manifest_sha256"],
        }
    elif args.command == "predict":
        config = make_icms_config(args.optimization, seed=args.seed, device=args.device)
        result = predict_fold(
            fold_directory=args.output,
            config=config,
            methods=args.methods,
            acknowledge_donor_outcomes=args.acknowledge_donor_outcomes,
            run_mode=args.run_mode,
            overwrite=args.overwrite,
        )
        summary = {
            "command": "predict",
            "output": result["output"],
            "target_animal": result["target_animal"],
            "prediction_sha256": result["prediction_sha256_before_target_open"],
            "target_outcomes_opened": False,
        }
    else:
        result = score_fold(
            fold_directory=args.output,
            acknowledge_target_outcomes=args.acknowledge_target_outcomes,
            run_mode=args.run_mode,
            overwrite=args.overwrite,
        )
        summary = {
            "command": "score",
            "output": result["output"],
            "target_animal": result["target_animal"],
            "metrics_sha256": result["metrics_sha256"],
            "randomized_causal_eligible": result["causal_effect_eligibility"]["animal_eligible"],
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
