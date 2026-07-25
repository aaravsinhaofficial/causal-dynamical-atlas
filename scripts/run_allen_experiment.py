#!/usr/bin/env python
"""Run the sealed Allen VBO omission experiment.

Examples
--------
Development smoke (the only mode intended before protocol freeze)::

    uv run python scripts/run_allen_experiment.py \
      --run-profile development --optimization smoke \
      --output results/allen-vbo/development-smoke

Locked fold after the protocol/code commit is frozen::

    uv run python scripts/run_allen_experiment.py \
      --stage prepare \
      --run-profile locked --fold 0 --acknowledge-locked \
      --optimization full --seed 0 --output results/allen-vbo/locked-fold-0

    uv run python scripts/run_allen_experiment.py \
      --stage predict \
      --run-profile locked --fold 0 --acknowledge-locked \
      --optimization full --seed 0 --output results/allen-vbo/locked-fold-0

    uv run python scripts/run_allen_experiment.py \
      --stage score \
      --run-profile locked --fold 0 --acknowledge-locked \
      --optimization full --seed 0 --output results/allen-vbo/locked-fold-0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cadence.experiments.allen import (
    LEARNED_METHODS,
    make_allen_config,
    run_allen_experiment,
)

SOURCE_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=SOURCE_ROOT / "data/processed/allen_vbo",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=SOURCE_ROOT
        / Path("data/manifests/allen_vbo_slc17a7_visp175_familiar_active_v1.1.0.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("prepare", "predict", "score", "all"),
        default="all",
        help="locked runs require three separate prepare/predict/score invocations",
    )
    parser.add_argument(
        "--run-profile",
        choices=("development", "locked"),
        default="development",
    )
    parser.add_argument("--optimization", choices=("smoke", "fast", "full"), default="smoke")
    parser.add_argument("--fold", type=int)
    parser.add_argument(
        "--acknowledge-locked",
        action="store_true",
        help="confirm that the protocol/code commit was frozen before opening locked outcomes",
    )
    parser.add_argument("--development-target", default="423606")
    parser.add_argument(
        "--development-donors",
        nargs="+",
        default=["539517", "448900"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=LEARNED_METHODS,
        default=list(LEARNED_METHODS),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.run_profile == "locked":
        if args.overwrite:
            parser.error("--overwrite is forbidden for locked Allen stages")
        if args.optimization != "full":
            parser.error("locked Allen stages require --optimization full")
        if args.seed != 0:
            parser.error("locked Allen stages require --seed 0")
        if tuple(args.methods) != LEARNED_METHODS:
            parser.error("locked Allen stages require all learned methods in canonical order")
    config = make_allen_config(
        args.optimization,
        seed=args.seed,
        device=args.device,
        methods=args.methods,
    )
    result = run_allen_experiment(
        processed_root=args.processed_root,
        manifest_path=args.manifest,
        output_directory=args.output,
        run_profile=args.run_profile,
        optimization=config,
        fold=args.fold,
        acknowledge_locked=args.acknowledge_locked,
        development_target=args.development_target,
        development_donors=args.development_donors,
        seed=args.seed,
        overwrite=args.overwrite,
        stage=args.stage,
    )
    summary = {
        "output": str(args.output),
        "run_profile": result["run_profile"],
        "stage": args.stage,
        "targets": result["targets"],
    }
    if "aggregate" in result:
        summary["aggregate"] = result["aggregate"]
        summary["prediction_sha256"] = result["protocol_audit"]["prediction_sha256_before_score"]
    elif "prediction_sha256" in result:
        summary["prediction_sha256"] = result["prediction_sha256"]
    elif "preparation_sha256" in result:
        summary["preparation_sha256"] = result["preparation_sha256"]
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
