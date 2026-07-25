#!/usr/bin/env python3
"""Run the leakage-safe CADENCE teacher benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

from cadence.experiments.teacher import (
    LEARNED_METHODS,
    compact_result_table,
    make_experiment_config,
    make_profile_teacher_config,
    run_teacher_experiment,
    teacher_experiment_scientific_sha256,
    validate_locked_teacher_experiment_config,
)
from cadence.protocol import ProtocolViolation, attest_preoutcome_freeze
from cadence.teacher import (
    generate_teacher_world,
    load_teacher_config,
    teacher_config_sha256,
    validate_locked_teacher_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train normal dynamics on donors, train the causal operator on donor "
            "perturbations, adapt targets on normal trials only, and score sealed "
            "paired target interventions."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/teacher.yaml"))
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output", type=Path, default=Path("results/teacher"))
    parser.add_argument("--partition", choices=("development", "locked"), default="development")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        metavar="INDEX",
        help="One or more indices in the selected teacher seed partition.",
    )
    parser.add_argument(
        "--run-seed",
        type=int,
        default=0,
        help="Optimization/initialization seed; teacher world seeds are separate.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=LEARNED_METHODS,
        default=list(LEARNED_METHODS),
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (default: CUDA when available, otherwise CPU).",
    )
    parser.add_argument("--normal-epochs", type=int)
    parser.add_argument("--intervention-epochs", type=int)
    parser.add_argument("--target-epochs", type=int)
    parser.add_argument(
        "--no-ablations",
        action="store_true",
        help="Skip proposed no-residual and no-target-adaptation ablations.",
    )
    parser.add_argument(
        "--acknowledge-locked",
        action="store_true",
        help=(
            "Required for the public-seed, post-freeze procedural partition; "
            "this is an audit boundary, not prospective seed secrecy."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _override_epochs(config, args: argparse.Namespace):
    if args.normal_epochs is not None:
        config = replace(
            config,
            normal_fit=replace(
                config.normal_fit,
                max_epochs=args.normal_epochs,
                patience=min(config.normal_fit.patience, args.normal_epochs),
            ),
        )
    if args.intervention_epochs is not None:
        config = replace(
            config,
            intervention_fit=replace(
                config.intervention_fit,
                max_epochs=args.intervention_epochs,
                patience=min(config.intervention_fit.patience, args.intervention_epochs),
            ),
        )
    if args.target_epochs is not None:
        config = replace(
            config,
            target_fit=replace(
                config.target_fit,
                max_epochs=args.target_epochs,
                patience=min(config.target_fit.patience, args.target_epochs),
            ),
        )
    return config


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    freeze_attestation = None
    if any(
        value is not None and value < 1
        for value in (
            args.normal_epochs,
            args.intervention_epochs,
            args.target_epochs,
        )
    ):
        raise SystemExit("epoch overrides must be positive")

    base = load_teacher_config(args.config)
    if args.partition == "locked":
        violations = []
        if args.profile != "full":
            violations.append("--profile must be full")
        if args.run_seed != 0:
            violations.append("--run-seed must be 0")
        if tuple(args.methods) != LEARNED_METHODS:
            violations.append("--methods must be the complete frozen ordered method set")
        if any(
            value is not None
            for value in (
                args.normal_epochs,
                args.intervention_epochs,
                args.target_epochs,
            )
        ):
            violations.append("epoch overrides are development-only")
        if args.no_ablations:
            violations.append("--no-ablations is development-only")
        if args.overwrite:
            violations.append("--overwrite is forbidden for post-freeze procedural artifacts")
        if len(set(args.seeds)) != len(args.seeds):
            violations.append("post-freeze procedural seed indices must be unique")
        if any(not 0 <= seed < len(base.seeds.locked) for seed in args.seeds):
            violations.append("post-freeze procedural seed index is outside the canonical cohort")
        try:
            validate_locked_teacher_config(base)
        except ProtocolViolation as error:
            violations.append(str(error))
        if violations:
            raise SystemExit(
                "invalid post-freeze procedural teacher evaluation scope: " + "; ".join(violations)
            )
    experiment = make_experiment_config(
        args.profile,
        seed=args.run_seed,
        device=args.device,
        learned_methods=args.methods,
    )
    experiment = replace(experiment, include_ablations=not args.no_ablations)
    experiment = _override_epochs(experiment, args)
    index_path = args.output / args.profile / "index.json"
    index_sha256_path = args.output / args.profile / "index.json.sha256"
    if args.partition == "locked":
        try:
            validate_locked_teacher_experiment_config(experiment)
        except ProtocolViolation as error:
            raise SystemExit(str(error)) from error
        existing_index_artifacts = [
            path for path in (index_path, index_sha256_path) if path.exists()
        ]
        if existing_index_artifacts:
            raise SystemExit(
                "post-freeze procedural index is append-only; existing "
                + ", ".join(str(path) for path in existing_index_artifacts)
            )
        if not args.acknowledge_locked:
            raise SystemExit(
                "Refusing to open the post-freeze procedural partition without "
                "--acknowledge-locked."
            )
        freeze_attestation = attest_preoutcome_freeze()
    teacher_config = make_profile_teacher_config(base, args.profile)
    results: list[dict[str, object]] = []

    for seed_index in args.seeds:
        world = generate_teacher_world(
            teacher_config,
            partition=args.partition,
            seed_index=seed_index,
            freeze_attestation=freeze_attestation,
        )
        dataset = world.generate_dataset()
        world_output = args.output / args.profile / f"{args.partition}-seed-{seed_index:02d}"
        payload = run_teacher_experiment(
            dataset,
            experiment,
            world_output,
            run_seed=args.run_seed,
            overwrite=args.overwrite,
            freeze_attestation=(None if freeze_attestation is None else asdict(freeze_attestation)),
        )
        completion_path = world_output / "completion.json"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        results.append(
            {
                "seed_index": seed_index,
                "world_id": world.ground_truth.world_id,
                "output": str(world_output),
                "aggregate": payload["aggregate"],
                "completion": str(completion_path),
                "artifact_sha256": completion["artifacts"],
            }
        )
        print(f"\nworld: {world.ground_truth.world_id}")
        print(compact_result_table(payload))

    index_path.parent.mkdir(parents=True, exist_ok=True)
    if index_path.exists() and not args.overwrite:
        raise SystemExit(
            f"Per-world runs completed, but refusing to overwrite {index_path}; "
            "pass --overwrite to refresh the index."
        )
    index_payload = {
        "schema_version": "cadence.teacher_index.v2",
        "partition": args.partition,
        "seed_material_public": True,
        "prospective_seed_secrecy": False,
        "evaluation_role": (
            "post_freeze_deterministic_procedural_audit"
            if args.partition == "locked"
            else "method_development"
        ),
        "eligible_for_biological_headline_conjunction": False,
        "teacher_config_sha256": teacher_config_sha256(teacher_config),
        "teacher_experiment_scientific_sha256": (teacher_experiment_scientific_sha256(experiment)),
        "learned_methods": list(experiment.learned_methods),
        "canonical_learned_method_set_complete": (
            tuple(experiment.learned_methods) == LEARNED_METHODS
        ),
        "preoutcome_freeze": (None if freeze_attestation is None else asdict(freeze_attestation)),
        "worlds": results,
    }
    index_text = (
        json.dumps(
            index_payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    mode = "x" if args.partition == "locked" else "w"
    with index_path.open(mode, encoding="utf-8") as stream:
        stream.write(index_text)
    index_sha256 = hashlib.sha256(index_text.encode()).hexdigest()
    with index_sha256_path.open(mode, encoding="utf-8") as stream:
        stream.write(f"{index_sha256}  {index_path.name}\n")
    print(f"\nWrote {index_path}")


if __name__ == "__main__":
    main()
