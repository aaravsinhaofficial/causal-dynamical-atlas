#!/usr/bin/env python3
"""Materialize deterministic CADENCE teacher-RNN release artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from cadence.protocol import ProtocolViolation, attest_preoutcome_freeze
from cadence.teacher import (
    StressCondition,
    generate_teacher_world,
    load_teacher_config,
    save_teacher_release,
    teacher_config_sha256,
    validate_locked_teacher_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paired counterfactual trials and exact ground truth for a "
            "procedural teacher-RNN world."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/teacher.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/teacher"))
    parser.add_argument("--partition", choices=("development", "locked"), default="development")
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument(
        "--all-seeds", action="store_true", help="Generate every seed in the partition."
    )
    parser.add_argument(
        "--world-seed",
        type=int,
        help=(
            "Explicit development-diagnostic seed; unavailable for the locked "
            "cohort and mutually exclusive with --all-seeds."
        ),
    )
    parser.add_argument(
        "--acknowledge-locked",
        action="store_true",
        help=(
            "Required for the public-seed, post-freeze procedural partition; "
            "this is an audit boundary, not prospective seed secrecy."
        ),
    )
    parser.add_argument("--eta", type=float, choices=(0.0, 0.5, 0.8, 1.0))
    parser.add_argument("--rho", type=float, choices=(0.0, 0.1, 0.25, 0.5))
    parser.add_argument("--target-neurons", type=int, choices=(32, 64, 128))
    parser.add_argument("--support", type=int, choices=(8, 16, 32, 64))
    parser.add_argument("--state-coverage", choices=("full", "narrow"))
    parser.add_argument(
        "--impossibility",
        choices=("none", "independent_target_direction"),
    )
    parser.add_argument("--impossibility-variant", type=int, choices=(-1, 1))
    parser.add_argument(
        "--omit-latents",
        action="store_true",
        help="Do not put diagnostic latent states in dataset.npz.",
    )
    parser.add_argument(
        "--omit-noise",
        action="store_true",
        help="Do not put paired exogenous variables in dataset.npz.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow replacement of existing artifacts."
    )
    return parser


def _artifact_paths(output_dir: Path) -> tuple[Path, ...]:
    return (
        output_dir / "dataset.npz",
        output_dir / "ground_truth.npz",
        output_dir / "manifest.json",
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    freeze_attestation = None
    if args.all_seeds and args.world_seed is not None:
        raise SystemExit("--all-seeds and --world-seed are mutually exclusive")
    locked_stress_overrides = [
        flag
        for flag, value in (
            ("--eta", args.eta),
            ("--rho", args.rho),
            ("--target-neurons", args.target_neurons),
            ("--support", args.support),
            ("--state-coverage", args.state_coverage),
            ("--impossibility", args.impossibility),
            ("--impossibility-variant", args.impossibility_variant),
        )
        if value is not None
    ]
    if args.partition == "locked" and args.world_seed is not None:
        raise SystemExit(
            "--world-seed is development-only; post-freeze procedural worlds "
            "must use a configured --seed-index"
        )
    if args.partition == "locked" and locked_stress_overrides:
        raise SystemExit(
            "stress overrides are development-only; post-freeze procedural "
            "worlds require the frozen default stress (received "
            + ", ".join(locked_stress_overrides)
            + ")"
        )
    config = load_teacher_config(args.config)
    if args.partition == "locked":
        if args.overwrite:
            raise SystemExit("--overwrite is forbidden for post-freeze procedural artifacts")
        try:
            validate_locked_teacher_config(config)
        except ProtocolViolation as error:
            raise SystemExit(str(error)) from error
        if not args.all_seeds and not 0 <= args.seed_index < len(config.seeds.locked):
            raise SystemExit("post-freeze procedural seed index is outside the canonical cohort")
    if args.partition == "locked" and not args.acknowledge_locked:
        raise SystemExit(
            "Refusing to open the post-freeze procedural partition without "
            "--acknowledge-locked. Use development worlds during model iteration."
        )
    if args.partition == "locked":
        freeze_attestation = attest_preoutcome_freeze()

    seeds = config.seeds.development if args.partition == "development" else config.seeds.locked
    seed_indices = range(len(seeds)) if args.all_seeds else (args.seed_index,)
    stress = StressCondition(
        eta=1.0 if args.eta is None else args.eta,
        rho=(config.dynamics.residual_ratio if args.rho is None else args.rho),
        target_neurons=args.target_neurons,
        support=args.support,
        state_coverage=("full" if args.state_coverage is None else args.state_coverage),
        impossibility=("none" if args.impossibility is None else args.impossibility),
        impossibility_variant=(
            1 if args.impossibility_variant is None else args.impossibility_variant
        ),
    )
    stress.validate()

    manifests: list[dict[str, object]] = []
    for seed_index in seed_indices:
        world = generate_teacher_world(
            config,
            partition=args.partition,
            seed_index=seed_index,
            world_seed=args.world_seed,
            stress=stress,
            freeze_attestation=freeze_attestation,
        )
        world_dir = args.output / world.ground_truth.world_id
        existing = [path for path in _artifact_paths(world_dir) if path.exists()]
        if existing and not args.overwrite:
            names = ", ".join(str(path) for path in existing)
            raise SystemExit(f"Refusing to overwrite existing artifacts: {names}")
        paths = save_teacher_release(
            world,
            world_dir,
            include_latents=not args.omit_latents,
            include_noise=not args.omit_noise,
        )
        manifests.append(
            {
                "world_id": world.ground_truth.world_id,
                "partition": args.partition,
                "seed_index": seed_index,
                "stress": stress.tag,
                "teacher_config_sha256": teacher_config_sha256(config),
                "seed_material_public": True,
                "prospective_seed_secrecy": False,
                "evaluation_role": (
                    "post_freeze_deterministic_procedural_audit"
                    if args.partition == "locked"
                    else "method_development"
                ),
                "eligible_for_biological_headline_conjunction": False,
                "preoutcome_freeze": (
                    None if freeze_attestation is None else asdict(freeze_attestation)
                ),
                "artifacts": {name: str(path) for name, path in paths.items()},
            }
        )
        # An explicit diagnostic seed names exactly one world.
        if args.world_seed is not None:
            break
    print(json.dumps(manifests, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
