"""Immutable animal-level splits and leakage audits."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ProtocolViolation(RuntimeError):
    """Raised when a result cannot support the advertised transfer claim."""


@dataclass(frozen=True, slots=True)
class FreezeAttestation:
    """Identity of the clean, tagged commit used to open locked outcomes."""

    commit: str
    tag: str
    tag_object: str


def attest_preoutcome_freeze(
    *,
    required_tag: str = "pre-outcome-v1.0.0",
    repository: str | Path = ".",
) -> FreezeAttestation:
    """Refuse a locked run unless tracked code is at the exact frozen tag.

    Untracked result files are intentionally ignored so several locked folds can
    be run from the same immutable checkout. Any change to a tracked file,
    staged or unstaged, invalidates the attestation.
    """

    root = Path(repository)

    def git(*arguments: str) -> str:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise ProtocolViolation(
                f"cannot attest pre-outcome git freeze: {' '.join(arguments)}"
            ) from error
        return result.stdout.strip()

    commit = git("rev-parse", "--verify", "HEAD")
    if git("cat-file", "-t", required_tag) != "tag":
        raise ProtocolViolation(
            f"locked outcomes require {required_tag} to be an annotated tag object"
        )
    tag_object = git("rev-parse", "--verify", required_tag)
    tagged_commit = git("rev-parse", "--verify", f"{required_tag}^{{commit}}")
    if commit != tagged_commit:
        raise ProtocolViolation(
            f"locked outcomes require HEAD at tag {required_tag}; "
            f"HEAD={commit}, tag={tagged_commit}"
        )
    worktree_status = git("status", "--porcelain")
    if worktree_status:
        raise ProtocolViolation("locked outcomes require a clean worktree")
    tags = git("tag", "--points-at", "HEAD").splitlines()
    if required_tag not in tags:
        raise ProtocolViolation(f"HEAD is not directly tagged {required_tag}")
    return FreezeAttestation(
        commit=commit,
        tag=required_tag,
        tag_object=tag_object,
    )


@dataclass(frozen=True, slots=True)
class TrialRecord:
    animal_id: str
    session_id: str
    trial_id: str
    is_intervention: bool
    intervention_strength: float
    has_event_overlap: bool = False


@dataclass(frozen=True, slots=True)
class AnimalFold:
    fold: int
    train_animals: tuple[str, ...]
    validation_animals: tuple[str, ...]
    test_animals: tuple[str, ...]

    def validate(self) -> None:
        train = set(self.train_animals)
        validation = set(self.validation_animals)
        test = set(self.test_animals)
        if not train or not validation or not test:
            raise ProtocolViolation("train, validation, and test animals must be nonempty")
        if train & validation or train & test or validation & test:
            raise ProtocolViolation("animal identities overlap across fold partitions")


def make_nested_leave_one_animal_out(
    animal_ids: list[str] | tuple[str, ...],
    *,
    validation_offset: int = 1,
) -> list[AnimalFold]:
    """Create deterministic outer-test and whole-animal inner-validation folds."""
    animals = tuple(sorted(set(animal_ids)))
    if len(animals) < 3:
        raise ValueError("nested animal folds require at least three animals")
    folds = []
    for test_index, test in enumerate(animals):
        validation = animals[(test_index + validation_offset) % len(animals)]
        if validation == test:
            validation = animals[(test_index + validation_offset + 1) % len(animals)]
        train = tuple(animal for animal in animals if animal not in {test, validation})
        fold = AnimalFold(test_index, train, (validation,), (test,))
        fold.validate()
        folds.append(fold)
    return folds


@dataclass(frozen=True, slots=True)
class SplitManifest:
    dataset: str
    dataset_version: str
    fold: AnimalFold
    target_support_trials: tuple[str, ...]
    target_validation_trials: tuple[str, ...]
    target_audit_normal_trials: tuple[str, ...]
    sealed_target_intervention_trials: tuple[str, ...]
    source_digests: tuple[tuple[str, str], ...]
    preprocessing_commit: str

    def validate(self, trial_records: list[TrialRecord]) -> None:
        self.fold.validate()
        lookup = {record.trial_id: record for record in trial_records}
        listed = (
            self.target_support_trials
            + self.target_validation_trials
            + self.target_audit_normal_trials
            + self.sealed_target_intervention_trials
        )
        if len(listed) != len(set(listed)):
            raise ProtocolViolation("a trial appears in multiple target partitions")
        missing = set(listed) - lookup.keys()
        if missing:
            raise ProtocolViolation(f"manifest references missing trials: {sorted(missing)[:3]}")
        test = set(self.fold.test_animals)
        for trial_id in (
            self.target_support_trials
            + self.target_validation_trials
            + self.target_audit_normal_trials
        ):
            record = lookup[trial_id]
            if record.animal_id not in test:
                raise ProtocolViolation("target support contains a non-test animal")
            if record.is_intervention or record.intervention_strength != 0:
                raise ProtocolViolation("target adaptation contains an intervention trial")
            if record.has_event_overlap:
                raise ProtocolViolation("target adaptation trial overlaps an intervention event")
        for trial_id in self.sealed_target_intervention_trials:
            record = lookup[trial_id]
            if record.animal_id not in test or not record.is_intervention:
                raise ProtocolViolation("sealed query set is not target-animal intervention data")

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


def write_manifest(path: Path, manifest: SplitManifest) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(manifest.canonical_json())
    payload["manifest_sha256"] = manifest.digest()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload["manifest_sha256"]


def mask_post_onset(
    values: ArrayLike,
    onset: int,
    *,
    sentinel: float = np.nan,
) -> NDArray[Any]:
    """Return a copy in which query outcomes cannot be consumed as inputs."""
    array = np.asarray(values).copy()
    if array.ndim < 2 or not 1 <= onset < array.shape[1]:
        raise ValueError("expected [trial, time, ...] and a valid onset")
    array[:, onset:] = sentinel
    return array


def assert_query_is_sealed(
    neural_inputs: ArrayLike,
    behavior_inputs: ArrayLike,
    onset: int,
) -> None:
    neural = np.asarray(neural_inputs)
    behavior = np.asarray(behavior_inputs)
    if not np.all(np.isnan(neural[:, onset:])):
        raise ProtocolViolation("post-onset neural target is mounted in the inference input")
    if not np.all(np.isnan(behavior[:, onset:])):
        raise ProtocolViolation("post-onset behavior target is mounted in the inference input")


def audit_preprocessing_fit_animals(
    fit_animal_ids: ArrayLike,
    fold: AnimalFold,
    *,
    target_normal_only: bool,
) -> None:
    """Reject global transforms fitted on held-out intervention outcomes."""
    fitted = set(np.asarray(fit_animal_ids).astype(str).tolist())
    test = set(fold.test_animals)
    if fitted & test and not target_normal_only:
        raise ProtocolViolation("preprocessing fit includes unrestricted target-animal data")
    allowed = set(fold.train_animals) | set(fold.validation_animals) | test
    if not fitted <= allowed:
        raise ProtocolViolation("preprocessing references animals absent from the fold")
