from __future__ import annotations

import subprocess

import numpy as np
import pytest

from cadence.protocol import (
    ProtocolViolation,
    SplitManifest,
    TrialRecord,
    assert_query_is_sealed,
    attest_preoutcome_freeze,
    make_nested_leave_one_animal_out,
    mask_post_onset,
)


def test_nested_folds_are_disjoint_and_cover_each_test_once() -> None:
    folds = make_nested_leave_one_animal_out(["m3", "m1", "m2", "m4"])
    assert [fold.test_animals[0] for fold in folds] == ["m1", "m2", "m3", "m4"]
    for fold in folds:
        fold.validate()
        assert len(fold.train_animals) == 2


def manifest() -> SplitManifest:
    fold = make_nested_leave_one_animal_out(["m1", "m2", "m3"])[0]
    return SplitManifest(
        "d",
        "v",
        fold,
        ("normal-fit",),
        ("normal-val",),
        ("normal-audit",),
        ("stim",),
        (("source.nwb", "abc"),),
        "commit",
    )


def records() -> list[TrialRecord]:
    return [
        TrialRecord("m1", "s", "normal-fit", False, 0),
        TrialRecord("m1", "s", "normal-val", False, 0),
        TrialRecord("m1", "s", "normal-audit", False, 0),
        TrialRecord("m1", "s", "stim", True, 5),
    ]


def test_manifest_allows_only_normal_target_adaptation() -> None:
    manifest().validate(records())
    contaminated = records()
    contaminated[0] = TrialRecord("m1", "s", "normal-fit", True, 1)
    with pytest.raises(ProtocolViolation, match="adaptation contains"):
        manifest().validate(contaminated)


def test_event_overlap_is_not_a_catch() -> None:
    contaminated = records()
    contaminated[0] = TrialRecord("m1", "s", "normal-fit", False, 0, True)
    with pytest.raises(ProtocolViolation, match="overlaps"):
        manifest().validate(contaminated)


def test_sealed_query_sentinel() -> None:
    array = np.ones((2, 6, 3))
    sealed = mask_post_onset(array, 2)
    assert_query_is_sealed(sealed, sealed[:, :, :1], 2)
    sealed[0, 4, 0] = 7
    with pytest.raises(ProtocolViolation, match="neural target"):
        assert_query_is_sealed(sealed, np.full((2, 6, 1), np.nan), 2)


def test_preoutcome_freeze_requires_exact_clean_tag(tmp_path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("frozen\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("results/\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt", ".gitignore"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "freeze"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "tag", "-a", "pre-outcome-v1.0.0", "-m", "pre-outcome freeze"],
        cwd=tmp_path,
        check=True,
    )

    attestation = attest_preoutcome_freeze(repository=tmp_path)
    assert attestation.tag == "pre-outcome-v1.0.0"
    assert len(attestation.commit) == 40
    assert len(attestation.tag_object) == 40

    # Ignored, append-only result artifacts do not invalidate later folds.
    result = tmp_path / "results" / "fold-0.json"
    result.parent.mkdir()
    result.write_text("{}\n", encoding="utf-8")
    attest_preoutcome_freeze(repository=tmp_path)

    (tmp_path / "untracked.py").write_text("raise RuntimeError\n", encoding="utf-8")
    with pytest.raises(ProtocolViolation, match="clean worktree"):
        attest_preoutcome_freeze(repository=tmp_path)
    (tmp_path / "untracked.py").unlink()

    tracked.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ProtocolViolation, match="clean worktree"):
        attest_preoutcome_freeze(repository=tmp_path)


def test_preoutcome_freeze_rejects_lightweight_tag(tmp_path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "tracked.txt").write_text("frozen\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "freeze"], cwd=tmp_path, check=True)
    subprocess.run(["git", "tag", "pre-outcome-v1.0.0"], cwd=tmp_path, check=True)

    with pytest.raises(ProtocolViolation, match="annotated tag object"):
        attest_preoutcome_freeze(repository=tmp_path)
