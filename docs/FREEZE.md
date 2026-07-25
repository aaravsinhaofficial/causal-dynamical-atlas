# Pre-outcome freeze and unsealing record

CADENCE uses an executable git boundary, not a prose-only promise. The
public-seed teacher procedural partition and both biological evaluation
profiles refuse to run unless the worktree is clean (apart from ignored result
artifacts) and `HEAD` is the
exact commit referenced by the annotated tag object
`pre-outcome-v1.0.0`. The check is implemented by
`cadence.protocol.attest_preoutcome_freeze` and records both the commit and tag
object identifiers. The teacher seeds are public and are not described as
sequestered; the tag boundary there freezes analysis choices rather than
concealing simulator outcomes.

The intended sequence is:

1. inspect only teacher development worlds, the four Allen development mice,
   and signal-blind ICMS metadata/normal support;
2. finalize the estimands, code, hyperparameters, baselines, and leakage tests;
3. commit and create the annotated `pre-outcome-v1.0.0` tag;
4. publish the annotated tag, then run the teacher procedural profile and each
   biological locked profile from that tagged checkout exactly once;
5. preserve every prediction-bundle SHA-256 digest before scoring;
6. commit immutable summary metrics, the conjunction table, and this record
   without changing the frozen analysis code.

Ignored result artifacts may accumulate between folds, but any nonignored
untracked file or staged/unstaged change invalidates the attestation. Locked stages
are append-only and may not overwrite a completed preparation, prediction, or
score. The reporter accepts locked results only from the canonical nonsymlink
paths in the README, rejects noncanonical resampling parameters, and itself
requires clean `HEAD` at the annotated tag for every complete frozen cohort.
It publishes each report as a new atomic, append-only directory with
authenticated input hashes, artifact SHA-256 sidecars, and a completion
manifest. Each run's authenticated manifest/completion chain records the frozen
commit, annotated tag object, canonical configuration digest, and applicable
committed processed-input index. The
eventual outcome commit and public tag are recorded here after the unsealing
run.

Before any target permission is changed, locked preparation durably publishes
an active-seal journal. Re-entry after a crash idempotently re-seals or restores
the target and preserves interrupted, uncommitted stage artifacts in a
quarantine directory rather than overwriting or deleting them. Append-only
therefore applies to completed stages; interrupted attempts are preserved and
retried. The active journal/registry is removed only after both the score
completion and the authenticated post-score restoration completion are
durable.

## Development evidence available before freeze

- Teacher development smoke, seeds 0--9, after removing validation-animal
  normal leakage: proposed condition-averaged neural skill has mean 0.00173,
  median 0.0131, range [-0.1175, 0.0623], and is positive in 8/10 worlds.
  Proposed neural pathwise skill has mean 0.00878 and is positive in 7/10.
  Behavior skill is positive in 10/10 (mean 0.5049), while latent-effect skill
  is positive in only 7/10 (mean 0.1270). The evaluation-only
  gauge-plus-true-observation diagnostic has mean neural skill 0.0278, and
  9.79% of readout query coordinates lie outside normal-rollout coordinate
  ranges on average. This mixed, near-zero neural result does not establish
  reliable latent, operator, or neural-map transfer; it exposes estimation
  failures and a measurable off-range burden that is consistent
  with, but does not establish, off-support extrapolation.
- Allen development full four-mouse LOAO: proposed neural causal skill has
  mean -0.0000863, range [-0.0001846, 0.0000002], and is positive in 1/4 mice.
  Proposed running causal skill has mean -0.0004914, range
  [-0.0016852, 0.0001936], and is positive in 2/4. Neither endpoint establishes
  transfer; the result is effectively at or below the zero-effect reference.
- DANDI:001868: metadata, normal calibration windows, and signal-blind
  preprocessing audits were inspected. Before the freeze, a schema-regression
  test also loaded one local ICMS83 example container to verify its exceptional
  channel wiring, array shapes, catch count, and normal-ITI validity. It did
  not print, score, summarize, or tune against an ICMS response trajectory.
  ICMS83 is catch-free and absolute-trajectory-only; none of the five
  randomized-effect mice was opened. The real-container test is now explicit
  opt-in and runs only when `CADENCE_RUN_REAL_ICMS_AUDIT=1` and
  `CADENCE_REAL_ICMS_EXAMPLE_DIR` identifies an existing directory containing
  `sub-ICMS83.nwb`; otherwise it is skipped. No biological stimulation
  response was used for model or protocol selection.

These failures are retained because the frozen test is allowed to falsify the
headline hypothesis. Biological evaluation outcomes cannot be used to revise
the method. Version 1 is explicitly fail-closed: it can establish failure of
the core predictive gates, but missing positive-claim uncertainty and
falsification artifacts remain `NOT_EVALUATED` and prevent a headline pass.

## Frozen-run recovery records

Pre-retry decisions and fail-closed state audits for numerical projection
exceptions are recorded in
[`ICMS92_RECOVERY.md`](ICMS92_RECOVERY.md) and
[`ALLEN_FOLD4_RECOVERY.md`](ALLEN_FOLD4_RECOVERY.md). Each record fixes a
ceiling of one identical-device retry before that retry is launched; they
forbid retry-until-pass, device hopping, outcome-dependent changes, and
untagged numerical amendments.
