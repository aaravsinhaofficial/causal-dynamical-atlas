# CADENCE

**A cross-animal causal dynamical atlas**

CADENCE asks a deliberately narrow, falsifiable question: after seeing only
normal activity from a new animal, can an intervention operator learned in
other animals predict that animal's complete neural and behavioral response?

The model separates a shared nonlinear normal flow \(F\), a state-dependent
low-rank intervention operator \(G\), animal-specific rank-two residual
dynamics, animal/session observation maps, and a shared behavior decoder. The
target animal contributes normal activity only. Its post-onset neural and
behavioral samples are stored separately, predictions are open-loop, and the
prediction bundle is SHA-256 hashed before scoring.

This is a prospective falsification project, not a guaranteed positive result.
The final leakage-safe nested topology already exposes the central risk. Across
ten teacher development smoke worlds, behavior skill was positive in all ten,
but mean neural condition skill was only 0.00173 (8/10 positive; range
-0.117 to 0.0623), mean neural pathwise skill was 0.00878 (7/10 positive), and
mean latent skill was 0.127 (7/10 positive). The coordinate-range diagnostic
is consistent with, but does not establish, off-support extrapolation. Thus
development does not establish reliable operator or observation-map transfer.
Across the four allowed Allen development mice, proposed neural skill averaged
-0.0000863 (1/4 positive) and running skill averaged -0.000491 (2/4 positive),
so neither endpoint established transfer. Those failures are preserved before
any of the 28 Allen evaluation mice are opened.

## Evaluations

- **Allen Visual Behavior Ophys 1.1.0.** A fixed 32-mouse VISp/Slc17a7 cohort:
  four development mice and 28 sequestered evaluation mice in five
  whole-animal folds. The sensory intervention is an eligible expected-image
  omission. Primary outputs are every recorded cell's event-rate trajectory
  and running speed over 0–2 s.
- **DANDI:001868 v0.260715.2016.** Six task mice and 45 source sessions
  containing ephys, ophys, and wheel, evaluated leave-one-animal-out on
  parameterized 700 ms ICMS. Frozen v1 scores sorted spikes and wheel; calcium
  is retained but not evaluated. Five mice are design-eligible for randomized
  catch contrasts; catch-free ICMS83 is absolute-trajectory-only.
- **Teacher RNNs.** Known shared operators, animal residuals, arbitrary
  observation maps, paired counterfactual twins, ten development worlds,
  twenty public-seed post-freeze procedural worlds, and an explicit
  non-identifiability construction. The procedural worlds test recovery under
  known ground truth but are not a blinded biological holdout.

Bulk raw arrays and per-animal processed artifacts are excluded from git.
Frozen manifests, the two processed provenance indexes, exact archive versions,
byte counts, hashes, and reconstruction commands are committed.

## Reproduce

Create the pinned environment and run the complete test suite:

```bash
uv sync --locked --extra dev
uv run pytest -q
uv run ruff check src scripts tests
```

Reconstruct the two biological datasets:

```bash
uv run python -m cadence.data.allen_vbo download \
  --manifest data/manifests/allen_vbo_slc17a7_visp175_familiar_active_v1.1.0.json \
  --destination data/raw/allen_vbo/nwb
uv run python scripts/preprocess_allen_vbo.py --all-normal

uv run python scripts/download_dandi_icms.py --scope all --workers 8 --audit-api
uv run python scripts/preprocess_dandi_icms.py
```

The exact pre-freeze development record is reconstructed with:

```bash
uv run python scripts/run_teacher_experiment.py \
  --partition development --profile smoke --seeds 0 1 2 3 4 5 6 7 8 9 \
  --run-seed 0 --methods proposed linear additive black_box --device cuda:0 \
  --output results/teacher-development-freeze

uv run python scripts/run_allen_experiment.py \
  --stage all --run-profile development --optimization full \
  --development-target 539517 \
  --development-donors 448900 484631 423606 \
  --methods proposed linear additive black_box --device cuda:1 --seed 0 \
  --output results/allen-vbo/development-final-full/mouse_539517
uv run python scripts/run_allen_experiment.py \
  --stage all --run-profile development --optimization full \
  --development-target 448900 \
  --development-donors 539517 484631 423606 \
  --methods proposed linear additive black_box --device cuda:1 --seed 0 \
  --output results/allen-vbo/development-final-full/mouse_448900
uv run python scripts/run_allen_experiment.py \
  --stage all --run-profile development --optimization full \
  --development-target 484631 \
  --development-donors 539517 448900 423606 \
  --methods proposed linear additive black_box --device cuda:1 --seed 0 \
  --output results/allen-vbo/development-final-full/mouse_484631
uv run python scripts/run_allen_experiment.py \
  --stage all --run-profile development --optimization full \
  --development-target 423606 \
  --development-donors 539517 448900 484631 \
  --methods proposed linear additive black_box --device cuda:1 --seed 0 \
  --output results/allen-vbo/development-final-full/mouse_423606

uv run python scripts/build_development_record.py
```

CUDA device selection is execution-only; CPU runs use the same scientific
configuration. The builder authenticates the ten teacher worlds, all four
development mice, their stage chains, and canonical processed-source
commitments before writing the append-only JSON/CSV/figure release.
The committed [machine-readable development record](results/releases/development/development_record.json)
and [diagnostic figure](results/releases/development/development_diagnostics.png)
are explicitly non-confirmatory.

Locked profiles additionally require an explicit acknowledgement and a clean
checkout at the exact annotated tag `pre-outcome-v1.0.0`. The executable
attestation prevents an accidental unsealing run from modified code.
Each biological workflow is split into preparation, prediction, and scoring;
run those as separate processes. For example:

```bash
# Teacher: public-seed deterministic audit, frozen to the post-freeze profile.
uv run python scripts/run_teacher_experiment.py \
  --partition locked --profile full \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --acknowledge-locked --output results/teacher-locked

# Allen: FOLD ranges from 0 through 4.
uv run python scripts/run_allen_experiment.py \
  --stage prepare --run-profile locked --optimization full --fold FOLD \
  --acknowledge-locked --output results/allen-vbo/locked-fold-FOLD
uv run python scripts/run_allen_experiment.py \
  --stage predict --run-profile locked --optimization full --fold FOLD \
  --acknowledge-locked --output results/allen-vbo/locked-fold-FOLD
uv run python scripts/run_allen_experiment.py \
  --stage score --run-profile locked --optimization full --fold FOLD \
  --acknowledge-locked --output results/allen-vbo/locked-fold-FOLD

# ICMS: TARGET is one of ICMS83, ICMS92, ICMS93, ICMS98, ICMS100, ICMS101.
uv run python scripts/run_icms_experiment.py prepare \
  --target TARGET --protocol-commit "$(git rev-parse HEAD)" \
  --optimization full --output results/icms/loao-TARGET
uv run python scripts/run_icms_experiment.py predict \
  --optimization full --acknowledge-donor-outcomes \
  --output results/icms/loao-TARGET
uv run python scripts/run_icms_experiment.py score \
  --acknowledge-target-outcomes --output results/icms/loao-TARGET
```

The literal `FOLD` and `TARGET` placeholders must be replaced. The
paper-scale commands are intentionally not wrapped in a convenience command:
the process boundary is part of the leakage defense. Aggregate completed
animal-level metrics with `scripts/aggregate_results.py`. Locked inputs are
accepted only from the exact one-shot paths shown above. Complete frozen
aggregation requires the preregistered 20,000 resamples and seed, runs through
the clean tagged reporter, and atomically writes a new append-only report with
input hashes, artifact sidecars, and a completion manifest. Failed and
`NOT_EVALUATED` gates are preserved.

Locked preparation writes a durable active-seal journal before changing target
file permissions. If a process is interrupted, uncommitted stage artifacts are
quarantined rather than overwritten or deleted, and re-entry idempotently
re-seals or restores the target. “Append-only” applies to completed stages; an
interrupted attempt is preserved and retried. The active journal/registry is
removed only after the score completion and the post-score restoration
completion plus its SHA-256 sidecar are durable.

## Claim boundary

The primary score is causal skill relative to predicting no effect, evaluated
over the full observed channel-by-time tensor with scales fit on target normal
support. Animals, not trials or sessions, are the biological replicates. The
headline claim passes only if neural and primary-behavior confidence intervals
are above zero, the proposed method clears the strongest eligible comparator
by at least 0.10 for both endpoints, proper-score and calibration gates pass,
and leakage/randomization controls remain null. A supplementary result can
support Gates 5--8 only when its canonical relative filename and observed
digest are bound into the runner's authenticated completion chain; a digest
claimed inside a metrics file is insufficient. Failed and unevaluated gates
are reported verbatim.
Frozen v1 is fail-closed: its biological runners do not emit every artifact
required for Gate 1 or Gates 5--8, so it can return `FAIL` but cannot return a
biological `PASS`. Missing evidence cannot be retrofitted after unsealing.

See the complete [frozen protocol](docs/PROTOCOL.md), [freeze and unsealing
record](docs/FREEZE.md), [dataset ledger](docs/DATASETS.md), and [related-work
boundary](docs/RELATED_WORK.md).
