# Allen fold-4 frozen-v1 recovery decision

Decision recorded before any recovery attempt at 2026-07-25T16:09:03Z.

## Incident

The failure suffix of the first `predict` attempt for locked Allen fold 4 was
last written at 2026-07-25T15:59:22.967009636Z and ends with:

```text
cadence.protocol.ProtocolViolation:
donor-delta projection failed after optimizer step
```

The exception arose during intervention selection, before target prediction
publication or any access to the five mice's sealed omission outcomes. No
fold-4 target response outcome was opened or summarized in making this
decision.

The 15-line failure suffix is preserved at
[`evidence/allen-fold4-predict-attempt-1-failure.log`](evidence/allen-fold4-predict-attempt-1-failure.log);
it is byte-identical to lines 535--549 of
`results/run-logs/allen-locked.log` in the tagged execution worktree:

```text
size:     1012 bytes
sha256:   833f240e2cbccc3cd744b25f3ddf09f463ab226714263106470f46f7db9eb239
```

The run attested the already-published frozen boundary and prepared fold:

```text
tag:                    pre-outcome-v1.0.0
commit:                 25465767879c36aaa5b8a7a24ddca3448d191d97
tag object:             119854cec77d72ae0e90ea9b7fbbff94e3652af2
configuration:          58c40f05375dbd892bf3bdb365e12191e036ac6e24fc3bc96eac65afcb2f23dd
canonical optimization: 87ed93e9d5439b4f826b7f83545f3f21f7f969e68479bca3b62585113c626819
runtime optimization:   6c9351be1613805a2cec44b63dff060192102035eb36db7a95181b1f3ff51238
preparation:            627fc5c62cc336fce0bf31dd27e7e070877ca4edcb6a7f0adf5d76113e04af2e
completion:             668b87f104f01589412b415ecced03eadf1e5c3622725c367918a79625974011
seal transaction:       f336b2efdfb358976a032ebcd3a942f346116cf132e12a1db786c378a03fb462
```

## Fail-closed state audit

At the pre-retry audit at 2026-07-25T16:07:45Z:

- the tagged execution checkout was clean and remained at the exact frozen
  commit;
- all six declared safe preparation artifacts authenticated against the
  preparation completion;
- the fold contained no model, checkpoint, prediction, prediction completion,
  score, restoration, quarantine, or temporary stage artifact;
- the mode-`0600` active registry authenticated against the preparation and
  contained all 15 expected entries for mice 453913, 456915, 477052, 533162,
  and 548950;
- every entry retained its recorded regular-file, nonsymlink device and inode
  identity;
- the ten processed source files retained sealed mode `0200`, the five
  experiment outcome copies retained sealed mode `0000`, and none was
  readable; and
- no Allen process remained alive.

Target-outcome bytes were deliberately not opened or rehashed during the
audit. Their identities, sealed permissions, and authenticated pre-seal
commitments were checked.

The source guard stacks the selection training groups' float32 donor-delta
tensors, subtracts their componentwise mean, recomputes that mean, and requires
its L2 norm to be at most the frozen absolute tolerance `1e-7` after every
optimizer step. The exception records neither the residual, method, nor step.
Float32 rounding or cancellation in the subtract-then-reduce operation against
the strict absolute threshold is a plausible explanation. The projection
itself runs outside autocast, although mixed precision can affect the training
trajectory. Instability cannot be excluded, and this record does not claim
either explanation as proven.

## Precommitted recovery rule

Version 1 permits exactly one outcome-blind recovery retry with the same tagged
source, prepared fold, configuration, seed, method set and order, configured
logical `cuda:1` device, and target-seal transaction, with no device remapping
or CPU fallback. The retry will begin only after the active ICMS92 recovery has
fully released GPU 1. It will use the frozen re-entry path, which authenticates
and re-seals all targets and preserves any interrupted artifacts.

From the root of the exact tagged execution worktree, the recovery command is:

```bash
uv run python scripts/run_allen_experiment.py \
  --stage predict \
  --run-profile locked \
  --optimization full \
  --fold 4 \
  --acknowledge-locked \
  --methods proposed linear additive black_box \
  --device cuda:1 \
  --seed 0 \
  --output results/allen-vbo/locked-fold-4
```

No preparation rerun, manual permission change, file deletion, `--overwrite`,
device remapping, CPU fallback, tolerance change, source change, seed change,
method change, or concurrent GPU-1 workload is allowed for this retry.
The frozen runner seeds its random-number generators but does not enforce
deterministic CUDA algorithms, so an identical retry can follow a different
numerical trajectory. That is why the retry count is fixed before launch.

If prediction succeeds and its completion authenticates, fold 4 will be scored
once using the frozen scoring command. Any failure of this one retry ends the
fold-4 frozen-v1 run; there will be no third attempt. A repeated projection
exception receives operational status `PROTOCOL_FAILED`, while the incomplete
primary Allen cohort and headline conjunction receive inferential disposition
`NOT_EVALUATED`, not a scientific `FAIL`. A different terminal error will be
preserved and reported without outcome-dependent recovery.

Any numerical or execution amendment must receive a new public protocol tag
and rerun all five Allen folds. Amended results may not be represented as
results of `pre-outcome-v1.0.0`.

## Post-retry record

Pending. The retry log, exact command, terminal status, artifact hashes, and
target restoration state will be appended here without changing the
precommitted rule above.
