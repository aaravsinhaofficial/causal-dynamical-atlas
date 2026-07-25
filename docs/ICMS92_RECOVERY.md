# ICMS92 frozen-v1 recovery decision

Decision recorded before any recovery attempt at 2026-07-25T16:00:59Z.

## Incident

The failure log from the first `predict` attempt for the frozen ICMS92 fold was
last written at 2026-07-25T15:50:08.768426748Z and ends with:

```text
cadence.protocol.ProtocolViolation:
training donor deltas failed exact post-step projection
```

The exception arose during donor intervention-model fitting, before a
prediction bundle or scoring artifact was published. No ICMS92 target response
outcome was opened or summarized in making this decision.

The failure log is preserved at
[`evidence/icms92-predict-attempt-1-failure.log`](evidence/icms92-predict-attempt-1-failure.log);
it is byte-identical to
`results/run-logs/icms92-predict-attempt-1-failure.log` in the tagged execution
worktree:

```text
size:     1082 bytes
sha256:   aed22efaf7814c4c57c7cb36ffa6504d22ebb8d9e6d7eda3a7fdacebfe6eed14
```

The run attested the already-published frozen boundary:

```text
tag:         pre-outcome-v1.0.0
commit:      25465767879c36aaa5b8a7a24ddca3448d191d97
tag object:  119854cec77d72ae0e90ea9b7fbbff94e3652af2
config:      0c6e8b46f16e9f2da4ad442f042e325586bed490b93a56a4d63274fd78875ff8
prepare:     0558195bf0d121dd89902df2eee6162114fe4813c9cfe339fdc5e6b7c6e363ff
completion:  11b262a5049c8f590d48daa2ba74638a0a541f444fb590c5561d756c9fcf8474
```

## Fail-closed state audit

At the pre-retry audit at 2026-07-25T16:00:59Z:

- the tagged execution checkout was clean and remained at the exact frozen
  commit;
- the fold contained only the completed preparation, its authenticated
  sidecars, and the physical target seal—no model, prediction, score,
  restoration, or temporary stage artifact had been published;
- `sub-ICMS92.h5` remained sealed at mode `000`, device `66305`, inode
  `933846027`, and size `82907570` bytes;
- the fold seal and active registry were byte-identical, each with SHA-256
  `82ec9a0c8aef0254c2adc9725fd5b269dd86f176e1c6df5924404204f83c4e12`;
- that digest matched `target_seal_transaction_sha256` in the authenticated
  preparation manifest; and
- the seal retained the expected target-container SHA-256
  `d203fd7c7f22cc2b133fd67473a176ab30dab67f79139b9f5fb9fd149ca6fdf6`.

The exception does not report the rejected residual. Source inspection shows
that the guard applies a one-pass float32 centering projection and rejects a
recomputed norm above the frozen absolute threshold of `1e-7`. This makes
rounding at the threshold a plausible explanation, but the record does not
claim that explanation as proven.

## Precommitted recovery rule

Version 1 permits exactly one recovery retry with the same tagged source,
prepared fold, configuration, seed, method set and order, configured logical
`cuda:1` device, and target seal, with no device remapping or CPU fallback. The
retry will begin only after the Allen evaluation has released GPU 1. It will
use the frozen re-entry path, which validates and re-seals the target and
preserves any interrupted artifacts.

The recovery command is:

```bash
uv run python scripts/run_icms_experiment.py predict \
  --output results/icms/loao-ICMS92 \
  --optimization full \
  --device cuda:1 \
  --seed 20260725 \
  --methods proposed linear additive black_box zero_effect condition_time nearest_donor \
  --acknowledge-donor-outcomes
```

No preparation rerun, manual permission change, file deletion, `--overwrite`,
device remapping, CPU fallback, tolerance change, source change, seed change,
or method change is allowed for this retry.

If prediction succeeds and authenticates, the target will be scored once using
the frozen scoring command. Any failure of this one retry ends the ICMS92
frozen-v1 fold; there will be no third attempt. If the same projection guard
recurs, the fold receives operational status `PROTOCOL_FAILED` and the
incomplete randomized ICMS-v1 exploratory cohort receives the inferential
disposition `NOT_EVALUATED`, not a scientific `FAIL`; the ICMS83 absolute-only
analysis remains separately descriptive. A different terminal error will be
preserved and reported without outcome-dependent recovery. There will be no
retry-until-pass or device-hopping. Any numerical amendment must receive a new
public protocol tag and rerun all six ICMS folds; amended results may not be
represented as results of `pre-outcome-v1.0.0`.

## Post-retry record

Pending. The retry log, exact command, terminal status, artifact hashes, and
target restoration state will be appended here without changing the
precommitted rule above.
