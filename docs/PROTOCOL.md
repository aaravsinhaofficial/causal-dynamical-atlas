# CADENCE frozen evaluation protocol

Protocol version: `1.0.0`

Freeze date: 2026-07-25

Status: outcomes sealed; no biological performance claim is made in this file.

## Question and claim boundary

CADENCE tests whether an intervention-response operator learned in donor animals
can be transported to a new animal after calibrating that animal on normal
activity only. The primary estimand is not decoding, manifold similarity, or
classification. It is the complete post-onset trajectory caused by withholding
or delivering an experimentally randomized input:

\[
\tau_i(t)=\mathbb{E}\!\left[Y_i(t; a)-Y_i(t;0)\mid i\right],\qquad t\geq0,
\]

where \(i\) is an animal, \(Y_i\) contains every prespecified neural channel and
primary behavioral endpoint. The predeclared descriptor \(a\) is supplied to
the frozen model at inference; no target-animal response under \(a\) is shown
during fitting, selection, adaptation, or prediction. Running speed is the
primary Allen behavioral endpoint; pupil area and licking are secondary and
cannot rescue a failed primary gate. This hierarchy was selected on the four
declared development mice before any evaluation-mouse outcome was opened.

Two biological evaluations have different evidential roles.

1. **Primary evaluation — Allen image omissions.** A fixed cohort contains 32
   mice: four method-development mice and 28 sequestered evaluation mice.
   Evaluation mice are scored by five-fold outer cross-fitting. Eligible
   omitted visual flashes are the intervention. Neural event-rate, running,
   pupil, and licking trajectories are observed together. Passing every
   prespecified gate would establish transport of a sensory-input perturbation
   in a single species, task, cortical area, cell class, and experience level.
2. **Direct-intervention evaluation — DANDI:001868 ICMS.** Six task mice are
   evaluated leave-one-animal-out. Intracortical microstimulation is an
   electrical intervention and catch trials are randomly interleaved in five
   mice. Because there are only six animals and one mouse has no catch trials,
   this analysis is explicitly exploratory regardless of how favorable its
   evaluable predictive metrics are. The source is a bioRxiv v1 manuscript accepted at *Science
   Advances* according to the authors' July 15, 2026 Zenodo record; no
   version-of-record DOI was located as of the freeze date.

The phrase *unseen intervention* means seen in donor animals but not used at any
stage for the target animal. It does not mean a globally novel intervention
class. Conclusions are restricted to the measured species, task, modality, and
intervention family.

## Information boundary

For every outer-test animal:

- fitting may use its normal trials and their full neural and behavioral
  trajectories;
- prediction may use scheduled task inputs and predeclared intervention
  descriptors. Allen and teacher use each query's last pre-onset neural and
  behavioral observation; ICMS uses the normal-audit context ensemble described
  below rather than intervention-trial-specific initialization;
- fitting, early stopping, preprocessing, cell selection, hyperparameter
  choice, calibration, and uncertainty calibration may not use any target
  intervention outcome;
- no post-onset neural value, behavioral value, animal-generated action/choice,
  reward, hit/miss label, or response time is mounted in the inference input;
- all predictions are serialized and SHA-256 hashed before the query outcomes
  are opened for scoring.

Per-animal centering, scaling, reliability filtering, and observation-map
estimation are fit on normal support only. There is no global PCA or responsive
cell selection using target intervention trials. Missing values remain
explicitly masked.

For Allen, a separate data-steward `prepare` process materializes the target
query-input and sealed-outcome files. The later `predict` process has been
tested with the target outcome files made unreadable and does not hash or open
them. For ICMS, `prepare` reads normal rows only and constructs a
target-independent physical query lattice; target stimulation rows are first
materialized by `score`. In both cases the scoring process verifies the
prediction bytes against their sidecar digest before it opens the target
outcome artifact.

## Fixed cohorts and splits

### Allen Visual Behavior Ophys

The cohort is release `1.1.0`, VISp, Slc17a7-IRES2-Cre, 175 µm, familiar active
sessions, one deterministically selected experiment per mouse, and at least 40
cells. The frozen manifest contains 32 distinct mice.

Mice are ordered by SHA-256 of `mouse_id || 20260725`. The four lowest hashes
(`539517`, `448900`, `484631`, and `423606`) are the only mice whose omission
outcomes may be used for implementation, model choice, baseline choice, or
uncertainty calibration. They never enter the headline estimate. The remaining
28 mice are assigned round-robin to five balanced outer folds. Each evaluation
mouse is a test mouse exactly once; all non-test mice are eligible donors only
after code and protocol freeze. The locked profile requires an explicit
`--acknowledge-locked` switch.

Within a target mouse, 160 clean normal windows are selected by a deterministic
SHA-256 ordering of presentation identifiers and divided 70/15/15% into adapter
fit, adapter early-stop, and untouched normal audit partitions. Other eligible
normal windows are never used for fitting; they form a separate effect-control
pool. All clean omission windows remain sealed queries.

A normal window is an active non-change, non-sham, non-omitted presentation
whose full \([-1,2]\) s window is in signal support and whose anchor is more
than 3 s from any omission, change, or sham event. An omission query is rejected
if another change, sham, or omission contaminates its analysis window.
Omissions occurred on approximately 5% of eligible presentations; change and
immediately pre-change presentations were not eligible. Controls are restricted
to that same eligible risk set and common signal support.

### DANDI:001868

The cohort is the immutable published version `0.260715.2016`. The task mice
are ICMS83, ICMS92, ICMS93, ICMS98, ICMS100, and ICMS101. Each outer fold has
one test mouse and one rotating whole-mouse validation animal; the remaining
four are training donors. All 45 task sessions containing behavior, ephys, and
ophys together are retained as the source cohort. Frozen v1 models and scores
use sorted spikes and wheel only; calcium remains stored but unevaluated. The
ten task-mouse assets lacking one source modality are not silently included.

Target calibration uses two normal-only sources. Catch trials must have
`current_uA == 0` and no electrical-stimulation interval overlapping the
complete \([-1,3]\) s window. Stimulation-free ITI windows are selected from
the complement of the union of every trial and stimulation interval after a
2.0 s guard on both sides, require the complete 4 s window, and are chosen at
evenly spaced ordinal positions without reading signal values. This
prespecified source is needed because ICMS83 has no catch trials in the
published task sessions.
Calibration is split 60/20/20 within session by a deterministic, signal-blind
hash of its normal-window identifier. Every positive-current outcome is sealed. Analyses use physical
electrode coordinates and pulse physics; the apparatus-specific raw channel
identifier is audit metadata and is never a cross-animal feature.
The source-authored `is_good_trial` flag is an additional prespecified
signal-support restriction applied before neural or wheel signals are read.
The released NWB column description defines `true` as matching two-photon
coverage of the stimulation window, `false` as no overlapping imaging frames,
and defaults to `true` when no source MAT key exists. It is not a hit/response
filter and is not used for tuning; the ICMS estimand is conditional on this
released cross-modal-coverage flag.

All six mice receive held-out absolute-trajectory predictions. ICMS83 has no
catch trials, so it is ineligible for the randomized causal-effect estimand:
its ITI-referenced effect is reported only as a nonrandomized sensitivity and
cannot contribute to the causal-skill gate. The primary ICMS causal-effect
summary therefore has five design-eligible mice and says `n=5` explicitly.
Every prespecified session/condition must still have validated same-block catch
support; otherwise the headline cohort is incomplete and `NOT_EVALUATED`.

The primary ICMS intervention family is restricted, without reading responses,
to the canonical 100 Hz, 70-pulse, 167 µs train whose recorded duration is
approximately 0.7 s. The release also contains 1,906 positive-current rows
whose recorded pulse count/duration is longer; these are outside the frozen
query lattice and cannot be silently mapped onto a 70-pulse prediction.
Frozen v1 labels these noncanonical rows `OUT_OF_SCOPE_NOT_SCORED`; an analysis
of them is `NOT_EVALUATED`, not a duration-generalization sensitivity.

The primary ICMS neural signal is curated sorted-unit activity. The prespecified
artifact-robust endpoint excludes the 2 ms pre-train through 5 ms post-train
interval. Frozen v1 files do not retain a distinct unmasked full-train spike
tensor, so a full-train spike sensitivity is `NOT_EVALUATED`. Volumetric
calcium is not treated as a 30 Hz whole-population trajectory: individual ROIs
are sampled in sparse z-stack bursts. It remains in processed files for a
future native-mask analysis and is `NOT_EVALUATED` in frozen v1.

## Model and fitting stages

For recording adapter \(i\) (an animal for teacher/Allen and an
animal--session pair for ICMS), CADENCE uses

\[
z_{t+1}=F_\theta(z_t,u_t)+G_\phi(z_t)a_t+R_i(z_t)+\delta_{g(i)}(a_t),\qquad
y_t^{(i)}=H_i(z_t).
\]

`F` is shared normal dynamics, `G` is a shared state-dependent low-rank
intervention operator, and `R_i` is a rank-two normal-dynamics residual. Each
recording adapter has an encoder, neural map, and affine behavior calibration:
animal-specific for teacher/Allen and session-specific but nested within animal
for ICMS. Donor intervention residuals are grouped by animal,
zero-centered and shrinkage-regularized; the target intervention residual is
never fitted. The point prediction uses the donor-distribution mean of zero,
while uncertainty draws integrate over the fitted donor distribution.
Training-donor deltas are projected to exact zero mean after every optimizer
step, including the final update.
For the teacher benchmark's primary neural endpoint, each learned method uses
a softplus-Poisson linear quasi-likelihood readout fit on frozen open-loop
target `normal_fit` rollouts. Adam uses the frozen readout weight decay of
0.2, and the additional explicit ridge coefficient is selected from the
frozen grid on `normal_val` only. The model's native neural decoder is reported
separately as a diagnostic.

Optimization is stage-locked:

1. fit shared normal dynamics and donor observation maps using normal donor
   trials;
2. freeze those parameters and fit the shared intervention operator plus donor
   random effects using donor interventions;
3. add each target observation map and fit it using only that target's normal
   support, with normal-only early stopping;
4. freeze every parameter, create open-loop target trajectories, and hash the
   prediction bundle;
5. mount sealed outcomes once for scoring.

Locked preparation first writes and fsyncs an active-seal journal, then changes
target permissions. An interrupted uncommitted attempt is quarantined, never
overwritten or deleted, and re-entry idempotently re-seals or restores the
target. Completed stages remain append-only. The journal/registry is removed
only after score completion and a separately hashed post-score restoration
completion are both durable.

Within each outer fold, candidate shared normal dynamics are fit only on
intervention-training donors. The validation animal receives normal-only
adapters with shared dynamics frozen and selects intervention epochs. A fresh
model then refits normal and intervention stages on all outer donors for those
fixed epoch counts before target adaptation.

Allen and teacher query rollouts encode only the final pre-onset sample. ICMS
is a marginal condition-response exception: each session's curve is averaged
over 16 signal-blind normal-audit contexts on the full physical lattice before
the target stimulation table opens; scoring later interpolates to realized
descriptors. It does not test trial-specific initialization. Future neural and
behavioral samples are never teacher-forced. Hyperparameters and architecture
are selected on teacher development worlds and whole-animal biological
validation folds, never outer-test outcomes. The teacher partition
historically named `locked` uses public, predeclared seeds and is materialized
only after the architecture and command line are committed. It is a
deterministic post-freeze procedural audit, not a prospectively secret or
blinded outcome set, and it is ineligible for a biological headline claim.

## Outcomes and estimators

The primary trajectory is \(0\)–\(2\) s for Allen and \(0\)–\(3\) s for ICMS.
Every time bin and every prespecified channel receives equal weight after
division by a scale estimated from that animal's normal support. Results are
computed within animal first and animals are weighted equally.
The scale is the sample standard deviation on target normal-fit support,
clipped to the within-animal 10th--90th percentiles of positive finite channel
standard deviations. A zero-effect-energy stratum has undefined skill; it is
retained in the stratum ledger but excluded from the finite-stratum mean.

The primary score is causal skill relative to predicting no effect:

\[
\mathrm{CS}=1-\frac{\sum(\widehat{\tau}-\tau)^2/s^2}
                         {\sum\tau^2/s^2}.
\]

Thus perfect prediction is 1, the no-effect model is 0, and a negative value is
worse than no-effect. Frozen v1 reports trajectory NRMSE and selected
time-resolved observed-space summaries. It does not emit response-summary or
split-half-reliability-ceiling artifacts. Proper energy scores and calibrated
bands are required for a positive headline claim; a missing field is explicitly
`NOT_EVALUATED`, never silently imputed as a pass. Absolute-trajectory scores are secondary because
shared baseline activity can make them deceptively easy.

For Allen, the primary quantity is an expected-effect score rather than the
pooled trial tensor in the displayed equation. Omission and matched-control
effects are first averaged within each
`preceding_image × flashes_since_change_bin × fallback_level` stratum. Causal
skill is computed on each resulting time-by-cell (or time-by-behavior)
expected-effect trajectory, and the finite stratum skills are then averaged
with equal stratum weight. Every stratum score and its support are retained.
The single pooled trial-level ratio is reported only as a secondary
sensitivity. This exception prevents frequent conditions from silently
dominating the frozen Allen estimand.

For Allen, the eligible-risk-set omission effect is estimated within mouse by
contrasting omission trajectories with the separate normal effect-control
pool. The expected image at an omission is the immediately preceding repeated
image. Matching uses this frozen hierarchy: exact expected image and
flashes-since-change; expected image and a prespecified flashes-since-change
bin (`1`, `2`, `3–4`, `5–8`, `9+`); expected image only; the same
flashes-since-change bin; then the complete eligible control pool. The fallback
level and effective control count are reported for every mouse.
For ICMS, effects are estimated within session and trial block by contrasting
each current/site condition with its randomly interleaved catch trials, then
aggregated equally across sessions. This randomized contrast is available for
five mice; the catch-free ICMS83 sensitivity is never pooled into it.

The positive-claim uncertainty specification combines equal-animal bootstrap
inference, donor-bootstrap model ensembles, target-normal-support bootstraps,
donor intervention random effects, and split-conformal simultaneous trajectory
bands calibrated on whole-animal cases disjoint from fitting, early stopping,
and model selection (or by a predeclared cross-fitted equivalent). The
whole-trajectory coverage lower bound is an exact one-sided 95% binomial bound
across independent animals, so even 5/5 ICMS animals cannot certify a lower
bound of 0.80. The current ICMS runner emits donor-random-effect draws, but
current producer schemas do not bind canonical supplementary-artifact
filenames and observed digests into their authenticated completion chains.
Coverage and proper-score fields therefore remain diagnostic: no external or
top-level evidence can enable a `PASS` for Gates 5--8. A dataset can still be
conclusively `FAIL` after an earlier gate while those later, unnecessary
positive-claim gates remain `NOT_EVALUATED`; it can never be declared `PASS`
that way.

## Locked comparisons

Dataset-specific matrices give comparisons the same allowed target-normal
support and descriptors. Learned models use the same encoded initialization;
fixed templates use the frozen matched-normal control. All matrices contain
CADENCE, linear and additive hierarchical models, and a black-box meta-GRU with
the same latent/hidden dimensions. No effect, equal-animal condition-time, and
nearest donor are included where defined. Allen additionally includes a
cell-functional ridge atlas and CADENCE without residual dynamics or target
adaptation. Teacher has separately listed native-decoder and
adaptation/residual diagnostics. Frozen ICMS v1 emits no CADENCE ablations.
Oracles are evaluation-only and excluded from the envelope.

Optimization budgets and early-stopping data are reported.
For the gain gate, the reporting layer constructs a deliberately conservative
post-outcome envelope: within each target and endpoint it takes the maximum
score over every available eligible non-oracle comparator, including
ablations. This is not validation-time method selection and does not describe a
deployable single baseline; it asks whether CADENCE clears all of them on each
target.

## Falsification and leakage tests

Gate 6 jointly consumes four exact paired animal-level sign-flip tests
(neural and behavior skill above zero, and neural and behavior gain above
zero) plus a tri-state (`PASS`, `FAIL`, or `NOT_EVALUATED`) result for each of:

- target-animal label permutation;
- donor intervention-semantic shuffle;
- animal-adapter shuffle;

Every supplementary Gate 5--8 result requires a canonical relative artifact
filename and observed digest bound into the runner's authenticated completion
chain. A bare pass flag or digest claimed inside a metrics file is
insufficient.

Gate 8 consumes:

- pseudo-onset on normal trials;
- a pre-onset effect test, which must be null.

Onset shifts, initial-state-only decoding, and the teacher impossibility pair
are additional diagnostics rather than hidden aliases for the declared gate
inputs; frozen v1 does not emit all of them. Sealed-sentinel invariance is an
executable leakage test, not a biological null.

Any target outcome hash changing before the explicit unseal step, any
non-finite sealed sentinel reaching a trainable module, or any target
intervention row appearing in a fitted transform invalidates that fold.
An unrun falsification cannot support a positive claim and is therefore
`NOT_EVALUATED`, not a successful null result.

## Headline pass/fail rule

A dataset passes only if all of the following hold jointly:

1. the randomized manipulation has a nonzero observed neural and behavioral
   effect;
2. the equal-animal 95% bootstrap confidence interval for neural causal skill
   is above zero;
3. the corresponding behavioral interval is above zero;
4. CADENCE improves causal skill by at least 0.10 over the per-target strongest
   eligible non-oracle envelope for both endpoints, with both
   confidence-interval lower bounds above zero;
5. the state-conditioned model improves validation-selected proper score over
   the donor condition-time template;
6. animal-level randomization tests reject the relevant nulls;
7. 90% simultaneous bands are not materially undercovered (exact one-sided
   95% binomial lower bound across animals at least 0.80);
8. pre-onset and pseudo-onset controls remain null.

Randomization-control evidence must retain the frozen procedure, test value,
and a canonical artifact filename plus observed digest bound into the
authenticated completion chain; a bare pass flag or claimed digest is
insufficient.
Pre-onset and pseudo-onset checks must retain a predeclared equivalence margin
and confidence interval wholly inside that margin.

The conjunction is reported verbatim, including failed and unevaluated gates.
Headline evaluation additionally requires the complete frozen replication
cohort: 28 Allen evaluation mice or five randomized ICMS mice. The procedural
teacher report is complete only with 80 target rows arranged as four targets
in each of 20 worlds. Teacher targets are averaged equally within world and the
20 independently seeded worlds—not the 80 nested targets—are its inferential
replication units, but teacher is never headline-eligible. Development cohorts
are likewise never headline-eligible. A positive teacher benchmark alone is not
evidence for a biological shared causal operator. A positive Allen result does
not imply direct neural-control transfer. A positive six-mouse ICMS result is
an exploratory within-task demonstration, not a prospective confirmatory
study.

Version 1 is a fail-closed falsification release. Its current frozen biological
runners do not emit everything needed to pass Gate 1 or Gates 5--8. Missing
components remain `NOT_EVALUATED`; an authenticated evaluated subcomponent
that contradicts a composite gate can still make that gate `FAIL`. This
version can therefore conclusively return `FAIL`, but it cannot return a
biological headline `PASS`. Uncalibrated marginal intervals are diagnostics
and are ineligible for the simultaneous-coverage gate. A future positive-claim
release would require the missing evidence and artifacts to be implemented and
frozen before any new outcome cohort is opened; missing evidence cannot be
retrofitted after this unsealing.
