# Dataset ledger

All raw inputs are public, immutable where the archive supports versioning, and
excluded from git. Committed manifests contain exact object identifiers,
versions, byte sizes, and content digests.

## DANDI:001868 — electrical intervention

- Published version: `0.260715.2016`
- License: CC BY 4.0
- Species: mouse
- Task: ICMS detection by wheel turn
- Task animals: 6
- Task sessions: 55 total; 45 contain behavior, ephys, and ophys together
- Modalities: curated sorted spikes, volumetric or planar two-photon calcium,
  continuous wheel position, randomized ICMS trains, catch trials
- Frozen v1 scored modalities: curated sorted spikes and wheel only. Calcium
  remains in processed files but is not read or scored by the v1 experiment.
- Intervention descriptors: current, frequency, pulse count, pulse width, and
  physical electrode coordinates
- Published-release size: 7,504,049,197 bytes across 85 assets
- Trimodal task subset: 6,812,254,225 bytes across 45 assets
- Processed trial windows: 16,640 positive-current interventions and 2,332
  normal calibration windows (1,400 randomized catches plus 932 guarded ITIs).
  The primary fixed-train analysis contains 14,734 exact 70-pulse, 100 Hz,
  167 µs, approximately 700 ms trials. The 1,906 longer recorded trains are
  labeled `OUT_OF_SCOPE_NOT_SCORED`; their analysis is `NOT_EVALUATED` in
  frozen v1.

The task protocol randomly interleaves ten conditions within each 100-trial
block: catch plus three stimulation contacts at three current amplitudes.
Currents may adapt between blocks. Consequently, causal contrasts are formed
within session and block, not across learning days.
The source `is_good_trial` column is a cross-modal support flag, not a response
label: its NWB description says `true` denotes matching two-photon coverage of
the stimulation window and `false` denotes no overlapping imaging frames
(defaulting to `true` if no source MAT key exists). Frozen v1 conditions its
eligible trial set on that released flag.

ICMS83 has no zero-current catch trials in the released trimodal sessions.
CADENCE therefore prespecifies signal-blind, stimulation-free ITI windows for
normal-only calibration in every animal. These windows lie wholly outside all
trial and stimulation intervals after a 2 s guard on both sides; they are not
pre-onset fragments taken from target intervention trials.

A local ICMS83 raw-container wiring/shape regression is explicit opt-in. It
runs only when `CADENCE_RUN_REAL_ICMS_AUDIT=1` and
`CADENCE_REAL_ICMS_EXAMPLE_DIR` names an existing directory containing
`sub-ICMS83.nwb`; otherwise pytest reports a skip reason. For example:

```bash
CADENCE_RUN_REAL_ICMS_AUDIT=1 \
CADENCE_REAL_ICMS_EXAMPLE_DIR=/absolute/path/to/icms_examples \
uv run pytest tests/test_dandi_icms.py::test_real_icms83_wiring_and_shapes
```

Its pre-freeze access is disclosed in `docs/FREEZE.md`; the default test suite
never opens that container.

The source study reports 30-fps frame acquisition with volumetric z-stacks
looped through depth. In the released volumetric DFF arrays, individual ROIs
are observed during their z-plane's acquisition burst, not as a simultaneous
30-Hz volume. CADENCE therefore writes separate interpolation-support and
direct-observation masks. Analyses must not call the resulting sparse array a
simultaneous 30-Hz population trace.

Reconstruction:

```bash
uv run python scripts/download_dandi_icms.py --scope all --workers 4 --audit-api
uv run python scripts/preprocess_dandi_icms.py
```

## Allen Visual Behavior Ophys — randomized sensory omission

- Release: `visual-behavior-ophys` project manifest `1.1.0`
- License: CC BY 4.0
- Species: mouse
- Frozen cohort: 32 unique mice, one experiment per mouse
- Cohort restriction: VISp, Slc17a7-IRES2-Cre, 175 µm, familiar, active
- Modalities: neural event rate, running speed, pupil area, lick events
- Intervention: omission of approximately 5% of eligible expected image
  flashes; change and immediately pre-change presentations were ineligible
- Frozen NWB total: approximately 35.4 GB
- Processed windows: 65,522 clean normal presentations and 4,707 eligible
  omissions; 6,768 prespecified cells across the 32 mice

The extraction accepts an omission only when its complete \([-1,2]\) s window
has common neural/behavioral support and is not contaminated by another
omission, change, or sham event. Normal calibration windows are active ordinary
image presentations from the same eligible risk set and at least 3 s away from
those events. Every clean normal is retained. A signal-blind subset of 160 is
used for normal-only target adaptation; the remainder is an untouched,
covariate-matched causal-effect control pool. Selection never consults omission
outcomes.

Reconstruction:

```bash
uv run python -m cadence.data.allen_vbo download \
  --manifest data/manifests/allen_vbo_slc17a7_visp175_familiar_active_v1.1.0.json \
  --destination data/raw/allen_vbo/nwb
uv run python scripts/preprocess_allen_vbo.py --all-normal
```

Each NWB is then extracted with the version-pinned manifest entry and content
hash verification. The repository's batch preprocessing command writes one
compressed directory per mouse.

## Teacher-RNN benchmark — known ground truth

The procedural benchmark fixes a nonlinear shared normal operator,
state-dependent rank-two intervention fields, exact rank-two animal residuals,
animal-specific observation maps with 64–128 neurons, three behaviors, and
negative-binomial neural noise. Intervention trials have counterfactual twins
with identical initial state, task drive, process innovations, and observation
noise.

Ten development-world seeds are available for method iteration. Twenty locked
world seeds are committed but may be materialized only after architecture
freeze. The generator additionally exposes explicit stress axes for conserved
intervention strength, animal residual strength, target neuron count, donor
support, and target state coverage. Those axes are diagnostic experiments, not
silently mixed into the 30 default worlds. A separate impossibility
construction produces two targets with identical normal distributions and
opposite private intervention directions.
