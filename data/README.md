# Data contract

Raw payloads and per-animal processed arrays are ignored. The small processed
source-index JSON files are committed; public inputs are reconstructed from
versioned manifests and verified checksums.

The primary biological split key is the animal identifier. An experiment,
session, imaging plane, electrode, unit, or cell identifier is never allowed to
place the same animal in both train and test sets.

For every held-out mouse:

1. ordinary, non-change, non-omitted presentations are available for
   observation-map calibration;
2. omission windows and every statistic derived from them are sealed;
3. shared model weights stay frozen during calibration;
4. the seal is opened once for final trajectory evaluation.

Preprocessing fits (normalization, PCA, cell selection, interpolation
hyperparameters) are either learned on training mice or on the held-out mouse's
normal calibration split, never on its omission responses.

For DANDI:001868, the corresponding normal-only support is a randomized
zero-current catch or a four-second ITI window separated from every trial and
electrical-stimulation interval by a two-second guard on both sides. Positive
current trials are never calibration data. Session-varying sorted units receive
session-specific observation maps grouped under an animal-level outer fold.
