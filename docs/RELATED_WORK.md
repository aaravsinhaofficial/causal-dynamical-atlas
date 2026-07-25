# Related work and contribution boundary

Last primary-source verification: **2026-07-25**.

This note defines the scientific claim narrowly enough to be falsifiable and separates it from nearby results in cross-animal alignment, latent dynamical systems, and perturbation modeling. Citation keys refer to [`paper/references.bib`](../paper/references.bib).

## The claim that would be new

The target result is not simply a shared embedding, a cross-animal decoder, or a good fit to stimulated trials. It is the following prospective test:

> Fit an intervention-conditioned dynamical operator using perturbation trials from donor animals. For a held-out animal, fit its observation map using only unperturbed activity. Without using any intervention trial from that animal for representation learning, normalization, model selection, or early stopping, predict the held-out animal's full time-resolved neural trajectory and behavior after an intervention that the model has never observed in that animal.

“Previously unseen intervention” here means **unseen in the held-out animal**. The intervention identity or a harmonized action descriptor may have been learned from donor animals; claiming generalization to a globally novel intervention would be a stronger, separate result.

For held-out animal \(i^\star\), the defensible protocol is:

1. Learn shared dynamics and intervention coupling from animals \(i\ne i^\star\), including their perturbation trials.
2. Estimate \(H_{i^\star}\), baselines, and any animal-specific residual using only trials or intervals with no intervention.
3. Freeze every learned parameter and preprocessing choice.
4. Initialize the rollout before intervention onset; supply only the intervention descriptor and pre-intervention history.
5. Score the entire post-intervention neural and behavioral time course in the held-out animal.
6. Repeat leave-one-animal-out. Sessions are repeated measures, not independent animals.

No target-animal evoked response may influence channel selection, PCA/CCA bases, scaling, trial rejection, hyperparameters, checkpoints, or uncertainty calibration. A method that sees even the first part of the target response is forecasting with target intervention data, not the zero-shot causal-transfer test above.

## Cross-animal dynamics and alignment

| Work | What the primary source establishes | Boundary relative to the target claim |
|---|---|---|
| Safaie et al. (`safaie2023preserved`) | PCA followed by CCA reveals strongly aligned latent trajectories in mouse and monkey motor cortex and mouse striatum during matched behavior. Decoders trained on aligned activity in one animal transfer movement or planned-target information to another. | This is compelling evidence for preserved **observational** dynamics. The data and test do not learn an intervention input map from donor animals or predict held-out animals' stimulation responses. |
| MARBLE (`gosztolai2025marble`) | Decomposes on-manifold activity into local flow fields and embeds their statistics with geometric deep learning. The resulting representations support comparisons and decoding across networks and animals. | It represents and compares local flow geometry; it is not an intervention-conditioned generative rollout in a normal-only calibrated held-out animal. |
| CANDY (`jiang2025candy`) | Uses behavior-anchored rank contrastive learning to align sessions and subjects while fitting a shared LDS. The preprint reports neural-to-behavior forecasting and adaptation to a held-out animal by freezing the shared LDS/behavior decoder and fitting subject-specific components. | This is the closest shared-dynamics alignment precursor. Its biological evaluations concern ordinary task activity and behavior decoding/forecasting; there is no intervention variable, donor perturbation training, or held-out stimulation-response endpoint. We cite the bioRxiv artifact as a **preprint**, with the related OpenReview identifier recorded in the BibTeX note. |
| CEBRA (`schneider2023cebra`) | Learns contrastive latent embeddings from neural data with behavioral or time labels, including multi-session and cross-animal analyses. | CEBRA establishes consistent representation and decoding, not an explicit state-transition operator that predicts causal response trajectories. |
| sa-SVAE (`jiang2024sasvae`) | A NeurIPS 2024 workshop extended abstract combining a structured VAE with behavior-guided contrastive alignment and a universal behavior decoder. | It learns behaviorally relevant shared dynamics but reports no intervention transfer. It should be described as a **workshop extended abstract**, not an archival main-conference paper. |
| Stitch-LFADS (`pandarinath2018lfads`) | LFADS learns single-trial nonlinear latent dynamics; its stitching construction shares encoder/generator/factor components while using session-specific read-in/read-out matrices for non-overlapping recordings. LFADS can also infer unexplained inputs around a task perturbation. | Stitching demonstrates a shared generator across sessions, mostly within an individual. Inferred inputs are not known causal action operators, and the paper does not train on donor-animal interventions then make a normal-only-calibrated neural-and-behavioral rollout in a new animal. |
| Shared Response Model (`chen2015srm`) | Factorizes time-locked multi-subject fMRI into subject-specific orthogonal maps and a low-dimensional shared response. | SRM is a functional alignment/factor model tied to shared stimulus timing. It contains neither autonomous/controlled neural dynamics nor an intervention forecast. |
| CCA stability (`gallego2020stability`) | CCA alignment exposes stable motor-cortical population dynamics across days despite unstable single-unit tuning. Safaie et al. extend this logic across animals. | CCA requires matched activity samples/conditions to learn a correspondence. Alignment quality and cross-day decoding do not identify a causal input-response operator. |
| Unsupervised domain adaptation (`jude2022domain`) | A sequential VAE/domain-adaptation model transfers a behavior decoder to unseen sessions from the same animal without decoder recalibration. | This addresses recording drift and observational decoder transfer, not cross-animal causal response prediction. |
| Distribution alignment (`dyer2017cryptography`) | Aligns distributions of neural population activity to enable movement decoding across recording conditions/subjects. | Distribution correspondence is not sufficient to identify how a perturbation moves the latent state. |

The common gap is therefore not “nobody has aligned animals.” Several methods do so well. The gap is the combination of **held-out-animal alignment from normal activity only** and **prospective transfer of an intervention-conditioned operator**, evaluated on full neural and behavioral response trajectories.

## Latent and controlled dynamical systems

Classical and modern state-space models supply most of the needed machinery:

- GPFA (`yu2009gpfa`) estimates smooth low-dimensional single-trial trajectories, but does not learn a causal control channel.
- Recurrent switching LDS models (`linderman2017rslds`, `glaser2020multipop`) approximate nonlinear flow using locally linear regimes and can include exogenous covariates. They do not by themselves solve cross-animal observation alignment or donor-to-recipient intervention transfer.
- LFADS (`pandarinath2018lfads`) provides expressive nonlinear dynamics, observation likelihoods, multi-session stitching, and optional inferred inputs. An inferred residual input is not automatically an identifiable experimental intervention effect.
- DPAD (`sani2024dpad`) explicitly prioritizes latent components that predict behavior while jointly modeling neural activity. It supplies useful neural/behavior forecast baselines, but not the cross-animal intervention protocol.
- SINDYc (`brunton2016sindyc`) identifies sparse state equations with known controls. It is an appropriate interpretable baseline once a common state and intervention vocabulary have been defined, but has no subject-specific observation model.
- Shenoy and Kao (`shenoy2021measurement`) formalize neural population dynamics as \(x_{t+1}=Ax_t+Bu_t\) and identify causal perturbation of states and inputs as a key experimental opportunity. That perspective motivates this study; it does not report the held-out-animal experiment.

These methods justify a hierarchical controlled model, but a flexible model class is not evidence of causal transfer. The decisive evidence must come from the animal-level intervention holdout.

## Perturbation-response modeling

### The closest causal generalization result

Sourmpis et al. (`sourmpis2026perturbations`) is the most important adjacent paper. It fits biologically constrained RNNs to **unperturbed** activity and tests out-of-distribution optogenetic inactivation. On the in-vivo dataset, recordings cover 6,182 units across 12 areas and 18 mice; the reconstructed model uses six task-relevant areas. Biological constraints such as Dale's law and local inhibition improve prediction of perturbation-induced changes in hit rate.

This result makes two points that the present project must respect:

1. Good held-out fit to normal activity is not evidence that a model has the right causal mechanism.
2. Perturbation testing is a necessary model-selection-independent evaluation.

It does **not** close the proposed gap. The in-vivo perturbation sessions provide behavior under inactivation but not simultaneous neural recording during stimulation, so the reported biological perturbation endpoint is change in behavioral hit frequency rather than a full measured post-intervention neural trajectory. The model is a pooled circuit reconstruction, not a leave-one-animal-out hierarchy in which perturbations from donor animals are transferred after normal-only calibration of a named held-out animal.

### Other perturbation studies

- Galgali et al. (`galgali2023residual`) use trial-to-trial residual dynamics to constrain recurrent contributions and relate the inferred structure to targeted perturbations. This is mechanistic analysis, not cross-animal prospective intervention forecasting.
- Zheng et al. (`zheng2025icmsrobustness`) directly apply ICMS in two macaques, measure neural-state recovery and reaction time in static versus moving-target contexts, and use an input-driven network to illustrate how feedback can confer robustness. The model is explanatory rather than a donor-trained, held-out-animal predictor.

Thus the safe novelty statement is not “the first model to predict a neural perturbation.” It is:

> A rigorously leakage-sealed demonstration, if the experiments support it, of
> full neural-and-behavioral intervention-response transfer to a held-out
> animal whose mapping was learned only from unperturbed activity.

That statement must remain conditional until the complete leave-one-animal-out results, uncertainty intervals, and leakage audit exist.

## ICMS response biology

ICMS is not a generic additive input. The literature gives several reasons to model action descriptors and uncertainty explicitly:

- Stoney et al. (`stoney1968extent`) quantified the current-dependent effective extent of cortical activation.
- Tehovnik et al. (`tehovnik2006direct`) distinguish direct and transsynaptic activation and emphasize the high excitability of axonal elements.
- Butovas and Schwarz (`butovas2003spatiotemporal`) measured current- and distance-dependent spatiotemporal population responses, including early excitation followed by suppression.
- Histed et al. (`histed2009direct`) showed that direct activation can be sparse and spatially distributed rather than a uniform sphere around the electrode.
- Lycke et al. (`lycke2023ultraflexible`) demonstrate low-threshold, spatially resolved, chronically stable ICMS with ultraflexible electrodes, supporting longitudinal response studies.

Consequences for modeling are concrete. The action input should encode at least electrode/channel, current or charge, pulse timing/train parameters, and—where available—anatomical position. A single binary “stimulated” flag conflates interventions with different recruitment mechanisms. Direct pulse-locked and delayed/polysynaptic responses should be scored separately or at least stratified because they need not share transferability.

## DANDI:001868 and its associated study

The published dataset version is `DANDI:001868/0.260715.2016` (`kim2026dandi001868`; DOI `10.48324/dandi.001868/0.260715.2016`). The DANDI metadata reports:

- 12 mice and 85 NWB files/sessions;
- six task-trained animals with chronic electrophysiology and two-photon imaging;
- six passive-control animals used for activation mapping, four also with electrophysiology;
- an ICMS-cued wheel-turn detection task in trunk S1;
- sorted units, ICMS and electrode metadata, behavioral trials/wheel signals, and processed imaging, with exact contents varying by cohort/session.

The linked study by Kim et al. (`kim2026learning`) is available as a 2026
bioRxiv v1 manuscript. The authors' July 15, 2026 Zenodo v2 record
([doi:10.5281/zenodo.21382755](https://doi.org/10.5281/zenodo.21382755))
describes it as accepted at *Science Advances*; a version-of-record DOI was not
located as of 2026-07-25. It reports learning-related reductions in detection
threshold, expanded recruitment at fixed current, enhanced/shorter-latency
responses in a subset of cells, stronger excitability of directly activated
pulse-locked neurons, and expansion of polysynaptically recruited neurons
associated with behavioral outcome. It characterizes plasticity; it does not
fit or test cross-animal intervention transfer.

The DANDI dataset is unusually relevant, but its headline “12 mice” must not be treated as 12 exchangeable task animals. The core simultaneous task cohort is six animals; passive controls have different measurements and no equivalent behavioral task. A rigorous analysis must therefore:

1. split by animal before any response-dependent preprocessing;
2. report the number of task animals, sessions, units, interventions, and usable trials in every fold;
3. verify which files truly contain unperturbed/catch trials—pre-stimulus windows alone are not full unperturbed trials;
4. harmonize channel/current/pulse descriptors across implants without using target evoked responses;
5. model longitudinal learning/session as a covariate or residual, not count sessions as independent biological replicates;
6. treat modality-missing sessions explicitly rather than silently pooling incompatible endpoints;
7. reserve every intervention trial of the held-out animal for final evaluation.

If the dataset does not contain enough truly unperturbed activity to estimate \(H_{i^\star}\), the zero-shot claim is not supported by replacing it with target-animal perturbation calibration. That would be a different, few-shot result and should be labeled as such.

## Minimum result set needed to support the contribution

The complete scientific program would report:

- leave-one-animal-out full-trajectory neural likelihood/error and behavioral forecasting;
- response-amplitude, latency, peak-time, recovery-time, and integrated-response metrics with animal-level confidence intervals;
- performance broken out by direct/pulse-locked versus delayed/polysynaptic response where labels permit;
- calibration and empirical coverage of predictive intervals;
- no-effect, population-mean, nearest-donor, target-normal-only, shared-dynamics-without-intervention, and shuffled-action baselines;
- ablations for shared operator, animal residual, action embedding, and observation-map calibration;
- negative controls that break animal/action pairing while preserving marginal statistics;
- a complete ledger proving that no held-out-animal intervention sample influenced fitting or selection.

Frozen v1 deliberately does not emit every response-summary, split-half,
calibrated-band, falsification, or ICMS calcium artifact in this aspirational
set. Those items are `NOT_EVALUATED`, and the fail-closed release cannot return
a biological `PASS`.

Neural prediction alone would not establish the stated neural-and-behavioral claim. Behavior prediction alone would reproduce only part of the closest perturbation-modeling precedent. Both must be prospective and time resolved.

## Identifier and status audit

| Key | Status recorded here | Verified identifier / primary record |
|---|---|---|
| `safaie2023preserved` | Nature article (2023) | [doi:10.1038/s41586-023-06714-0](https://doi.org/10.1038/s41586-023-06714-0) |
| `gosztolai2025marble` | Nature Methods article (2025) | [doi:10.1038/s41592-024-02582-2](https://doi.org/10.1038/s41592-024-02582-2); [arXiv:2304.03376](https://arxiv.org/abs/2304.03376) |
| `jiang2025candy` | bioRxiv preprint (2025) | [doi:10.1101/2025.11.11.686428](https://doi.org/10.1101/2025.11.11.686428); [OpenReview uvTea5Rfek](https://openreview.net/forum?id=uvTea5Rfek) |
| `schneider2023cebra` | Nature article (2023) | [doi:10.1038/s41586-023-06031-6](https://doi.org/10.1038/s41586-023-06031-6); [arXiv:2204.00673](https://arxiv.org/abs/2204.00673) |
| `jiang2024sasvae` | NeurIPS 2024 workshop extended abstract | [OpenReview OmkS4CEQzX](https://openreview.net/forum?id=OmkS4CEQzX); [official workshop listing](https://nips.cc/virtual/2024/101454) |
| `pandarinath2018lfads` | Nature Methods article (2018) | [doi:10.1038/s41592-018-0109-9](https://doi.org/10.1038/s41592-018-0109-9) |
| `chen2015srm` | NeurIPS proceedings paper (2015) | [official proceedings](https://proceedings.neurips.cc/paper/2015/hash/b3967a0e938dc2a6340e258630febd5a-Abstract.html) |
| `gallego2020stability` | Nature Neuroscience article (2020) | [doi:10.1038/s41593-019-0555-4](https://doi.org/10.1038/s41593-019-0555-4) |
| `sourmpis2026perturbations` | eLife Version of Record (2026) | [doi:10.7554/eLife.106827.3](https://doi.org/10.7554/eLife.106827.3) |
| `zheng2025icmsrobustness` | Nature Communications article (2025) | [doi:10.1038/s41467-025-58421-1](https://doi.org/10.1038/s41467-025-58421-1) |
| `kim2026learning` | bioRxiv v1; accepted at Science Advances per authors' 2026-07-15 record | [bioRxiv doi:10.64898/2026.06.05.730421](https://doi.org/10.64898/2026.06.05.730421); [Zenodo doi:10.5281/zenodo.21382755](https://doi.org/10.5281/zenodo.21382755) |
| `kim2026dandi001868` | DANDI dataset, published version | [doi:10.48324/dandi.001868/0.260715.2016](https://doi.org/10.48324/dandi.001868/0.260715.2016); [DANDI record](https://dandiarchive.org/dandiset/001868/0.260715.2016) |

Bibliographic metadata were checked against publisher pages, Crossref records, official proceedings/OpenReview/arXiv pages, and the DANDI API. The bibliography deliberately does not infer publication status from repository prose or third-party summaries.
