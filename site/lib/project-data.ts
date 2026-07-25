export const repositoryUrl =
  "https://github.com/aaravsinhaofficial/causal-dynamical-atlas";

export const frozenTagUrl = `${repositoryUrl}/tree/pre-outcome-v1.0.0`;

export const navigation = [
  { label: "Question", href: "#question" },
  { label: "Model", href: "#model" },
  { label: "Protocol", href: "#protocol" },
  { label: "Datasets", href: "#datasets" },
  { label: "Development", href: "#development" },
  { label: "Outcome", href: "#outcome" },
] as const;

export const datasets = [
  {
    code: "01 / PRIMARY",
    id: "allen",
    name: "Allen image omissions",
    description:
      "A randomized sensory omission asks whether donor-learned intervention dynamics transport to a new animal.",
    facts: [
      ["Cohort", "32 mice"],
      ["Evaluation", "28 sequestered mice"],
      ["Design", "5 whole-animal folds"],
      ["Window", "0–2 s"],
    ],
    modalities: "Event rate · running · pupil · licking",
    role: "Primary biological evaluation",
  },
  {
    code: "02 / EXPLORATORY",
    id: "icms",
    name: "DANDI:001868 ICMS",
    description:
      "A direct electrical intervention tests leave-one-animal-out transport across a fixed physical stimulation lattice.",
    facts: [
      ["Task cohort", "6 mice"],
      ["Causal eligible", "5 mice"],
      ["Source", "45 trimodal sessions"],
      ["Window", "0–3 s"],
    ],
    modalities: "Sorted spikes · wheel · 700 ms ICMS",
    role: "Exploratory direct-intervention evaluation",
  },
  {
    code: "03 / PROCEDURAL",
    id: "teacher",
    name: "Teacher RNN worlds",
    description:
      "Known operators and paired counterfactual twins expose what the method can recover when ground truth is available.",
    facts: [
      ["Development", "10 worlds"],
      ["Post-freeze audit", "20 public seeds"],
      ["Readout", "64–128 neurons"],
      ["Behavior", "3 channels"],
    ],
    modalities: "Known latent flow · exact counterfactuals",
    role: "Procedural benchmark, never headline-eligible",
  },
] as const;
export const protocolSteps = [
  {
    number: "01",
    label: "Learn normal flow",
    detail:
      "Fit shared dynamics and donor observation maps on normal donor trials.",
    access: "Normal donor activity",
  },
  {
    number: "02",
    label: "Learn the intervention operator",
    detail:
      "Freeze normal dynamics; fit the shared operator and donor random effects.",
    access: "Donor interventions",
  },
  {
    number: "03",
    label: "Calibrate the new animal",
    detail:
      "Fit only its observation map and residual dynamics with normal support.",
    access: "Target normal only",
  },
  {
    number: "04",
    label: "Predict open-loop",
    detail:
      "Freeze every parameter, serialize trajectories, and bind the bytes with SHA-256.",
    access: "Outcomes sealed",
  },
  {
    number: "05",
    label: "Score once",
    detail:
      "Verify the prediction digest, mount post-onset outcomes, and preserve every failed or unevaluated gate.",
    access: "Explicit unseal",
  },
] as const;

export const gateBlueprints = [
  {
    number: 1,
    short: "Observed effect",
    title: "Randomized manipulation has a neural and behavioral effect",
  },
  {
    number: 2,
    short: "Neural skill",
    title: "Neural causal-skill 95% confidence interval is above zero",
  },
  {
    number: 3,
    short: "Behavior skill",
    title: "Behavior causal-skill 95% confidence interval is above zero",
  },
  {
    number: 4,
    short: "Comparator gain",
    title: "At least 0.10 gain over the strongest eligible comparator",
  },
  {
    number: 5,
    short: "Proper score",
    title: "State-conditioned model improves the selected proper score",
  },
  {
    number: 6,
    short: "Randomization",
    title: "Animal-level randomization tests reject the relevant nulls",
  },
  {
    number: 7,
    short: "Coverage",
    title: "Simultaneous trajectory bands are not materially undercovered",
  },
  {
    number: 8,
    short: "Onset controls",
    title: "Pre-onset and pseudo-onset controls remain null",
  },
] as const;
