# CADENCE project site

This is a static Next.js presentation layer for the frozen CADENCE protocol and
its pre-outcome development record. It does not read scientific result
directories at runtime and it never substitutes development values for final
outcomes.

## Local use

```bash
npm install
npm run dev
```

Production validation:

```bash
npm run lint
npm run typecheck
npm run build
```

The static export is written to `out/`.

## Development data

`data/development-summary.json` is a small display derivative of the committed
pre-outcome release:

`results/releases/development/development_record.json`

Its source SHA-256 is retained in the derivative and displayed next to the
diagnostics. The committed release figure is copied verbatim to
`public/development-diagnostics.png`.

## Final outcome population

The site looks for `data/final-outcome.json` at build time. If it is absent,
every confirmatory panel renders **NOT YET POPULATED**. The file is deliberately
gitignored so a result must be supplied intentionally.

After the append-only final report exists, populate the site from the repository
root with the source-bound builder:

```bash
uv run python scripts/build_final_editorial_artifacts.py \
  --generated-at YYYY-MM-DD
```

The builder authenticates `summary.json`, `report.complete.json`, and both
sidecars before writing this presentation derivative, the final paper figures,
and the machine-derived TeX include. `data/final-outcome.example.json` is
illustrative only. The accepted site schema is
`cadence-site-final-outcome-v1`, validated by `lib/final-outcome.ts`.

Population rules:

- Preserve reporter tri-state strings verbatim: `PASS`, `FAIL`, and
  `NOT_EVALUATED`. Do not map absent or null values to zero or failure.
- Set the global headline directly from
  `summary.analyses["allen_vbo:locked"].conjunction.overall_status`. Allen is
  the primary evaluation; do not combine it with exploratory ICMS.
- Set `release.sourceSummarySha256` to
  `report.complete.json.artifacts["summary.json"]`. The generated site JSON is a
  derived presentation artifact, not a reporter-authenticated completion
  artifact; its observed digest is displayed separately. The loader requires
  the source digest to be exactly 64 lowercase hexadecimal characters.
- Allen uses the `allen_vbo:locked` conjunction. ICMS randomized causal results
  use `icms:randomized_n5`. Teacher uses `teacher:locked` and remains
  non-headline. `icms:absolute_only` is descriptive and must not be mixed into
  randomized ICMS gates.
- For eligible Allen and randomized-ICMS Gate 2/3 endpoint cards, use
  `primary_summaries.neural_skill` and `primary_summaries.behavior_skill`
  (including their bootstrap intervals). If those summaries are absent because
  the cohort is ineligible or incomplete, leave endpoint values null.
- Teacher descriptive intervals may come from `method_summaries`; label the
  replication as world-level and non-headline.
- Copy gate `gate_id`, `criterion`, and `status` in 1–8 order. The display
  `evidence` string must be a declared concise derivative of canonical gate
  details, or the canonical details JSON itself—not an invented conclusion.
- Cohort completeness and `gates_evaluable` are eligibility metadata, never
  evidence of `PASS`.

The status precedence already belongs to the reporter (`FAIL` >
`NOT_EVALUATED` > `PASS`). The site must display `conjunction.overall_status`
instead of recomputing it.
