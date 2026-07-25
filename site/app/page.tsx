import Image from "next/image";

import { BrandMark } from "@/components/BrandMark";
import { ModelDiagram } from "@/components/ModelDiagram";
import { OutcomeExplorer } from "@/components/OutcomeExplorer";
import { SignalLoom } from "@/components/SignalLoom";
import development from "@/data/development-summary.json";
import { loadFinalOutcome } from "@/lib/final-outcome";
import {
  datasets,
  frozenTagUrl,
  navigation,
  protocolSteps,
  repositoryUrl,
} from "@/lib/project-data";

export const dynamic = "force-static";

function SectionHeading({
  index,
  eyebrow,
  title,
  detail,
  light = false,
}: {
  index: string;
  eyebrow: string;
  title: string;
  detail?: string;
  light?: boolean;
}) {
  return (
    <header className={`section-heading ${light ? "section-heading--light" : ""}`}>
      <div className="section-heading__index" aria-hidden="true">
        {index}
      </div>
      <div className="section-heading__copy">
        <span className="eyebrow">{eyebrow}</span>
        <h2>{title}</h2>
      </div>
      {detail && <p className="section-heading__detail">{detail}</p>}
    </header>
  );
}

function formatDevelopmentValue(value: number): string {
  const absolute = Math.abs(value);
  if (absolute < 0.001) return value.toFixed(6);
  return value.toFixed(3);
}

export default function Home() {
  const finalOutcome = loadFinalOutcome();

  return (
    <>
      <a className="skip-link" href="#main-content">
        Skip to main content
      </a>

      <header className="site-header">
        <a className="site-header__brand" href="#top" aria-label="CADENCE home">
          <BrandMark />
        </a>
        <nav className="site-nav" aria-label="Primary navigation">
          {navigation.map((item) => (
            <a href={item.href} key={item.href}>
              {item.label}
            </a>
          ))}
        </nav>
        <a className="header-repository" href={repositoryUrl}>
          Repository
          <span aria-hidden="true">↗</span>
        </a>
      </header>

      <main id="main-content">
        <section className="hero" id="top" aria-labelledby="hero-title">
          <div className="hero__grid">
            <div className="hero__copy">
              <div className="hero__status">
                <span aria-hidden="true" />
                {finalOutcome.state === "ready"
                  ? "FINAL OUTCOME RELEASE"
                  : "PRE-OUTCOME RELEASE"}{" "}
                · PROTOCOL 1.0.0
              </div>
              <p className="hero__kicker">
                A cross-animal causal dynamical atlas
              </p>
              <h1 id="hero-title">
                Can a causal operator travel from{" "}
                <em>one brain to another?</em>
              </h1>
              <p className="hero__lede">
                CADENCE makes one narrow prediction: after seeing only normal
                activity from a new animal, forecast its complete neural and
                behavioral response to an intervention.
              </p>
              <div className="hero__actions">
                <a className="button button--lime" href="#protocol">
                  Inspect the seal
                  <span aria-hidden="true">↓</span>
                </a>
                <a className="button button--ghost" href="#outcome">
                  Outcome board
                  <span aria-hidden="true">→</span>
                </a>
              </div>
              <p className="hero__boundary">
                Prospective falsification, not a guaranteed positive result.
              </p>
            </div>
            <SignalLoom />
          </div>

          <dl className="hero-facts" aria-label="Frozen design facts">
            <div>
              <dt>32</dt>
              <dd>Allen mice in the frozen cohort</dd>
            </div>
            <div>
              <dt>28</dt>
              <dd>sequestered Allen evaluation mice</dd>
            </div>
            <div>
              <dt>05</dt>
              <dd>whole-animal outer folds</dd>
            </div>
            <div>
              <dt>SHA-256</dt>
              <dd>predictions bound before scoring</dd>
            </div>
          </dl>
        </section>

        <section className="question section-shell" id="question">
          <SectionHeading
            index="01"
            eyebrow="The question"
            title="Transport an effect, not just a representation."
            detail="The estimand is a complete caused trajectory. Decoding accuracy, manifold similarity, and classification do not answer it."
          />

          <div className="question__body">
            <div className="question__statement">
              <p className="question__big">
                Learn how an intervention changes dynamics in donor animals.
                Calibrate a new animal on normal activity alone. Then predict
                every prespecified post-onset channel.
              </p>
              <div
                className="equation"
                role="math"
                aria-label="Individual causal effect over time"
              >
                <span className="equation__symbol">τᵢ(t)</span>
                <span className="equation__equals">=</span>
                <span className="equation__expression">
                  𝔼 [ Yᵢ(t; a) − Yᵢ(t; 0) | i ]
                </span>
                <span className="equation__domain">t ≥ 0</span>
              </div>
            </div>

            <aside className="question__aside">
              <span className="eyebrow">Claim boundary</span>
              <ol>
                <li>
                  <span>01</span>
                  <p>
                    “Unseen” means unseen for the <strong>target animal</strong>,
                    not a globally novel intervention class.
                  </p>
                </li>
                <li>
                  <span>02</span>
                  <p>
                    Animals—not trials, cells, or sessions—are the biological
                    replicates.
                  </p>
                </li>
                <li>
                  <span>03</span>
                  <p>
                    A positive benchmark alone cannot establish a biological
                    shared operator.
                  </p>
                </li>
              </ol>
            </aside>
          </div>
        </section>

        <section className="model section-shell section-shell--ink" id="model">
          <SectionHeading
            index="02"
            eyebrow="Model anatomy"
            title="Shared where it should generalize. Individual where it must fit."
            detail="The decomposition makes the transport assumption inspectable instead of hiding it inside one black box."
            light
          />

          <div
            className="model__formula"
            role="math"
            aria-label="CADENCE state update equation"
          >
            <span>zₜ₊₁</span>
            <span>=</span>
            <b>Fθ(zₜ, uₜ)</b>
            <span>+</span>
            <b>Gφ(zₜ)aₜ</b>
            <span>+</span>
            <b>Rᵢ(zₜ)</b>
            <span>+</span>
            <b>δg(i)(aₜ)</b>
          </div>

          <ModelDiagram />

          <div className="model-notes">
            <article>
              <span>SHARED</span>
              <h3>Normal flow + intervention operator</h3>
              <p>
                Donor animals identify a shared nonlinear flow and a
                state-dependent, low-rank perturbation field.
              </p>
            </article>
            <article>
              <span>INDIVIDUAL</span>
              <h3>Residual dynamics + observation map</h3>
              <p>
                Rank-two residuals and recording maps adapt the shared latent
                system without reading target intervention outcomes.
              </p>
            </article>
            <article>
              <span>HELD FIXED</span>
              <h3>Target intervention residual</h3>
              <p>
                The target residual is never fitted. Point predictions use the
                zero-centered donor distribution mean.
              </p>
            </article>
          </div>
        </section>

        <section className="protocol section-shell" id="protocol">
          <SectionHeading
            index="03"
            eyebrow="Leakage-sealed protocol"
            title="The process boundary is part of the experiment."
            detail="Preparation, prediction, and scoring are separate stages. Target outcomes stay unreadable until prediction bytes are durable."
          />

          <ol className="protocol-steps">
            {protocolSteps.map((step, index) => (
              <li className="protocol-step" key={step.number}>
                <div className="protocol-step__index">{step.number}</div>
                <div className="protocol-step__copy">
                  <span className="protocol-step__access">{step.access}</span>
                  <h3>{step.label}</h3>
                  <p>{step.detail}</p>
                </div>
                {index < protocolSteps.length - 1 && (
                  <span className="protocol-step__connector" aria-hidden="true">
                    ↓
                  </span>
                )}
              </li>
            ))}
          </ol>

          <div className="seal-ledger">
            <div className="seal-ledger__title">
              <span className="seal-ledger__icon" aria-hidden="true">
                ◈
              </span>
              <div>
                <span className="eyebrow">Fail-closed ledger</span>
                <h3>No missing evidence becomes a pass.</h3>
              </div>
            </div>
            <div className="seal-ledger__rules">
              <p>Normal-only transforms</p>
              <p>Open-loop rollouts</p>
              <p>Digest-verified predictions</p>
              <p>Failed + unevaluated gates retained</p>
            </div>
            <a href={`${frozenTagUrl}/docs/PROTOCOL.md`}>
              Read protocol 1.0.0 <span aria-hidden="true">↗</span>
            </a>
          </div>
        </section>

        <section className="datasets section-shell" id="datasets">
          <SectionHeading
            index="04"
            eyebrow="Evaluation atlas"
            title="Three tests. Three different evidential roles."
            detail="The biological datasets ask complementary questions; the teacher benchmark checks recovery under known ground truth."
          />

          <div className="dataset-grid">
            {datasets.map((dataset) => (
              <article className={`dataset-card dataset-card--${dataset.id}`} key={dataset.id}>
                <header>
                  <span className="dataset-card__code">{dataset.code}</span>
                  <span className="dataset-card__pulse" aria-hidden="true" />
                </header>
                <h3>{dataset.name}</h3>
                <p className="dataset-card__description">{dataset.description}</p>
                <dl>
                  {dataset.facts.map(([label, value]) => (
                    <div key={label}>
                      <dt>{label}</dt>
                      <dd>{value}</dd>
                    </div>
                  ))}
                </dl>
                <p className="dataset-card__modalities">{dataset.modalities}</p>
                <footer>{dataset.role}</footer>
              </article>
            ))}
          </div>

          <div className="dataset-ledger">
            <p>
              <strong>Public, versioned inputs.</strong> Frozen manifests retain
              archive identifiers, byte counts, content digests, and
              reconstruction commands; bulk arrays remain outside git.
            </p>
            <a href={`${frozenTagUrl}/docs/DATASETS.md`}>
              Dataset ledger <span aria-hidden="true">↗</span>
            </a>
          </div>
        </section>

        <section className="development section-shell" id="development">
          <SectionHeading
            index="05"
            eyebrow="Development diagnostics"
            title="Warning signs are evidence, too."
            detail="These values were preserved before evaluation outcomes were opened. They guide skepticism; they cannot support a confirmatory claim."
          />

          <div className="development-banner" role="note">
            <span>DEVELOPMENT · NON-CONFIRMATORY</span>
            <p>
              {development.date} · {development.role.replaceAll("_", " ")} ·
              biological headline eligible: <strong>no</strong>
            </p>
          </div>

          <figure className="diagnostic-figure">
            <div className="diagnostic-figure__frame">
              <Image
                src="/development-diagnostics.png"
                alt="Two-panel pre-outcome diagnostic. Teacher development shows behavior causal skill positive across ten worlds while neural skill stays near zero and sometimes negative. Allen development shows proposed neural and running causal skill near zero across four mice, mostly negative."
                width={1760}
                height={704}
                sizes="(max-width: 800px) 100vw, 1200px"
                loading="eager"
              />
            </div>
            <figcaption>
              <span>Committed release figure</span>
              <p>
                The zero line is the no-effect predictor. Positive skill is
                better; negative skill is worse.
              </p>
            </figcaption>
          </figure>

          <div className="development-groups">
            <article className="development-group">
              <header>
                <span className="eyebrow">Teacher · smoke profile</span>
                <h3>Known ground truth, uneven transfer</h3>
                <p>{development.teacher.worlds} development worlds</p>
              </header>
              <div className="development-metrics">
                {development.teacher.metrics.map((metric) => (
                  <div className="development-metric" key={metric.id}>
                    <span>{metric.label}</span>
                    <strong>{formatDevelopmentValue(metric.mean)}</strong>
                    <small>
                      {metric.positive}/{metric.n} positive
                    </small>
                  </div>
                ))}
              </div>
              <p className="development-group__note">
                Mean coordinate fraction outside the normal-rollout range:{" "}
                <strong>
                  {(development.teacher.outsideNormalRolloutRangeMean * 100).toFixed(2)}%
                </strong>
                . This is consistent with—but does not establish—off-support
                extrapolation.
              </p>
            </article>

            <article className="development-group development-group--allen">
              <header>
                <span className="eyebrow">Allen · full profile</span>
                <h3>No development transfer established</h3>
                <p>{development.allen.developmentMice} held-out development mice</p>
              </header>
              <div className="development-metrics">
                {development.allen.metrics.map((metric) => (
                  <div className="development-metric" key={metric.id}>
                    <span>{metric.label}</span>
                    <strong>{formatDevelopmentValue(metric.mean)}</strong>
                    <small>
                      {metric.positive}/{metric.n} positive
                    </small>
                  </div>
                ))}
              </div>
              <p className="development-group__note">
                Neither primary neural skill nor running skill established
                transfer. These failures remain visible before all 28 evaluation
                mice are opened.
              </p>
            </article>
          </div>

          <div className="provenance-line">
            <span>source</span>
            <code>{development.source.path}</code>
            <span>sha256</span>
            <code title={development.source.sha256}>
              {development.source.sha256.slice(0, 16)}…
            </code>
          </div>
        </section>

        <section className="outcome section-shell section-shell--night" id="outcome">
          <SectionHeading
            index="06"
            eyebrow="Final outcome board"
            title="The interface knows the difference between zero and unknown."
            detail="This board is driven only by the separately supplied final outcome file. Missing fields remain visibly missing."
            light
          />

          <OutcomeExplorer load={finalOutcome} />

          <div className="outcome-boundary">
            <span className="eyebrow">v1 claim ceiling</span>
            <p>
              Frozen v1 is fail-closed. Its biological runners can conclusively
              return <strong>FAIL</strong>, but missing positive-claim artifacts
              mean they cannot return a biological <strong>PASS</strong>.
            </p>
          </div>
        </section>
      </main>

      <footer className="site-footer">
        <div className="site-footer__top">
          <BrandMark />
          <p>
            A prospective test of cross-animal intervention-response transport.
          </p>
          <a href="#top">
            Back to top <span aria-hidden="true">↑</span>
          </a>
        </div>
        <div className="site-footer__bottom">
          <span>Protocol frozen 25 July 2026</span>
          <span>Pre-outcome tag: pre-outcome-v1.0.0</span>
          <a href={repositoryUrl}>Code + records ↗</a>
        </div>
      </footer>
    </>
  );
}
