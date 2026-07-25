"use client";

import { useState } from "react";

import type {
  DatasetOutcome,
  FinalOutcomeLoad,
  OutcomeStatus,
} from "@/lib/final-outcome";
import { gateBlueprints } from "@/lib/project-data";

type DisplayGate = {
  number: number;
  title: string;
  status: OutcomeStatus | "NOT YET POPULATED";
  evidence: string;
};

type DisplayDataset = Omit<DatasetOutcome, "status" | "gates"> & {
  status: OutcomeStatus | "NOT YET POPULATED";
  gates: DisplayGate[];
};

const emptyDatasets: DisplayDataset[] = [
  {
    id: "allen",
    label: "Allen image omissions",
    evidentialRole: "Primary biological evaluation",
    status: "NOT YET POPULATED",
    replication: "28 evaluation mice required",
    summary:
      "Confirmatory metrics have not been loaded. Development values are intentionally not carried into this panel.",
    endpoints: [],
    gates: gateBlueprints.map((gate) => ({
      number: gate.number,
      title: gate.title,
      status: "NOT YET POPULATED",
      evidence: "Awaiting source-bound final evidence.",
    })),
  },
  {
    id: "icms",
    label: "DANDI ICMS",
    evidentialRole: "Exploratory direct-intervention evaluation",
    status: "NOT YET POPULATED",
    replication: "5 randomized causal-eligible mice required",
    summary:
      "Final exploratory metrics have not been loaded. The catch-free animal remains outside the randomized causal estimand.",
    endpoints: [],
    gates: gateBlueprints.map((gate) => ({
      number: gate.number,
      title: gate.title,
      status: "NOT YET POPULATED",
      evidence: "Awaiting source-bound final evidence.",
    })),
  },
];

function statusClass(status: DisplayDataset["status"]): string {
  return status.toLowerCase().replaceAll("_", "-").replaceAll(" ", "-");
}

function StatusBadge({ status }: { status: DisplayDataset["status"] }) {
  return (
    <span className={`result-status result-status--${statusClass(status)}`}>
      <span className="result-status__dot" aria-hidden="true" />
      {status}
    </span>
  );
}

function formatValue(value: number | null): string {
  if (value === null) return "—";
  const absolute = Math.abs(value);
  if (absolute !== 0 && absolute < 0.001) return value.toExponential(2);
  return value.toFixed(3);
}

export function OutcomeExplorer({ load }: { load: FinalOutcomeLoad }) {
  const ready = load.state === "ready";
  const datasets: DisplayDataset[] = ready ? load.data.datasets : emptyDatasets;
  const [activeId, setActiveId] = useState(datasets[0]?.id ?? "allen");
  const active = datasets.find((dataset) => dataset.id === activeId) ?? datasets[0];

  if (!active) return null;

  return (
    <div className="outcome-explorer">
      <div className="outcome-release">
        <div>
          <span className="eyebrow">outcome release</span>
          <h3>{ready ? load.data.release.label : "Awaiting final JSON"}</h3>
        </div>

        {ready ? (
          <StatusBadge status={load.data.headline.status} />
        ) : (
          <StatusBadge
            status={load.state === "invalid" ? "NOT_EVALUATED" : "NOT YET POPULATED"}
          />
        )}
      </div>

      {ready ? (
        <div className="outcome-file">
          <div className="outcome-file__copy">
            <p>{load.data.headline.summary}</p>
            <small>
              This site JSON is a derived presentation artifact. The source-summary
              digest binds it back to the immutable report; the site JSON itself is
              not a reporter completion artifact.
            </small>
          </div>
          <dl>
            <div>
              <dt>Protocol</dt>
              <dd>v{load.data.release.protocolVersion}</dd>
            </div>
            <div>
              <dt>Released</dt>
              <dd>{load.data.release.generatedAt}</dd>
            </div>
            <div>
              <dt>Source summary digest</dt>
              <dd title={load.data.release.sourceSummarySha256}>
                {load.data.release.sourceSummarySha256.slice(0, 14)}…
              </dd>
            </div>
            <div>
              <dt>Site JSON digest</dt>
              <dd title={load.observedSha256}>{load.observedSha256.slice(0, 14)}…</dd>
            </div>
          </dl>
        </div>
      ) : (
        <div
          className={`outcome-empty ${
            load.state === "invalid" ? "outcome-empty--invalid" : ""
          }`}
        >
          <span className="outcome-empty__code" aria-hidden="true">
            {load.state === "invalid" ? "!" : "∅"}
          </span>
          <div>
            <strong>
              {load.state === "invalid"
                ? "FINAL DATA INVALID"
                : "NOT YET POPULATED"}
            </strong>
            <p>
              {load.state === "invalid"
                ? load.error
                : `No ${load.expectedPath} was present at build time. The interface will not infer a result from development diagnostics.`}
            </p>
          </div>
        </div>
      )}

      <div className="outcome-tabs" role="tablist" aria-label="Evaluation dataset">
        {datasets.map((dataset) => (
          <button
            type="button"
            role="tab"
            id={`tab-${dataset.id}`}
            aria-controls={`panel-${dataset.id}`}
            aria-selected={active.id === dataset.id}
            tabIndex={active.id === dataset.id ? 0 : -1}
            onClick={() => setActiveId(dataset.id)}
            key={dataset.id}
          >
            <span>{dataset.label}</span>
            <StatusBadge status={dataset.status} />
          </button>
        ))}
      </div>

      <div
        className="outcome-panel"
        id={`panel-${active.id}`}
        role="tabpanel"
        aria-labelledby={`tab-${active.id}`}
      >
        <div className="outcome-panel__summary">
          <div>
            <span className="eyebrow">{active.evidentialRole}</span>
            <h3>{active.label}</h3>
          </div>
          <div className="outcome-panel__replication">
            <span>replication</span>
            <strong>{active.replication}</strong>
          </div>
          <p>{active.summary}</p>
        </div>

        {active.endpoints.length > 0 && (
          <section className="endpoint-grid" aria-label={`${active.label} endpoints`}>
            {active.endpoints.map((endpoint) => (
              <article className="endpoint-card" key={endpoint.label}>
                <span className="endpoint-card__label">{endpoint.label}</span>
                <strong>{formatValue(endpoint.value)}</strong>
                <p>
                  95% CI {formatValue(endpoint.ciLow)} to{" "}
                  {formatValue(endpoint.ciHigh)}
                </p>
                {endpoint.status && <StatusBadge status={endpoint.status} />}
                {endpoint.note && <small>{endpoint.note}</small>}
              </article>
            ))}
          </section>
        )}

        <section className="gate-list" aria-label={`${active.label} claim gates`}>
          <div className="gate-list__header">
            <span>Gate</span>
            <span>Prespecified requirement</span>
            <span>State</span>
          </div>
          {active.gates.map((gate) => (
            <article className="gate-row" key={`${active.id}-${gate.number}`}>
              <span className="gate-row__number">
                {String(gate.number).padStart(2, "0")}
              </span>
              <div>
                <h4>{gate.title}</h4>
                <p>{gate.evidence}</p>
              </div>
              <StatusBadge status={gate.status} />
            </article>
          ))}
        </section>
      </div>
    </div>
  );
}
