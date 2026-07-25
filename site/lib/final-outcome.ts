import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

export type OutcomeStatus = "PASS" | "FAIL" | "NOT_EVALUATED";

export type EndpointResult = {
  label: string;
  value: number | null;
  ciLow: number | null;
  ciHigh: number | null;
  status?: OutcomeStatus;
  note?: string;
};

export type GateResult = {
  number: number;
  title: string;
  status: OutcomeStatus;
  evidence: string;
};

export type DatasetOutcome = {
  id: "allen" | "icms" | string;
  label: string;
  evidentialRole: string;
  status: OutcomeStatus;
  replication: string;
  summary: string;
  endpoints: EndpointResult[];
  gates: GateResult[];
};

export type FinalOutcome = {
  schema: "cadence-site-final-outcome-v1";
  release: {
    label: string;
    generatedAt: string;
    protocolVersion: string;
    sourceSummarySha256: string;
  };
  headline: {
    status: OutcomeStatus;
    summary: string;
  };
  datasets: DatasetOutcome[];
};

export type FinalOutcomeLoad =
  | {
      state: "absent";
      expectedPath: string;
    }
  | {
      state: "invalid";
      expectedPath: string;
      error: string;
    }
  | {
      state: "ready";
      data: FinalOutcome;
      observedSha256: string;
      sourceName: string;
    };

const statuses = new Set<OutcomeStatus>(["PASS", "FAIL", "NOT_EVALUATED"]);

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isString(value: unknown): value is string {
  return typeof value === "string";
}

function isNullableNumber(value: unknown): value is number | null {
  return value === null || (typeof value === "number" && Number.isFinite(value));
}

function isStatus(value: unknown): value is OutcomeStatus {
  return typeof value === "string" && statuses.has(value as OutcomeStatus);
}

function isSha256(value: unknown): value is string {
  return typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
}

function isEndpoint(value: unknown): value is EndpointResult {
  if (!isObject(value)) return false;
  return (
    isString(value.label) &&
    isNullableNumber(value.value) &&
    isNullableNumber(value.ciLow) &&
    isNullableNumber(value.ciHigh) &&
    (value.status === undefined || isStatus(value.status)) &&
    (value.note === undefined || isString(value.note))
  );
}

function isGate(value: unknown): value is GateResult {
  if (!isObject(value)) return false;
  return (
    typeof value.number === "number" &&
    Number.isInteger(value.number) &&
    isString(value.title) &&
    isStatus(value.status) &&
    isString(value.evidence)
  );
}

function isDataset(value: unknown): value is DatasetOutcome {
  if (!isObject(value)) return false;
  return (
    isString(value.id) &&
    isString(value.label) &&
    isString(value.evidentialRole) &&
    isStatus(value.status) &&
    isString(value.replication) &&
    isString(value.summary) &&
    Array.isArray(value.endpoints) &&
    value.endpoints.every(isEndpoint) &&
    Array.isArray(value.gates) &&
    value.gates.every(isGate)
  );
}

function isFinalOutcome(value: unknown): value is FinalOutcome {
  if (!isObject(value) || value.schema !== "cadence-site-final-outcome-v1") {
    return false;
  }
  if (!isObject(value.release) || !isObject(value.headline)) {
    return false;
  }
  return (
    isString(value.release.label) &&
    isString(value.release.generatedAt) &&
    isString(value.release.protocolVersion) &&
    isSha256(value.release.sourceSummarySha256) &&
    isStatus(value.headline.status) &&
    isString(value.headline.summary) &&
    Array.isArray(value.datasets) &&
    value.datasets.length > 0 &&
    value.datasets.every(isDataset)
  );
}

export function loadFinalOutcome(): FinalOutcomeLoad {
  const path = join(process.cwd(), "data", "final-outcome.json");
  const expectedPath = "data/final-outcome.json";

  if (!existsSync(path)) {
    return { state: "absent", expectedPath };
  }

  try {
    const bytes = readFileSync(path);
    const parsed: unknown = JSON.parse(bytes.toString("utf8"));
    if (!isFinalOutcome(parsed)) {
      return {
        state: "invalid",
        expectedPath,
        error: "The file does not match cadence-site-final-outcome-v1.",
      };
    }

    return {
      state: "ready",
      data: parsed,
      observedSha256: createHash("sha256").update(bytes).digest("hex"),
      sourceName: path.split(/[\\/]/).at(-1) ?? "final-outcome.json",
    };
  } catch (error) {
    return {
      state: "invalid",
      expectedPath,
      error: error instanceof Error ? error.message : "Unknown parse error",
    };
  }
}
