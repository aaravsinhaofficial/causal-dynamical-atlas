const nodes = [
  {
    symbol: "Fθ",
    label: "Normal flow",
    detail: "shared nonlinear dynamics",
    className: "model-node--cyan",
  },
  {
    symbol: "Gφ",
    label: "Intervention",
    detail: "state-dependent, low rank",
    className: "model-node--lime",
  },
  {
    symbol: "Ri",
    label: "Residual",
    detail: "animal-specific, rank two",
    className: "model-node--coral",
  },
  {
    symbol: "Hi",
    label: "Observation",
    detail: "animal / session map",
    className: "model-node--paper",
  },
] as const;

export function ModelDiagram() {
  return (
    <section className="model-diagram" aria-label="CADENCE model anatomy">
      <div className="model-diagram__rail" aria-hidden="true">
        <span>shared</span>
        <span>target normal only</span>
      </div>
      <div className="model-diagram__nodes">
        {nodes.map((node, index) => (
          <div className="model-node-wrap" key={node.symbol}>
            <article className={`model-node ${node.className}`}>
              <span className="model-node__symbol" aria-hidden="true">
                {node.symbol}
              </span>
              <div>
                <h3>{node.label}</h3>
                <p>{node.detail}</p>
              </div>
            </article>
            {index < nodes.length - 1 && (
              <span className="model-arrow" aria-hidden="true">
                →
              </span>
            )}
          </div>
        ))}
      </div>
      <div className="model-diagram__readout">
        <span className="eyebrow">open-loop readout</span>
        <p>
          Complete neural <em>and</em> behavioral trajectories
        </p>
        <span className="model-diagram__hash">prediction bytes → SHA-256</span>
      </div>
    </section>
  );
}
