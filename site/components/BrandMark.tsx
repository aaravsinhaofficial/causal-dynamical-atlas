export function BrandMark({ compact = false }: { compact?: boolean }) {
  return (
    <span className={compact ? "brand brand--compact" : "brand"}>
      <svg
        className="brand__mark"
        viewBox="0 0 36 36"
        aria-hidden="true"
      >
        <path d="M4 24.5C9.5 24.5 9.5 11.5 15 11.5s5.5 13 11 13c2.3 0 3.7-2.2 6-6.5" />
        <circle cx="4" cy="24.5" r="2.2" />
        <circle cx="15" cy="11.5" r="2.2" />
        <circle cx="26" cy="24.5" r="2.2" />
      </svg>
      <span className="brand__word">CADENCE</span>
      {!compact && <span className="brand__descriptor">causal dynamical atlas</span>}
    </span>
  );
}
