export function SignalLoom() {
  return (
    <figure className="signal-loom" aria-labelledby="loom-caption">
      <svg
        className="signal-loom__graphic"
        viewBox="0 0 680 560"
        role="img"
        aria-label="Normal trajectories from donor animals converge on a shared model, then fan into predicted intervention trajectories for a target animal."
      >
        <defs>
          <pattern id="grid" width="44" height="44" patternUnits="userSpaceOnUse">
            <path d="M44 0H0V44" fill="none" className="loom-grid" />
          </pattern>
          <clipPath id="loom-clip">
            <rect x="0" y="0" width="680" height="560" rx="2" />
          </clipPath>
        </defs>

        <rect width="680" height="560" fill="url(#grid)" />
        <g clipPath="url(#loom-clip)">
          <path
            className="loom-trace loom-trace--muted loom-trace--a"
            d="M-28 102C73 39 119 159 205 114s121-79 199-19 112 72 304 3"
          />
          <path
            className="loom-trace loom-trace--muted loom-trace--b"
            d="M-31 171c95-48 148 54 229 5s122-56 202-9 159 45 310-11"
          />
          <path
            className="loom-trace loom-trace--normal loom-trace--c"
            d="M-36 261c87-77 147 54 231 2s121-84 196-22c84 70 163 64 320-15"
          />
          <path
            className="loom-trace loom-trace--normal loom-trace--d"
            d="M-37 330c83-53 151 43 233-8 91-57 127-72 202-6 77 68 167 53 313-24"
          />

          <path
            className="loom-trace loom-trace--intervention loom-trace--e"
            d="M-26 412c104-38 147 31 233-6 69-30 113-31 159 12 35 33 57 47 91 27 54-31 104-119 251-143"
          />
          <path
            className="loom-trace loom-trace--intervention loom-trace--f"
            d="M-24 467c98-25 155 14 230-4 72-18 114-36 159 6 29 27 55 61 100 43 60-24 102-128 243-151"
          />

          <line x1="378" y1="55" x2="378" y2="514" className="loom-seal" />
          <circle cx="378" cy="241" r="8" className="loom-node" />
          <circle cx="378" cy="418" r="8" className="loom-node loom-node--hot" />
          <circle cx="546" cy="382" r="5" className="loom-node loom-node--target" />
        </g>

        <g className="loom-labels">
          <text x="27" y="42">
            OBSERVED
          </text>
          <text x="404" y="42">
            OPEN-LOOP
          </text>
          <text x="27" y="527">
            normal support
          </text>
          <text x="404" y="527">
            predicted response
          </text>
          <text x="391" y="231" className="loom-labels__seal">
            onset
          </text>
        </g>
      </svg>
      <figcaption id="loom-caption" className="signal-loom__caption">
        <span>01</span>
        The target contributes normal activity—never a post-onset response.
      </figcaption>
    </figure>
  );
}
