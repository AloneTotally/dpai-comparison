import React, { useState, useMemo, useCallback } from "react";
import Papa from "papaparse";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Legend
} from "recharts";

// ---- Design tokens ----
const COLORS = {
  bg: "#F6F7F5",
  panel: "#FFFFFF",
  ink: "#1B2430",
  inkFaint: "#6B7480",
  grid: "#DEE3E0",
  accent: "#0F7173",
  accentSoft: "#E4F0EF",
  warn: "#C1502E",
  series: ["#0F7173", "#C1502E", "#4A5FC1", "#8A8B3C", "#A8574A", "#2E7D9C"],
};

const FONT_IMPORT = `@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');`;

// ---- Synthetic stub data generator ----
// Retarded-diffusion-style approach to plateau: n(t) = n_eq * (1 - exp(-(t/tau)^p))
function genCurve(n_eq, tau, p, times) {
  return times.map((t) => ({
    t,
    n: n_eq * (1 - Math.exp(-Math.pow(t / tau, p))),
  }));
}

const MIN_T = 0;
const LOG_TIMES = (() => {
  const pts = [MIN_T];
  for (let e = 0; e <= 5.3; e += 0.06) pts.push(Math.pow(10, e));
  return pts;
})();

const STUB_FAMILIES = [
  { name: "Untextured", n_eq: 3.42e-4, tau: 5800, p: 0.85 },
  { name: "Recess (30% removed)", n_eq: 2.55e-4, tau: 900, p: 0.9 },
  { name: "Cone array", n_eq: 3.05e-4, tau: 1500, p: 0.88 },
  { name: "Hierarchical lattice", n_eq: 3.3e-4, tau: 650, p: 0.92 },
];

function buildStubDataset() {
  const rows = [];
  STUB_FAMILIES.forEach((fam) => {
    const curve = genCurve(fam.n_eq, fam.tau, fam.p, LOG_TIMES);
    curve.forEach((pt) => rows.push({ geometry: fam.name, t: pt.t, n: pt.n }));
  });
  return rows;
}

const PRESET_TIMES = [1000, 15000, 60000];

function formatMol(v) {
  if (v === undefined || v === null || isNaN(v)) return "—";
  return v.toExponential(2) + " mol";
}
function formatTime(t) {
  if (t >= 1000) return (t / 1000).toFixed(t % 1000 === 0 ? 0 : 1) + "k s";
  return t + " s";
}

// interpolate n(t) for a geometry's sorted points at query time
function interpAt(points, tQuery) {
  if (!points.length) return null;
  if (tQuery <= points[0].t) return points[0].n;
  if (tQuery >= points[points.length - 1].t) return points[points.length - 1].n;
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i], b = points[i + 1];
    if (tQuery >= a.t && tQuery <= b.t) {
      if (b.t === a.t) return a.n;
      const frac = (tQuery - a.t) / (b.t - a.t);
      return a.n + frac * (b.n - a.n);
    }
  }
  return points[points.length - 1].n;
}

export default function GeometryComparisonTool() {
  const [rawRows, setRawRows] = useState(null); // null => use stub
  const [tProcess, setTProcess] = useState(15000);
  const [logSlider, setLogSlider] = useState(Math.log10(15000));
  const [fileName, setFileName] = useState(null);
  const [parseError, setParseError] = useState(null);
  const [hidden, setHidden] = useState({});

  const isStub = rawRows === null;
  const rows = rawRows || buildStubDataset();

  const geometries = useMemo(() => {
    const seen = [];
    rows.forEach((r) => { if (!seen.includes(r.geometry)) seen.push(r.geometry); });
    return seen;
  }, [rows]);

  const seriesByGeom = useMemo(() => {
    const map = {};
    geometries.forEach((g) => {
      map[g] = rows
        .filter((r) => r.geometry === g)
        .sort((a, b) => a.t - b.t);
    });
    return map;
  }, [rows, geometries]);

  // Chart data: merge onto union of time points is heavy; instead sample at each series' own times
  // Build a chart-friendly array keyed by t for each series (recharts wants one array w/ multiple keys)
  const chartData = useMemo(() => {
    const tset = new Set();
    rows.forEach((r) => tset.add(r.t));
    const times = Array.from(tset).sort((a, b) => a - b);
    return times.map((t) => {
      const entry = { t };
      geometries.forEach((g) => {
        const pts = seriesByGeom[g];
        entry[g] = pts.length ? interpAt(pts, t) : null;
      });
      return entry;
    });
  }, [rows, geometries, seriesByGeom]);

  const ranking = useMemo(() => {
    return geometries
      .map((g) => ({
        geometry: g,
        n: interpAt(seriesByGeom[g], tProcess),
        n_eq: seriesByGeom[g].length ? seriesByGeom[g][seriesByGeom[g].length - 1].n : null,
      }))
      .sort((a, b) => (b.n ?? -Infinity) - (a.n ?? -Infinity));
  }, [geometries, seriesByGeom, tProcess]);

  const presetRankings = useMemo(() => {
    return PRESET_TIMES.map((pt) => ({
      t: pt,
      order: geometries
        .map((g) => ({ geometry: g, n: interpAt(seriesByGeom[g], pt) }))
        .sort((a, b) => (b.n ?? -Infinity) - (a.n ?? -Infinity))
        .map((r) => r.geometry),
    }));
  }, [geometries, seriesByGeom]);

  const rankingStable = useMemo(() => {
    if (presetRankings.length < 2) return true;
    const first = presetRankings[0].order.join("|");
    return presetRankings.every((p) => p.order.join("|") === first);
  }, [presetRankings]);

  const handleFile = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setFileName(file.name);
    setParseError(null);
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (res) => {
        try {
          const cols = res.meta.fields.map((f) => f.trim().toLowerCase());
          const geomKey = res.meta.fields[cols.indexOf("geometry")];
          const tKey = res.meta.fields[cols.indexOf("t")];
          const nKey = res.meta.fields[cols.indexOf("n")];
          if (!geomKey || !tKey || !nKey) {
            setParseError("CSV needs columns: geometry, t, n (case-insensitive).");
            return;
          }
          const parsed = res.data
            .filter((r) => r[geomKey] !== undefined && r[tKey] !== undefined && r[nKey] !== undefined)
            .map((r) => ({ geometry: String(r[geomKey]), t: Number(r[tKey]), n: Number(r[nKey]) }));
          if (!parsed.length) {
            setParseError("No valid rows found.");
            return;
          }
          setRawRows(parsed);
        } catch (err) {
          setParseError("Could not parse file: " + err.message);
        }
      },
      error: (err) => setParseError(err.message),
    });
  }, []);

  const resetToStub = () => {
    setRawRows(null);
    setFileName(null);
    setParseError(null);
  };

  const onSliderChange = (e) => {
    const lv = parseFloat(e.target.value);
    setLogSlider(lv);
    setTProcess(Math.round(Math.pow(10, lv)));
  };

  const toggleHidden = (g) => {
    setHidden((h) => ({ ...h, [g]: !h[g] }));
  };

  return (
    <div style={{
      fontFamily: "'IBM Plex Sans', sans-serif",
      background: COLORS.bg,
      color: COLORS.ink,
      padding: "28px",
      minHeight: "100%",
      boxSizing: "border-box",
    }}>
      <style>{FONT_IMPORT}</style>
      <style>{`
        .mono { font-family: 'IBM Plex Mono', monospace; }
        input[type=range] {
          -webkit-appearance: none;
          height: 4px;
          background: ${COLORS.grid};
          border-radius: 2px;
          outline: none;
        }
        input[type=range]::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 16px; height: 16px;
          border-radius: 50%;
          background: ${COLORS.accent};
          cursor: pointer;
          border: 2px solid white;
          box-shadow: 0 0 0 1px ${COLORS.accent};
        }
        .chip {
          cursor: pointer;
          user-select: none;
          transition: opacity 0.15s ease;
        }
        .btn {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 12px;
          letter-spacing: 0.02em;
          padding: 7px 14px;
          border-radius: 4px;
          border: 1px solid ${COLORS.grid};
          background: white;
          cursor: pointer;
          color: ${COLORS.ink};
        }
        .btn:hover { border-color: ${COLORS.accent}; color: ${COLORS.accent}; }
        .btn-primary {
          background: ${COLORS.accent};
          color: white;
          border-color: ${COLORS.accent};
        }
        .btn-primary:hover { opacity: 0.9; color: white; }
      `}</style>

      {/* Header */}
      <div style={{ marginBottom: "22px", display: "flex", justifyContent: "space-between", alignItems: "flex-end", flexWrap: "wrap", gap: "12px" }}>
        <div>
          <div className="mono" style={{ fontSize: "11px", color: COLORS.accent, letterSpacing: "0.08em", marginBottom: "4px" }}>
            n(t) UPTAKE COMPARISON
          </div>
          <h1 style={{ fontSize: "22px", fontWeight: 600, margin: 0, letterSpacing: "-0.01em" }}>
            Geometry ranking at fixed operating time
          </h1>
        </div>
        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          <label className="btn" style={{ position: "relative", overflow: "hidden" }}>
            Upload CSV
            <input
              type="file"
              accept=".csv"
              onChange={handleFile}
              style={{ position: "absolute", inset: 0, opacity: 0, cursor: "pointer" }}
            />
          </label>
          {!isStub && (
            <button className="btn" onClick={resetToStub}>Reset to demo data</button>
          )}
        </div>
      </div>

      {parseError && (
        <div className="mono" style={{ fontSize: "12px", color: COLORS.warn, marginBottom: "14px", padding: "10px 14px", background: "#FBEAE5", borderRadius: "4px" }}>
          {parseError}
        </div>
      )}

      <div className="mono" style={{ fontSize: "11px", color: COLORS.inkFaint, marginBottom: "20px" }}>
        {isStub
          ? "Showing placeholder data — upload a CSV with columns: geometry, t, n"
          : `Loaded: ${fileName} · ${geometries.length} geometries · ${rows.length} rows`}
      </div>

      {/* Chart panel */}
      <div style={{ background: COLORS.panel, borderRadius: "8px", padding: "20px 20px 8px", border: `1px solid ${COLORS.grid}`, marginBottom: "20px" }}>
        <ResponsiveContainer width="100%" height={380}>
          <LineChart data={chartData} margin={{ top: 10, right: 24, left: 8, bottom: 8 }}>
            <CartesianGrid stroke={COLORS.grid} strokeDasharray="0" vertical={false} />
            <XAxis
              dataKey="t"
              type="number"
              domain={[0, "auto"]}
              tickFormatter={(t) => formatTime(t)}
              tick={{ fontSize: 11, fill: COLORS.inkFaint, fontFamily: "IBM Plex Mono" }}
              stroke={COLORS.grid}
            />
            <YAxis
              tickFormatter={(v) => v.toExponential(1)}
              tick={{ fontSize: 11, fill: COLORS.inkFaint, fontFamily: "IBM Plex Mono" }}
              stroke={COLORS.grid}
              width={64}
              label={{ value: "n(t) — mol CO₂ / film", angle: -90, position: "insideLeft", fontSize: 11, fill: COLORS.inkFaint }}
            />
            <Tooltip
              formatter={(v, name) => [formatMol(v), name]}
              labelFormatter={(t) => `t = ${formatTime(t)}`}
              contentStyle={{ fontFamily: "IBM Plex Mono", fontSize: 12, border: `1px solid ${COLORS.grid}`, borderRadius: 4 }}
            />
            <Legend
              onClick={(e) => toggleHidden(e.dataKey)}
              wrapperStyle={{ fontFamily: "IBM Plex Mono", fontSize: 12, cursor: "pointer" }}
            />
            <ReferenceLine
              x={tProcess}
              stroke={COLORS.warn}
              strokeWidth={1.5}
              strokeDasharray="4 3"
              label={{ value: `t_process = ${formatTime(tProcess)}`, position: "top", fontSize: 11, fill: COLORS.warn, fontFamily: "IBM Plex Mono" }}
            />
            {geometries.map((g, i) => (
              <Line
                key={g}
                dataKey={g}
                type="monotone"
                stroke={COLORS.series[i % COLORS.series.length]}
                strokeWidth={2}
                dot={false}
                hide={!!hidden[g]}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Slider control */}
      <div style={{ background: COLORS.panel, borderRadius: "8px", padding: "18px 20px", border: `1px solid ${COLORS.grid}`, marginBottom: "20px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
          <div className="mono" style={{ fontSize: "12px", color: COLORS.inkFaint }}>OPERATING TIME t_process</div>
          <div className="mono" style={{ fontSize: "16px", fontWeight: 600, color: COLORS.accent }}>{formatTime(tProcess)}</div>
        </div>
        <input
          type="range"
          min={0}
          max={5.3}
          step={0.01}
          value={logSlider}
          onChange={onSliderChange}
          style={{ width: "100%" }}
        />
        <div style={{ display: "flex", gap: "8px", marginTop: "12px" }}>
          {PRESET_TIMES.map((pt) => (
            <button
              key={pt}
              className={"btn" + (tProcess === pt ? " btn-primary" : "")}
              onClick={() => { setTProcess(pt); setLogSlider(Math.log10(pt)); }}
            >
              {formatTime(pt)}
            </button>
          ))}
        </div>
      </div>

      {/* Ranking table */}
      <div style={{ background: COLORS.panel, borderRadius: "8px", padding: "18px 20px", border: `1px solid ${COLORS.grid}`, marginBottom: "20px" }}>
        <div className="mono" style={{ fontSize: "12px", color: COLORS.inkFaint, marginBottom: "12px" }}>
          RANKING AT t = {formatTime(tProcess)}
        </div>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ borderBottom: `1px solid ${COLORS.grid}` }}>
              {["#", "Geometry", "n(t_process)", "n_eq", "% of n_eq reached"].map((h) => (
                <th key={h} className="mono" style={{ textAlign: h === "Geometry" ? "left" : "right", fontSize: 11, color: COLORS.inkFaint, fontWeight: 500, padding: "6px 8px" }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ranking.map((r, idx) => (
              <tr key={r.geometry} style={{ borderBottom: `1px solid ${COLORS.grid}`, background: idx === 0 ? COLORS.accentSoft : "transparent" }}>
                <td className="mono" style={{ padding: "9px 8px", fontSize: 13, color: idx === 0 ? COLORS.accent : COLORS.ink, fontWeight: idx === 0 ? 600 : 400 }}>{idx + 1}</td>
                <td style={{ padding: "9px 8px", fontSize: 13, fontWeight: idx === 0 ? 600 : 400 }}>{r.geometry}</td>
                <td className="mono" style={{ padding: "9px 8px", fontSize: 13, textAlign: "right" }}>{formatMol(r.n)}</td>
                <td className="mono" style={{ padding: "9px 8px", fontSize: 13, textAlign: "right", color: COLORS.inkFaint }}>{formatMol(r.n_eq)}</td>
                <td className="mono" style={{ padding: "9px 8px", fontSize: 13, textAlign: "right", color: COLORS.inkFaint }}>
                  {r.n && r.n_eq ? ((r.n / r.n_eq) * 100).toFixed(0) + "%" : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Crossover sensitivity check */}
      <div style={{ background: COLORS.panel, borderRadius: "8px", padding: "18px 20px", border: `1px solid ${COLORS.grid}` }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
          <div className="mono" style={{ fontSize: "12px", color: COLORS.inkFaint }}>RANKING STABILITY ACROSS PRESET TIMES</div>
          <div className="mono" style={{
            fontSize: "11px", padding: "3px 10px", borderRadius: "3px",
            background: rankingStable ? "#E4F0EF" : "#FBEAE5",
            color: rankingStable ? COLORS.accent : COLORS.warn,
          }}>
            {rankingStable ? "STABLE" : "CROSSOVER DETECTED"}
          </div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          {presetRankings.map((p) => (
            <div key={p.t} style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <div className="mono" style={{ fontSize: 11, color: COLORS.inkFaint, width: "70px", flexShrink: 0 }}>{formatTime(p.t)}</div>
              <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
                {p.order.map((g, i) => (
                  <span key={g} className="mono" style={{
                    fontSize: 11, padding: "3px 8px", borderRadius: "3px",
                    background: i === 0 ? COLORS.accentSoft : COLORS.bg,
                    color: i === 0 ? COLORS.accent : COLORS.inkFaint,
                    border: `1px solid ${i === 0 ? COLORS.accent : COLORS.grid}`,
                  }}>
                    {i + 1}. {g}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
        {!rankingStable && (
          <div className="mono" style={{ fontSize: 11, color: COLORS.warn, marginTop: "12px", lineHeight: 1.5 }}>
            The top-ranked geometry changes depending on operating time. Report the crossover
            explicitly rather than a single winner — the correct choice depends on your actual t_process.
          </div>
        )}
      </div>
    </div>
  );
}
