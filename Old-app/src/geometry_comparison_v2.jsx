import React, { useState, useMemo, useCallback } from "react";
import Papa from "papaparse";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer,
} from "recharts";

import uptakeFile from "./results/uptake-file-cones-v1.txt?raw";
import volumeFile from "./results/volume-file-cones-v1.csv?raw";
import areaFile from "./results/area-file-cones-v1.txt?raw";
// import uptakeFile from "./results/uptake-file-v2.txt?raw";
// import volumeFile from "./results/volume-file-v2.csv?raw";
// import areaFile from "./results/area-file-v2.txt?raw";

const FONT_IMPORT = `@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');`;

const C = {
  bg: "#F6F7F5", panel: "#FFFFFF", ink: "#1B2430", faint: "#6B7480",
  grid: "#DEE3E0", accent: "#0F7173", accentSoft: "#E4F0EF", warn: "#C1502E",
};

// Color encodes R_texture, dash encodes H_texture
const R_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"];
const H_DASHES = ["0", "6 3", "2 2"];

const PRESET_TIMES = [1000, 15000, 60000];

// ---- Formatters ----
const fmtTime = t => t >= 1000 ? (t / 1000).toFixed(t % 1000 === 0 ? 0 : 1) + "k s" : t + " s";
const fmtMol  = v => (v == null || isNaN(v)) ? "—" : v.toExponential(2) + " mol";
const fmtVol  = v => (v == null || isNaN(v)) ? "—" : (v * 1e13).toFixed(2) + " ×10⁻¹³ m³";
const fmtArea = v => (v == null || isNaN(v)) ? "—" : (v * 1e8).toFixed(3) + " ×10⁻⁸ m²";
const fmtR    = r => (r * 1e6).toFixed(1) + " µm";
const fmtH    = h => (h * 1e6).toFixed(0) + " µm";

// ---- Parsers ----
function parseNumericCell(value) {
  if (value == null) return null;
  const cleaned = String(value).trim();
  if (!cleaned) return null;
  if (["-", "--", "null", "nan", "none"].includes(cleaned.toLowerCase())) return null;
  const parsed = Number(cleaned);
  return Number.isFinite(parsed) ? parsed : null;
}

function findColumnIndex(headers, candidates) {
  if (!headers) return -1;
  const normalizedHeaders = headers.map(h => String(h ?? "").toLowerCase().replace(/[^a-z0-9]+/g, "_"));
  for (const candidate of candidates) {
    const normalizedCandidate = String(candidate).toLowerCase().replace(/[^a-z0-9]+/g, "_");
    const index = normalizedHeaders.findIndex(name => name === normalizedCandidate || name.includes(normalizedCandidate) || normalizedCandidate.includes(name));
    if (index >= 0) return index;
  }
  return -1;
}

// COMSOL whitespace-delimited .txt: honor the header names so R_texture/H_cone are mapped correctly
function parseCOMSOLTxt(text) {
  const rows = [];
  const lines = text.replace(/\r/g, "").split("\n");
  let headers = null;

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) continue;

    if (line.startsWith("%")) {
      const headerText = line.replace(/^%\s*/, "").trim();
      if (/(R_texture|H_texture|H_cone|Total moles|intopA|intopV|Volume|value)/i.test(headerText)) {
        headers = headerText.split(/\s{2,}/).filter(Boolean).map(h => h.trim());
      }
      continue;
    }

    const vals = line.split(/\s+/).filter(Boolean).map(parseNumericCell);
    if (vals.length < 4) continue;

    const rIndex = findColumnIndex(headers, ["R_texture", "R"]);
    const hIndex = findColumnIndex(headers, ["H_texture", "H_cone", "H"]);
    const tIndex = findColumnIndex(headers, ["t", "time"]);
    const valueIndex = findColumnIndex(headers, ["moles", "intopA", "intopV", "Volume", "value"]);

    let R = null;
    let H = null;
    let t = null;
    let value = null;

    if (rIndex >= 0 && hIndex >= 0 && tIndex >= 0 && valueIndex >= 0) {
      R = vals[rIndex];
      H = vals[hIndex];
      t = vals[tIndex];
      value = vals[valueIndex];
    } else {
      // Fallback for cone-style exports that use a simpler 4-column layout.
      const maybeConeLayout = vals[0] != null && vals[1] != null && vals[2] != null && vals[3] != null;
      if (maybeConeLayout) {
        const hasHAndR = vals[0] < 1e-4 && vals[1] < 1e-4 && vals[2] >= 0 && vals[3] != null;
        if (hasHAndR) {
          R = vals[1];
          H = vals[0];
          t = vals[2];
          value = vals[3];
        }
      }
    }

    if (R != null && H != null && t != null && value != null) {
      rows.push([R, H, t, value]);
    }
  }

  return rows;
}

// Volume .csv: honor the header names so R_texture/H_cone are mapped correctly
function parseVolumeCsv(text) {
  const lines = text.replace(/\r/g, "").split("\n");
  const entries = [];
  let headers = null;

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) continue;

    if (line.startsWith("%")) {
      const headerText = line.replace(/^%\s*/, "").trim();
      if (/(R_texture|H_texture|H_cone|Volume|intopV|intopA)/i.test(headerText)) {
        headers = Papa.parse(headerText, { header: false }).data?.[0] ?? [];
      }
      continue;
    }

    const row = Papa.parse(line, { header: false }).data?.[0] ?? [];
    const vals = row.map(parseNumericCell);
    if (vals.length < 4) continue;

    const rIndex = findColumnIndex(headers, ["R_texture", "R"]);
    const hIndex = findColumnIndex(headers, ["H_texture", "H_cone", "H"]);
    const valueIndex = findColumnIndex(headers, ["Volume", "intopV", "intopA", "value"]);

    const R = rIndex >= 0 ? vals[rIndex] : vals[0];
    const H = hIndex >= 0 ? vals[hIndex] : vals[1];
    const V_unit = valueIndex >= 0 ? vals[valueIndex] : vals[3];

    if (R != null && H != null && V_unit != null) {
      entries.push({ R, H, V_unit });
    }
  }

  return entries;
}

// ---- Helpers ----
const geoKey   = (R, H) => `${R}|${H}`;
const geoLabel = (R, H) => `R=${fmtR(R)} H=${fmtH(H)}`;

function interpAt(points, tQuery) {
  if (!points.length) return null;
  if (tQuery <= points[0].t) return points[0].n;
  if (tQuery >= points[points.length - 1].t) return points[points.length - 1].n;
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i], b = points[i + 1];
    if (tQuery >= a.t && tQuery <= b.t) {
      const frac = (tQuery - a.t) / (b.t - a.t);
      return a.n + frac * (b.n - a.n);
    }
  }
  return points[points.length - 1].n;
}

// ---- Main component ----
export default function COMSOLAnalysis() {
  const [uptakeRaw, setUptakeRaw] = useState(() => parseCOMSOLTxt(uptakeFile));
  const [volumeRaw, setVolumeRaw] = useState(() => parseVolumeCsv(volumeFile));
  const [areaRaw,   setAreaRaw]   = useState(() => parseCOMSOLTxt(areaFile));
  const [fileErrors, setFileErrors] = useState({});
  const [tProcess,  setTProcess]  = useState(15000);
  const [logSlider, setLogSlider] = useState(Math.log10(15000));
  const [filterH,   setFilterH]   = useState("all");
  const [filterR,   setFilterR]   = useState("all");

  // ---- Process uptake into series ----
  const { seriesByGeom, allR, allH, geomList } = useMemo(() => {
    if (!uptakeRaw) return { seriesByGeom: {}, allR: [], allH: [], geomList: [] };
    const sbg = {};
    for (const [R, H, t, n] of uptakeRaw) {
      const key = geoKey(R, H);
      if (!sbg[key]) sbg[key] = { R, H, label: geoLabel(R, H), points: [] };
      sbg[key].points.push({ t, n });
    }
    Object.values(sbg).forEach(g => g.points.sort((a, b) => a.t - b.t));
    const allR = [...new Set(Object.values(sbg).map(g => g.R))].sort((a, b) => a - b);
    const allH = [...new Set(Object.values(sbg).map(g => g.H))].sort((a, b) => a - b);
    return { seriesByGeom: sbg, allR, allH, geomList: Object.keys(sbg) };
  }, [uptakeRaw]);

  // ---- Volume and area lookup maps ----
  const volMap = useMemo(() => {
    if (!volumeRaw) return {};
    const m = {};
    volumeRaw.forEach(({ R, H, V_unit }) => { m[geoKey(R, H)] = V_unit; });
    return m;
  }, [volumeRaw]);

  const areaMap = useMemo(() => {
    if (!areaRaw) return {};
    const m = {};
    areaRaw.forEach(([R, H, , area]) => { m[geoKey(R, H)] = area; });
    return m;
  }, [areaRaw]);

  // ---- Style per geometry ----
  const styleOf = useCallback(key => {
    const g = seriesByGeom[key];
    if (!g) return { stroke: C.ink, strokeDasharray: "0" };
    return {
      stroke: R_COLORS[allR.indexOf(g.R) % R_COLORS.length],
      strokeDasharray: H_DASHES[allH.indexOf(g.H) % H_DASHES.length],
    };
  }, [seriesByGeom, allR, allH]);

  // ---- Filtered geometries ----
  const visibleGeoms = useMemo(() => geomList.filter(k => {
    const g = seriesByGeom[k];
    if (filterH !== "all" && Math.abs(g.H - parseFloat(filterH)) > 1e-12) return false;
    if (filterR !== "all" && Math.abs(g.R - parseFloat(filterR)) > 1e-12) return false;
    return true;
  }), [geomList, seriesByGeom, filterH, filterR]);

  // ---- Chart data ----
  const chartData = useMemo(() => {
    if (!uptakeRaw) return [];
    const tset = new Set(uptakeRaw.map(r => r[2]));
    const times = [...tset].sort((a, b) => a - b);
    return times.map(t => {
      const entry = { t };
      visibleGeoms.forEach(k => { entry[k] = interpAt(seriesByGeom[k].points, t); });
      return entry;
    });
  }, [uptakeRaw, visibleGeoms, seriesByGeom]);

  // ---- Ranking ----
  const ranking = useMemo(() => geomList.map(k => {
    const g = seriesByGeom[k];
    const pts = g.points;
    const n_t  = interpAt(pts, tProcess);
    const n_eq = pts.length ? pts[pts.length - 1].n : null;
    return {
      key: k, label: g.label, R: g.R, H: g.H, n_t, n_eq,
      V_unit: volMap[k] ?? null,
      intopA: areaMap[k] ?? null,
      pct: n_t && n_eq ? (n_t / n_eq) * 100 : null,
    };
  }).sort((a, b) => (b.n_t ?? -Infinity) - (a.n_t ?? -Infinity)), [geomList, seriesByGeom, tProcess, volMap, areaMap]);

  // ---- Crossover ----
  const presetRankings = useMemo(() => PRESET_TIMES.map(pt => ({
    t: pt,
    order: [...geomList].sort((a, b) =>
      (interpAt(seriesByGeom[b]?.points ?? [], pt) ?? -Infinity) -
      (interpAt(seriesByGeom[a]?.points ?? [], pt) ?? -Infinity)
    ),
  })), [geomList, seriesByGeom]);

  const rankingStable = useMemo(() => {
    if (presetRankings.length < 2) return true;
    const first = presetRankings[0].order.join("|");
    return presetRankings.every(p => p.order.join("|") === first);
  }, [presetRankings]);

  // ---- File handlers ----
  const makeFileHandler = useCallback((key, parser, setter) => e => {
    const file = e.target.files?.[0];
    if (!file) return;
    file.text().then(text => {
      try {
        setter(parser(text));
        setFileErrors(prev => ({ ...prev, [key]: null }));
      } catch (err) {
        setFileErrors(prev => ({ ...prev, [key]: err.message }));
      }
    });
  }, []);

  const handleUptake = makeFileHandler("uptake", parseCOMSOLTxt, setUptakeRaw);
  const handleVolume = makeFileHandler("volume", parseVolumeCsv, setVolumeRaw);
  const handleArea   = makeFileHandler("area",   parseCOMSOLTxt, setAreaRaw);

  const hasData = !!uptakeRaw;
  const MONO = { fontFamily: "'IBM Plex Mono', monospace" };

  return (
    <div style={{ fontFamily: "'IBM Plex Sans', sans-serif", background: C.bg, color: C.ink, padding: 28, minHeight: "100%", boxSizing: "border-box" }}>
      <style>{FONT_IMPORT}</style>
      <style>{`
        .mono { font-family: 'IBM Plex Mono', monospace; }
        .btn { font-family: 'IBM Plex Mono', monospace; font-size: 12px; letter-spacing: .02em; padding: 6px 13px; border-radius: 4px; border: 1px solid ${C.grid}; background: white; cursor: pointer; color: ${C.ink}; transition: border-color .15s, color .15s; }
        .btn:hover { border-color: ${C.accent}; color: ${C.accent}; }
        .btn.on { background: ${C.accent}; color: white; border-color: ${C.accent}; }
        .up-btn { position: relative; overflow: hidden; display: inline-flex; }
        .up-btn input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
        .up-btn.loaded { border-color: ${C.accent}; color: ${C.accent}; }
        select { font-family: 'IBM Plex Mono', monospace; font-size: 12px; padding: 5px 10px; border-radius: 4px; border: 1px solid ${C.grid}; background: white; color: ${C.ink}; cursor: pointer; }
        input[type=range] { -webkit-appearance: none; height: 4px; background: ${C.grid}; border-radius: 2px; outline: none; width: 100%; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 16px; height: 16px; border-radius: 50%; background: ${C.accent}; cursor: pointer; border: 2px solid white; box-shadow: 0 0 0 1px ${C.accent}; }
        tr:hover td { background: rgba(0,0,0,.02); }
      `}</style>

      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <div className="mono" style={{ fontSize: 11, color: C.accent, letterSpacing: "0.08em", marginBottom: 4 }}>COMSOL PARAMETRIC SWEEP · CO₂ UPTAKE ANALYSIS</div>
        <h1 style={{ fontSize: 22, fontWeight: 600, margin: 0, letterSpacing: "-0.01em" }}>Microtexture comparison tool</h1>
      </div>

      {/* File upload panel */}
      <div style={{ background: C.panel, borderRadius: 8, padding: "16px 20px", border: `1px solid ${C.grid}`, marginBottom: 20 }}>
        <div className="mono" style={{ fontSize: 11, color: C.faint, marginBottom: 14, letterSpacing: "0.06em" }}>LOAD COMSOL EXPORT FILES</div>
        <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
          {[
            { key: "uptake", label: "Uptake · n_CO2 vs t", hint: "whitespace-delimited .txt",  loaded: !!uptakeRaw, handler: handleUptake },
            { key: "volume", label: "Volume · V_unit",      hint: "pivoted .csv (optional)",    loaded: !!volumeRaw, handler: handleVolume },
            { key: "area",   label: "Area · intopA",        hint: "whitespace-delimited .txt (optional)", loaded: !!areaRaw, handler: handleArea },
          ].map(({ key, label, hint, loaded, handler }) => (
            <div key={key} style={{ display: "flex", flexDirection: "column", gap: 6, minWidth: 180 }}>
              <span className="mono" style={{ fontSize: 10, color: C.faint, letterSpacing: "0.06em" }}>{label}</span>
              <label className={`btn up-btn${loaded ? " loaded" : ""}`}>
                {loaded ? "✓ loaded" : "upload file"}
                <input type="file" onChange={handler} />
              </label>
              <span className="mono" style={{ fontSize: 10, color: loaded ? C.accent : C.faint }}>{loaded ? "ready" : hint}</span>
              {fileErrors[key] && <span className="mono" style={{ fontSize: 10, color: C.warn }}>{fileErrors[key]}</span>}
            </div>
          ))}
        </div>
      </div>

      {!hasData && (
        <div style={{ background: C.panel, borderRadius: 8, padding: "48px 24px", border: `1px solid ${C.grid}`, textAlign: "center" }}>
          <div className="mono" style={{ fontSize: 13, color: C.faint, lineHeight: 1.8 }}>
            Upload <strong>uptake-file.txt</strong> to start.<br />
            Volume and area files add extra columns to the ranking table.
          </div>
        </div>
      )}

      {hasData && (<>

        {/* Summary + filters + legend row */}
        <div style={{ display: "flex", gap: 12, marginBottom: 16, alignItems: "center", flexWrap: "wrap" }}>
          <span className="mono" style={{ fontSize: 11, color: C.faint }}>
            {geomList.length} geometries · {allR.length} R values · {allH.length} H values
          </span>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <span className="mono" style={{ fontSize: 11, color: C.faint }}>filter:</span>
            <select value={filterR} onChange={e => setFilterR(e.target.value)}>
              <option value="all">all R</option>
              {allR.map(r => <option key={r} value={r}>R = {fmtR(r)}</option>)}
            </select>
            <select value={filterH} onChange={e => setFilterH(e.target.value)}>
              <option value="all">all H</option>
              {allH.map(h => <option key={h} value={h}>H = {fmtH(h)}</option>)}
            </select>
            {(filterR !== "all" || filterH !== "all") && (
              <button className="btn" style={{ fontSize: 11, padding: "4px 10px" }}
                onClick={() => { setFilterR("all"); setFilterH("all"); }}>
                clear
              </button>
            )}
          </div>

          {/* Inline legend */}
          <div style={{ marginLeft: "auto", display: "flex", gap: 20, flexWrap: "wrap", alignItems: "center" }}>
            <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
              <span className="mono" style={{ fontSize: 10, color: C.faint }}>COLOR = R</span>
              {allR.map((r, i) => (
                <span key={r} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <span style={{ width: 18, height: 3, background: R_COLORS[i], display: "inline-block", borderRadius: 2 }} />
                  <span className="mono" style={{ fontSize: 10, color: C.ink }}>{fmtR(r)}</span>
                </span>
              ))}
            </div>
            <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
              <span className="mono" style={{ fontSize: 10, color: C.faint }}>DASH = H</span>
              {allH.map((h, i) => (
                <span key={h} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <svg width="22" height="6" aria-hidden="true">
                    <line x1="0" y1="3" x2="22" y2="3" stroke={C.faint} strokeWidth="2" strokeDasharray={H_DASHES[i]} />
                  </svg>
                  <span className="mono" style={{ fontSize: 10, color: C.ink }}>{fmtH(h)}</span>
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Chart */}
        <div style={{ background: C.panel, borderRadius: 8, padding: "20px 20px 8px", border: `1px solid ${C.grid}`, marginBottom: 20 }}>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={chartData} margin={{ top: 10, right: 24, left: 8, bottom: 8 }}>
              <CartesianGrid stroke={C.grid} strokeDasharray="0" vertical={false} />
              <XAxis
                dataKey="t" type="number" domain={[0, "auto"]}
                tickFormatter={fmtTime}
                tick={{ fontSize: 11, fill: C.faint, ...MONO }}
                stroke={C.grid}
              />
              <YAxis
                tickFormatter={v => v.toExponential(1)}
                tick={{ fontSize: 11, fill: C.faint, ...MONO }}
                stroke={C.grid} width={68}
                label={{ value: "n(t) — mol CO₂", angle: -90, position: "insideLeft", fontSize: 11, fill: C.faint }}
              />
              <Tooltip
                formatter={(v, name) => [fmtMol(typeof v === "number" ? v : null), seriesByGeom[name]?.label ?? name]}
                labelFormatter={l => `t = ${fmtTime(Number(l))}`}
                contentStyle={{ ...MONO, fontSize: 12, border: `1px solid ${C.grid}`, borderRadius: 4 }}
              />
              <ReferenceLine
                x={tProcess}
                stroke={C.warn} strokeWidth={1.5} strokeDasharray="4 3"
                label={{ value: `t = ${fmtTime(tProcess)}`, position: "top", fontSize: 11, fill: C.warn, ...MONO }}
              />
              {visibleGeoms.map(k => {
                const s = styleOf(k);
                return (
                  <Line key={k} dataKey={k} type="monotone"
                    stroke={s.stroke} strokeWidth={1.8}
                    strokeDasharray={s.strokeDasharray}
                    dot={false} connectNulls name={k}
                  />
                );
              })}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Time slider */}
        <div style={{ background: C.panel, borderRadius: 8, padding: "16px 20px", border: `1px solid ${C.grid}`, marginBottom: 20 }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
            <span className="mono" style={{ fontSize: 12, color: C.faint }}>OPERATING TIME  t_process</span>
            <span className="mono" style={{ fontSize: 16, fontWeight: 600, color: C.accent }}>{fmtTime(tProcess)}</span>
          </div>
          <input type="range" min={0} max={5.3} step={0.01} value={logSlider}
            onChange={e => { const v = parseFloat(e.target.value); setLogSlider(v); setTProcess(Math.round(10 ** v)); }}
          />
          <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
            {PRESET_TIMES.map(pt => (
              <button key={pt} className={`btn${tProcess === pt ? " on" : ""}`}
                onClick={() => { setTProcess(pt); setLogSlider(Math.log10(pt)); }}>
                {fmtTime(pt)}
              </button>
            ))}
          </div>
        </div>

        {/* Ranking table */}
        <div style={{ background: C.panel, borderRadius: 8, padding: "18px 20px", border: `1px solid ${C.grid}`, marginBottom: 20, overflowX: "auto" }}>
          <div className="mono" style={{ fontSize: 12, color: C.faint, marginBottom: 12 }}>
            RANKING AT t = {fmtTime(tProcess)}
          </div>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ borderBottom: `1px solid ${C.grid}` }}>
                {[
                  { label: "#",         align: "left"  },
                  { label: "R",         align: "left"  },
                  { label: "H",         align: "left"  },
                  { label: "n(t)",      align: "right" },
                  { label: "n_eq",      align: "right" },
                  { label: "% sat.",    align: "right" },
                  ...(volumeRaw ? [{ label: "V_unit",  align: "right" }] : []),
                  ...(areaRaw   ? [{ label: "intopA",  align: "right" }] : []),
                ].map(({ label, align }) => (
                  <th key={label} className="mono"
                    style={{ textAlign: align, fontSize: 11, color: C.faint, fontWeight: 500, padding: "6px 8px" }}>
                    {label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ranking.map((r, idx) => {
                const ri = allR.indexOf(r.R);
                return (
                  <tr key={r.key} style={{ borderBottom: `1px solid ${C.grid}`, background: idx === 0 ? C.accentSoft : "transparent" }}>
                    <td className="mono" style={{ padding: "8px 8px", fontSize: 12 }}>
                      <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <span style={{ color: idx === 0 ? C.accent : C.faint, fontWeight: idx === 0 ? 600 : 400 }}>{idx + 1}</span>
                        <span style={{ width: 10, height: 3, background: R_COLORS[ri] ?? C.ink, display: "inline-block", borderRadius: 1 }} />
                      </span>
                    </td>
                    <td className="mono" style={{ padding: "8px 8px", fontSize: 12, fontWeight: idx === 0 ? 600 : 400 }}>{fmtR(r.R)}</td>
                    <td className="mono" style={{ padding: "8px 8px", fontSize: 12, fontWeight: idx === 0 ? 600 : 400 }}>{fmtH(r.H)}</td>
                    <td className="mono" style={{ padding: "8px 8px", fontSize: 12, textAlign: "right" }}>{fmtMol(r.n_t)}</td>
                    <td className="mono" style={{ padding: "8px 8px", fontSize: 12, textAlign: "right", color: C.faint }}>{fmtMol(r.n_eq)}</td>
                    <td className="mono" style={{ padding: "8px 8px", fontSize: 12, textAlign: "right", color: C.faint }}>
                      {r.pct != null ? r.pct.toFixed(0) + "%" : "—"}
                    </td>
                    {volumeRaw && (
                      <td className="mono" style={{ padding: "8px 8px", fontSize: 11, textAlign: "right", color: C.faint }}>{fmtVol(r.V_unit)}</td>
                    )}
                    {areaRaw && (
                      <td className="mono" style={{ padding: "8px 8px", fontSize: 11, textAlign: "right", color: C.faint }}>{fmtArea(r.intopA)}</td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Crossover panel */}
        <div style={{ background: C.panel, borderRadius: 8, padding: "18px 20px", border: `1px solid ${C.grid}` }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <div className="mono" style={{ fontSize: 12, color: C.faint }}>RANKING STABILITY ACROSS PRESET TIMES</div>
            <div className="mono" style={{
              fontSize: 11, padding: "3px 10px", borderRadius: 3,
              background: rankingStable ? C.accentSoft : "#FBEAE5",
              color: rankingStable ? C.accent : C.warn,
            }}>
              {rankingStable ? "STABLE" : "CROSSOVER DETECTED"}
            </div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {presetRankings.map(p => (
              <div key={p.t} style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
                <div className="mono" style={{ fontSize: 11, color: C.faint, width: 70, flexShrink: 0, paddingTop: 3 }}>{fmtTime(p.t)}</div>
                <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                  {p.order.slice(0, 5).map((k, i) => {
                    const g = seriesByGeom[k];
                    const ri = allR.indexOf(g?.R);
                    return (
                      <span key={k} className="mono" style={{
                        fontSize: 10, padding: "3px 8px", borderRadius: 3,
                        background: i === 0 ? C.accentSoft : C.bg,
                        color: i === 0 ? C.accent : C.faint,
                        border: `1px solid ${i === 0 ? C.accent : C.grid}`,
                        display: "flex", alignItems: "center", gap: 5,
                      }}>
                        <span style={{ width: 8, height: 8, borderRadius: "50%", background: R_COLORS[ri] ?? C.faint, display: "inline-block", flexShrink: 0 }} />
                        {i + 1}. {g?.label ?? k}
                      </span>
                    );
                  })}
                  {p.order.length > 5 && (
                    <span className="mono" style={{ fontSize: 10, color: C.faint, padding: "3px 6px" }}>
                      +{p.order.length - 5} more
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
          {!rankingStable && (
            <div className="mono" style={{ fontSize: 11, color: C.warn, marginTop: 14, lineHeight: 1.7 }}>
              The top geometry changes with operating time. Report the crossover explicitly — the right choice depends on your target t_process.
            </div>
          )}
        </div>

      </>)}
    </div>
  );
}