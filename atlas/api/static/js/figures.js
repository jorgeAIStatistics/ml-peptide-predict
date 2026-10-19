/* ATLAS — scientific figures.
   Hand-drawn SVG so figures match the rest of the instrument and the
   application carries no runtime dependency. Every figure re-renders on
   container resize, so axis type stays at its true size instead of being
   scaled by a viewBox. */

import { svg, el, fmt, fmtCount, tip, hideTip, titleCase } from './ui.js';

/* ---------------- axis helpers ---------------- */

export function niceTicks(min, max, target = 6) {
  const span = max - min;
  if (!(span > 0)) return [min];
  const raw = span / target;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) {
    out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return out;
}

function decadeTicks(maxLog) {
  const out = [];
  for (let d = 0; d <= Math.ceil(maxLog); d++) out.push(d);
  return out;
}

/** Re-render `draw` whenever the host resizes. */
function responsive(host, draw) {
  draw();
  if (host._ro) host._ro.disconnect();
  host._ro = new ResizeObserver(() => draw());
  host._ro.observe(host);
}

/* ============================================================
   Evidence coverage landscape
   ============================================================
   The primary discovery view. Literature volume on a log axis against
   how completely the molecular identity resolved, with trials carrying
   posted results as point area.

   The quadrants are the point: they separate compounds that are
   characterised from those that are studied but unidentified, identified
   but unstudied, or neither. That is a research map, not a scatter plot.
   ============================================================ */

const STATUS_COLOUR = {
  resolved: 'var(--teal)',
  ambiguous_multiple_candidates: 'var(--amber)',
  unresolved_no_database_match: 'var(--copper)',
  mixture_not_a_single_molecular_entity: 'var(--blue)',
  supplier_defined_identity_unverified: 'var(--stone-ink)',
};

export function coverageLandscape(host, rows, { onSelect, selectedKey } = {}) {
  const draw = () => {
    const width = Math.max(360, host.clientWidth || 720);
    const height = Math.min(560, Math.max(380, width * 0.52));
    host.textContent = '';

    const pad = { t: 26, r: 26, b: 52, l: 64 };
    const iw = width - pad.l - pad.r;
    const ih = height - pad.t - pad.b;

    const logOf = (n) => Math.log10(Math.max(0, n || 0) + 1);
    const maxLog = Math.max(1, ...rows.map((r) => logOf(r.publications)));
    const sx = (n) => pad.l + (logOf(n) / maxLog) * iw;
    const sy = (c) => pad.t + ih - (Math.max(0, Math.min(1, c || 0))) * ih;
    const maxTrials = Math.max(1, ...rows.map((r) => Math.max(0, r.trials_with_results || 0)));
    const sr = (t) => 3.5 + Math.sqrt(Math.max(0, t || 0) / maxTrials) * 9;

    const root = svg('svg', {
      width, height, viewBox: `0 0 ${width} ${height}`,
      role: 'img',
      'aria-label':
        'Evidence coverage landscape: literature volume against molecular identity completeness',
    });

    // --- quadrant fields, drawn first so points sit above them
    const midX = pad.l + iw * 0.45;
    const midY = pad.t + ih * 0.42;
    const fields = [
      { x: midX, y: pad.t, w: pad.l + iw - midX, h: midY - pad.t,
        label: 'characterised', hint: 'identity resolved, literature present' },
      { x: pad.l, y: pad.t, w: midX - pad.l, h: midY - pad.t,
        label: 'identified, little studied', hint: '' },
      { x: midX, y: midY, w: pad.l + iw - midX, h: pad.t + ih - midY,
        label: 'studied, identity unresolved', hint: '' },
      { x: pad.l, y: midY, w: midX - pad.l, h: pad.t + ih - midY,
        label: 'dark', hint: 'neither identified nor studied' },
    ];
    for (const f of fields) {
      root.append(svg('rect', {
        x: f.x, y: f.y, width: Math.max(0, f.w), height: Math.max(0, f.h),
        fill: f.label === 'characterised' ? 'var(--teal-wash)'
          : f.label === 'dark' ? 'var(--copper-wash)' : 'transparent',
        opacity: 0.5,
      }));
    }
    root.append(svg('line', {
      x1: midX, x2: midX, y1: pad.t, y2: pad.t + ih,
      stroke: 'var(--rule)', 'stroke-dasharray': '2 4',
    }));
    root.append(svg('line', {
      x1: pad.l, x2: pad.l + iw, y1: midY, y2: midY,
      stroke: 'var(--rule)', 'stroke-dasharray': '2 4',
    }));
    // Quadrant labels collide below about 560px, where the fields are narrower
    // than the words. The shading and axes still carry the meaning.
    if (width >= 560) {
      for (const f of fields) {
        const t = svg('text', {
          class: 'ax-text', x: f.x + 8, y: f.y + 14,
          'font-style': 'italic', fill: 'var(--ink-5)',
        });
        t.textContent = f.label;
        root.append(t);
      }
    }

    // --- axes
    for (const d of decadeTicks(maxLog)) {
      const x = pad.l + (d / maxLog) * iw;
      root.append(svg('line', { class: 'ax-grid', x1: x, x2: x, y1: pad.t, y2: pad.t + ih }));
      const t = svg('text', { class: 'ax-text', x, y: pad.t + ih + 16, 'text-anchor': 'middle' });
      t.textContent = d === 0 ? '0' : `10${'⁰¹²³⁴⁵⁶⁷⁸⁹'[d] || `^${d}`}`;
      root.append(t);
    }
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      const y = sy(v);
      root.append(svg('line', { class: 'ax-grid', x1: pad.l, x2: pad.l + iw, y1: y, y2: y }));
      const t = svg('text', { class: 'ax-text', x: pad.l - 9, y: y + 3.5, 'text-anchor': 'end' });
      t.textContent = `${(v * 100).toFixed(0)}%`;
      root.append(t);
    }
    root.append(svg('line', { class: 'ax-line', x1: pad.l, x2: pad.l + iw, y1: pad.t + ih, y2: pad.t + ih }));
    root.append(svg('line', { class: 'ax-line', x1: pad.l, x2: pad.l, y1: pad.t, y2: pad.t + ih }));

    const xt = svg('text', { class: 'ax-title', x: pad.l + iw / 2, y: height - 8, 'text-anchor': 'middle' });
    xt.textContent = 'indexed publications';
    root.append(xt);
    const yt = svg('text', {
      class: 'ax-title', x: 13, y: pad.t + ih / 2, 'text-anchor': 'middle',
      transform: `rotate(-90 13 ${pad.t + ih / 2})`,
    });
    yt.textContent = 'identity completeness';
    root.append(yt);

    // --- points
    const ordered = [...rows].sort((a, b) => (a.trials_with_results || 0) - (b.trials_with_results || 0));
    for (const r of ordered) {
      const cx = sx(r.publications);
      const cy = sy(r.completeness);
      const isSel = r.key === selectedKey;
      const g = svg('g', {
        class: 'pt', tabindex: '0', role: 'button',
        'aria-label':
          `${r.name}: ${fmtCount(r.publications)} publications, identity ` +
          `${r.identity_status}, completeness ${Math.round((r.completeness || 0) * 100)}%`,
      });
      const show = (ev) => {
        const b = ev.currentTarget.getBoundingClientRect();
        tip(
          `<strong>${r.name}</strong>` +
          `${fmtCount(r.publications)} publications · ${fmtCount(r.human_pk_publications)} human PK<br>` +
          `${fmtCount(r.trials_with_results)} trials with results<br>` +
          `identity ${titleCase(r.identity_status || '')} · ` +
          `${Math.round((r.completeness || 0) * 100)}% complete`,
          ev.clientX || b.right, ev.clientY || b.top);
      };
      g.addEventListener('pointerenter', show);
      g.addEventListener('pointermove', show);
      g.addEventListener('pointerleave', hideTip);
      g.addEventListener('focus', show);
      g.addEventListener('blur', hideTip);
      if (onSelect) {
        g.addEventListener('click', () => onSelect(r.key));
        g.addEventListener('keydown', (e) => {
          if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSelect(r.key); }
        });
      }
      if (isSel) {
        g.append(svg('circle', {
          cx, cy, r: sr(r.trials_with_results) + 5,
          fill: 'none', stroke: 'var(--teal)', 'stroke-width': 1.5,
        }));
      }
      g.append(svg('circle', {
        cx, cy, r: sr(r.trials_with_results),
        fill: STATUS_COLOUR[r.identity_status] || 'var(--ink-4)',
        'fill-opacity': r.publications > 0 ? 0.68 : 0.3,
        stroke: STATUS_COLOUR[r.identity_status] || 'var(--ink-4)',
        'stroke-width': 1,
      }));
      root.append(g);
    }

    host.append(root);
  };
  responsive(host, draw);
}

/* ============================================================
   Line figure — concentration-time and similar trajectories
   ============================================================ */

export function lineFigure(host, config) {
  const draw = () => {
    const width = Math.max(320, host.clientWidth || 640);
    const height = config.height || 320;
    host.textContent = '';

    const pad = { t: 12, r: 18, b: 44, l: 64 };
    const iw = width - pad.l - pad.r;
    const ih = height - pad.t - pad.b;
    const all = config.series.flatMap((s) => s.points);
    if (!all.length) return;

    const xs = all.map((p) => p[0]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    const logY = !!config.logY;
    let y0, y1 = Math.max(...all.map((p) => p[1]));
    if (logY) {
      const pos = all.map((p) => p[1]).filter((v) => v > 0);
      y0 = 10 ** Math.floor(Math.log10(pos.length ? Math.min(...pos) : 1e-6));
      y1 = 10 ** Math.ceil(Math.log10(y1 || 1));
    } else { y0 = 0; y1 = y1 > 0 ? y1 * 1.08 : 1; }

    const sx = (v) => pad.l + (x1 === x0 ? 0 : ((v - x0) / (x1 - x0)) * iw);
    const sy = (v) => logY
      ? pad.t + ih - (Math.log10(Math.max(v, y0) / y0) / Math.log10(y1 / y0)) * ih
      : pad.t + ih - ((v - y0) / (y1 - y0)) * ih;

    const root = svg('svg', {
      width, height, viewBox: `0 0 ${width} ${height}`,
      role: 'img', 'aria-label': config.ariaLabel || 'line figure',
    });

    const yTicks = logY
      ? (() => { const t = []; for (let d = Math.log10(y0); d <= Math.log10(y1) + 1e-9; d++) t.push(10 ** d); return t; })()
      : niceTicks(y0, y1, 6);
    for (const v of yTicks) {
      const y = sy(v);
      root.append(svg('line', { class: 'ax-grid', x1: pad.l, x2: pad.l + iw, y1: y, y2: y }));
      const t = svg('text', { class: 'ax-text', x: pad.l - 8, y: y + 3.5, 'text-anchor': 'end' });
      t.textContent = fmt(v, 3);
      root.append(t);
    }
    for (const v of niceTicks(x0, x1, 7)) {
      const x = sx(v);
      root.append(svg('line', { class: 'ax-grid', x1: x, x2: x, y1: pad.t, y2: pad.t + ih }));
      const t = svg('text', { class: 'ax-text', x, y: pad.t + ih + 16, 'text-anchor': 'middle' });
      t.textContent = fmt(v, 3);
      root.append(t);
    }

    // uncertainty band, drawn under the lines
    if (config.band) {
      const { lower, upper } = config.band;
      const up = upper.map((p) => `${sx(p[0]).toFixed(2)},${sy(p[1]).toFixed(2)}`);
      const dn = [...lower].reverse().map((p) => `${sx(p[0]).toFixed(2)},${sy(p[1]).toFixed(2)}`);
      root.append(svg('polygon', {
        points: [...up, ...dn].join(' '),
        fill: 'var(--teal)', opacity: 0.1,
      }));
    }

    for (const d of config.doseTimes || []) {
      if (d < x0 || d > x1) continue;
      root.append(svg('line', {
        x1: sx(d), x2: sx(d), y1: pad.t, y2: pad.t + ih,
        stroke: 'var(--copper)', 'stroke-width': 1, 'stroke-dasharray': '2 3', opacity: 0.75,
      }));
    }

    root.append(svg('line', { class: 'ax-line', x1: pad.l, x2: pad.l + iw, y1: pad.t + ih, y2: pad.t + ih }));
    root.append(svg('line', { class: 'ax-line', x1: pad.l, x2: pad.l, y1: pad.t, y2: pad.t + ih }));

    const COLOURS = ['var(--teal)', 'var(--copper)', 'var(--blue)', 'var(--amber)'];
    config.series.forEach((s, i) => {
      const pts = s.points.filter((p) => Number.isFinite(p[1]) && (!logY || p[1] > 0));
      if (pts.length < 2) return;
      const d = pts.map((p, j) => `${j ? 'L' : 'M'}${sx(p[0]).toFixed(2)},${sy(p[1]).toFixed(2)}`).join('');
      root.append(svg('path', {
        d, fill: 'none', stroke: s.colour || COLOURS[i % COLOURS.length],
        'stroke-width': s.width || 1.7,
        'stroke-dasharray': s.dashed ? '4 3' : null,
      }));
    });

    // reference observations, plotted as instrument marks
    for (const r of config.references || []) {
      root.append(svg('line', {
        x1: sx(r.x) - 5, x2: sx(r.x) + 5, y1: sy(r.y), y2: sy(r.y),
        stroke: 'var(--copper)', 'stroke-width': 1.6,
      }));
      root.append(svg('line', {
        x1: sx(r.x), x2: sx(r.x), y1: sy(r.y) - 5, y2: sy(r.y) + 5,
        stroke: 'var(--copper)', 'stroke-width': 1.6,
      }));
      if (r.label) {
        const t = svg('text', { class: 'ax-text', x: sx(r.x) + 8, y: sy(r.y) - 6, fill: 'var(--copper)' });
        t.textContent = r.label;
        root.append(t);
      }
    }

    const xt = svg('text', { class: 'ax-title', x: pad.l + iw / 2, y: height - 6, 'text-anchor': 'middle' });
    xt.textContent = config.xLabel || '';
    root.append(xt);
    const yt = svg('text', {
      class: 'ax-title', x: 12, y: pad.t + ih / 2, 'text-anchor': 'middle',
      transform: `rotate(-90 12 ${pad.t + ih / 2})`,
    });
    yt.textContent = config.yLabel || '';
    root.append(yt);

    // Crosshair readout against the first series. `cursorIndex` lets several
    // figures share one cursor, so a time read on one is marked on all.
    const primary = config.series[0];
    const cross = svg('line', { stroke: 'var(--ink-3)', 'stroke-width': 1, 'stroke-dasharray': '2 2', y1: pad.t, y2: pad.t + ih, opacity: 0 });
    const dot = svg('circle', { r: 3.5, fill: 'var(--surface)', stroke: 'var(--teal)', 'stroke-width': 2, opacity: 0 });
    const hit = svg('rect', { x: pad.l, y: pad.t, width: iw, height: ih, fill: 'transparent' });
    root.append(cross, dot, hit);

    const place = (i) => {
      const p = primary.points[i];
      if (!p) { cross.setAttribute('opacity', 0); dot.setAttribute('opacity', 0); return; }
      cross.setAttribute('x1', sx(p[0])); cross.setAttribute('x2', sx(p[0]));
      cross.setAttribute('opacity', 1);
      dot.setAttribute('cx', sx(p[0])); dot.setAttribute('cy', sy(p[1]));
      dot.setAttribute('opacity', 1);
    };
    if (config.cursorIndex != null) place(config.cursorIndex);

    hit.addEventListener('pointermove', (ev) => {
      const box = root.getBoundingClientRect();
      const xv = x0 + ((ev.clientX - box.left - pad.l) / iw) * (x1 - x0);
      let best = 0, bd = Infinity;
      primary.points.forEach((p, i) => { const d = Math.abs(p[0] - xv); if (d < bd) { bd = d; best = i; } });
      place(best);
      const p = primary.points[best];
      const lines = config.series.map((s) => {
        const q = s.points[best];
        return q ? `${s.name}: ${fmt(q[1], 4)}` : null;
      }).filter(Boolean).join('<br>');
      tip(`<strong>${config.xLabel?.split(' ')[0] || 'x'} = ${fmt(p[0], 4)}</strong>${lines}`, ev.clientX, ev.clientY);
      if (config.onCursor) config.onCursor(best);
    });
    hit.addEventListener('pointerleave', () => {
      hideTip();
      if (config.cursorIndex == null) {
        cross.setAttribute('opacity', 0); dot.setAttribute('opacity', 0);
      }
    });

    host.append(root);
  };
  responsive(host, draw);
}

/* ============================================================
   Distribution strip — where a compound sits among all compounds
   ============================================================ */

export function distributionStrip(host, values, highlight, { label: axisLabel, log = true } = {}) {
  const draw = () => {
    const width = Math.max(200, host.clientWidth || 300);
    const height = 44;
    host.textContent = '';
    const pad = { l: 8, r: 8, t: 8, b: 16 };
    const iw = width - pad.l - pad.r;
    const f = (v) => (log ? Math.log10(Math.max(0, v) + 1) : v);
    const max = Math.max(1, ...values.map(f));
    const sx = (v) => pad.l + (f(v) / max) * iw;

    const root = svg('svg', { width, height, viewBox: `0 0 ${width} ${height}`, role: 'img',
      'aria-label': `${axisLabel}: this compound against all registry entries` });
    root.append(svg('line', { x1: pad.l, x2: pad.l + iw, y1: 18, y2: 18, stroke: 'var(--rule)' }));
    for (const v of values) {
      root.append(svg('line', {
        x1: sx(v), x2: sx(v), y1: 12, y2: 24,
        stroke: 'var(--ink-5)', 'stroke-width': 1, opacity: 0.45,
      }));
    }
    if (highlight !== null && highlight !== undefined) {
      root.append(svg('line', {
        x1: sx(highlight), x2: sx(highlight), y1: 8, y2: 28,
        stroke: 'var(--teal)', 'stroke-width': 2,
      }));
    }
    const t = svg('text', { class: 'ax-text', x: pad.l, y: height - 2 });
    t.textContent = axisLabel || '';
    root.append(t);
    host.append(root);
  };
  responsive(host, draw);
}

/* ============================================================
   Horizontal bars with a reference line
   ============================================================ */

export function barFigure(host, { rows, refValue, refLabel, xLabel }) {
  const draw = () => {
    const width = Math.max(300, host.clientWidth || 600);
    host.textContent = '';
    const rowH = 22;
    const pad = { t: 10, r: 52, b: 32, l: 152 };
    const ih = rows.length * rowH;
    const height = ih + pad.t + pad.b;
    const iw = width - pad.l - pad.r;
    const max = Math.max(...rows.map((r) => r.value), refValue || 0) * 1.08;
    const sx = (v) => pad.l + (v / max) * iw;

    const root = svg('svg', { width, height, viewBox: `0 0 ${width} ${height}`, role: 'img',
      'aria-label': `${xLabel} by model` });
    for (const v of niceTicks(0, max, 6)) {
      const x = sx(v);
      root.append(svg('line', { class: 'ax-grid', x1: x, x2: x, y1: pad.t, y2: pad.t + ih }));
      const t = svg('text', { class: 'ax-text', x, y: pad.t + ih + 15, 'text-anchor': 'middle' });
      t.textContent = fmt(v, 2);
      root.append(t);
    }
    rows.forEach((r, i) => {
      const y = pad.t + i * rowH + 3;
      const h = rowH - 9;
      root.append(svg('rect', {
        x: pad.l, y, width: Math.max(1, sx(r.value) - pad.l), height: h,
        fill: r.worse ? 'var(--copper)' : 'var(--teal)',
        opacity: r.highlight ? 1 : 0.72,
      }));
      const lab = svg('text', { class: 'ax-text', x: pad.l - 9, y: y + h - 2, 'text-anchor': 'end' });
      lab.textContent = r.label;
      root.append(lab);
      const val = svg('text', { class: 'ax-text', x: sx(r.value) + 6, y: y + h - 2 });
      val.textContent = r.value.toFixed(3);
      root.append(val);
    });
    if (refValue != null) {
      const x = sx(refValue);
      root.append(svg('line', {
        x1: x, x2: x, y1: pad.t, y2: pad.t + ih,
        stroke: 'var(--ink-3)', 'stroke-width': 1.2, 'stroke-dasharray': '3 3',
      }));
      const t = svg('text', { class: 'ax-text', x, y: pad.t - 1, 'text-anchor': 'middle' });
      t.textContent = refLabel || 'baseline';
      root.append(t);
    }
    const xt = svg('text', { class: 'ax-title', x: pad.l + iw / 2, y: height - 4, 'text-anchor': 'middle' });
    xt.textContent = xLabel;
    root.append(xt);
    host.append(root);
  };
  responsive(host, draw);
}

/* ============================================================
   Predicted versus observed
   ============================================================
   The plot a model has to survive. The identity line is where a perfect
   model would lie; vertical reach shows each prediction's interval, and
   colour marks whether the observation fell inside it.
   ============================================================ */

export function predictedObserved(host, d, { colourBy = 'covered', onHover } = {}) {
  const draw = () => {
    const width = Math.max(320, host.clientWidth || 640);
    const height = Math.min(520, Math.max(340, width * 0.72));
    host.textContent = '';
    const pad = { t: 14, r: 16, b: 46, l: 56 };
    const iw = width - pad.l - pad.r;
    const ih = height - pad.t - pad.b;

    const all = [...d.observed, ...d.predicted];
    const lo = Math.floor(Math.min(...all) * 2) / 2;
    const hi = Math.ceil(Math.max(...all) * 2) / 2;
    const sx = (v) => pad.l + ((v - lo) / (hi - lo)) * iw;
    const sy = (v) => pad.t + ih - ((v - lo) / (hi - lo)) * ih;

    const root = svg('svg', { width, height, viewBox: `0 0 ${width} ${height}`,
      role: 'img', 'aria-label': 'Predicted against observed pKi on held-out molecules' });

    for (const v of niceTicks(lo, hi, 6)) {
      root.append(svg('line', { class: 'ax-grid', x1: sx(v), x2: sx(v), y1: pad.t, y2: pad.t + ih }));
      root.append(svg('line', { class: 'ax-grid', x1: pad.l, x2: pad.l + iw, y1: sy(v), y2: sy(v) }));
      const a = svg('text', { class: 'ax-text', x: sx(v), y: pad.t + ih + 16, 'text-anchor': 'middle' });
      a.textContent = fmt(v, 3); root.append(a);
      const b = svg('text', { class: 'ax-text', x: pad.l - 8, y: sy(v) + 3.5, 'text-anchor': 'end' });
      b.textContent = fmt(v, 3); root.append(b);
    }

    // identity line, and the +/- 1 log-unit corridor
    root.append(svg('line', { x1: sx(lo), y1: sy(lo), x2: sx(hi), y2: sy(hi),
      stroke: 'var(--ink-3)', 'stroke-width': 1.2 }));
    for (const off of [-1, 1]) {
      root.append(svg('line', {
        x1: sx(lo), y1: sy(lo + off), x2: sx(hi), y2: sy(hi + off),
        stroke: 'var(--rule-strong)', 'stroke-width': 1, 'stroke-dasharray': '3 3' }));
    }

    for (let i = 0; i < d.observed.length; i++) {
      const x = sx(d.observed[i]);
      const covered = d.covered[i];
      const colour = colourBy === 'modality'
        ? (d.modality[i] === 'peptide_like' ? 'var(--blue)' : 'var(--teal)')
        : (covered ? 'var(--teal)' : 'var(--copper)');
      const g = svg('g', { class: 'pt', tabindex: '0', role: 'button',
        'aria-label': `${d.molecule_ids[i]}: observed ${fmt(d.observed[i], 3)}, predicted ${fmt(d.predicted[i], 3)}` });
      g.append(svg('line', {
        x1: x, x2: x, y1: sy(d.interval_low[i]), y2: sy(d.interval_high[i]),
        stroke: colour, 'stroke-width': 1, opacity: 0.22 }));
      g.append(svg('circle', { cx: x, cy: sy(d.predicted[i]), r: 3.2,
        fill: colour, 'fill-opacity': 0.72, stroke: colour, 'stroke-width': 0.8 }));
      const show = (ev) => {
        tip(`<strong>${d.molecule_ids[i]}</strong>`
          + `observed ${fmt(d.observed[i], 4)} · predicted ${fmt(d.predicted[i], 4)}<br>`
          + `residual ${fmt(d.residual[i], 3)} · similarity ${fmt(d.similarity[i], 3)}<br>`
          + `${covered ? 'inside' : 'outside'} its interval · ${titleCase(d.modality[i])}`,
          ev.clientX || 0, ev.clientY || 0);
        if (onHover) onHover(i);
      };
      g.addEventListener('pointerenter', show);
      g.addEventListener('pointermove', show);
      g.addEventListener('pointerleave', () => { hideTip(); if (onHover) onHover(null); });
      g.addEventListener('focus', show);
      g.addEventListener('blur', () => { hideTip(); if (onHover) onHover(null); });
      root.append(g);
    }

    const xt = svg('text', { class: 'ax-title', x: pad.l + iw / 2, y: height - 6, 'text-anchor': 'middle' });
    xt.textContent = 'observed pKi'; root.append(xt);
    const yt = svg('text', { class: 'ax-title', x: 12, y: pad.t + ih / 2,
      'text-anchor': 'middle', transform: `rotate(-90 12 ${pad.t + ih / 2})` });
    yt.textContent = 'predicted pKi'; root.append(yt);
    host.append(root);
  };
  responsive(host, draw);
}

/* ============================================================
   Applicability map — error against similarity to training
   ============================================================ */

export function applicabilityMap(host, d) {
  const draw = () => {
    const width = Math.max(300, host.clientWidth || 600);
    const height = 280;
    host.textContent = '';
    const pad = { t: 14, r: 16, b: 46, l: 56 };
    const iw = width - pad.l - pad.r;
    const ih = height - pad.t - pad.b;
    const err = d.residual.map(Math.abs);
    const maxErr = Math.max(...err, 1);
    const sx = (v) => pad.l + v * iw;
    const sy = (v) => pad.t + ih - (v / maxErr) * ih;

    const root = svg('svg', { width, height, viewBox: `0 0 ${width} ${height}`,
      role: 'img', 'aria-label': 'Absolute error against similarity to the training set' });
    for (const v of niceTicks(0, maxErr, 5)) {
      root.append(svg('line', { class: 'ax-grid', x1: pad.l, x2: pad.l + iw, y1: sy(v), y2: sy(v) }));
      const t = svg('text', { class: 'ax-text', x: pad.l - 8, y: sy(v) + 3.5, 'text-anchor': 'end' });
      t.textContent = fmt(v, 2); root.append(t);
    }
    for (const v of [0, 0.2, 0.4, 0.6, 0.8, 1]) {
      root.append(svg('line', { class: 'ax-grid', x1: sx(v), x2: sx(v), y1: pad.t, y2: pad.t + ih }));
      const t = svg('text', { class: 'ax-text', x: sx(v), y: pad.t + ih + 16, 'text-anchor': 'middle' });
      t.textContent = v.toFixed(1); root.append(t);
    }

    // binned mean error, which is the trend the cloud is hiding
    const bins = [[0, 0.4], [0.4, 0.55], [0.55, 0.7], [0.7, 1.01]];
    const means = [];
    for (const [a, b] of bins) {
      const sel = err.filter((_, i) => d.similarity[i] >= a && d.similarity[i] < b);
      if (sel.length >= 3) means.push({ x: (a + Math.min(b, 1)) / 2, y: sel.reduce((p, c) => p + c, 0) / sel.length, n: sel.length });
    }

    for (let i = 0; i < err.length; i++) {
      root.append(svg('circle', { cx: sx(d.similarity[i]), cy: sy(err[i]), r: 2.6,
        fill: d.covered[i] ? 'var(--teal)' : 'var(--copper)', 'fill-opacity': 0.4 }));
    }
    if (means.length > 1) {
      root.append(svg('path', {
        d: means.map((m, i) => `${i ? 'L' : 'M'}${sx(m.x).toFixed(1)},${sy(m.y).toFixed(1)}`).join(''),
        fill: 'none', stroke: 'var(--ink)', 'stroke-width': 1.8 }));
      for (const m of means) {
        root.append(svg('circle', { cx: sx(m.x), cy: sy(m.y), r: 3.6, fill: 'var(--ink)' }));
      }
    }

    const xt = svg('text', { class: 'ax-title', x: pad.l + iw / 2, y: height - 6, 'text-anchor': 'middle' });
    xt.textContent = 'max Tanimoto to training set'; root.append(xt);
    const yt = svg('text', { class: 'ax-title', x: 12, y: pad.t + ih / 2,
      'text-anchor': 'middle', transform: `rotate(-90 12 ${pad.t + ih / 2})` });
    yt.textContent = 'absolute error'; root.append(yt);
    host.append(root);
  };
  responsive(host, draw);
}

/* ============================================================
   Reliability diagram — measured against nominal coverage
   ============================================================ */

export function reliabilityFigure(host, points) {
  const draw = () => {
    const width = Math.max(260, host.clientWidth || 400);
    const height = Math.min(340, width);
    host.textContent = '';
    const pad = { t: 14, r: 16, b: 44, l: 50 };
    const iw = width - pad.l - pad.r;
    const ih = height - pad.t - pad.b;
    const s = (v) => ({ x: pad.l + ((v - 0.4) / 0.6) * iw, y: pad.t + ih - ((v - 0.4) / 0.6) * ih });

    const root = svg('svg', { width, height, viewBox: `0 0 ${width} ${height}`,
      role: 'img', 'aria-label': 'Measured coverage against nominal level' });
    for (const v of [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) {
      const p = s(v);
      root.append(svg('line', { class: 'ax-grid', x1: p.x, x2: p.x, y1: pad.t, y2: pad.t + ih }));
      root.append(svg('line', { class: 'ax-grid', x1: pad.l, x2: pad.l + iw, y1: p.y, y2: p.y }));
      const a = svg('text', { class: 'ax-text', x: p.x, y: pad.t + ih + 16, 'text-anchor': 'middle' });
      a.textContent = v.toFixed(1); root.append(a);
      const b = svg('text', { class: 'ax-text', x: pad.l - 8, y: p.y + 3.5, 'text-anchor': 'end' });
      b.textContent = v.toFixed(1); root.append(b);
    }
    const p0 = s(0.4), p1 = s(1.0);
    root.append(svg('line', { x1: p0.x, y1: p0.y, x2: p1.x, y2: p1.y,
      stroke: 'var(--ink-3)', 'stroke-width': 1.2, 'stroke-dasharray': '4 3' }));

    const path = points.map((pt, i) => {
      const a = s(pt.nominal), b = s(pt.empirical);
      return `${i ? 'L' : 'M'}${a.x.toFixed(1)},${b.y.toFixed(1)}`;
    }).join('');
    root.append(svg('path', { d: path, fill: 'none', stroke: 'var(--teal)', 'stroke-width': 1.8 }));
    for (const pt of points) {
      const a = s(pt.nominal), b = s(pt.empirical);
      const g = svg('g', { class: 'pt', tabindex: '0',
        'aria-label': `nominal ${pt.nominal}, measured ${pt.empirical.toFixed(3)}` });
      g.append(svg('circle', { cx: a.x, cy: b.y, r: 4, fill: 'var(--teal)' }));
      const show = (ev) => tip(`<strong>nominal ${pt.nominal.toFixed(2)}</strong>`
        + `measured ${pt.empirical.toFixed(3)}<br>median width ${pt.median_width.toFixed(2)} log units`,
        ev.clientX || 0, ev.clientY || 0);
      g.addEventListener('pointerenter', show);
      g.addEventListener('pointerleave', hideTip);
      g.addEventListener('focus', show);
      g.addEventListener('blur', hideTip);
      root.append(g);
    }
    const xt = svg('text', { class: 'ax-title', x: pad.l + iw / 2, y: height - 6, 'text-anchor': 'middle' });
    xt.textContent = 'nominal coverage'; root.append(xt);
    const yt = svg('text', { class: 'ax-title', x: 12, y: pad.t + ih / 2,
      'text-anchor': 'middle', transform: `rotate(-90 12 ${pad.t + ih / 2})` });
    yt.textContent = 'measured'; root.append(yt);
    host.append(root);
  };
  responsive(host, draw);
}

/* ============================================================
   Tornado — which parameter a conclusion rests on
   ============================================================ */

export function tornadoFigure(host, { rows, reference, quantity, unit }) {
  const draw = () => {
    const width = Math.max(300, host.clientWidth || 560);
    host.textContent = '';
    const rowH = 30;
    const pad = { t: 24, r: 20, b: 34, l: 86 };
    const ih = rows.length * rowH;
    const height = ih + pad.t + pad.b;
    const iw = width - pad.l - pad.r;

    const vals = rows.flatMap((r) => [r.low, r.high]).filter((v) => v != null);
    const lo = Math.min(reference, ...vals);
    const hi = Math.max(reference, ...vals);
    const padFrac = (hi - lo) * 0.08 || 1;
    const x0 = lo - padFrac, x1 = hi + padFrac;
    const sx = (v) => pad.l + ((v - x0) / (x1 - x0)) * iw;

    const root = svg('svg', { width, height, viewBox: `0 0 ${width} ${height}`,
      role: 'img', 'aria-label': `Parameter sensitivity of ${quantity}` });
    for (const v of niceTicks(x0, x1, 5)) {
      root.append(svg('line', { class: 'ax-grid', x1: sx(v), x2: sx(v), y1: pad.t, y2: pad.t + ih }));
      const t = svg('text', { class: 'ax-text', x: sx(v), y: pad.t + ih + 15, 'text-anchor': 'middle' });
      t.textContent = fmt(v, 3); root.append(t);
    }
    rows.forEach((r, i) => {
      const y = pad.t + i * rowH + rowH / 2;
      if (r.low != null && r.high != null) {
        const a = Math.min(sx(r.low), sx(r.high)), b = Math.max(sx(r.low), sx(r.high));
        root.append(svg('rect', { x: a, y: y - 8, width: Math.max(2, b - a), height: 16,
          fill: 'var(--teal)', 'fill-opacity': 0.26, stroke: 'var(--teal)', 'stroke-width': 1 }));
      }
      const lab = svg('text', { class: 'ax-text', x: pad.l - 10, y: y + 3.5, 'text-anchor': 'end' });
      lab.textContent = r.parameter; root.append(lab);
      const sw = svg('text', { class: 'ax-text', x: pad.l + iw + 4, y: y + 3.5, fill: 'var(--ink-3)' });
      sw.textContent = r.swing != null ? `${(r.swing * 100).toFixed(0)}%` : '—';
      root.append(sw);
    });
    const rx = sx(reference);
    root.append(svg('line', { x1: rx, x2: rx, y1: pad.t - 6, y2: pad.t + ih,
      stroke: 'var(--ink)', 'stroke-width': 1.4 }));
    const rt = svg('text', { class: 'ax-text', x: rx, y: pad.t - 10, 'text-anchor': 'middle', fill: 'var(--ink)' });
    rt.textContent = `${fmt(reference, 4)}${unit ? ` ${unit}` : ''}`;
    root.append(rt);
    host.append(root);
  };
  responsive(host, draw);
}
