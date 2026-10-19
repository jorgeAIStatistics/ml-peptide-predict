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

    // crosshair readout against the first series
    const primary = config.series[0];
    const cross = svg('line', { stroke: 'var(--ink-3)', 'stroke-width': 1, 'stroke-dasharray': '2 2', y1: pad.t, y2: pad.t + ih, opacity: 0 });
    const dot = svg('circle', { r: 3.5, fill: 'var(--surface)', stroke: 'var(--teal)', 'stroke-width': 2, opacity: 0 });
    const hit = svg('rect', { x: pad.l, y: pad.t, width: iw, height: ih, fill: 'transparent' });
    root.append(cross, dot, hit);
    hit.addEventListener('pointermove', (ev) => {
      const box = root.getBoundingClientRect();
      const xv = x0 + ((ev.clientX - box.left - pad.l) / iw) * (x1 - x0);
      let best = 0, bd = Infinity;
      primary.points.forEach((p, i) => { const d = Math.abs(p[0] - xv); if (d < bd) { bd = d; best = i; } });
      const p = primary.points[best];
      cross.setAttribute('x1', sx(p[0])); cross.setAttribute('x2', sx(p[0])); cross.setAttribute('opacity', 1);
      dot.setAttribute('cx', sx(p[0])); dot.setAttribute('cy', sy(p[1])); dot.setAttribute('opacity', 1);
      const lines = config.series.map((s) => {
        const q = s.points[best];
        return q ? `${s.name}: ${fmt(q[1], 4)}` : null;
      }).filter(Boolean).join('<br>');
      tip(`<strong>${config.xLabel?.split(' ')[0] || 'x'} = ${fmt(p[0], 4)}</strong>${lines}`, ev.clientX, ev.clientY);
    });
    hit.addEventListener('pointerleave', () => {
      cross.setAttribute('opacity', 0); dot.setAttribute('opacity', 0); hideTip();
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
