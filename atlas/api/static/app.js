/* ============================================================
   ATLAS workbench — no build step, no runtime dependencies.
   Charts are hand-drawn SVG so the figures match the rest of the
   instrument and the app works offline.
   ============================================================ */

const API = '';

const state = {
  meta: null,
  rows: [],
  filtered: [],
  models: [],
  verification: null,
  selected: null,
  sort: { key: 'publications', dir: 'desc' },
  lastRun: null,
  logY: false,
};

/* ---------------- utilities ---------------- */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

async function getJSON(path) {
  const res = await fetch(API + path);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function postJSON(path, body) {
  const res = await fetch(API + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({ error: 'malformed response' }));
  if (!res.ok) throw new Error(data.error || `${res.status}`);
  return data;
}

/** Significant-figure formatting that stays readable across magnitudes. */
function fmt(v, sig = 4) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e6 || a < 1e-4) return v.toExponential(2);
  return String(Number(v.toPrecision(sig)));
}

function fmtInt(v) {
  if (v === null || v === undefined) return '—';
  if (v < 0) return 'n/m';           // -1 marks "not measured", never zero
  return v.toLocaleString('en-US');
}

const titleCase = (s) => (s || '').replace(/_/g, ' ');

function statusClass(status) {
  if (!status) return 'status-unresolved';
  if (status.startsWith('resolved')) return 'status-resolved';
  if (status.startsWith('ambiguous')) return 'status-ambiguous';
  if (status.startsWith('mixture')) return 'status-mixture';
  if (status.startsWith('supplier')) return 'status-supplier';
  return 'status-unresolved';
}

function statusShort(status) {
  if (!status) return 'unknown';
  return {
    resolved: 'resolved',
    ambiguous_multiple_candidates: 'ambiguous',
    unresolved_no_database_match: 'unresolved',
    mixture_not_a_single_molecular_entity: 'mixture',
    supplier_defined_identity_unverified: 'supplier-defined',
  }[status] || status;
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return node;
}

const SVGNS = 'http://www.w3.org/2000/svg';
function svg(tag, attrs = {}) {
  const node = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined) continue;
    node.setAttribute(k, String(v));
  }
  return node;
}

/* ---------------- tooltip ---------------- */

const tip = $('#tooltip');

function showTip(html, x, y) {
  tip.innerHTML = html;
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  let left = x + 14;
  let top = y + 14;
  if (left + r.width > window.innerWidth - 10) left = x - r.width - 14;
  if (top + r.height > window.innerHeight - 10) top = y - r.height - 14;
  tip.style.left = `${Math.max(6, left)}px`;
  tip.style.top = `${Math.max(6, top)}px`;
}
const hideTip = () => { tip.hidden = true; };

/* ---------------- axis ticks ---------------- */

function niceTicks(min, max, target = 6) {
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

function logTicks(min, max) {
  const lo = Math.floor(Math.log10(min));
  const hi = Math.ceil(Math.log10(max));
  const major = [];
  const minor = [];
  for (let d = lo; d <= hi; d++) {
    const base = 10 ** d;
    if (base >= min && base <= max) major.push(base);
    for (let m = 2; m <= 9; m++) {
      const v = base * m;
      if (v >= min && v <= max) minor.push(v);
    }
  }
  return { major, minor };
}

/* ---------------- line chart ---------------- */

/**
 * Draw a multi-series line chart into `host`.
 * Re-renders on container resize so axis text stays at its true size
 * rather than being scaled by a viewBox.
 */
function lineChart(host, config) {
  const draw = () => {
    const width = Math.max(320, host.clientWidth || 640);
    const height = config.height || 330;
    host.textContent = '';

    const pad = { t: 10, r: 16, b: 42, l: 62 };
    const iw = width - pad.l - pad.r;
    const ih = height - pad.t - pad.b;

    const all = config.series.flatMap((s) => s.points);
    if (!all.length) return;

    const xs = all.map((p) => p[0]);
    let ys = all.map((p) => p[1]);
    const x0 = Math.min(...xs);
    const x1 = Math.max(...xs);

    const logY = !!config.logY;
    let y0;
    let y1 = Math.max(...ys);
    if (logY) {
      const pos = ys.filter((v) => v > 0);
      const lowest = pos.length ? Math.min(...pos) : 1e-6;
      // Floor the axis a decade below the smallest positive value so a
      // decaying profile does not run off the bottom of the panel.
      y0 = 10 ** Math.floor(Math.log10(lowest));
      y1 = 10 ** Math.ceil(Math.log10(y1 || 1));
    } else {
      y0 = 0;
      y1 = y1 > 0 ? y1 * 1.06 : 1;
    }

    const sx = (v) => pad.l + (x1 === x0 ? 0 : ((v - x0) / (x1 - x0)) * iw);
    const sy = (v) => {
      if (logY) {
        const c = Math.max(v, y0);
        return pad.t + ih - (Math.log10(c / y0) / Math.log10(y1 / y0)) * ih;
      }
      return pad.t + ih - ((v - y0) / (y1 - y0)) * ih;
    };

    const root = svg('svg', {
      class: 'chart', width, height,
      viewBox: `0 0 ${width} ${height}`,
      role: 'img',
      'aria-label': config.ariaLabel || 'line chart',
    });

    // --- y grid + labels
    if (logY) {
      const { major, minor } = logTicks(y0, y1);
      for (const v of minor) {
        root.append(svg('line', {
          class: 'ax-grid-minor', x1: pad.l, x2: pad.l + iw, y1: sy(v), y2: sy(v),
        }));
      }
      for (const v of major) {
        const y = sy(v);
        root.append(svg('line', { class: 'ax-grid', x1: pad.l, x2: pad.l + iw, y1: y, y2: y }));
        const t = svg('text', { class: 'ax-text', x: pad.l - 8, y: y + 3.5, 'text-anchor': 'end' });
        t.textContent = fmt(v, 2);
        root.append(t);
      }
    } else {
      for (const v of niceTicks(y0, y1, 6)) {
        const y = sy(v);
        root.append(svg('line', { class: 'ax-grid', x1: pad.l, x2: pad.l + iw, y1: y, y2: y }));
        const t = svg('text', { class: 'ax-text', x: pad.l - 8, y: y + 3.5, 'text-anchor': 'end' });
        t.textContent = fmt(v, 3);
        root.append(t);
      }
    }

    // --- x grid + labels
    for (const v of niceTicks(x0, x1, 7)) {
      const x = sx(v);
      root.append(svg('line', { class: 'ax-grid', x1: x, x2: x, y1: pad.t, y2: pad.t + ih }));
      const t = svg('text', { class: 'ax-text', x, y: pad.t + ih + 16, 'text-anchor': 'middle' });
      t.textContent = fmt(v, 3);
      root.append(t);
    }

    // --- dose markers
    for (const d of config.doseTimes || []) {
      if (d < x0 || d > x1) continue;
      root.append(svg('line', {
        class: 'dose-tick', x1: sx(d), x2: sx(d), y1: pad.t, y2: pad.t + ih,
      }));
    }

    // --- axes
    root.append(svg('line', {
      class: 'ax-line', x1: pad.l, x2: pad.l + iw, y1: pad.t + ih, y2: pad.t + ih,
    }));
    root.append(svg('line', {
      class: 'ax-line', x1: pad.l, x2: pad.l, y1: pad.t, y2: pad.t + ih,
    }));

    const xt = svg('text', {
      class: 'ax-title', x: pad.l + iw / 2, y: height - 6, 'text-anchor': 'middle',
    });
    xt.textContent = config.xLabel || '';
    root.append(xt);

    const yt = svg('text', {
      class: 'ax-title', x: 12, y: pad.t + ih / 2,
      'text-anchor': 'middle', transform: `rotate(-90 12 ${pad.t + ih / 2})`,
    });
    yt.textContent = config.yLabel || '';
    root.append(yt);

    // --- series
    config.series.forEach((s, i) => {
      const pts = s.points.filter((p) => Number.isFinite(p[1]) && (!logY || p[1] > 0));
      if (pts.length < 2) return;
      const d = pts.map((p, j) => `${j ? 'L' : 'M'}${sx(p[0]).toFixed(2)},${sy(p[1]).toFixed(2)}`).join('');
      if (i === 0 && config.fillFirst && !logY) {
        const area = `${d}L${sx(pts.at(-1)[0]).toFixed(2)},${sy(y0).toFixed(2)}L${sx(pts[0][0]).toFixed(2)},${sy(y0).toFixed(2)}Z`;
        root.append(svg('path', { class: 'series-area', d: area }));
      }
      root.append(svg('path', { class: `series-line ${s.cls || ''}`, d }));
    });

    // --- hover readout against the primary series
    const primary = config.series[0];
    const cross = svg('line', { class: 'crosshair', y1: pad.t, y2: pad.t + ih, opacity: 0 });
    const dot = svg('circle', { class: 'readout-dot', r: 3.5, opacity: 0 });
    root.append(cross, dot);

    const hit = svg('rect', {
      class: 'chart-hit', x: pad.l, y: pad.t, width: iw, height: ih,
    });
    root.append(hit);

    const move = (ev) => {
      const box = root.getBoundingClientRect();
      const px = ev.clientX - box.left;
      const xv = x0 + ((px - pad.l) / iw) * (x1 - x0);
      let best = 0;
      let bestD = Infinity;
      primary.points.forEach((p, i) => {
        const dd = Math.abs(p[0] - xv);
        if (dd < bestD) { bestD = dd; best = i; }
      });
      const p = primary.points[best];
      cross.setAttribute('x1', sx(p[0]));
      cross.setAttribute('x2', sx(p[0]));
      cross.setAttribute('opacity', 1);
      dot.setAttribute('cx', sx(p[0]));
      dot.setAttribute('cy', sy(p[1]));
      dot.setAttribute('opacity', 1);

      const lines = config.series.map((s) => {
        const q = s.points[best];
        return q ? `${s.name}: ${fmt(q[1], 4)}` : null;
      }).filter(Boolean).join('<br>');
      showTip(`<strong>t = ${fmt(p[0], 4)} h</strong>${lines}`, ev.clientX, ev.clientY);
    };

    hit.addEventListener('pointermove', move);
    hit.addEventListener('pointerleave', () => {
      cross.setAttribute('opacity', 0);
      dot.setAttribute('opacity', 0);
      hideTip();
    });

    host.append(root);
  };

  draw();
  if (host._ro) host._ro.disconnect();
  host._ro = new ResizeObserver(() => draw());
  host._ro.observe(host);
}

/* ---------------- view switching ---------------- */

function setView(name) {
  $$('.view').forEach((v) => {
    const on = v.id === `view-${name}`;
    v.classList.toggle('is-active', on);
    v.hidden = !on;
  });
  $$('.rail-item').forEach((b) => b.classList.toggle('is-active', b.dataset.view === name));
  if (location.hash.slice(1) !== name) location.hash = name;
  if (name === 'verification' && !state.verification) loadVerification();
  if (name === 'gaps') renderGaps();
  window.scrollTo(0, 0);
}

/* ---------------- 01 compound atlas ---------------- */

function applyFilters() {
  const q = $('#atlas-search').value.trim().toLowerCase();
  const area = $('#atlas-area').value;
  const status = $('#atlas-status').value;

  let rows = state.rows.filter((r) => {
    if (area && r.research_area !== area) return false;
    if (status && r.identity_status !== status) return false;
    if (q) {
      const hay = [r.name, r.preferred_name, ...(r.aliases || [])]
        .filter(Boolean).join(' ').toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });

  const { key, dir } = state.sort;
  rows = rows.slice().sort((a, b) => {
    const av = a[key];
    const bv = b[key];
    const an = typeof av === 'number';
    let c;
    if (an || typeof bv === 'number') c = (av ?? -Infinity) - (bv ?? -Infinity);
    else c = String(av ?? '').localeCompare(String(bv ?? ''));
    return dir === 'asc' ? c : -c;
  });

  state.filtered = rows;
  renderAtlasTable();
}

function renderAtlasTable() {
  const tbody = $('#atlas-tbody');
  tbody.textContent = '';
  $('#atlas-count').textContent =
    `${state.filtered.length} of ${state.rows.length} entries`;

  $$('#atlas-table thead th').forEach((th) => {
    if (!th.dataset.sort) return;
    if (th.dataset.sort === state.sort.key) {
      th.setAttribute('aria-sort', state.sort.dir === 'asc' ? 'ascending' : 'descending');
    } else th.removeAttribute('aria-sort');
  });

  for (const r of state.filtered) {
    const nPermit = (r.permitted || []).length;
    const meter = el('span', { class: 'permit-meter' });
    for (let i = 0; i < 5; i++) {
      meter.append(el('i', { class: i < nPermit ? 'on' : '' }));
    }
    meter.append(el('span', { class: 'permit-n', text: `${nPermit}` }));

    const tr = el('tr', {
      class: state.selected === r.key ? 'is-selected' : '',
      tabindex: '0',
      onclick: () => selectCompound(r.key),
      onkeydown: (e) => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectCompound(r.key); }
      },
    },
      el('td', {},
        el('span', { class: 'cell-name', text: r.name }),
        r.preferred_name && r.preferred_name.toLowerCase() !== r.name.toLowerCase()
          ? el('span', { class: 'cell-alias', text: r.preferred_name })
          : null),
      el('td', {},
        el('span', { class: `status ${statusClass(r.identity_status)}`, text: statusShort(r.identity_status) })),
      el('td', {}, el('span', { class: 'modality', text: titleCase(r.modality) })),
      el('td', { class: 'num', text: r.molecular_weight ? fmt(r.molecular_weight, 5) : '—' }),
      el('td', { class: `num ${!r.publications ? 'cell-zero' : ''}`, text: fmtInt(r.publications) }),
      el('td', { class: `num ${!r.human_pk_publications ? 'cell-zero' : ''}`, text: fmtInt(r.human_pk_publications) }),
      el('td', { class: `num ${!r.trials_with_results ? 'cell-zero' : ''}`, text: fmtInt(r.trials_with_results) }),
      el('td', {}, el('span', { class: `density density-${r.density || 'none'}`, text: r.density || 'none' })),
      el('td', {}, meter),
    );
    tbody.append(tr);
  }
}

async function selectCompound(key) {
  state.selected = key;
  renderAtlasTable();
  const host = $('#atlas-inspector');
  host.textContent = '';
  host.append(el('div', { class: 'empty-state' }, el('p', { text: 'Loading…' })));
  try {
    const d = await getJSON(`/api/compound/${encodeURIComponent(key)}`);
    renderInspector(d);
  } catch (err) {
    host.textContent = '';
    host.append(el('div', { class: 'empty-state' },
      el('h2', { text: 'Could not load compound' }),
      el('p', { text: String(err.message) })));
  }
}

function provField(label, field, { link = false, unit = '', identifier = false } = {}) {
  if (!field || field.value === null || field.value === undefined) {
    return [el('dt', { text: label }), el('dd', { class: 'unknown', text: 'not resolved' })];
  }
  const dd = el('dd', {});
  // Identifiers are numbers but not quantities: a PubChem CID must read
  // 71306398, never 7.13e+7.
  const text = (typeof field.value === 'number' && !identifier)
    ? fmt(field.value, 6)
    : String(field.value);
  if (link && field.url) {
    dd.append(el('a', { href: field.url, target: '_blank', rel: 'noopener', text }));
  } else {
    dd.append(document.createTextNode(text + (unit ? ` ${unit}` : '')));
  }
  dd.append(el('span', { class: 'prov', text: `${field.source_db}:${field.source_id}` }));
  return [el('dt', { text: label }), dd];
}

function renderInspector(d) {
  const host = $('#atlas-inspector');
  host.textContent = '';
  const f = d.identity_fields || {};

  host.append(el('div', { class: 'insp-head' },
    el('h2', { class: 'insp-title', text: d.name }),
    el('p', { class: 'insp-pref', text: `${d.key} · ${titleCase(d.modality)}` })));

  // identity
  const kv = el('dl', { class: 'kv' });
  kv.append(...provField('Preferred name', f.preferred_name));
  kv.append(...provField('ChEMBL', f.chembl_id, { link: true }));
  kv.append(...provField('PubChem CID', f.pubchem_cid, { link: true, identifier: true }));
  kv.append(...provField('Mol. weight', f.molecular_weight_g_per_mol, { unit: 'g/mol' }));
  kv.append(...provField('Formula', f.molecular_formula));
  kv.append(...provField('InChIKey', f.inchikey));
  kv.append(...provField('Max phase', f.max_clinical_phase, { identifier: true }));
  kv.append(...provField('First approval', f.first_approval_year, { identifier: true }));
  host.append(el('div', { class: 'insp-section' },
    el('h3', { class: 'insp-heading', text: 'Resolved identity' }),
    kv,
    d.identity_status
      ? el('p', { class: 'metric-note', text: `Identity status: ${statusShort(d.identity_status)}` })
      : null));

  if (f.smiles) {
    host.append(el('div', { class: 'insp-section' },
      el('h3', { class: 'insp-heading', text: 'Structure (SMILES)' }),
      el('code', { class: 'smiles', text: f.smiles.value }),
      el('p', { class: 'metric-note', text: `source ${f.smiles.source_db}:${f.smiles.source_id}` })));
  }

  if (d.identity_note) {
    host.append(el('div', { class: 'insp-section' },
      el('h3', { class: 'insp-heading', text: 'Nomenclature note' }),
      el('p', { class: 'analysis-why', text: d.identity_note })));
  }

  if ((d.caveats || []).length) {
    const box = el('div', { class: 'insp-section' },
      el('h3', { class: 'insp-heading', text: `Identity caveats (${d.caveats.length})` }));
    for (const c of d.caveats) {
      box.append(el('div', { class: 'caveat' },
        el('span', { class: 'caveat-mark', text: '▲', 'aria-hidden': 'true' }),
        el('span', { text: c })));
    }
    host.append(box);
  }

  // literature
  const maxN = Math.max(1, d.publications || 1);
  const bars = el('div', { class: 'lit-bars' });
  const litRows = [
    ['Publications', d.publications, false],
    ['Human-tagged', d.human_publications, false],
    ['Human PK', d.human_pk_publications, false],
    ['Randomized trials', d.randomized_trials, true],
    ['Registered trials', d.registered_trials, true],
    ['Trials w/ results', d.trials_with_results, true],
  ];
  for (const [label, n, dim] of litRows) {
    const pct = n > 0 ? Math.max(1.5, (n / maxN) * 100) : 0;
    bars.append(el('div', { class: 'lit-bar' },
      el('span', { text: label }),
      el('div', { class: 'lit-track' },
        el('div', { class: `lit-fill ${dim ? 'lit-dim' : ''}`, style: `right:${100 - pct}%` })),
      el('span', { class: 'lit-val', text: fmtInt(n) })));
  }
  host.append(el('div', { class: 'insp-section' },
    el('h3', { class: 'insp-heading', text: 'Measured evidence volume' }),
    bars,
    el('p', { class: 'metric-note', text: 'Counts are query hits, not compound-specific evidence. "n/m" means not measured.' })));

  // analyses
  const kinds = state.meta.analysis_kinds;
  const box = el('div', { class: 'insp-section' },
    el('h3', { class: 'insp-heading', text: 'Evidence gate' }));
  for (const k of kinds) {
    const ok = (d.permitted || []).includes(k.value);
    const reason = (d.refused || {})[k.value];
    if (!ok && !reason) continue;
    const row = el('div', { class: 'analysis-row' },
      el('span', { class: `analysis-mark ${ok ? 'yes' : 'no'}`, text: ok ? '●' : '○', 'aria-hidden': 'true' }),
      el('div', {},
        el('span', { class: 'analysis-name', text: k.label }),
        !ok && reason ? el('span', { class: 'analysis-reason', text: reason }) : null));
    box.append(row);
  }
  host.append(box);

  // sources
  if ((d.sources || []).length) {
    const list = el('ul', { class: 'src-list' });
    for (const s of d.sources) {
      list.append(el('li', { class: 'src-item' },
        el('span', { class: 'src-db', text: s.source_db }),
        s.url
          ? el('a', { href: s.url, target: '_blank', rel: 'noopener', text: s.title || s.source_id })
          : document.createTextNode(s.title || s.source_id),
        el('span', { class: 'src-meta', text: `retrieved ${(s.retrieved_at || '').slice(0, 19).replace('T', ' ')}` }),
        s.license_note ? el('span', { class: 'src-meta', text: s.license_note }) : null));
    }
    host.append(el('div', { class: 'insp-section' },
      el('h3', { class: 'insp-heading', text: 'Retrieval provenance' }),
      list,
      el('p', { class: 'metric-note', text: `Snapshot ${d.snapshot_id}` })));
  }

  const studioBtn = el('button', {
    class: 'btn', style: 'width:100%',
    text: 'Open in Model Studio →',
    onclick: () => { $('#studio-compound').value = d.key; setView('studio'); },
  });
  host.append(el('div', { class: 'insp-section' }, studioBtn));
}

/* ---------------- 02 model studio ---------------- */

function renderModelControls() {
  const sel = $('#studio-model');
  sel.textContent = '';
  for (const m of state.models) {
    sel.append(el('option', { value: m.name, text: m.display_name }));
  }
  sel.onchange = () => renderParamInputs();
  renderParamInputs();
}

const DEFAULTS = {
  CL: 1.0, V: 10.0, ka: 1.5, F: 0.8, V1: 5.0, Q: 2.0, V2: 20.0,
};

function currentModel() {
  return state.models.find((m) => m.name === $('#studio-model').value);
}

function renderParamInputs() {
  const m = currentModel();
  if (!m) return;
  $('#studio-model-desc').textContent = m.description;

  const host = $('#studio-params');
  host.textContent = '';
  for (const p of m.parameters) {
    host.append(el('div', { class: 'param-row' },
      el('label', { for: `p-${p.name}`, text: p.name, title: p.description }),
      el('input', {
        type: 'number', id: `p-${p.name}`, step: 'any',
        value: DEFAULTS[p.name] ?? 1,
        'data-param': p.name,
        'aria-label': `${p.name} (${p.unit}) — ${p.description}`,
      }),
      el('span', { class: 'unit', text: p.unit })));
  }

  const eqs = $('#studio-equations');
  eqs.textContent = '';
  for (const e of m.equations) eqs.append(el('li', { text: e }));
}

async function runSimulation() {
  const m = currentModel();
  if (!m) return;
  const btn = $('#studio-run');
  const errBox = $('#studio-error');
  errBox.hidden = true;
  btn.disabled = true;
  btn.textContent = 'Solving…';

  const parameters = {};
  for (const input of $$('#studio-params input')) {
    parameters[input.dataset.param] = Number(input.value);
  }
  const body = {
    model: m.name,
    parameters,
    dose_mg: Number($('#studio-dose').value),
    n_doses: Number($('#studio-ndoses').value),
    compound_key: $('#studio-compound').value || null,
  };
  const iv = $('#studio-interval').value;
  const hz = $('#studio-horizon').value;
  if (iv) body.interval_h = Number(iv);
  if (hz) body.horizon_h = Number(hz);

  try {
    const run = await postJSON('/api/simulate', body);
    state.lastRun = run;
    renderRun(run);
  } catch (err) {
    errBox.textContent = String(err.message);
    errBox.hidden = false;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run simulation';
  }
}

function renderRun(run) {
  const host = $('#studio-canvas');
  host.textContent = '';

  // --- evidence banner
  const clsLetter = (run.evidence_class || 'D')[0];
  const banner = el('div', { class: `evidence-banner cls-${clsLetter}` },
    el('span', { class: 'evidence-class', text: clsLetter, 'aria-hidden': 'true' }),
    el('div', {},
      el('p', { class: 'evidence-label', text: run.evidence_label }),
      el('p', { class: 'evidence-text', text: run.interpretation })));

  if (run.abstention) {
    const ab = el('div', { class: 'abstention' },
      el('p', { class: 'abstention-head' },
        el('span', { text: '⊘', 'aria-hidden': 'true' }),
        el('span', { text: `abstention · ${run.abstention.reason}` })),
      el('p', { text: run.abstention.explanation }));
    if ((run.abstention.what_would_resolve_it || []).length) {
      const ul = el('ul', {});
      for (const r of run.abstention.what_would_resolve_it) ul.append(el('li', { text: r }));
      ab.append(ul);
    }
    banner.querySelector('div').append(ab);
  }
  host.append(banner);

  // --- concentration chart
  const chartPanel = el('div', { class: 'panel chart-panel' });
  const logBtn = el('button', {
    class: `toggle ${state.logY ? 'on' : ''}`,
    text: 'log y',
    'aria-pressed': state.logY ? 'true' : 'false',
    onclick: () => { state.logY = !state.logY; renderRun(run); },
  });
  chartPanel.append(el('div', { class: 'chart-head' },
    el('h2', { class: 'chart-title', text: 'Concentration–time profile' }),
    el('div', { class: 'chart-tools' }, logBtn)));

  const chartBody = el('div', { class: 'chart-body' });
  chartPanel.append(chartBody);

  const doseTimes = [];
  for (let i = 0; i < run.n_doses; i++) doseTimes.push(i * run.interval_h);

  chartPanel.append(el('div', { class: 'chart-legend' },
    el('span', {}, el('i', {}), 'central concentration (mg/L)'),
    run.n_doses > 1 ? el('span', {}, el('i', { class: 'dash' }), 'dose event') : null));
  host.append(chartPanel);

  lineChart(chartBody, {
    series: [{
      name: 'C',
      points: run.times_h.map((t, i) => [t, run.concentration_mg_per_l[i]]),
    }],
    xLabel: 'time (h)',
    yLabel: 'concentration (mg/L)',
    logY: state.logY,
    doseTimes,
    fillFirst: true,
    ariaLabel: `Concentration against time for ${run.model_display_name}`,
  });

  // --- NCA metrics
  const n = run.nca;
  const metrics = el('div', { class: 'metric-wrap' });
  const addMetric = (label, value, unit, note, noteCls) => {
    metrics.append(el('div', { class: 'metric' },
      el('p', { class: 'metric-label', text: label }),
      el('p', { class: 'metric-value' },
        document.createTextNode(value),
        unit ? el('span', { class: 'u', text: unit }) : null),
      note ? el('p', { class: `metric-note ${noteCls || ''}`, text: note }) : null));
  };
  addMetric('Cmax (observed)', fmt(n.cmax_mg_per_l), 'mg/L', `at t = ${fmt(n.tmax_h)} h`);
  addMetric('AUC to last sample', fmt(n.auc_last_mg_h_per_l), 'mg·h/L', 'linear-up / log-down');
  addMetric('AUC extrapolated to ∞',
    n.auc_inf_mg_h_per_l === null ? '—' : fmt(n.auc_inf_mg_h_per_l), 'mg·h/L',
    n.extrapolated_fraction === null
      ? 'terminal slope not estimable'
      : `${(n.extrapolated_fraction * 100).toFixed(1)}% extrapolated — ${n.auc_inf_is_reliable ? 'within the 20% convention' : 'exceeds 20%, extrapolation-dominated'}`,
    n.auc_inf_is_reliable ? 'good' : 'warn');
  addMetric('Analytic AUC(0,∞)',
    n.analytic_auc_inf_mg_h_per_l === null ? '—' : fmt(n.analytic_auc_inf_mg_h_per_l),
    'mg·h/L', 'exact: F·Dose/CL', 'good');
  addMetric('Terminal half-life',
    n.terminal_half_life_h === null ? '—' : fmt(n.terminal_half_life_h), 'h',
    n.terminal_reason
      ? n.terminal_reason
      : `fitted on ${n.terminal_n_points} points, r² = ${n.terminal_r_squared === null ? '—' : n.terminal_r_squared.toFixed(6)}`);
  addMetric('Mass-balance error',
    run.diagnostics.mass_balance_relative_error.toExponential(2), 'rel.',
    run.diagnostics.mass_balance_relative_error < 1e-9
      ? 'conserved to solver tolerance'
      : 'above 1e-9 — inspect the model',
    run.diagnostics.mass_balance_relative_error < 1e-9 ? 'good' : 'warn');

  const ncaPanel = el('div', { class: 'panel' },
    el('div', { class: 'chart-head' },
      el('h2', { class: 'chart-title', text: 'Non-compartmental analysis of the simulated profile' })),
    metrics);

  if (run.diagnostics.is_flip_flop) {
    ncaPanel.append(el('p', {
      class: 'control-error',
      style: 'margin:14px 18px',
      text: 'Flip-flop kinetics: absorption is slower than elimination, so the '
        + 'terminal slope reflects absorption. Reading clearance from this '
        + 'terminal phase would give the wrong answer.',
    }));
  }

  ncaPanel.append(el('div', { class: 'run-foot' },
    el('span', { text: `run ${run.run_id}` }),
    el('span', { text: `model ${run.model}` }),
    el('span', { text: `solver ${run.diagnostics.solver.method} rtol=${run.diagnostics.solver.rtol}` }),
    el('span', { text: `t½ = ${fmt(run.diagnostics.terminal_half_life_h)} h` })));
  host.append(ncaPanel);

  // --- compartment / mass-balance chart
  const stateNames = Object.keys(run.states);
  const total = run.times_h.map((_, i) =>
    stateNames.reduce((acc, k) => acc + run.states[k][i], 0));
  const clsFor = ['', 's2', 's3', ''];
  const series = stateNames.map((k, i) => ({
    name: k.replace('_mg', ''),
    cls: clsFor[i % clsFor.length],
    points: run.times_h.map((t, j) => [t, run.states[k][j]]),
  }));
  series.push({ name: 'total', cls: 's3', points: run.times_h.map((t, i) => [t, total[i]]) });

  const massPanel = el('div', { class: 'panel chart-panel' },
    el('div', { class: 'chart-head' },
      el('h2', { class: 'chart-title', text: 'Compartment amounts and conservation' })));
  const massBody = el('div', { class: 'chart-body' });
  massPanel.append(massBody);
  massPanel.append(el('div', { class: 'chart-legend' },
    ...series.map((s, i) => el('span', {},
      el('i', { class: s.cls }), s.name)),
    el('span', { text: '— the flat total is the conservation check' })));
  host.append(massPanel);

  lineChart(massBody, {
    series,
    xLabel: 'time (h)',
    yLabel: 'amount (mg)',
    height: 260,
    doseTimes,
    ariaLabel: 'Drug amount in each compartment over time, with the conserved total',
  });
}

/* ---------------- 03 evidence & gaps ---------------- */

function renderGaps() {
  const kinds = state.meta.analysis_kinds;
  const rows = state.rows;

  const summary = $('#gap-summary');
  summary.textContent = '';
  const stat = (n, of, label) => {
    summary.append(el('div', { class: 'gap-stat' },
      el('p', { class: 'gap-stat-n' },
        document.createTextNode(String(n)),
        el('span', { class: 'of', text: ` / ${of}` })),
      el('p', { class: 'gap-stat-l', text: label })));
  };
  stat(rows.filter((r) => (r.publications || 0) === 0).length, rows.length,
    'entries with no indexed publication matching their name or aliases');
  stat(rows.filter((r) => (r.trials_with_results || 0) === 0).length, rows.length,
    'entries with no registered trial that has posted results');
  stat(rows.filter((r) => r.identity_status !== 'resolved').length, rows.length,
    'entries whose molecular identity is not fully resolved');
  stat(rows.filter((r) => (r.permitted || []).includes('human_pk_simulation')).length,
    rows.length,
    'entries with source-backed parameters for a human PK simulation');

  const head = $('#gap-head');
  head.textContent = '';
  head.append(el('th', { scope: 'col', text: 'Compound' }));
  for (const k of kinds) {
    head.append(el('th', { scope: 'col', class: 'rot', text: k.label }));
  }

  const tbody = $('#gap-tbody');
  tbody.textContent = '';
  for (const r of rows) {
    const tr = el('tr', {},
      el('td', {},
        el('span', { class: 'cell-name', text: r.name }),
        el('span', { class: 'cell-alias', text: statusShort(r.identity_status) })));
    for (const k of kinds) {
      const ok = (r.permitted || []).includes(k.value);
      const reason = (r.refused || {})[k.value];
      let cls = 'cell-na';
      let label = `${k.label}: not applicable`;
      if (ok) { cls = 'cell-yes'; label = `${k.label}: permitted`; }
      else if (reason) { cls = 'cell-no'; label = `${k.label}: refused — ${reason}`; }

      const btn = el('button', {
        class: 'cell-btn', type: 'button', 'aria-label': `${r.name}. ${label}`,
        onmouseenter: (e) => showTip(`<strong>${r.name}</strong>${label}`, e.clientX, e.clientY),
        onmousemove: (e) => showTip(`<strong>${r.name}</strong>${label}`, e.clientX, e.clientY),
        onmouseleave: hideTip,
        onfocus: (e) => {
          const b = e.target.getBoundingClientRect();
          showTip(`<strong>${r.name}</strong>${label}`, b.right, b.top);
        },
        onblur: hideTip,
      }, el('i', { class: `cell-dot ${cls}` }));
      tr.append(el('td', { class: 'cell' }, btn));
    }
    tbody.append(tr);
  }
}

/* ---------------- 04 verification ---------------- */

async function loadVerification() {
  const body = $('#verify-body');
  body.textContent = '';
  body.append(el('div', { class: 'empty-state' }, el('p', { text: 'Running verification suite…' })));
  try {
    const v = await getJSON('/api/verification');
    state.verification = v;
    renderVerification(v);
  } catch (err) {
    body.textContent = '';
    body.append(el('div', { class: 'empty-state' }, el('p', { text: String(err.message) })));
  }
}

function renderVerification(v) {
  const banner = $('#verify-banner');
  banner.textContent = '';
  banner.classList.toggle('is-bad', !v.all_passed);
  banner.append(
    el('span', { class: 'verify-score', text: `${v.n_passed}/${v.n_checks}` }),
    el('p', { class: 'verify-note', text: v.note }));

  const body = $('#verify-body');
  body.textContent = '';
  for (const r of v.reports) {
    const block = el('div', { class: 'verify-model' },
      el('div', { class: 'verify-model-head' },
        el('span', { class: r.passed ? 'verify-pass' : 'verify-fail', text: r.passed ? '✓' : '✗' }),
        el('span', { text: r.model }),
        el('span', { style: 'margin-left:auto', text: `${r.n_passed}/${r.n_checks}` })));
    for (const c of r.checks) {
      block.append(el('div', { class: 'check-row' },
        el('span', { class: `check-mark ${c.passed ? 'ok' : 'bad'}`, text: c.passed ? '✓' : '✗' }),
        el('span', { class: 'check-name', text: c.name }),
        el('span', {
          class: 'check-metric',
          text: c.metric === null ? '—' : `${c.metric.toExponential(2)} ≤ ${c.tolerance.toExponential(0)}`,
        }),
        el('span', { class: 'check-detail', text: c.detail })));
    }
    body.append(block);
  }
}

/* ---------------- resizer ---------------- */

function installResizer() {
  const split = $('#view-atlas .split');
  const handle = el('button', {
    class: 'resizer', type: 'button',
    'aria-label': 'Resize inspector panel. Use left and right arrow keys.',
  });
  split.insertBefore(handle, $('#atlas-inspector'));

  const setWidth = (px) => {
    const w = Math.min(640, Math.max(280, px));
    document.documentElement.style.setProperty('--side-w', `${w}px`);
  };

  let dragging = false;
  handle.addEventListener('pointerdown', (e) => {
    dragging = true;
    handle.setPointerCapture(e.pointerId);
  });
  handle.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    setWidth(split.getBoundingClientRect().right - e.clientX);
  });
  handle.addEventListener('pointerup', () => { dragging = false; });
  handle.addEventListener('keydown', (e) => {
    const cur = parseInt(getComputedStyle(document.documentElement)
      .getPropertyValue('--side-w'), 10) || 400;
    if (e.key === 'ArrowLeft') { setWidth(cur + 24); e.preventDefault(); }
    if (e.key === 'ArrowRight') { setWidth(cur - 24); e.preventDefault(); }
  });
}

/* ---------------- boot ---------------- */

function renderRailLegend() {
  const list = $('#legend-identity');
  list.textContent = '';
  for (const [status, n] of Object.entries(state.meta.identity_status_counts)) {
    list.append(el('li', {},
      el('span', { class: `status ${statusClass(status)}`, text: statusShort(status) }),
      el('span', { class: 'legend-count', text: String(n) })));
  }
}

function populateSelects() {
  const area = $('#atlas-area');
  for (const a of state.meta.research_areas) {
    area.append(el('option', { value: a, text: titleCase(a) }));
  }
  const status = $('#atlas-status');
  for (const s of Object.keys(state.meta.identity_status_counts)) {
    status.append(el('option', { value: s, text: statusShort(s) }));
  }
  const comp = $('#studio-compound');
  for (const r of [...state.rows].sort((a, b) => a.name.localeCompare(b.name))) {
    comp.append(el('option', { value: r.key, text: r.name }));
  }
}

async function boot() {
  try {
    const [meta, registry, models] = await Promise.all([
      getJSON('/api/meta'),
      getJSON('/api/registry'),
      getJSON('/api/models'),
    ]);
    state.meta = meta;
    state.rows = registry.rows;
    state.models = models.models;

    $('#chip-snapshot').textContent = `snapshot ${meta.snapshot_id || '—'}`;
    $('#chip-entries').textContent = `${meta.n_entries} entries`;
    $('#scope-note').textContent = meta.scope_note;

    const hpk = meta.permitted_counts.human_pk_simulation;
    $('#chip-verify').textContent = `human PK permitted ${hpk}/${meta.n_entries}`;
    $('#chip-verify').classList.toggle('is-bad', hpk === 0);

    renderRailLegend();
    populateSelects();
    renderModelControls();
    installResizer();
    applyFilters();

    // wiring
    $('#atlas-search').addEventListener('input', applyFilters);
    $('#atlas-area').addEventListener('change', applyFilters);
    $('#atlas-status').addEventListener('change', applyFilters);
    $$('#atlas-table thead th.sortable').forEach((th) => {
      th.addEventListener('click', () => {
        const key = th.dataset.sort;
        state.sort = {
          key,
          dir: state.sort.key === key && state.sort.dir === 'desc' ? 'asc' : 'desc',
        };
        applyFilters();
      });
    });
    $$('.rail-item').forEach((b) => b.addEventListener('click', () => setView(b.dataset.view)));
    $('#studio-run').addEventListener('click', runSimulation);
    $('#studio-reset').addEventListener('click', () => {
      renderParamInputs();
      $('#studio-dose').value = 10;
      $('#studio-ndoses').value = 1;
      $('#studio-interval').value = '';
      $('#studio-horizon').value = '';
      $('#studio-error').hidden = true;
    });
    window.addEventListener('hashchange', () => {
      const v = location.hash.slice(1);
      if (v) setView(v);
    });

    const initial = location.hash.slice(1);
    if (initial) setView(initial);
  } catch (err) {
    document.body.prepend(el('div', {
      class: 'scope-strip',
      style: 'background:var(--oxide-soft);color:var(--oxide)',
      text: `Could not reach the ATLAS API: ${err.message}`,
    }));
  }
}

boot();
