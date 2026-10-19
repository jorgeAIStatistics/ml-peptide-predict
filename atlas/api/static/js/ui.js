/* ATLAS — DOM and formatting primitives.
   Everything the interface draws goes through these, so notation,
   tabular figures and unit treatment stay consistent. */

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

/** Build an element. Children may be nodes, strings, or null (skipped). */
export function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') n.className = v;
    else if (k === 'text') n.textContent = v;
    else if (k === 'html') n.innerHTML = v;
    else if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  add(n, kids);
  return n;
}

export function add(parent, kids) {
  for (const c of kids.flat(3)) {
    if (c === null || c === undefined || c === false || c === '') continue;
    parent.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return parent;
}

const SVGNS = 'http://www.w3.org/2000/svg';
export function svg(tag, attrs = {}, ...kids) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined) continue;
    if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, String(v));
  }
  for (const c of kids.flat(3)) if (c) n.append(c);
  return n;
}

export function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); return node; }

/* ---------------- numeric notation ---------------- */

/** Significant-figure formatting that stays readable across magnitudes. */
export function fmt(v, sig = 4) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e6 || a < 1e-4) return v.toExponential(2);
  return String(Number(v.toPrecision(sig)));
}

/** Counts. A negative value marks "not measured", which is not zero. */
export function fmtCount(v) {
  if (v === null || v === undefined) return '—';
  if (v < 0) return 'n/m';
  return v.toLocaleString('en-US');
}

export function fmtPct(v, places = 0) {
  return v === null || v === undefined ? '—' : `${(v * 100).toFixed(places)}%`;
}

export const titleCase = (s) => (s || '').replace(/_/g, ' ');

/** A measurement: value and unit set apart typographically. */
export function measure(value, unit, { sig = 4, cls = '' } = {}) {
  return el('span', { class: `num ${cls}` },
    typeof value === 'number' ? fmt(value, sig) : value,
    unit ? el('span', { class: 'unit', text: unit }) : null);
}

export function label(text) { return el('span', { class: 'label', text }); }

/* ---------------- status vocabulary ---------------- */

export const STATUS_SHORT = {
  resolved: 'resolved',
  ambiguous_multiple_candidates: 'ambiguous',
  unresolved_no_database_match: 'unresolved',
  mixture_not_a_single_molecular_entity: 'mixture',
  supplier_defined_identity_unverified: 'supplier-defined',
};

export function statusClass(s) {
  if (!s) return 'st-unresolved';
  if (s.startsWith('resolved')) return 'st-resolved';
  if (s.startsWith('ambiguous')) return 'st-ambiguous';
  if (s.startsWith('mixture')) return 'st-mixture';
  if (s.startsWith('supplier')) return 'st-supplier';
  return 'st-unresolved';
}

export function statusMark(s) {
  return el('span', { class: `mark-status ${statusClass(s)}`, text: STATUS_SHORT[s] || s || 'unknown' });
}

export function evidenceChip(letter) {
  return el('span', { class: `ev ev-${letter}`, text: letter, title: `evidence class ${letter}` });
}

/* ---------------- layout primitives ---------------- */

export function masthead({ eyebrow, title, lede, actions }) {
  return el('header', { class: 'masthead' },
    el('div', { class: 'masthead-eyebrow' }, label(eyebrow || '')),
    el('div', { class: 'row row-wrap' },
      el('h1', { text: title }),
      el('span', { class: 'spacer' }),
      actions || null),
    lede ? el('p', { text: lede }) : null);
}

export function region(title, { caption, actions, id } = {}, ...kids) {
  const head = title
    ? el('div', { class: 'region-head' },
        el('h2', { text: title }),
        actions || null)
    : null;
  return add(el('section', { class: 'region', id: id || null }, head,
    caption ? el('p', { class: 'fig-caption', text: caption }) : null), kids);
}

/** A figure with a number, title and caption, as in a paper. */
export function figure(n, title, body, caption) {
  return el('figure', { class: 'figure' },
    el('div', { class: 'figure-head' },
      el('span', { class: 'fig-mark', text: `Fig. ${String(n).padStart(2, '0')}` }),
      el('p', { class: 'figure-title', text: title })),
    el('div', { class: 'figure-body' }, body),
    caption ? el('figcaption', { class: 'fig-caption', text: caption }) : null);
}

export function emptyState(glyph, title, body) {
  return el('div', { class: 'empty' },
    el('p', { class: 'empty-glyph', 'aria-hidden': 'true', text: glyph }),
    el('h3', { text: title }),
    el('p', { text: body }));
}

/* ---------------- tooltip ---------------- */

let tipEl = null;
export function tip(html, x, y) {
  if (!tipEl) {
    tipEl = el('div', { class: 'tip', role: 'tooltip' });
    document.body.append(tipEl);
  }
  tipEl.innerHTML = html;
  tipEl.hidden = false;
  const r = tipEl.getBoundingClientRect();
  let left = x + 14, top = y + 14;
  if (left + r.width > innerWidth - 10) left = x - r.width - 14;
  if (top + r.height > innerHeight - 10) top = y - r.height - 14;
  tipEl.style.left = `${Math.max(6, left)}px`;
  tipEl.style.top = `${Math.max(6, top)}px`;
}
export function hideTip() { if (tipEl) tipEl.hidden = true; }
