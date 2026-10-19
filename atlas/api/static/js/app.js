/* ATLAS — application shell.
   The spine groups modules by biological scale rather than numbering them,
   because the scales are the product's organising idea. A persistent specimen
   bar carries the selected compound between modules, so moving from the Atlas
   to the Studio does not lose the research context. */

import { $, $$, el, clear, fmtCount, titleCase, label, region } from './ui.js';
import { api } from './api.js';
import { MolecularAtlas, renderSpecimen } from './atlas.js';
import { EvidenceLandscape } from './workspaces.js';
import { SimulationStudio } from './studio.js';
import { ModelIntelligence } from './models.js';
import { InterfaceExplorer } from './interfaces.js';

const state = {
  meta: null,
  rows: [],
  selected: null,
  view: 'home',
};

/* ---------------- module definitions ---------------- */

const GLYPHS = {
  home: 'M3 10.5 12 3l9 7.5V21H3z|M9 21v-7h6v7',
  atlas: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18z|M12 3v18|M3 12h18|M5.6 6.3c3 2 9.8 2 12.8 0|M5.6 17.7c3-2 9.8-2 12.8 0',
  systems: 'M6 6h.01|M18 6h.01|M6 18h.01|M18 18h.01|M12 12h.01|M6 6l6 6|M18 6l-6 6|M6 18l6-6|M18 18l-6-6',
  studio: 'M3 18c3-9 6 3 9-4s6 2 9-6|M3 3v18h18',
  interaction: 'M9 12a3 3 0 1 0 0-6 3 3 0 0 0 0 6z|M15 18a3 3 0 1 0 0-6 3 3 0 0 0 0 6z|M11 10.5 13 13.5',
  interfaces: 'M4 3v18|M20 3v18|M4 7c5 0 5 3 10 3M4 17c5 0 5-3 10-3|M15 10h.01|M15 14h.01|M9 12h.01',
  gero: 'M12 21s-7-4.4-7-10a7 7 0 0 1 14 0c0 5.6-7 10-7 10z|M12 11v6|M9.5 13.5 12 11l2.5 2.5',
  models: 'M4 19V9|M10 19V5|M16 19v-7|M22 19H2',
  evidence: 'M5 4h11l3 3v13H5z|M16 4v3h3|M8 12h8|M8 16h5',
};

const MODULES = [
  { group: 'Molecular', items: [
    { id: 'home', name: 'Research Home', glyph: 'home' },
    { id: 'atlas', name: 'Molecular Atlas', glyph: 'atlas' },
  ]},
  { group: 'Biological', items: [
    { id: 'systems', name: 'Systems Explorer', glyph: 'systems' },
    { id: 'gero', name: 'Geroscience Atlas', glyph: 'gero' },
  ]},
  { group: 'Analysis', items: [
    { id: 'studio', name: 'Simulation Studio', glyph: 'studio' },
    { id: 'interaction', name: 'Interaction Lab', glyph: 'interaction' },
    { id: 'interfaces', name: 'Interface Explorer', glyph: 'interfaces' },
    { id: 'models', name: 'Model Intelligence', glyph: 'models' },
  ]},
  { group: 'Evidence', items: [
    { id: 'evidence', name: 'Evidence Library', glyph: 'evidence' },
  ]},
];

function glyph(name) {
  const paths = (GLYPHS[name] || '').split('|');
  const ns = 'http://www.w3.org/2000/svg';
  const s = document.createElementNS(ns, 'svg');
  s.setAttribute('viewBox', '0 0 24 24');
  s.setAttribute('fill', 'none');
  s.setAttribute('stroke', 'currentColor');
  s.setAttribute('stroke-width', '1.5');
  s.setAttribute('stroke-linecap', 'round');
  s.setAttribute('stroke-linejoin', 'round');
  s.setAttribute('aria-hidden', 'true');
  for (const d of paths) {
    const p = document.createElementNS(ns, 'path');
    p.setAttribute('d', d);
    s.append(p);
  }
  return s;
}

/* ---------------- shell ---------------- */

function buildSpine() {
  const spine = $('#spine');
  clear(spine);
  for (const group of MODULES) {
    const g = el('div', { class: 'spine-group' },
      el('p', { class: 'spine-caption', text: group.group }));
    for (const m of group.items) {
      g.append(el('button', {
        class: 'module', type: 'button', 'data-view': m.id,
        'aria-current': state.view === m.id ? 'page' : null,
        title: m.name,
        onclick: () => go(m.id),
      },
        el('span', { class: 'module-glyph' }, glyph(m.glyph)),
        el('span', { class: 'module-name', text: m.name })));
    }
    spine.append(g);
  }
}

function syncSpine() {
  $$('#spine .module').forEach((b) => {
    if (b.dataset.view === state.view) b.setAttribute('aria-current', 'page');
    else b.removeAttribute('aria-current');
  });
}

/** The persistent research context: which compound is in hand. */
function renderSpecimenBar() {
  const bar = $('#specimen-bar');
  clear(bar);
  if (!state.selected) { bar.hidden = true; return; }
  const row = state.rows.find((r) => r.key === state.selected);
  if (!row) { bar.hidden = true; return; }
  bar.hidden = false;
  bar.append(
    el('span', { class: 'label', text: 'in hand' }),
    el('button', {
      class: 'specimen-chip', type: 'button',
      onclick: () => go('atlas', { key: state.selected }),
    }, row.name),
    el('span', { class: 'fine muted num', text: `${fmtCount(row.publications)} pubs` }),
    el('span', { class: 'fine muted', text: titleCase(row.identity_status || '') }),
    el('span', { class: 'spacer' }),
    el('button', {
      class: 'btn btn-ghost fine', type: 'button', text: 'Simulation Studio →',
      onclick: () => go('studio'),
    }),
    el('button', {
      class: 'btn btn-ghost fine', type: 'button', text: 'clear',
      onclick: () => { state.selected = null; renderSpecimenBar(); mount(); },
    }));
}

/* ---------------- routing ---------------- */

const workspaces = {};

function go(view, opts = {}) {
  state.view = view;
  if (opts.key !== undefined) state.selected = opts.key;
  const hash = opts.key ? `#${view}/${opts.key}` : `#${view}`;
  if (location.hash !== hash) location.hash = hash;
  syncSpine();
  renderSpecimenBar();
  mount(opts);
  $('#canvas').scrollTop = 0;
}

function mount(opts = {}) {
  const canvas = $('#canvas');
  $$('.workspace', canvas).forEach((w) => w.classList.remove('is-active'));
  let host = $(`#ws-${state.view}`);
  if (!host) {
    host = el('section', { class: 'workspace', id: `ws-${state.view}` });
    canvas.append(host);
  }
  host.classList.add('is-active');

  switch (state.view) {
    case 'home':
      renderHome(host);
      break;
    case 'atlas':
      if (state.selected && opts.key !== null) {
        renderSpecimen(host, state.selected, state, {
          onBack: () => { const k = state.selected; state.selected = k; go('atlas', { key: null }); },
          onOpenStudio: () => go('studio'),
        });
      } else {
        workspaces.atlas = workspaces.atlas || new MolecularAtlas(host, {
          state, onSelect: (key) => go('atlas', { key }),
        });
        workspaces.atlas.host = host;
        workspaces.atlas.render();
      }
      break;
    case 'studio':
      workspaces.studio = workspaces.studio || new SimulationStudio(host, { state });
      workspaces.studio.host = host;
      workspaces.studio.render();
      break;
    case 'evidence':
      workspaces.evidence = workspaces.evidence || new EvidenceLandscape(host, {
        state, onSelect: (key) => go('atlas', { key }),
      });
      workspaces.evidence.host = host;
      workspaces.evidence.render();
      break;
    case 'models':
      workspaces.models = workspaces.models || new ModelIntelligence(host, { state });
      workspaces.models.host = host;
      workspaces.models.render();
      break;
    case 'interfaces':
      workspaces.interfaces = workspaces.interfaces
        || new InterfaceExplorer(host, { state });
      workspaces.interfaces.host = host;
      workspaces.interfaces.render();
      break;
    default:
      renderUnbuilt(host, state.view);
  }
}

/* ---------------- research home ---------------- */

function renderHome(host) {
  clear(host);
  const m = state.meta || {};
  const rows = state.rows || [];
  const counts = m.permitted_counts || {};
  const dark = rows.filter((r) => !r.publications).length;
  const unresolved = rows.filter((r) => r.identity_status !== 'resolved').length;

  host.append(el('header', { class: 'masthead home-masthead' },
    el('div', { class: 'masthead-eyebrow' }, label('Research Home')),
    el('h1', { class: 't-hero' }, 'What this instrument can', el('br'), 'currently answer'),
    el('p', { class: 'home-lede' },
      'ATLAS reports measured evidence where it exists, computes where a model '
      + 'has been validated for the purpose, and refuses where neither holds. '
      + 'The counts below are the honest state of the registry, not targets.')));

  const stat = (n, of, title, body, tone) => el('article', { class: `stat stat-${tone || 'plain'}` },
    el('p', { class: 'stat-n num' }, String(n), of ? el('span', { class: 'stat-of', text: ` / ${of}` }) : null),
    el('h3', { text: title }),
    el('p', { class: 'fine muted', text: body }));

  host.append(region('Evidence state', {
    caption: 'Measured from the current snapshot. Each figure is a count, not an estimate.',
  }, el('div', { class: 'stat-row' },
    stat(rows.length, null, 'molecular entities', 'resolved against ChEMBL and PubChem', 'teal'),
    stat(unresolved, rows.length, 'identities not fully resolved', 'ambiguous, unresolved, mixture or supplier-defined', 'amber'),
    stat(dark, rows.length, 'with no indexed literature', 'no PubMed record matched the name or its aliases', 'copper'),
    stat(counts.human_pk_simulation ?? 0, rows.length, 'can support a human PK simulation',
      'only where a label supplied source-backed parameters', 'teal'))));

  const gateRow = (kind, n) => {
    const of = rows.length;
    return el('div', { class: 'cap' },
      el('div', { class: 'cap-head' },
        el('span', { text: titleCase(kind) }),
        el('span', { class: 'num fine', text: `${n} / ${of}` })),
      el('div', { class: 'cap-track' },
        el('div', { class: 'cap-fill', style: `width:${(n / of) * 100}%` })));
  };
  const caps = el('div', { class: 'caps' });
  for (const k of m.analysis_kinds || []) {
    caps.append(gateRow(k.value, counts[k.value] ?? 0));
  }
  host.append(region('What the gate permits', {
    caption: 'Three capabilities are permitted for no compound at all: human PK '
      + 'simulation without extracted parameters, quantitative safety prediction, '
      + 'and combination effects. That is a finding about the evidence, not a '
      + 'missing feature.',
  }, caps));

  host.append(region('Begin', {}, el('div', { class: 'entry-row' },
    entry('Molecular Atlas', 'Position every compound by how much is known about it.', () => go('atlas')),
    entry('Evidence Library', 'See which questions the evidence can and cannot answer.', () => go('evidence')),
    entry('Model Intelligence', 'Held-out performance, calibration and applicability.', () => go('models')),
    entry('Simulation Studio', 'Run a mechanistic model and read its diagnostics.', () => go('studio')),
    entry('Interface Explorer', 'Follow one residue across ten solved complexes.', () => go('interfaces')))));
}

function entry(title, body, onclick) {
  return el('button', { class: 'entry', type: 'button', onclick },
    el('h3', { text: title }),
    el('p', { class: 'fine muted', text: body }),
    el('span', { class: 'entry-arrow', 'aria-hidden': 'true', text: '→' }));
}

/* ---------------- honest unbuilt states ---------------- */

const UNBUILT = {
  systems: {
    title: 'Systems Explorer',
    lede: 'A biological network canvas over compounds, receptors, pathways, '
      + 'tissues and aging hallmarks.',
    needs: [
      'A typed knowledge graph. The schema and traversal are not implemented, so '
      + 'there are no edges to draw.',
      'Edge-level provenance, so every relationship can be traced to the study '
      + 'that reported it.',
      'Causal versus associative edge typing, so the canvas cannot imply '
      + 'mechanism where only correlation exists.',
    ],
    why: 'Rendering a force-directed graph now would mean inventing relationships '
      + 'to fill the canvas. The module stays empty until the graph holds real edges.',
  },
  gero: {
    title: 'Geroscience Atlas',
    lede: 'The twelve hallmarks of aging, linked to pathways, biomarkers and the '
      + 'compound registry.',
    needs: [
      'The hallmark framework as structured data, with mechanisms, biomarkers and '
      + 'measurement approaches per hallmark.',
      'Evidence-linked compound-to-hallmark relationships, classified as '
      + 'demonstrated, indirect, hypothesised or contradicted.',
      'Longitudinal datasets, to separate a biomarker change from a functional '
      + 'outcome.',
    ],
    why: 'Drawing compounds onto hallmarks without that evidence would turn '
      + 'pathway overlap into an implied longevity claim, which is the specific '
      + 'error this platform exists to avoid.',
  },
  interaction: {
    title: 'Interaction Laboratory',
    lede: 'Evidence-aware comparison of several compounds’ mechanisms.',
    needs: [
      'Shared-target data across compounds, which requires the knowledge graph.',
      'Direct combination studies. The gate already refuses combination effects '
      + 'for all seven registry formulations because none has been studied as a '
      + 'combination.',
    ],
    why: 'Component evidence does not add. Until combination-specific measurements '
      + 'exist, the honest output is the refusal the gate already returns.',
  },
};

function renderUnbuilt(host, view) {
  const spec = UNBUILT[view] || { title: titleCase(view), lede: '', needs: [], why: '' };
  clear(host);
  host.append(el('header', { class: 'masthead' },
    el('div', { class: 'masthead-eyebrow' }, label(spec.title)),
    el('h1', { text: spec.title }),
    el('p', { text: spec.lede })));
  const ul = el('ul', { class: 'bullets' });
  for (const n of spec.needs) ul.append(el('li', { text: n }));
  host.append(region('Not yet built', {
    caption: 'This module is navigable so the architecture is visible, and empty '
      + 'because the evidence behind it is not in place.',
  },
    el('p', { class: 'label', text: 'what it needs' }), ul,
    el('div', { class: 'note note-copper' },
      el('p', { text: spec.why }))));
}

/* ---------------- command palette ---------------- */

function installPalette() {
  const overlay = $('#palette');
  const input = $('#palette-input');
  const list = $('#palette-list');

  const close = () => { overlay.hidden = true; input.value = ''; };
  const open = () => {
    overlay.hidden = false;
    input.value = '';
    fill('');
    input.focus();
  };

  function fill(q) {
    clear(list);
    const query = q.trim().toLowerCase();
    const mods = MODULES.flatMap((g) => g.items)
      .filter((m) => !query || m.name.toLowerCase().includes(query))
      .map((m) => ({ kind: 'module', label: m.name, hint: 'workspace', run: () => go(m.id) }));
    const comps = (state.rows || [])
      .filter((r) => query && [r.name, r.preferred_name, ...(r.aliases || [])]
        .filter(Boolean).join(' ').toLowerCase().includes(query))
      .slice(0, 8)
      .map((r) => ({ kind: 'compound', label: r.name, hint: titleCase(r.identity_status || ''),
        run: () => go('atlas', { key: r.key }) }));
    const items = [...comps, ...mods].slice(0, 12);
    if (!items.length) {
      list.append(el('li', { class: 'palette-empty', text: 'No match.' }));
      return;
    }
    items.forEach((it, i) => {
      list.append(el('li', {},
        el('button', {
          type: 'button', class: 'palette-item', 'data-idx': i,
          onclick: () => { close(); it.run(); },
        },
          el('span', { class: 'palette-kind', text: it.kind }),
          el('span', { class: 'palette-label', text: it.label }),
          el('span', { class: 'palette-hint fine muted', text: it.hint }))));
    });
  }

  input.addEventListener('input', () => fill(input.value));
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') close();
    if (e.key === 'Enter') { const b = list.querySelector('.palette-item'); if (b) b.click(); }
    if (e.key === 'ArrowDown') { e.preventDefault(); list.querySelector('.palette-item')?.focus(); }
  });
  overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });
  document.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); open(); }
    if (e.key === 'Escape' && !overlay.hidden) close();
  });
  $('#palette-open').addEventListener('click', open);
}

/* ---------------- theme ---------------- */

function installTheme() {
  const btn = $('#theme-toggle');
  const stored = (() => { try { return localStorage.getItem('atlas-theme'); } catch { return null; } })();
  const apply = (t) => {
    document.documentElement.setAttribute('data-theme', t);
    btn.setAttribute('aria-pressed', String(t === 'dark'));
    btn.title = t === 'dark' ? 'Switch to light' : 'Switch to dark';
  };
  // Light is the instrument's defined appearance; dark is opt-in only.
  apply(stored === 'dark' ? 'dark' : 'light');
  btn.addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    apply(next);
    try { localStorage.setItem('atlas-theme', next); } catch { /* private mode */ }
  });
}

/* ---------------- boot ---------------- */

async function boot() {
  installTheme();
  try {
    const [meta, registry] = await Promise.all([api.meta(), api.registry()]);
    state.meta = meta;
    state.rows = registry.rows;
  } catch (err) {
    $('#canvas').append(el('div', { class: 'empty' },
      el('h3', { text: 'Could not reach the ATLAS API' }),
      el('p', { text: String(err.message) })));
    return;
  }

  $('#readout').append(
    el('span', { class: 'readout-item' }, 'snapshot ', el('b', { text: state.meta.snapshot_id || '—' })),
    el('span', { class: 'readout-item' }, el('b', { text: String(state.meta.n_entries) }), ' entities'),
    el('span', { class: 'readout-item' }, 'human PK ',
      el('b', { text: `${state.meta.permitted_counts?.human_pk_simulation ?? 0}/${state.meta.n_entries}` })));
  $('#scope-note').textContent = state.meta.scope_note;

  buildSpine();
  installPalette();
  window.addEventListener('hashchange', fromHash);
  fromHash();
}

function fromHash() {
  const raw = location.hash.slice(1);
  if (!raw) { go('home'); return; }
  const [view, key] = raw.split('/');
  state.view = view || 'home';
  state.selected = key || (view === 'atlas' ? null : state.selected);
  syncSpine();
  renderSpecimenBar();
  mount({ key: key || null });
}

boot();
