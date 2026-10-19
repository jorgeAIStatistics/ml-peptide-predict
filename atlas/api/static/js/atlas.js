/* ATLAS — Molecular Atlas.
   A molecular exploration environment. The default view is a coverage
   landscape of the whole registry; the sortable register is an advanced
   browsing mode, not the centrepiece. Selecting a compound opens a research
   workspace whose subject is the molecule's identity, not a table row. */

import {
  $, el, add, clear, fmt, fmtCount, fmtPct, titleCase, label, measure,
  masthead, region, figure, statusMark, STATUS_SHORT,
} from './ui.js';
import { coverageLandscape, distributionStrip } from './figures.js';
import { api } from './api.js';

const MODALITY_ORDER = [
  'peptide', 'modified_peptide', 'peptide_hormone_analog', 'protein',
  'small_molecule', 'nucleotide_cofactor', 'metal_coordination_complex',
  'mixture_or_formulation', 'unknown',
];

export class MolecularAtlas {
  constructor(host, { onSelect, state }) {
    this.host = host;
    this.onSelect = onSelect;
    this.state = state;
    this.mode = 'landscape';
    this.query = '';
    this.area = '';
    this.sort = { key: 'publications', dir: 'desc' };
  }

  get rows() {
    let rows = this.state.rows || [];
    if (this.area) rows = rows.filter((r) => r.research_area === this.area);
    if (this.query) {
      const q = this.query.toLowerCase();
      rows = rows.filter((r) =>
        [r.name, r.preferred_name, ...(r.aliases || [])]
          .filter(Boolean).join(' ').toLowerCase().includes(q));
    }
    return rows;
  }

  render() {
    const host = clear(this.host);
    const counts = this.state.meta?.identity_status_counts || {};
    const n = (this.state.rows || []).length;

    host.append(masthead({
      eyebrow: 'Molecular Atlas',
      title: 'The registry as a coverage landscape',
      lede:
        `${n} molecular entities, positioned by how much literature exists for `
        + 'them and how completely their identity resolved against ChEMBL and '
        + 'PubChem. Position is measured, not assigned: a compound sits low and '
        + 'left because the evidence is thin, not because it is unimportant.',
      actions: this.modeSwitch(),
    }));

    host.append(this.filterBar());

    if (this.mode === 'landscape') this.renderLandscape(host);
    else this.renderRegister(host);

    host.append(this.strata());
  }

  modeSwitch() {
    const mk = (id, text) => el('button', {
      type: 'button', 'aria-pressed': String(this.mode === id),
      text, onclick: () => { this.mode = id; this.render(); },
    });
    return el('div', { class: 'segmented' },
      mk('landscape', 'landscape'), mk('register', 'register'));
  }

  filterBar() {
    const search = el('input', {
      type: 'search', id: 'atlas-q', placeholder: 'name or alias…',
      value: this.query, autocomplete: 'off', spellcheck: 'false',
      oninput: (e) => {
        this.query = e.target.value;
        const focus = document.activeElement === e.target;
        this.render();
        if (focus) { const n = $('#atlas-q'); n.focus(); n.setSelectionRange(n.value.length, n.value.length); }
      },
    });
    const areas = el('select', {
      id: 'atlas-area',
      onchange: (e) => { this.area = e.target.value; this.render(); },
    }, el('option', { value: '', text: 'all research areas' }));
    for (const a of this.state.meta?.research_areas || []) {
      areas.append(el('option', { value: a, text: titleCase(a), selected: a === this.area || null }));
    }
    return el('div', { class: 'toolbar' },
      el('div', { class: 'field' }, label('search'), search),
      el('div', { class: 'field' }, label('research area'), areas),
      el('span', { class: 'spacer' }),
      el('p', { class: 'num fine muted', text: `${this.rows.length} of ${(this.state.rows || []).length} shown` }));
  }

  renderLandscape(host) {
    const body = el('div', { class: 'landscape-body' });
    const r = region(null, {}, figure(1,
      'Evidence coverage of the molecular registry',
      body,
      'Horizontal: indexed publications matching the compound name or its '
      + 'aliases, on a log scale. Vertical: fraction of core identity fields '
      + 'resolved. Point area: registered trials that have posted results. '
      + 'Colour: identity status. Select a point to open its research workspace.'));
    r.append(this.legend());
    host.append(r);
    coverageLandscape(body, this.rows, {
      onSelect: (key) => this.onSelect(key),
      selectedKey: this.state.selected,
    });
  }

  legend() {
    const swatch = (colour, text, shape) => el('span', { class: 'lg-item' },
      el('i', { class: `lg-dot ${shape || ''}`, style: `background:${colour}` }), text);
    return el('div', { class: 'legend' },
      swatch('var(--teal)', 'resolved'),
      swatch('var(--amber)', 'ambiguous'),
      swatch('var(--copper)', 'unresolved'),
      swatch('var(--blue)', 'mixture'),
      swatch('var(--stone-ink)', 'supplier-defined'),
      el('span', { class: 'spacer' }),
      el('span', { class: 'lg-item fine muted', text: 'larger point = more trials with posted results' }));
  }

  renderRegister(host) {
    const cols = [
      ['name', 'Compound', ''],
      ['identity_status', 'Identity', ''],
      ['modality', 'Modality', ''],
      ['molecular_weight', 'MW', 'num'],
      ['publications', 'Publications', 'num'],
      ['human_pk_publications', 'Human PK', 'num'],
      ['trials_with_results', 'Trials ⊕', 'num'],
      ['completeness', 'Complete', 'num'],
    ];
    const head = el('tr', {});
    for (const [key, title, cls] of cols) {
      const th = el('th', {
        scope: 'col', class: `${cls} sortable`,
        'aria-sort': this.sort.key === key ? (this.sort.dir === 'asc' ? 'ascending' : 'descending') : null,
        text: title,
        onclick: () => {
          this.sort = { key, dir: this.sort.key === key && this.sort.dir === 'desc' ? 'asc' : 'desc' };
          this.render();
        },
      });
      head.append(th);
    }

    const { key, dir } = this.sort;
    const rows = [...this.rows].sort((a, b) => {
      const av = a[key], bv = b[key];
      let c;
      if (typeof av === 'number' || typeof bv === 'number') c = (av ?? -Infinity) - (bv ?? -Infinity);
      else c = String(av ?? '').localeCompare(String(bv ?? ''));
      return dir === 'asc' ? c : -c;
    });

    const body = el('tbody', {});
    for (const r of rows) {
      body.append(el('tr', {
        tabindex: '0',
        class: r.key === this.state.selected ? 'is-selected' : '',
        onclick: () => this.onSelect(r.key),
        onkeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); this.onSelect(r.key); } },
      },
        el('td', {}, el('span', { class: 'cell-name', text: r.name }),
          r.preferred_name && r.preferred_name.toLowerCase() !== r.name.toLowerCase()
            ? el('span', { class: 'cell-sub', text: r.preferred_name }) : null),
        el('td', {}, statusMark(r.identity_status)),
        el('td', {}, el('span', { class: 'fine muted', text: titleCase(r.modality) })),
        el('td', { class: 'num', text: r.molecular_weight ? fmt(r.molecular_weight, 5) : '—' }),
        el('td', { class: 'num', text: fmtCount(r.publications) }),
        el('td', { class: 'num', text: fmtCount(r.human_pk_publications) }),
        el('td', { class: 'num', text: fmtCount(r.trials_with_results) }),
        el('td', { class: 'num', text: r.completeness != null ? fmtPct(r.completeness) : '—' })));
    }

    host.append(region('Register', {
      caption: 'Advanced browsing. Every column sorts; the landscape above is '
        + 'the same data arranged by evidence rather than by name.',
    }, el('div', { class: 'table-wrap' },
      el('table', { class: 'register' }, el('thead', {}, head), body))));
  }

  /** Research areas as strata, with their evidence profile. */
  strata() {
    const byArea = new Map();
    for (const r of this.state.rows || []) {
      if (!byArea.has(r.research_area)) byArea.set(r.research_area, []);
      byArea.get(r.research_area).push(r);
    }
    const wrap = el('div', { class: 'strata' });
    for (const [area, rows] of [...byArea].sort((a, b) => b[1].length - a[1].length)) {
      const resolved = rows.filter((r) => r.identity_status === 'resolved').length;
      const dark = rows.filter((r) => !r.publications).length;
      const pubs = rows.reduce((a, r) => a + (r.publications || 0), 0);
      const bar = el('div', { class: 'stratum-bar' });
      for (const r of [...rows].sort((a, b) => (b.publications || 0) - (a.publications || 0))) {
        bar.append(el('button', {
          class: `stratum-seg st-seg-${(r.identity_status || '').split('_')[0]}`,
          type: 'button',
          title: `${r.name} — ${fmtCount(r.publications)} publications`,
          'aria-label': `${r.name}, ${fmtCount(r.publications)} publications`,
          onclick: () => this.onSelect(r.key),
        }));
      }
      wrap.append(el('div', { class: 'stratum' },
        el('div', { class: 'stratum-head' },
          el('h3', { text: titleCase(area) }),
          el('span', { class: 'num fine muted', text: `${rows.length} entries` })),
        bar,
        el('p', { class: 'fine muted' },
          `${resolved} resolved · ${dark} with no indexed literature · `,
          el('span', { class: 'num', text: fmtCount(pubs) }), ' publications in total')));
    }
    return region('Research areas', {
      caption: 'Each segment is one compound, ordered by literature volume and '
        + 'coloured by identity status. Select a segment to open that compound.',
    }, wrap);
  }
}

/* ============================================================
   Specimen — the compound research workspace
   ============================================================ */

export async function renderSpecimen(host, key, state, { onBack, onOpenStudio } = {}) {
  clear(host);
  host.append(el('div', { class: 'loading', text: 'Resolving…' }));

  let d;
  try {
    d = await api.compound(key);
  } catch (err) {
    clear(host).append(el('div', { class: 'empty' },
      el('h3', { text: 'Could not load compound' }), el('p', { text: String(err.message) })));
    return;
  }

  // Identity renders immediately. The affinity call fits a model on first use,
  // so it is awaited separately and filled in rather than held in front of the
  // page the researcher asked for.
  const affinityPromise = api.predict({ compound_key: key }).catch(() => null);

  clear(host);
  const f = d.identity_fields || {};
  const status = d.identity_status;

  /* --- specimen header: identity as the subject --------- */
  const formula = f.molecular_formula?.value;
  host.append(el('header', { class: 'specimen' },
    el('div', { class: 'specimen-nav' },
      el('button', {
        class: 'btn btn-ghost', type: 'button', text: '← Atlas',
        onclick: () => onBack && onBack(),
      }),
      el('span', { class: 'fine muted', text: `${d.key} · ${titleCase(d.research_area)}` })),
    el('div', { class: 'specimen-main' },
      el('div', { class: 'specimen-id' },
        el('h1', { class: 't-hero', text: d.name }),
        f.preferred_name
          ? el('p', { class: 'specimen-pref num', text: f.preferred_name.value })
          : null,
        el('div', { class: 'row row-wrap specimen-marks' },
          statusMark(status),
          el('span', { class: 'fine muted', text: titleCase(d.modality) }),
          d.approved ? el('span', { class: 'tag tag-teal', text: 'approved drug' }) : null)),
      formula
        ? el('div', { class: 'specimen-formula' },
            label('molecular formula'),
            el('p', { class: 'formula', html: subscriptFormula(formula) }),
            f.molecular_weight_g_per_mol
              ? el('p', { class: 'fine muted' }, measure(f.molecular_weight_g_per_mol.value, 'g/mol', { sig: 6 }))
              : null)
        : el('div', { class: 'specimen-formula specimen-absent' },
            label('molecular formula'),
            el('p', { class: 'fine muted', text: 'not resolved' })))));

  /* --- identity ledger with per-field provenance -------- */
  const ledger = el('dl', { class: 'ledger' });
  const fieldRow = (name, field, opts = {}) => {
    ledger.append(el('dt', { text: name }));
    if (!field || field.value === null || field.value === undefined) {
      ledger.append(el('dd', { class: 'absent', text: 'not resolved' }));
      return;
    }
    const text = opts.identifier || typeof field.value !== 'number'
      ? String(field.value) : fmt(field.value, 6);
    ledger.append(el('dd', {},
      field.url ? el('a', { href: field.url, target: '_blank', rel: 'noopener', text })
        : document.createTextNode(text),
      el('span', { class: 'prov', text: `${field.source_db}:${field.source_id}` })));
  };
  fieldRow('ChEMBL', f.chembl_id, { identifier: true });
  fieldRow('PubChem CID', f.pubchem_cid, { identifier: true });
  fieldRow('InChIKey', f.inchikey, { identifier: true });
  fieldRow('Molecule type', f.molecule_type, { identifier: true });
  fieldRow('Max clinical phase', f.max_clinical_phase, { identifier: true });
  fieldRow('First approval', f.first_approval_year, { identifier: true });

  host.append(region('Molecular identity', {
    caption: 'Each field carries the database that supplied it. No single source '
      + 'holds a complete identity, so they are attributed individually.',
  }, ledger,
    f.smiles ? el('div', { class: 'structure' },
      label('structure · smiles'),
      el('code', { class: 'smiles', text: f.smiles.value }),
      el('p', { class: 'fine muted', text: `source ${f.smiles.source_db}:${f.smiles.source_id}` })) : null,
    d.identity_note ? el('div', { class: 'note' },
      label('nomenclature'), el('p', { text: d.identity_note })) : null,
    (d.caveats || []).length ? el('div', { class: 'caveats' },
      ...(d.caveats.map((c) => el('div', { class: 'caveat' },
        el('span', { class: 'caveat-mark', 'aria-hidden': 'true', text: '▲' }),
        el('span', { text: c }))))) : null));

  /* --- evidence position among all compounds ------------ */
  const all = state.rows || [];
  const strips = el('div', { class: 'strips' });
  const stripFor = (title, valueKey, log = true) => {
    const holder = el('div', { class: 'strip' });
    strips.append(el('div', {},
      el('div', { class: 'row' },
        label(title), el('span', { class: 'spacer' }),
        el('span', { class: 'num fine', text: fmtCount(d[valueKey]) })),
      holder));
    distributionStrip(holder, all.map((r) => r[valueKey] || 0), d[valueKey] || 0,
      { label: '', log });
  };
  stripFor('publications', 'publications');
  stripFor('human-tagged', 'human_publications');
  stripFor('human pharmacokinetics', 'human_pk_publications');
  stripFor('trials with posted results', 'trials_with_results');

  host.append(region('Evidence position', {
    caption: 'Each strip places this compound among all 46 registry entries on a '
      + 'log scale. Counts are query hits, not compound-specific evidence; '
      + '"n/m" means a measurement was attempted and failed, which is not zero.',
  }, strips));

  /* --- what may be computed ----------------------------- */
  const kinds = state.meta?.analysis_kinds || [];
  const gate = el('div', { class: 'gate' });
  for (const k of kinds) {
    const ok = (d.permitted || []).includes(k.value);
    const reason = (d.refused || {})[k.value];
    if (!ok && !reason) continue;
    gate.append(el('div', { class: `gate-row ${ok ? 'is-ok' : 'is-no'}` },
      el('span', { class: 'gate-mark', 'aria-hidden': 'true', text: ok ? '●' : '○' }),
      el('div', {},
        el('span', { class: 'gate-name', text: k.label }),
        !ok && reason ? el('span', { class: 'gate-reason num', text: reason }) : null)));
  }
  host.append(region('What may be computed', {
    caption: 'The evidence gate decides this per compound. A refusal names the '
      + 'reason rather than hiding the capability.',
    actions: el('button', {
      class: 'btn', type: 'button', text: 'Open in Simulation Studio →',
      onclick: () => onOpenStudio && onOpenStudio(d.key),
    }),
  }, gate));

  /* --- affinity, filled in when the gate has answered --- */
  const affinitySlot = el('div', { class: 'affinity-slot' },
    el('div', { class: 'loading', text: 'Evaluating target engagement…' }));
  host.append(affinitySlot);
  affinityPromise.then((a) => {
    clear(affinitySlot);
    if (a) affinitySlot.append(affinityRegion(a));
  });

  /* --- provenance --------------------------------------- */
  if ((d.sources || []).length) {
    const list = el('ul', { class: 'sources' });
    for (const s of d.sources) {
      list.append(el('li', {},
        el('span', { class: 'src-db', text: s.source_db }),
        s.url ? el('a', { href: s.url, target: '_blank', rel: 'noopener', text: s.title || s.source_id })
          : document.createTextNode(s.title || s.source_id),
        el('span', { class: 'src-meta', text: `retrieved ${(s.retrieved_at || '').slice(0, 19).replace('T', ' ')}` }),
        s.license_note ? el('span', { class: 'src-meta', text: s.license_note }) : null));
    }
    host.append(region('Retrieval provenance', {
      caption: `Evidence snapshot ${d.snapshot_id}.`,
    }, list));
  }
}

function affinityRegion(a) {
  const letter = { A_documented_experimental_engagement: 'A',
    B_prediction_within_validated_domain: 'B',
    C_exploratory_screening_hypothesis: 'C',
    D_unsupported_target_relationship: 'D' }[a.target_relevance] || 'D';

  const head = el('div', { class: `relevance rel-${letter}` },
    el('div', { class: 'row' },
      el('span', { class: `ev ev-${letter}`, text: letter }),
      el('strong', { text: a.target_relevance_label || '' }),
      el('span', { class: 'spacer' }),
      el('span', { class: 'fine muted', text: a.target || '' })),
    el('p', { class: 'fine', text: a.target_rationale || '' }));

  const body = el('div', {});
  if (a.answer_kind === 'measurement') {
    const m = a.measurement;
    body.append(el('div', { class: 'readings' },
      reading('measured pKi', m.pki, '', m.is_discordant ? `range ${fmt(m.pki_min)}–${fmt(m.pki_max)}` : 'median of reported values'),
      reading('Ki', m.ki_nm < 1000 ? m.ki_nm : m.ki_nm / 1000, m.ki_nm < 1000 ? 'nM' : 'µM', 'derived from pKi'),
      reading('independent studies', m.n_documents, '', `${m.n_replicates} activity records`)));
    if (m.is_discordant) {
      body.append(el('div', { class: 'caveat' },
        el('span', { class: 'caveat-mark', text: '▲', 'aria-hidden': 'true' }),
        el('span', { text: `Reported values disagree across a factor of `
          + `${Math.round(10 ** (m.pki_max - m.pki_min))} in affinity. The median `
          + 'summarises disagreeing measurements, not a settled value.' })));
    }
  } else if (a.answer_kind === 'prediction') {
    const p = a.prediction;
    body.append(el('div', { class: 'readings' },
      reading('predicted pKi', p.predicted_pki, '', `interval ${fmt(p.interval_low)}–${fmt(p.interval_high)}`),
      reading('≈ Ki', p.predicted_ki_nm < 1000 ? p.predicted_ki_nm : p.predicted_ki_nm / 1000,
        p.predicted_ki_nm < 1000 ? 'nM' : 'µM', 'derived'),
      reading('similarity to training', a.max_similarity, '', `band ${a.similarity_band}`)));
  } else {
    body.append(el('p', { class: 'fine', text: a.abstention?.explanation || '' }));
    const ul = el('ul', { class: 'bullets' });
    for (const x of a.abstention?.what_would_resolve_it || []) ul.append(el('li', { text: x }));
    body.append(ul);
  }

  return region('Target engagement', {
    caption: 'Only a measurement may become an interaction edge in the knowledge '
      + 'graph. A prediction never does, at any class.',
  }, head, body);
}

function reading(name, value, unit, note) {
  return el('div', { class: 'reading' },
    label(name),
    el('p', { class: 'reading-value' }, measure(value, unit)),
    note ? el('p', { class: 'fine muted', text: note }) : null);
}

/** Render a molecular formula with real subscripts. */
function subscriptFormula(f) {
  return String(f).replace(/([A-Za-z\]\)])(\d+)/g, '$1<sub>$2</sub>');
}
