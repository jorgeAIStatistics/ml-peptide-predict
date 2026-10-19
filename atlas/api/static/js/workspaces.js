/* ATLAS — analytical workspaces.
   Each is composed for its own question rather than from a shared dashboard
   template: the Studio is an instrument with three coordinated zones, the
   Evidence Library is a landscape of what is and is not known, and Model
   Intelligence puts validation beside every number it reports. */

import {
  $, el, add, clear, fmt, fmtCount, fmtPct, titleCase, label, measure,
  masthead, region, figure, statusMark,
} from './ui.js';
import { lineFigure, barFigure } from './figures.js';
import { api } from './api.js';

/* ============================================================
   Scientific interpretation — shared across workspaces
   ============================================================ */

export function interpretationBlock(i) {
  const box = el('details', { class: 'interp' },
    el('summary', { class: 'interp-summary' },
      el('span', { class: 'interp-sym num', text: i.symbol }),
      el('span', { class: 'interp-val num', text: i.mathematics?.stated_value
        || (i.value == null ? '—' : fmt(i.value)) }),
      el('span', { class: 'interp-name fine muted', text: i.name })));

  const sect = (title, ...kids) => add(el('div', { class: 'interp-sect' },
    el('p', { class: 'label', text: title })), kids);

  const maths = el('div', {},
    el('p', { class: 'fine', text: i.what_is_measured?.definition || '' }),
    i.mathematics?.transformation
      ? el('code', { class: 'formula', text: i.mathematics.transformation }) : null);
  if ((i.derived || []).length) {
    const dl = el('dl', { class: 'ledger ledger-tight' });
    for (const d of i.derived) {
      dl.append(el('dt', { text: d.label }),
        el('dd', {}, String(d.value), el('span', { class: 'prov', text: d.how })));
    }
    maths.append(dl);
  }
  box.append(sect('Result and mathematics', maths));

  const bio = el('div', {});
  const list = (t, items, cls) => {
    if (!(items || []).length) return;
    bio.append(el('p', { class: 'fine muted interp-sub', text: t }));
    const ul = el('ul', { class: `bullets ${cls || ''}` });
    for (const x of items) ul.append(el('li', { text: x }));
    bio.append(ul);
  };
  list('Establishes', i.biological_meaning?.establishes);
  list('Does not establish', i.biological_meaning?.does_not_establish, 'bullets-negative');
  list('Common misreadings', i.biological_meaning?.common_misreadings, 'bullets-negative');
  box.append(sect('Biological meaning', bio));

  if ((i.why_this_result || []).length) {
    const ul = el('ul', { class: 'bullets' });
    for (const c of i.why_this_result) {
      ul.append(el('li', {}, el('strong', { text: `${titleCase(c.kind)}: ` }), c.detail));
    }
    box.append(sect('Why this result', ul));
  }

  const u = i.uncertainty || {};
  const unc = el('div', {});
  if (u.dominant_driver) unc.append(el('p', { class: 'fine', text: u.dominant_driver }));
  if (u.applicability) unc.append(el('p', { class: 'fine muted', text: u.applicability }));
  const vm = i.validation?.measurements_that_would_test_it || [];
  if (vm.length) {
    unc.append(el('p', { class: 'fine muted interp-sub', text: 'What would test this' }));
    const ul = el('ul', { class: 'bullets' });
    for (const x of vm) ul.append(el('li', { text: x }));
    unc.append(ul);
  }
  const oa = i.validation?.open_assumptions || [];
  if (oa.length) {
    unc.append(el('p', { class: 'fine muted interp-sub', text: 'Open assumptions' }));
    const ul = el('ul', { class: 'bullets bullets-negative' });
    for (const x of oa) ul.append(el('li', { text: x }));
    unc.append(ul);
  }
  box.append(sect('Uncertainty and validation', unc));
  return box;
}

function interpretations(list, heading) {
  if (!(list || []).length) return null;
  const wrap = el('div', { class: 'interps' });
  for (const i of list) wrap.append(interpretationBlock(i));
  return region(heading || 'What these numbers mean', {
    caption: 'Each value is explained from its own definition and from the '
      + 'metadata of whatever produced it. Expand one for the mathematics, what '
      + 'it does and does not establish, and what would test it.',
  }, wrap);
}

/* ============================================================
   Simulation Studio
   ============================================================ */

export class SimulationStudio {
  constructor(host, { state }) {
    this.host = host;
    this.state = state;
    this.models = null;
    this.model = null;
    this.params = {};
    this.dose = 10;
    this.nDoses = 1;
    this.run = null;
    this.logY = false;
    this.pkStore = null;
    this.busy = false;
  }

  async render() {
    const host = clear(this.host);
    host.append(masthead({
      eyebrow: 'Simulation Studio',
      title: 'Mechanistic computation',
      lede: 'Numerical integration of compartmental models. Every trajectory is '
        + 'solver output, and every run reports its mass-balance error and the '
        + 'evidence class it may be described under.',
    }));

    if (!this.models) {
      try {
        this.models = (await api.models()).models;
        this.pkStore = await api.pkParameters();
      } catch (err) {
        host.append(region(null, {}, el('p', { class: 'fine', text: String(err.message) })));
        return;
      }
      this.model = this.models[0];
      this.resetParams();
    }

    const grid = el('div', { class: 'studio' },
      this.configZone(), this.canvasZone());
    host.append(grid);
  }

  resetParams() {
    const DEF = { CL: 1.0, V: 10.0, ka: 1.5, F: 0.8, V1: 5.0, Q: 2.0, V2: 20.0 };
    this.params = {};
    for (const p of this.model.parameters) this.params[p.name] = DEF[p.name] ?? 1;
  }

  configZone() {
    const zone = el('aside', { class: 'studio-config' });

    const sel = el('select', {
      onchange: (e) => {
        this.model = this.models.find((m) => m.name === e.target.value);
        this.resetParams();
        this.render();
      },
    });
    for (const m of this.models) {
      sel.append(el('option', { value: m.name, text: m.display_name, selected: m.name === this.model.name || null }));
    }

    zone.append(el('div', { class: 'config-block' },
      label('model'), sel,
      el('p', { class: 'fine muted', text: this.model.description })));

    const params = el('div', { class: 'params' });
    for (const p of this.model.parameters) {
      params.append(el('div', { class: 'param' },
        el('label', { class: 'param-name num', for: `p-${p.name}`, title: p.description, text: p.name }),
        el('input', {
          type: 'number', id: `p-${p.name}`, step: 'any', value: this.params[p.name],
          'aria-label': `${p.name} in ${p.unit}. ${p.description}`,
          oninput: (e) => { this.params[p.name] = Number(e.target.value); },
        }),
        el('span', { class: 'param-unit num', text: p.unit })));
    }
    zone.append(el('div', { class: 'config-block' },
      el('p', { class: 'label', text: 'parameters' }), params,
      el('p', { class: 'fine muted', text: 'Units are fixed by the model and dimension-checked server-side.' })));

    const dosing = el('div', { class: 'params' },
      el('div', { class: 'param' },
        el('label', { class: 'param-name num', for: 's-dose', text: 'Dose' }),
        el('input', { type: 'number', id: 's-dose', step: 'any', value: this.dose,
          oninput: (e) => { this.dose = Number(e.target.value); } }),
        el('span', { class: 'param-unit num', text: 'mg' })),
      el('div', { class: 'param' },
        el('label', { class: 'param-name num', for: 's-n', text: 'Doses' }),
        el('input', { type: 'number', id: 's-n', min: '1', max: '200', step: '1', value: this.nDoses,
          oninput: (e) => { this.nDoses = Number(e.target.value); } }),
        el('span', { class: 'param-unit num', text: 'n' })));
    zone.append(el('div', { class: 'config-block' },
      el('p', { class: 'label', text: 'dosing' }), dosing));

    // Attribution and, where it exists, the source-backed parameter set.
    const compSel = el('select', {
      onchange: (e) => { this.compound = e.target.value || null; this.render(); },
    }, el('option', { value: '', text: '— exploratory only —' }));
    for (const r of [...(this.state.rows || [])].sort((a, b) => a.name.localeCompare(b.name))) {
      compSel.append(el('option', { value: r.key, text: r.name, selected: r.key === this.compound || null }));
    }
    const attrib = el('div', { class: 'config-block' },
      label('attribute to compound'), compSel,
      el('p', { class: 'fine muted', text: 'Naming a compound asks the evidence gate whether the result may be called its pharmacokinetics.' }));

    const entry = this.compound && this.pkStore?.entries?.[this.compound];
    if (entry && entry.status === 'usable') {
      attrib.append(el('div', { class: 'note note-teal' },
        el('div', { class: 'row' }, el('span', { class: 'src-db', text: 'openfda' }),
          el('strong', { class: 'fine', text: `${entry.brand_name || entry.generic_name} — source-backed` })),
        el('p', { class: 'fine muted', text: entry.validation?.verdict || '' }),
        el('button', {
          class: 'btn', type: 'button', text: 'Load label parameters',
          onclick: () => {
            this.model = this.models.find((m) => m.name === entry.model) || this.model;
            this.params = { ...entry.parameters };
            this.dose = entry.dose_mg;
            this.render();
          },
        })));
    } else if (entry?.blocked_reason) {
      attrib.append(el('p', { class: 'fine muted', text: `No validated parameters: ${entry.blocked_reason}` }));
    }
    zone.append(attrib);

    zone.append(el('div', { class: 'config-actions' },
      el('button', {
        class: 'btn btn-primary', type: 'button', id: 'studio-run',
        text: this.busy ? 'Solving…' : 'Run simulation',
        disabled: this.busy || null,
        onclick: () => this.execute(),
      }),
      el('button', { class: 'btn', type: 'button', text: 'Reset', onclick: () => { this.resetParams(); this.render(); } })));

    if (this.error) zone.append(el('p', { class: 'error-note', role: 'alert', text: this.error }));

    const eqs = el('ol', { class: 'equations' });
    for (const e of this.model.equations) eqs.append(el('li', { class: 'num', text: e }));
    zone.append(el('div', { class: 'config-block' }, el('p', { class: 'label', text: 'equations integrated' }), eqs));

    return zone;
  }

  canvasZone() {
    const zone = el('div', { class: 'studio-canvas' });
    if (!this.run) {
      zone.append(el('div', { class: 'empty empty-tall' },
        el('p', { class: 'empty-glyph', 'aria-hidden': 'true', text: '∫' }),
        el('h3', { text: 'No run yet' }),
        el('p', { class: 'fine muted', text: 'Choose a model and parameters, then run the solver. '
          + 'Results carry their evidence class, mass-balance diagnostic and run identifier.' })));
      return zone;
    }

    const run = this.run;
    const cls = (run.evidence_class || 'D')[0];
    zone.append(el('div', { class: `verdict verdict-${cls}` },
      el('span', { class: `ev ev-${cls}`, text: cls }),
      el('div', {},
        el('p', { class: 'verdict-label', text: run.evidence_label }),
        el('p', { class: 'fine', text: run.interpretation }),
        run.abstention ? el('div', { class: 'abstain' },
          el('p', { class: 'abstain-head num', text: `⊘ ${run.abstention.reason}` }),
          el('p', { class: 'fine', text: run.abstention.explanation })) : null)));

    const body = el('div', { class: 'fig-host' });
    const doseTimes = [];
    for (let i = 0; i < run.n_doses; i++) doseTimes.push(i * run.interval_h);
    const toggle = el('button', {
      class: 'btn btn-ghost fine', type: 'button',
      'aria-pressed': String(this.logY),
      text: this.logY ? 'linear y' : 'log y',
      onclick: () => { this.logY = !this.logY; this.render(); },
    });
    const fig = figure(1, 'Concentration–time profile', body,
      'Solver output. Dashed verticals mark dose events; the crosshair reads '
      + 'the nearest sampled point.');
    fig.querySelector('.figure-head').append(el('span', { class: 'spacer' }), toggle);
    zone.append(el('div', { class: 'studio-figure' }, fig));
    lineFigure(body, {
      series: [{ name: 'C', points: run.times_h.map((t, i) => [t, run.concentration_mg_per_l[i]]) }],
      xLabel: 'time (h)', yLabel: 'concentration (mg/L)',
      logY: this.logY, doseTimes, height: 320,
      ariaLabel: `Concentration against time for ${run.model_display_name}`,
    });

    const n = run.nca;
    const readings = el('div', { class: 'readings' },
      reading('Cmax', n.cmax_mg_per_l * 1000, 'ng/mL', `at ${fmt(n.tmax_h)} h`),
      reading('AUC(0,∞)', (n.analytic_auc_inf_mg_h_per_l ?? 0) * 1000, 'ng·h/mL', 'exact: F·Dose/CL'),
      reading('terminal t½', n.terminal_half_life_h, 'h',
        n.terminal_r_squared != null ? `r² = ${n.terminal_r_squared.toFixed(6)}` : ''),
      reading('mass balance', run.diagnostics.mass_balance_relative_error, 'rel.',
        run.diagnostics.mass_balance_relative_error < 1e-9 ? 'conserved to tolerance' : 'above 1e-9'));
    zone.append(region('Non-compartmental analysis', {
      caption: `Run ${run.run_id} · solver ${run.diagnostics.solver.method} at rtol `
        + `${run.diagnostics.solver.rtol}.`,
    }, readings));

    const interp = interpretations(run.interpretations, 'What each quantity means');
    if (interp) zone.append(interp);
    return zone;
  }

  async execute() {
    this.busy = true; this.error = null; this.render();
    try {
      this.run = await api.simulate({
        model: this.model.name, parameters: this.params,
        dose_mg: this.dose, n_doses: this.nDoses,
        compound_key: this.compound || null,
      });
    } catch (err) {
      this.error = String(err.message);
      this.run = null;
    } finally {
      this.busy = false;
      this.render();
    }
  }
}

function reading(name, value, unit, note) {
  return el('div', { class: 'reading' },
    label(name),
    el('p', { class: 'reading-value' }, measure(value, unit)),
    note ? el('p', { class: 'fine muted', text: note }) : null);
}

/* ============================================================
   Evidence Library — a landscape of what is and is not known
   ============================================================ */

export class EvidenceLandscape {
  constructor(host, { state, onSelect }) {
    this.host = host; this.state = state; this.onSelect = onSelect;
    this.focus = null;
  }

  render() {
    const host = clear(this.host);
    const rows = this.state.rows || [];
    const kinds = this.state.meta?.analysis_kinds || [];

    host.append(masthead({
      eyebrow: 'Evidence Library',
      title: 'What the evidence can and cannot answer',
      lede: 'Missing evidence is reported as a finding, not as an error. Each '
        + 'refusal below names the reason the gate returned and what would '
        + 'resolve it.',
    }));

    // Reason tally — the shape of the registry's ignorance.
    const tally = new Map();
    for (const r of rows) {
      for (const reason of Object.values(r.refused || {})) {
        tally.set(reason, (tally.get(reason) || 0) + 1);
      }
    }
    const bars = el('div', { class: 'reasons' });
    const max = Math.max(...tally.values(), 1);
    for (const [reason, n] of [...tally].sort((a, b) => b[1] - a[1])) {
      bars.append(el('button', {
        class: `reason ${this.focus === reason ? 'is-focus' : ''}`, type: 'button',
        onclick: () => { this.focus = this.focus === reason ? null : reason; this.render(); },
      },
        el('div', { class: 'reason-head' },
          el('span', { class: 'num fine', text: reason }),
          el('span', { class: 'num', text: String(n) })),
        el('div', { class: 'reason-track' },
          el('div', { class: 'reason-fill', style: `width:${(n / max) * 100}%` }))));
    }
    host.append(region('Why analyses are refused', {
      caption: 'Counted across the whole registry. Select a reason to see which '
        + 'compounds it applies to.',
    }, bars));

    // Per-compound capability profile, as a dense legible field.
    const field = el('div', { class: 'cap-field' });
    const shown = this.focus
      ? rows.filter((r) => Object.values(r.refused || {}).includes(this.focus))
      : rows;
    for (const r of [...shown].sort((a, b) => (b.permitted || []).length - (a.permitted || []).length)) {
      const marks = el('div', { class: 'cap-marks' });
      for (const k of kinds) {
        const ok = (r.permitted || []).includes(k.value);
        const reason = (r.refused || {})[k.value];
        marks.append(el('span', {
          class: `cap-mark ${ok ? 'ok' : reason ? 'no' : 'na'}`,
          title: `${k.label}: ${ok ? 'permitted' : reason || 'not applicable'}`,
          'aria-label': `${r.name}. ${k.label}: ${ok ? 'permitted' : reason || 'not applicable'}`,
          tabindex: '0',
        }));
      }
      field.append(el('button', {
        class: 'cap-row', type: 'button',
        onclick: () => this.onSelect(r.key),
      },
        el('span', { class: 'cap-name', text: r.name }),
        statusMark(r.identity_status),
        el('span', { class: 'spacer' }),
        marks));
    }
    host.append(region(this.focus ? `Compounds refused for ${this.focus}` : 'Capability profile', {
      caption: 'One row per compound; one mark per analysis kind, in the order '
        + 'listed in the gate. Filled marks are permitted, hatched are refused.',
      actions: this.focus ? el('button', {
        class: 'btn btn-ghost', type: 'button', text: 'clear filter',
        onclick: () => { this.focus = null; this.render(); },
      }) : null,
    }, field));
  }
}

/* ============================================================
   Model Intelligence
   ============================================================ */

export class ModelIntelligence {
  constructor(host, { state }) {
    this.host = host; this.state = state; this.data = null; this.result = null;
  }

  async render() {
    const host = clear(this.host);
    host.append(masthead({
      eyebrow: 'Model Intelligence',
      title: 'What the model is, and how far it can be trusted',
      lede: 'Held-out performance under splits of increasing honesty, the domain '
        + 'the model may be used on, and the calibration measured for its intervals.',
    }));

    if (!this.data) {
      host.append(el('div', { class: 'loading', text: 'Loading model…' }));
      try { this.data = await api.modelCard(); } catch (err) {
        clear(host).append(el('p', { class: 'fine', text: String(err.message) })); return;
      }
      this.render(); return;
    }

    const { card, report, calibration, peptide_findings: pep } = this.data;
    const hp = card.headline_performance, rs = card.random_split_comparison;

    host.append(region('Model identity', {
      caption: card.quality_tier_meaning,
      actions: el('span', { class: 'tag tag-teal', text: titleCase(card.quality_tier) }),
    }, el('div', { class: 'readings' },
      reading('MAE (scaffold split)', hp.mae_log_units, 'log', 'the honest figure'),
      reading('R² (scaffold split)', hp.r2, '', `Spearman ${hp.spearman.toFixed(3)}`),
      reading('MAE (random split)', rs.mae_log_units, 'log',
        `${fmtPct(rs.fraction_test_scaffolds_seen_in_training)} of test scaffolds already seen`),
      reading('interval coverage', calibration.empirical_coverage, '',
        `nominal ${calibration.nominal_coverage}, previously ${calibration.previous_empirical_coverage}`)),
      el('dl', { class: 'ledger' },
        el('dt', { text: 'Endpoint' }), el('dd', { text: card.endpoint }),
        el('dt', { text: 'Target' }), el('dd', { text: card.target }),
        el('dt', { text: 'Family' }), el('dd', { text: card.model_family }),
        el('dt', { text: 'In domain' }), el('dd', { class: 'prose', text: card.intended_domain }),
        el('dt', { text: 'Out of domain' }), el('dd', { class: 'prose', text: card.out_of_domain }))));

    // Negative transfer, stated as the measured finding it is.
    const nt = card.negative_transfer;
    host.append(region('Measured limits', {}, el('div', { class: 'note note-copper' },
      el('p', {}, el('strong', { text: nt.verdict }), ' ',
        `Best model MAE ${nt.best_model_mae.toFixed(2)} against a mean-predictor `
        + `baseline of ${nt.baseline_mean_mae.toFixed(2)} log units (${nt.experiment}).`)),
      pep ? el('div', { class: 'note note-amber' },
        el('p', {}, el('strong', { text: 'Peptide model not deployed. ' }), pep.decision)) : null));

    // Per-split performance.
    if (report?.available) {
      const bySplit = {};
      for (const r of report.results) (bySplit[r.split] ||= []).push(r);
      let n = 1;
      for (const [split, rows] of Object.entries(bySplit)) {
        const base = rows.find((r) => r.model === 'baseline_mean');
        const meta = report.splits.find((s) => s.kind === split) || {};
        const body = el('div', { class: 'fig-host' });
        host.append(region(null, {}, figure(n++, titleCase(split), body,
          `Train ${meta.n_train} · test ${meta.n_test} · `
          + `${fmtPct(meta.leakage?.fraction_test_scaffolds_seen || 0)} of test scaffolds `
          + 'already seen in training. Bars past the dashed line are worse than '
          + 'predicting the training mean.')));
        barFigure(body, {
          rows: rows.filter((r) => r.model !== 'baseline_mean')
            .sort((a, b) => a.mae - b.mae)
            .map((r) => ({ label: r.model, value: r.mae,
              worse: base ? r.mae >= base.mae : false,
              highlight: r.model === 'fingerprint_knn' })),
          refValue: base?.mae, refLabel: `baseline ${base ? base.mae.toFixed(2) : ''}`,
          xLabel: 'mean absolute error (log units)',
        });
      }
    }

    const interp = interpretations(this.data.interpretations, 'What the performance numbers mean');
    if (interp) host.append(interp);

    host.append(this.console());
  }

  console() {
    const input = el('input', { type: 'text', id: 'mi-smiles', placeholder: 'paste a structure…',
      autocomplete: 'off', spellcheck: 'false' });
    const sel = el('select', { id: 'mi-compound' }, el('option', { value: '', text: '— choose —' }));
    for (const r of [...(this.state.rows || [])].sort((a, b) => a.name.localeCompare(b.name))) {
      sel.append(el('option', { value: r.key, text: r.name }));
    }
    const out = el('div', { id: 'mi-out' });

    const run = async () => {
      const smiles = input.value.trim(), key = sel.value;
      if (!smiles && !key) { clear(out).append(el('p', { class: 'error-note', text: 'Enter a structure or choose an entry.' })); return; }
      clear(out).append(el('div', { class: 'loading', text: 'Evaluating…' }));
      try {
        const r = await api.predict(smiles ? { smiles } : { compound_key: key });
        clear(out);
        const letter = { A_documented_experimental_engagement: 'A',
          B_prediction_within_validated_domain: 'B',
          C_exploratory_screening_hypothesis: 'C',
          D_unsupported_target_relationship: 'D' }[r.target_relevance] || 'D';
        out.append(el('div', { class: `relevance rel-${letter}` },
          el('div', { class: 'row' }, el('span', { class: `ev ev-${letter}`, text: letter }),
            el('strong', { text: r.target_relevance_label || '' })),
          el('p', { class: 'fine', text: r.target_rationale || '' }),
          el('p', { class: 'fine muted', text: r.may_create_graph_edge
            ? 'May become an interaction edge — it is a measurement.'
            : 'Will not become an interaction edge.' })));
        if (r.answer_kind === 'measurement') {
          const m = r.measurement;
          out.append(el('div', { class: 'readings' },
            reading('measured pKi', m.pki, '', `range ${fmt(m.pki_min)}–${fmt(m.pki_max)}`),
            reading('Ki', m.ki_nm, 'nM', 'derived'),
            reading('studies', m.n_documents, '', `${m.n_replicates} records`)));
        } else if (r.answer_kind === 'prediction') {
          const p = r.prediction;
          out.append(el('div', { class: 'readings' },
            reading('predicted pKi', p.predicted_pki, '', `interval ${fmt(p.interval_low)}–${fmt(p.interval_high)}`),
            reading('≈ Ki', p.predicted_ki_nm, 'nM', 'derived'),
            reading('similarity', r.max_similarity, '', `band ${r.similarity_band}`)));
        } else {
          out.append(el('p', { class: 'fine', text: r.abstention?.explanation || '' }));
        }
        const i = interpretations(r.interpretations);
        if (i) out.append(i);
      } catch (err) {
        clear(out).append(el('p', { class: 'error-note', text: String(err.message) }));
      }
    };
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') run(); });
    sel.addEventListener('change', () => { input.value = ''; });

    return region('Prediction console', {
      caption: 'A measurement outranks a prediction; a prediction is returned only '
        + 'where applicability and target relevance both hold.',
    }, el('div', { class: 'toolbar' },
      el('div', { class: 'field', style: 'flex:1;min-width:200px' }, label('smiles'), input),
      el('div', { class: 'field' }, label('or registry entry'), sel),
      el('button', { class: 'btn btn-primary', type: 'button', text: 'Evaluate', onclick: run })),
      out);
  }
}
