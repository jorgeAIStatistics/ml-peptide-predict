/* ATLAS — Simulation Studio.

   Three coordinated zones: an instrument-style configuration column, a
   simulation canvas, and an interpretation inspector. Changing a parameter
   re-solves after a short settle, and a run in flight is cancelled when it is
   superseded rather than being allowed to land out of order.

   The figures are linked: a time cursor moved on one is drawn on all of them,
   because the question "what is happening in the peripheral compartment at the
   moment of peak plasma concentration" should not require reading two charts
   and holding a number in your head. */

import {
  $, el, add, clear, fmt, titleCase, label, measure, masthead, region, figure,
} from './ui.js';
import { lineFigure, tornadoFigure } from './figures.js';
import { interpretations } from './interpret.js';
import { api } from './api.js';

const SETTLE_MS = 260;

const DEFAULTS = { CL: 1.0, V: 10.0, ka: 1.5, F: 0.8, V1: 5.0, Q: 2.0, V2: 20.0 };

export class SimulationStudio {
  constructor(host, { state }) {
    this.host = host;
    this.state = state;
    this.models = null;
    this.model = null;
    this.params = {};
    this.dose = 10;
    this.nDoses = 1;
    this.compound = null;
    this.run = null;
    this.baseline = null;      // a held scenario to compare against
    this.sensitivity = null;
    this.logY = false;
    this.live = true;
    this.cursor = null;        // shared time index across linked figures
    this.busy = false;
    this.error = null;
    this.pkStore = null;
    this._timer = null;
    this._abort = null;
    this._figures = [];
  }

  /* ---------------- lifecycle ---------------- */

  async render() {
    if (!this.models) {
      clear(this.host).append(el('div', { class: 'loading', text: 'Loading models…' }));
      try {
        this.models = (await api.models()).models;
        this.pkStore = await api.pkParameters();
      } catch (err) {
        clear(this.host).append(el('p', { class: 'error-note', text: String(err.message) }));
        return;
      }
      this.model = this.models[0];
      this.resetParams();
      this.solve();
    }
    this.paint();
  }

  resetParams() {
    this.params = {};
    for (const p of this.model.parameters) this.params[p.name] = DEFAULTS[p.name] ?? 1;
  }

  /** Re-solve after the controls settle, cancelling anything in flight. */
  schedule() {
    if (!this.live) return;
    clearTimeout(this._timer);
    this._timer = setTimeout(() => this.solve(), SETTLE_MS);
  }

  async solve() {
    if (this._abort) this._abort.abort();
    this._abort = new AbortController();
    const signal = this._abort.signal;
    this.busy = true;
    this.error = null;
    this.paintStatus();

    const body = {
      model: this.model.name,
      parameters: this.params,
      dose_mg: this.dose,
      n_doses: this.nDoses,
      compound_key: this.compound || null,
    };
    try {
      const [run, sens] = await Promise.all([
        api.simulate(body, { signal }),
        api.sensitivity({ ...body, fraction: 0.2 }, { signal }).catch(() => null),
      ]);
      if (signal.aborted) return;
      this.run = run;
      this.sensitivity = sens;
      this.error = null;
    } catch (err) {
      if (err.name === 'AbortError' || signal.aborted) return;
      this.error = String(err.message);
    } finally {
      if (!signal.aborted) {
        this.busy = false;
        this.paint();
      }
    }
  }

  /* ---------------- composition ---------------- */

  paint() {
    const host = clear(this.host);
    this._figures = [];
    host.append(masthead({
      eyebrow: 'Simulation Studio',
      title: 'Mechanistic computation',
      lede: 'Parameters on the left drive the figures on the right. Changing one '
        + 're-solves the system; a run that is superseded is cancelled rather '
        + 'than allowed to land late.',
    }));
    host.append(el('div', { class: 'studio' }, this.configZone(), this.canvasZone()));
  }

  /** Only the busy indicator, so a live update does not rebuild the controls. */
  paintStatus() {
    const dot = $('#studio-status');
    if (dot) {
      dot.className = `solve-status ${this.busy ? 'is-busy' : ''}`;
      dot.title = this.busy ? 'solving' : 'settled';
    }
  }

  configZone() {
    const zone = el('aside', { class: 'studio-config' });

    const sel = el('select', {
      onchange: (e) => {
        this.model = this.models.find((m) => m.name === e.target.value);
        this.resetParams();
        this.baseline = null;
        this.solve();
      },
    });
    for (const m of this.models) {
      sel.append(el('option', { value: m.name, text: m.display_name,
        selected: m.name === this.model.name || null }));
    }
    zone.append(el('div', { class: 'config-block' },
      el('div', { class: 'row' }, label('model'),
        el('span', { class: 'spacer' }),
        el('span', { class: 'solve-status', id: 'studio-status', title: 'settled' })),
      sel,
      el('p', { class: 'fine muted', text: this.model.description })));

    // Parameters: a number field and a slider on the same value, because
    // reading a value and sweeping it are different actions.
    const params = el('div', { class: 'params' });
    for (const p of this.model.parameters) {
      const value = this.params[p.name];
      const max = p.upper != null ? p.upper : Math.max(value * 4, 1);
      const min = p.lower != null && p.lower > 0 ? p.lower : Math.min(value / 4, value);
      const number = el('input', {
        type: 'number', id: `p-${p.name}`, step: 'any', value,
        'aria-label': `${p.name} in ${p.unit}. ${p.description}`,
        oninput: (e) => {
          this.params[p.name] = Number(e.target.value);
          const s = $(`#s-${p.name}`);
          if (s) s.value = e.target.value;
          this.schedule();
        },
      });
      const slider = el('input', {
        type: 'range', id: `s-${p.name}`, min, max,
        step: (max - min) / 200, value,
        'aria-label': `${p.name} slider`,
        oninput: (e) => {
          this.params[p.name] = Number(e.target.value);
          number.value = Number(e.target.value).toPrecision(4);
          this.schedule();
        },
      });
      params.append(el('div', { class: 'param-live' },
        el('div', { class: 'param' },
          el('label', { class: 'param-name num', for: `p-${p.name}`, title: p.description, text: p.name }),
          number,
          el('span', { class: 'param-unit num', text: p.unit })),
        slider));
    }
    zone.append(el('div', { class: 'config-block' },
      el('p', { class: 'label', text: 'parameters' }), params,
      el('p', { class: 'fine muted', text: 'Units are fixed by the model and dimension-checked server-side.' })));

    const dosing = el('div', { class: 'params' },
      this.numberRow('Dose', 's-dose', this.dose, 'mg', (v) => { this.dose = v; }),
      this.numberRow('Doses', 's-n', this.nDoses, 'n', (v) => { this.nDoses = v; }));
    zone.append(el('div', { class: 'config-block' },
      el('p', { class: 'label', text: 'dosing' }), dosing));

    zone.append(this.attributionBlock());

    zone.append(el('div', { class: 'config-actions' },
      el('button', {
        class: 'btn', type: 'button',
        text: this.baseline ? 'Clear baseline' : 'Hold as baseline',
        title: 'Keep the current curve to compare against',
        onclick: () => {
          this.baseline = this.baseline ? null : {
            times: this.run?.times_h, conc: this.run?.concentration_mg_per_l,
            params: { ...this.params }, dose: this.dose,
          };
          this.paint();
        },
      }),
      el('button', { class: 'btn', type: 'button', text: 'Reset',
        onclick: () => { this.resetParams(); this.solve(); } })));

    if (this.error) zone.append(el('p', { class: 'error-note', role: 'alert', text: this.error }));

    const eqs = el('ol', { class: 'equations' });
    for (const e of this.model.equations) eqs.append(el('li', { class: 'num', text: e }));
    zone.append(el('div', { class: 'config-block' },
      el('p', { class: 'label', text: 'equations integrated' }), eqs));

    return zone;
  }

  numberRow(name, id, value, unit, set) {
    return el('div', { class: 'param' },
      el('label', { class: 'param-name num', for: id, text: name }),
      el('input', { type: 'number', id, step: 'any', value,
        oninput: (e) => { set(Number(e.target.value)); this.schedule(); } }),
      el('span', { class: 'param-unit num', text: unit }));
  }

  attributionBlock() {
    const sel = el('select', {
      onchange: (e) => { this.compound = e.target.value || null; this.solve(); },
    }, el('option', { value: '', text: '— exploratory only —' }));
    for (const r of [...(this.state.rows || [])].sort((a, b) => a.name.localeCompare(b.name))) {
      sel.append(el('option', { value: r.key, text: r.name, selected: r.key === this.compound || null }));
    }
    const block = el('div', { class: 'config-block' },
      label('attribute to compound'), sel,
      el('p', { class: 'fine muted', text: 'Naming a compound asks the evidence gate whether the result may be called its pharmacokinetics.' }));

    const entry = this.compound && this.pkStore?.entries?.[this.compound];
    if (entry && entry.status === 'usable') {
      block.append(el('div', { class: 'note note-teal' },
        el('div', { class: 'row' }, el('span', { class: 'src-db', text: 'openfda' }),
          el('strong', { class: 'fine', text: `${entry.brand_name || entry.generic_name}` })),
        el('p', { class: 'fine muted', text: entry.validation?.verdict || '' }),
        el('button', { class: 'btn', type: 'button', text: 'Load label parameters',
          onclick: () => {
            this.model = this.models.find((m) => m.name === entry.model) || this.model;
            this.params = { ...entry.parameters };
            this.dose = entry.dose_mg;
            this.solve();
          } })));
    } else if (entry?.blocked_reason) {
      block.append(el('p', { class: 'fine muted', text: `No validated parameters: ${entry.blocked_reason}` }));
    }
    return block;
  }

  /* ---------------- canvas ---------------- */

  canvasZone() {
    const zone = el('div', { class: 'studio-canvas' });
    if (!this.run) {
      zone.append(el('div', { class: 'empty empty-tall' },
        el('p', { class: 'empty-glyph', 'aria-hidden': 'true', text: '∫' }),
        el('h3', { text: this.busy ? 'Solving…' : 'No run yet' }),
        el('p', { class: 'fine muted', text: 'Adjust a parameter to solve the system.' })));
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

    zone.append(this.concentrationFigure());
    zone.append(this.compartmentFigure());
    zone.append(this.readings());
    if (this.sensitivity) zone.append(this.sensitivityFigure());

    const interp = interpretations(run.interpretations, 'What each quantity means');
    if (interp) zone.append(interp);
    return zone;
  }

  /** Register a figure so the shared cursor can be drawn on it. */
  linked(host, config) {
    const cfg = {
      ...config,
      onCursor: (i) => this.setCursor(i),
      cursorIndex: this.cursor,
    };
    this._figures.push({ host, cfg });
    lineFigure(host, cfg);
  }

  setCursor(i) {
    this.cursor = i;
    for (const f of this._figures) {
      f.cfg.cursorIndex = i;
      lineFigure(f.host, f.cfg);
    }
  }

  concentrationFigure() {
    const run = this.run;
    const body = el('div', { class: 'fig-host' });
    const doseTimes = [];
    for (let i = 0; i < run.n_doses; i++) doseTimes.push(i * run.interval_h);

    const series = [{
      name: 'C', points: run.times_h.map((t, i) => [t, run.concentration_mg_per_l[i]]),
    }];
    if (this.baseline?.times) {
      series.push({ name: 'baseline', dashed: true, colour: 'var(--ink-4)', width: 1.3,
        points: this.baseline.times.map((t, i) => [t, this.baseline.conc[i]]) });
    }

    // Observed values from the label, where the compound has them.
    const references = [];
    const entry = this.compound && this.pkStore?.entries?.[this.compound];
    for (const t of entry?.validation?.targets || []) {
      if (t.name === 'cmax') {
        references.push({ x: run.nca.tmax_h, y: t.observed / 1000, label: `observed Cmax ${fmt(t.observed, 3)} ng/mL` });
      }
    }

    const tools = el('div', { class: 'row' },
      el('button', { class: 'btn btn-ghost fine', type: 'button',
        'aria-pressed': String(this.logY), text: this.logY ? 'linear y' : 'log y',
        onclick: () => { this.logY = !this.logY; this.paint(); } }),
      el('label', { class: 'live-toggle fine' },
        el('input', { type: 'checkbox', checked: this.live || null,
          onchange: (e) => { this.live = e.target.checked; } }),
        'live'));

    const fig = figure(1, 'Concentration–time profile', body,
      'Solver output. Dashed verticals mark dose events. '
      + (references.length ? 'Crosshair marks are the values the label reports. ' : '')
      + 'Hovering moves a cursor shared with the compartment figure below.');
    fig.querySelector('.figure-head').append(el('span', { class: 'spacer' }), tools);

    const wrap = el('div', { class: 'studio-figure' }, fig);
    queueMicrotask(() => this.linked(body, {
      series, xLabel: 'time (h)', yLabel: 'concentration (mg/L)',
      logY: this.logY, doseTimes, references, height: 300,
      ariaLabel: `Concentration against time for ${run.model_display_name}`,
    }));
    return wrap;
  }

  compartmentFigure() {
    const run = this.run;
    const body = el('div', { class: 'fig-host' });
    const names = Object.keys(run.states);
    const total = run.times_h.map((_, i) => names.reduce((a, k) => a + run.states[k][i], 0));
    const series = names.map((k) => ({
      name: k.replace('_mg', ''),
      points: run.times_h.map((t, j) => [t, run.states[k][j]]),
    }));
    series.push({ name: 'total', colour: 'var(--ink-3)', dashed: true, width: 1.2,
      points: run.times_h.map((t, i) => [t, total[i]]) });

    const fig = figure(2, 'Compartment amounts and conservation', body,
      'Each compartment in milligrams. The flat total is the conservation check: '
      + `mass balance holds to ${run.diagnostics.mass_balance_relative_error.toExponential(1)} relative.`);
    const wrap = el('div', { class: 'studio-figure' }, fig);
    queueMicrotask(() => this.linked(body, {
      series, xLabel: 'time (h)', yLabel: 'amount (mg)', height: 240,
      ariaLabel: 'Drug amount in each compartment over time, with the conserved total',
    }));
    return wrap;
  }

  readings() {
    const n = this.run.nca;
    const cursorNote = this.cursor != null && this.run.times_h[this.cursor] != null
      ? `cursor at ${fmt(this.run.times_h[this.cursor], 4)} h — `
        + `${fmt(this.run.concentration_mg_per_l[this.cursor] * 1000, 4)} ng/mL`
      : null;
    return region('Non-compartmental analysis', {
      caption: `Run ${this.run.run_id} · solver ${this.run.diagnostics.solver.method} `
        + `at rtol ${this.run.diagnostics.solver.rtol}.`,
      actions: cursorNote ? el('span', { class: 'cursor-readout num fine', text: cursorNote }) : null,
    }, el('div', { class: 'readings' },
      reading('Cmax', n.cmax_mg_per_l * 1000, 'ng/mL', `at ${fmt(n.tmax_h)} h`),
      reading('AUC(0,∞)', (n.analytic_auc_inf_mg_h_per_l ?? 0) * 1000, 'ng·h/mL', 'exact: F·Dose/CL'),
      reading('terminal t½', n.terminal_half_life_h, 'h',
        n.terminal_r_squared != null ? `r² = ${n.terminal_r_squared.toFixed(6)}` : ''),
      reading('mass balance', this.run.diagnostics.mass_balance_relative_error, 'rel.',
        this.run.diagnostics.mass_balance_relative_error < 1e-9 ? 'conserved to tolerance' : 'above 1e-9')));
  }

  sensitivityFigure() {
    const s = this.sensitivity;
    const quantity = this.sensQuantity || 'cmax';
    const rows = s.parameters.map((p) => ({
      parameter: p.parameter,
      low: p.effects[quantity]?.low,
      high: p.effects[quantity]?.high,
      swing: p.effects[quantity]?.swing,
    })).sort((a, b) => (b.swing ?? 0) - (a.swing ?? 0));

    const body = el('div', { class: 'fig-host' });
    const unit = { cmax: 'ng/mL', auc: 'ng·h/mL', half_life: 'h' }[quantity];
    const switcher = el('div', { class: 'segmented' });
    for (const [id, text] of [['cmax', 'Cmax'], ['auc', 'AUC'], ['half_life', 't½']]) {
      switcher.append(el('button', { type: 'button', text,
        'aria-pressed': String(quantity === id),
        onclick: () => { this.sensQuantity = id; this.paint(); } }));
    }

    const QUANTITY_NAME = { cmax: 'Cmax', auc: 'AUC', half_life: 'terminal half-life' };
    const fig = figure(3, 'Which parameter the result rests on', body,
      `Each parameter varied ±${(s.fraction * 100).toFixed(0)}% while the others are `
      + `held, and ${QUANTITY_NAME[quantity]} recomputed. The vertical line is the `
      + 'reference value; the percentage at the right is the resulting swing. '
      + `A parameter showing 0% is one ${QUANTITY_NAME[quantity]} does not depend `
      + `on. ${s.method}`);
    fig.querySelector('.figure-head').append(el('span', { class: 'spacer' }), switcher);
    queueMicrotask(() => tornadoFigure(body, {
      rows, reference: s.reference[quantity], quantity, unit,
    }));
    return el('div', { class: 'studio-figure' }, fig);
  }
}

function reading(name, value, unit, note) {
  return el('div', { class: 'reading' },
    label(name),
    el('p', { class: 'reading-value' }, measure(value, unit)),
    note ? el('p', { class: 'fine muted', text: note }) : null);
}
