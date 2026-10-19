/* ATLAS — Model Intelligence.

   A model is presented by what it got wrong as much as by what it got right.
   The diagnostics here are drawn from 396 held-out molecules of the
   scaffold-disjoint test role, so the predicted-versus-observed plot, the
   applicability map and the reliability diagram are measurements rather than
   illustrations.

   The figures are linked: hovering a molecule in one highlights it in the
   others, because "which compounds are the model worst on, and are they the
   distant ones" is a single question asked across three views. */

import {
  $, el, clear, fmt, fmtPct, titleCase, label, measure, masthead, region, figure,
} from './ui.js';
import {
  predictedObserved, applicabilityMap, reliabilityFigure, barFigure,
} from './figures.js';
import { interpretations } from './interpret.js';
import { api } from './api.js';

export class ModelIntelligence {
  constructor(host, { state }) {
    this.host = host;
    this.state = state;
    this.card = null;
    this.calibration = null;
    this.colourBy = 'covered';
  }

  async render() {
    if (!this.card) {
      clear(this.host).append(el('div', { class: 'loading', text: 'Loading model…' }));
      try {
        [this.card, this.calibration] = await Promise.all([api.modelCard(), api.calibration()]);
      } catch (err) {
        clear(this.host).append(el('p', { class: 'error-note', text: String(err.message) }));
        return;
      }
    }
    this.paint();
  }

  paint() {
    const host = clear(this.host);
    const { card, report, calibration, peptide_findings: pep } = this.card;
    const hp = card.headline_performance;
    const rs = card.random_split_comparison;
    const diag = this.calibration?.report?.diagnostics;
    const reliability = this.calibration?.report?.reliability;

    host.append(masthead({
      eyebrow: 'Model Intelligence',
      title: 'What the model is, and how far it can be trusted',
      lede: 'Held-out performance under splits of increasing honesty, the domain '
        + 'the model may be used on, and the calibration measured for its intervals.',
      actions: el('span', { class: 'tag tag-teal', text: titleCase(card.quality_tier) }),
    }));

    /* --- identity --------------------------------------- */
    host.append(region('Model identity', {
      caption: card.quality_tier_meaning,
    },
      el('div', { class: 'readings' },
        reading('MAE', hp.mae_log_units, 'log', 'scaffold split — the honest figure'),
        reading('R²', hp.r2, '', `Spearman ${hp.spearman.toFixed(3)}`),
        reading('MAE (random split)', rs.mae_log_units, 'log',
          `${fmtPct(rs.fraction_test_scaffolds_seen_in_training)} of test scaffolds already seen`),
        reading('coverage', calibration.empirical_coverage, '',
          `nominal ${calibration.nominal_coverage}, was ${calibration.previous_empirical_coverage}`)),
      el('dl', { class: 'ledger' },
        el('dt', { text: 'Endpoint' }), el('dd', { text: card.endpoint }),
        el('dt', { text: 'Target' }), el('dd', { text: card.target }),
        el('dt', { text: 'Family' }), el('dd', { text: card.model_family }),
        el('dt', { text: 'In domain' }), el('dd', { class: 'prose', text: card.intended_domain }),
        el('dt', { text: 'Out of domain' }), el('dd', { class: 'prose', text: card.out_of_domain }))));

    /* --- held-out diagnostics, linked ------------------- */
    if (diag) {
      const poHost = el('div', { class: 'fig-host' });
      const amHost = el('div', { class: 'fig-host' });

      const colourSwitch = el('div', { class: 'segmented' },
        ...[['covered', 'by coverage'], ['modality', 'by modality']].map(([id, t]) =>
          el('button', { type: 'button', text: t, 'aria-pressed': String(this.colourBy === id),
            onclick: () => { this.colourBy = id; this.paint(); } })));

      const nOut = diag.covered.filter((c) => !c).length;
      const poFig = figure(1, 'Predicted against observed', poHost,
        `${diag.n} held-out molecules from the scaffold-disjoint test role. The `
        + 'solid diagonal is where a perfect model would lie; the dashed lines '
        + 'are one log unit either side. Vertical reach is each prediction’s '
        + `90% interval, and ${nOut} of ${diag.n} observations fell outside theirs.`);
      poFig.querySelector('.figure-head').append(el('span', { class: 'spacer' }), colourSwitch);

      const amFig = figure(2, 'Applicability: error against similarity', amHost,
        'Each point is one held-out molecule; the heavy line is the mean '
        + 'absolute error per similarity band. A model that degrades with '
        + 'distance from its training set shows it here.');

      host.append(region('Held-out diagnostics', {
        caption: 'Measured on molecules the model never saw, whose scaffolds were '
          + 'absent from training, selection and calibration alike.',
      }, el('div', { class: 'diag-grid' },
        el('div', {}, poFig), el('div', {}, amFig))));

      queueMicrotask(() => {
        predictedObserved(poHost, diag, { colourBy: this.colourBy });
        applicabilityMap(amHost, diag);
      });

      /* --- error by molecular family ------------------- */
      const byMod = {};
      diag.modality.forEach((m, i) => {
        (byMod[m] ||= []).push(Math.abs(diag.residual[i]));
      });
      const famHost = el('div', { class: 'fig-host' });
      const famRows = Object.entries(byMod).map(([m, errs]) => ({
        label: `${titleCase(m)} (${errs.length})`,
        value: errs.reduce((a, b) => a + b, 0) / errs.length,
        worse: false,
      })).sort((a, b) => a.value - b.value);

      const relHost = el('div', { class: 'fig-host' });
      host.append(region('Error and calibration by subgroup', {
        caption: 'The model is not equally good everywhere, and its intervals are '
          + 'not equally reliable everywhere. Both are reported rather than averaged.',
      }, el('div', { class: 'diag-grid' },
        el('div', {}, figure(3, 'Mean absolute error by molecular family', famHost,
          'Peptide-like and small-molecule-like molecules evaluated separately '
          + 'on the same held-out set.')),
        el('div', {}, figure(4, 'Reliability of the prediction intervals', relHost,
          'Measured coverage against the nominal level. A model whose intervals '
          + 'mean what they claim lies on the dashed diagonal at every level, '
          + 'not only at the one it was calibrated for.')))));

      queueMicrotask(() => {
        barFigure(famHost, { rows: famRows, xLabel: 'mean absolute error (log units)' });
        if (reliability) reliabilityFigure(relHost, reliability);
      });
    }

    /* --- measured limits -------------------------------- */
    const nt = card.negative_transfer;
    host.append(region('Measured limits', {}, el('div', { class: 'note note-copper' },
      el('p', {}, el('strong', { text: nt.verdict }), ' ',
        `Best model MAE ${nt.best_model_mae.toFixed(2)} against a mean-predictor `
        + `baseline of ${nt.baseline_mean_mae.toFixed(2)} log units (${nt.experiment}).`)),
      pep ? el('div', { class: 'note note-amber' },
        el('p', {}, el('strong', { text: 'Peptide model not deployed. ' }), pep.decision)) : null));

    /* --- split comparison ------------------------------- */
    if (report?.available) {
      const bySplit = {};
      for (const r of report.results) (bySplit[r.split] ||= []).push(r);
      let n = 5;
      const grid = el('div', { class: 'split-grid' });
      for (const [split, rows] of Object.entries(bySplit)) {
        const base = rows.find((r) => r.model === 'baseline_mean');
        const meta = report.splits.find((s) => s.kind === split) || {};
        const body = el('div', { class: 'fig-host' });
        grid.append(el('div', {}, figure(n++, titleCase(split), body,
          `Train ${meta.n_train} · test ${meta.n_test} · `
          + `${fmtPct(meta.leakage?.fraction_test_scaffolds_seen || 0)} of test scaffolds `
          + 'already seen. Bars past the dashed line are worse than the mean.')));
        queueMicrotask(() => barFigure(body, {
          rows: rows.filter((r) => r.model !== 'baseline_mean')
            .sort((a, b) => a.mae - b.mae)
            .map((r) => ({ label: r.model, value: r.mae,
              worse: base ? r.mae >= base.mae : false,
              highlight: r.model === 'fingerprint_knn' })),
          refValue: base?.mae, refLabel: `baseline ${base ? base.mae.toFixed(2) : ''}`,
          xLabel: 'MAE (log units)',
        }));
      }
      host.append(region('Performance by split', {
        caption: 'The same model under four constructions. The gap between the '
          + 'random and scaffold splits is how much a random split flatters it.',
      }, grid));
    }

    const interp = interpretations(this.card.interpretations, 'What the performance numbers mean');
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
      if (!smiles && !key) {
        clear(out).append(el('p', { class: 'error-note', text: 'Enter a structure or choose an entry.' }));
        return;
      }
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
      caption: 'A measurement outranks a prediction; a prediction is returned '
        + 'only where applicability and target relevance both hold.',
    }, el('div', { class: 'toolbar toolbar-flush' },
      el('div', { class: 'field', style: 'flex:1;min-width:200px' }, label('smiles'), input),
      el('div', { class: 'field' }, label('or registry entry'), sel),
      el('button', { class: 'btn btn-primary', type: 'button', text: 'Evaluate', onclick: run })),
      out);
  }
}

function reading(name, value, unit, note) {
  return el('div', { class: 'reading' },
    label(name),
    el('p', { class: 'reading-value' }, measure(value, unit)),
    note ? el('p', { class: 'fine muted', text: note }) : null);
}
