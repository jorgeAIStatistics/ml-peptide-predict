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

    const estimatePanel = el('div', { class: 'toolbar toolbar-flush' },
      el('div', { class: 'field', style: 'flex:1;min-width:200px' }, label('smiles'), input),
      el('div', { class: 'field' }, label('or registry entry'), sel),
      el('button', { class: 'btn btn-primary', type: 'button', text: 'Evaluate', onclick: run }));

    const rankPanel = this.rankPanel();
    rankPanel.setAttribute('hidden', '');

    const modes = el('div', { class: 'segmented' },
      ...[['estimate', 'Estimate one'], ['rank', 'Rank a pair']].map(([id, text], i) =>
        el('button', {
          type: 'button', text, 'aria-pressed': String(i === 0),
          onclick: (e) => {
            const on = id === 'rank';
            rankPanel.toggleAttribute('hidden', !on);
            estimatePanel.toggleAttribute('hidden', on);
            for (const b of e.currentTarget.parentElement.children) {
              b.setAttribute('aria-pressed', String(b === e.currentTarget));
            }
          },
        })));

    const consoleRegion = region('Prediction console', {
      caption: 'A measurement outranks a prediction; a prediction is returned '
        + 'only where applicability and target relevance both hold. Ranking a '
        + 'pair is a narrower claim with its own gate, and it is the one this '
        + 'model passes.',
      actions: modes,
    }, estimatePanel, rankPanel, out);
    return consoleRegion;
  }

  /* --- ranking a pair -------------------------------------------------
     A different claim from an estimate, with its own deployment gate: the
     affinity gate fails and this one passes, so the console offers an
     ordering where it refuses a number. Every refusal below comes from the
     service, not from the UI. */
  rankPanel() {
    const pick = (side) => {
      const s = el('select', { id: `mi-rank-${side}` },
        el('option', { value: '', text: '— choose —' }));
      for (const r of [...(this.state.rows || [])].sort((a, b) => a.name.localeCompare(b.name))) {
        s.append(el('option', { value: r.key, text: r.name }));
      }
      return s;
    };
    const a = pick('a');
    const b = pick('b');
    const smiA = el('input', { type: 'text', id: 'mi-rank-smiles-a', placeholder: 'or paste a structure…',
      autocomplete: 'off', spellcheck: 'false' });
    const smiB = el('input', { type: 'text', id: 'mi-rank-smiles-b', placeholder: 'or paste a structure…',
      autocomplete: 'off', spellcheck: 'false' });
    const out = el('div', { id: 'mi-rank-out' });

    const compare = async () => {
      const body = {};
      if (smiA.value.trim()) body.a_smiles = smiA.value.trim();
      else if (a.value) body.a_compound_key = a.value;
      if (smiB.value.trim()) body.b_smiles = smiB.value.trim();
      else if (b.value) body.b_compound_key = b.value;
      if (!(body.a_smiles || body.a_compound_key) || !(body.b_smiles || body.b_compound_key)) {
        clear(out).append(el('p', { class: 'error-note',
          text: 'Choose or paste two molecules.' }));
        return;
      }
      // Labels must describe what was actually sent. A pasted structure takes
      // precedence over the dropdown, so reading the dropdown's selected text
      // would attribute the answer to a molecule that was not compared.
      const labels = {
        a: body.a_smiles ? 'the first structure' : (a.selectedOptions[0]?.text || 'the first'),
        b: body.b_smiles ? 'the second structure' : (b.selectedOptions[0]?.text || 'the second'),
      };
      clear(out).append(el('div', { class: 'loading', text: 'Comparing…' }));
      try {
        clear(out).append(this.rankResult(await api.rank(body), labels));
      } catch (err) {
        clear(out).append(el('p', { class: 'error-note', text: String(err.message) }));
      }
    };

    return el('div', {},
      el('div', { class: 'toolbar toolbar-flush' },
        el('div', { class: 'field', style: 'flex:1;min-width:160px' }, label('first'), a, smiA),
        el('div', { class: 'field', style: 'flex:1;min-width:160px' }, label('second'), b, smiB),
        el('button', { class: 'btn btn-primary', type: 'button', text: 'Compare', onclick: compare })),
      out);
  }

  rankResult(r, labels) {
    const nameA = r.a?.name || labels.a;
    const nameB = r.b?.name || labels.b;
    const frag = el('div', {});

    if (r.answer_kind === 'abstention') {
      frag.append(el('div', { class: 'note note-amber' },
        el('p', {}, el('strong', { text: 'No ordering offered. ' }),
          r.refusal_explanation || r.refusal_reason || '')),
        (r.what_would_resolve_it || []).length
          ? el('ul', { class: 'fine' },
            ...r.what_would_resolve_it.map((w) => el('li', { text: w })))
          : null);
      return frag;
    }

    const measured = r.basis === 'measurement';
    const tie = r.tighter === 'indistinguishable';
    const winner = r.tighter === 'a' ? nameA : nameB;
    const loser = r.tighter === 'a' ? nameB : nameA;

    frag.append(el('div', { class: `relevance rel-${measured ? 'A' : 'B'}` },
      el('div', { class: 'row' },
        el('span', { class: `ev ev-${measured ? 'A' : 'B'}`, text: measured ? 'A' : 'B' }),
        el('strong', {
          text: tie
            ? `${nameA} and ${nameB} are indistinguishable`
            : `${winner} binds more tightly than ${loser}`,
        })),
      el('p', { class: 'fine', text: measured
        ? 'Both have measured affinities, so this compares measurements.'
        : 'A model ordering, not an affinity estimate.' })));

    /* An activity-cliff pair is the case where this ranker is weakest, so the
       warning goes above the numbers rather than into the footnotes. Its band
       accuracy is quoted in place of the pooled figure, which would flatter
       it. */
    if (r.is_activity_cliff) {
      frag.append(el('div', { class: 'note note-amber' },
        el('p', {}, el('strong', { text: 'Least reliable case. ' }),
          `These structures are ${r.pair_similarity.toFixed(2)} similar — a `
          + 'near-identical pair with a real affinity gap is an activity '
          + 'cliff, and measured accuracy here is '
          + `${(r.confidence * 100).toFixed(1)}% against `
          + `${(r.pooled_accuracy * 100).toFixed(1)}% pooled. Treat this `
          + 'ordering with more caution than a less similar pair, not less.'),
        r.band_meets_gate === false
          ? el('p', { class: 'fine' },
            `On its own evidence this band does not clear the ranking gate: its `
            + `95% lower bound is ${(r.confidence_interval[0] * 100).toFixed(1)}% `
            + 'against the 80% required. The gate passed on the pooled domain, '
            + 'which this band is part of and drags down.')
          : null));
    }

    const readings = el('div', { class: 'readings' });
    if (!measured && r.confidence != null) {
      const ci = r.confidence_interval || [];
      readings.append(reading(
        r.similarity_band ? `accuracy, ${r.similarity_band} band` : 'ranking accuracy',
        r.confidence * 100, '%',
        ci[0] != null ? `95% interval ${(ci[0] * 100).toFixed(1)}–${(ci[1] * 100).toFixed(1)}%` : null));
    }
    const floor = r.gate?.pair_similarity_min;
    readings.append(reading('pair similarity', r.pair_similarity, '',
      floor != null ? `series floor ${floor}` : null));
    if (r.margin_log_units != null) {
      readings.append(reading('margin', r.margin_log_units, 'log',
        r.margin_is_meaningful ? 'above the validated threshold' : 'below it — reported as a tie'));
    }
    frag.append(readings);

    frag.append(el('div', { class: 'note note-copper' },
      el('p', {}, el('strong', { text: 'What this is not. ' }),
        'No affinity number is returned for either molecule. The affinity '
        + 'deployment gate is unchanged and still fails; only the ranking '
        + 'gate passes.')));

    if ((r.limitations || []).length) {
      frag.append(el('ul', { class: 'fine muted' },
        ...r.limitations.map((l) => el('li', { text: l }))));
    }
    return frag;
  }
}

function reading(name, value, unit, note) {
  return el('div', { class: 'reading' },
    label(name),
    el('p', { class: 'reading-value' }, measure(value, unit)),
    note ? el('p', { class: 'fine muted', text: note }) : null);
}
