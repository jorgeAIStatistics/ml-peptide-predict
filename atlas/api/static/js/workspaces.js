/* ATLAS — Evidence Library.
   Missing evidence is treated as a finding with a shape, not as an error. */

import {
  el, clear, fmt, fmtCount, titleCase, label, masthead, region, statusMark,
} from './ui.js';

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

