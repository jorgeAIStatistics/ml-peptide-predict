/* ATLAS — the scientific interpretation renderer.
   Four coordinated views per quantity, behind native <details> so the
   disclosure is keyboard accessible and findable by browser search. */

import { el, add, fmt, titleCase } from './ui.js';
import { region } from './ui.js';

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

export function interpretations(list, heading) {
  if (!(list || []).length) return null;
  const wrap = el('div', { class: 'interps' });
  for (const i of list) wrap.append(interpretationBlock(i));
  return region(heading || 'What these numbers mean', {
    caption: 'Each value is explained from its own definition and from the '
      + 'metadata of whatever produced it. Expand one for the mathematics, what '
      + 'it does and does not establish, and what would test it.',
  }, wrap);
}

