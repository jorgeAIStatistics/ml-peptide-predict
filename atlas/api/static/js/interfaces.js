/* ATLAS — Interface Explorer.

   Ten experimentally determined MC4R–peptide complexes, drawn so that a
   residue can be followed across four linked views at once: where it sits in
   the structure, which contacts it makes, what its side-chain chemistry is,
   and how many of the ten structures support it.

   The vocabulary matters more than the picture. Every structure here is
   experimental, but the typing laid over it is not uniformly so, and the
   interface says which is which at the point of use rather than in a footnote:

     * a **candidate** hydrogen bond, because these structures are 2.58–3.30 Å,
       carry no hydrogens, and no donor-hydrogen-acceptor angle was ever
       checked — a heavy-atom distance alone over-counts;
     * an **inferred** charged group or ring on the peptide side, because these
       ligands are largely non-canonical and a residue name carries no
       chemistry for 2,4-dichlorophenylalanine;
     * a **transferred** pocket position on MC1R, MC3R or MC5R, for which no
       peptide-bound structure exists at all.

   And the thing this workspace is not: the peptide affinity model fails its
   deployment gate and publishes no numeric estimate. A drawn salt bridge is
   not evidence that a peptide binds tightly, and the explorer offers no
   affinity for any of the ligands it draws. */

import {
  $$, el, clear, fmt, titleCase, label, measure, masthead,
  region, figure, emptyState, tip, hideTip,
} from './ui.js';
import { api } from './api.js';
import { StructureViewer, viewerLegend, KIND_COLOUR } from './structure3d.js';

/* Which evidence class each interaction kind's mark carries. The backend
   states it per kind; this maps it to the instrument's existing chip
   vocabulary so an inferred mark looks like every other qualified value. */
const EVIDENCE_TONE = {
  experimental: 'teal',
  geometric: 'blue',
  candidate: 'amber',
  inferred: 'amber',
  illustrative: 'stone',
  transferred: 'copper',
};

/* Descriptor groups for the residue chemistry panel, in the order a reader
   would ask about them. Names are the backend's DESCRIPTOR_NAMES. */
const CHEM_GROUPS = [
  ['Size and shape', ['heavy_atoms', 'side_chain_volume', 'molar_refractivity',
    'rotatable_bonds', 'branching_index']],
  ['Polarity', ['crippen_logp', 'tpsa', 'h_bond_donors', 'h_bond_acceptors']],
  ['Aromaticity', ['aromatic_rings', 'aromatic_atoms']],
  ['Heteroatoms', ['halogens', 'sulfur_atoms', 'nitrogen_atoms', 'oxygen_atoms']],
  ['Ionisation', ['formal_charge_graph', 'estimated_charge', 'is_acidic',
    'is_basic', 'is_ionisable']],
  ['Conformation', ['backbone_constrained', 'glycine_like']],
];

const DEFAULT_OVERLAYS = ['salt_bridge', 'hbond_candidate', 'aromatic_stacked',
  'aromatic_tshaped', 'cation_pi'];

export class InterfaceExplorer {
  constructor(host, { state }) {
    this.host = host;
    this.state = state;
    this.index = null;
    this.detail = null;
    this.pdbId = null;
    this.loading = false;
    this.error = null;
    this.selected = null;          // consensus pocket position
    this.scope = 'interface';
    this.enabled = new Set(DEFAULT_OVERLAYS);
    this.viewer = null;
    this.pending = null;
  }

  /* ---------------- data ---------------- */

  async render() {
    if (!this.index) {
      clear(this.host).append(
        el('div', { class: 'loading', text: 'Loading solved interfaces…' }));
      try {
        this.index = await api.interfaces();
      } catch (err) {
        this.error = err.message;
        this.paint();
        return;
      }
      if (this.index.available && (this.index.structures || []).length) {
        this.pdbId = this.index.structures[0].pdb_id;
      }
    }
    if (this.pdbId && (!this.detail || this.detail.structure?.pdb_id !== this.pdbId)) {
      await this.loadDetail(this.pdbId);
      return;
    }
    this.paint();
  }

  async loadDetail(pdbId) {
    this.pdbId = pdbId;
    this.loading = true;
    this.selected = null;
    this.paint();
    try {
      this.detail = await api.interface(pdbId);
      this.error = null;
    } catch (err) {
      this.detail = null;
      this.error = err.message;
    }
    this.loading = false;
    this.paint();
  }

  /** Interaction kinds in the backend's declared order.
      JSON object key order is not dependable -- Flask sorts it -- so the
      order is sent as its own list and honoured here. */
  kinds() {
    const kinds = this.index?.interaction_kinds || {};
    const order = this.index?.interaction_kind_order || Object.keys(kinds);
    return order.filter((k) => kinds[k]).map((k) => [k, kinds[k]]);
  }

  /** The consensus pocket positions, as a set, for the viewer's pick targets. */
  pocketPositions() {
    return new Set((this.index?.pocket?.positions || []).map((p) => p.position));
  }

  supportMap() {
    const out = {};
    for (const p of this.index?.pocket?.positions || []) out[String(p.position)] = p.support;
    return out;
  }

  /* ---------------- paint ---------------- */

  paint() {
    if (this.viewer) { this.viewer.destroy(); this.viewer = null; }
    this.pending = null;
    const host = clear(this.host);

    host.append(masthead({
      eyebrow: 'Interface Explorer',
      title: 'Ten solved interfaces, residue by residue',
      lede: 'Every structure here is an experimentally determined MC4R–peptide '
        + 'complex. What is drawn over it — a hydrogen-bond candidate, an '
        + 'inferred charged group, a pocket position transferred to another '
        + 'receptor — is labelled with what kind of claim it is.',
      actions: this.index?.pocket
        ? el('span', { class: 'tag tag-teal',
          text: `${this.index.pocket.n_positions} pocket positions` })
        : null,
    }));

    if (this.error) {
      host.append(el('p', { class: 'error-note', text: this.error }));
      return;
    }
    if (!this.index?.available) {
      host.append(region('Not available', {},
        emptyState('◇', 'The interaction profiles are not in this snapshot',
          this.index?.reason
          || 'Run atlas.ml.interaction_features and atlas.ml.receptor_pocket.')));
      return;
    }

    host.append(this.standingBand());

    const split = el('div', { class: 'iface' });
    split.append(this.rail());
    const canvas = el('div', { class: 'iface-canvas' });
    split.append(canvas);
    host.append(split);

    if (this.loading) {
      canvas.append(el('div', { class: 'loading', text: `Reading ${this.pdbId}…` }));
      return;
    }
    if (!this.detail) {
      canvas.append(emptyState('◇', 'No complex selected',
        'Choose one of the ten solved complexes.'));
      return;
    }

    canvas.append(this.viewerRegion());
    canvas.append(this.contactMapRegion());
    canvas.append(this.inspectorRegion());
    canvas.append(this.measurementRegion());
    canvas.append(this.panelRegion());
    canvas.append(this.caveatRegion());
    this.mountViewer();
  }

  /** Construct the viewer, synchronously, once its host is in the document.

      It was deferred to requestAnimationFrame at first, which looked tidy and
      was wrong: a frame is never scheduled while the page is hidden, so the
      figure silently stayed empty. The viewer measures its host, so building
      it before layout gives a minimum-width first draw; its ResizeObserver
      then redraws at the true size. An empty figure is the worse failure. */
  mountViewer() {
    const pending = this.pending;
    this.pending = null;
    if (!pending) return;
    this.viewer = new StructureViewer(pending.host, pending.geometry, {
      enabled: this.enabled,
      selected: this.selected,
      scope: this.scope,
      pocketPositions: this.pocketPositions(),
      support: this.supportMap(),
      onSelect: (pos) => this.select(pos),
    });
    clear(pending.legendHost).append(viewerLegend(
      this.kinds(), this.enabled, this.overlayCounts()));
    this.syncScopeNote();
  }

  /* ---------------- the standing statements ---------------- */

  standingBand() {
    const mp = this.index.model_position || {};
    const classes = this.index.evidence_classes || {};
    const band = el('div', { class: 'iface-band' });

    band.append(el('div', { class: 'iface-band-main' },
      el('p', { class: 'label', text: 'what this is, and is not' }),
      el('p', { class: 'fine', text: mp.statement || '' }),
      el('p', { class: 'fine muted', text: this.index.restriction || '' })));

    const key = el('div', { class: 'iface-key' });
    for (const [name, spec] of Object.entries(classes)) {
      key.append(el('details', { class: `iface-class iface-class-${EVIDENCE_TONE[name] || 'stone'}` },
        el('summary', {},
          el('span', { class: 'iface-class-mark', 'aria-hidden': 'true' }),
          el('span', { class: 'iface-class-name', text: spec.label })),
        el('p', { class: 'fine', text: spec.detail })));
    }
    band.append(el('div', {}, el('p', { class: 'label', text: 'the six classes of mark' }), key));
    return band;
  }

  /* ---------------- left rail ---------------- */

  rail() {
    const rail = el('aside', { class: 'iface-rail', 'aria-label': 'Complex and overlays' });

    /* -- complex selector -- */
    const list = el('div', { class: 'iface-list' });
    for (const s of this.index.structures || []) {
      const active = s.pdb_id === this.pdbId;
      list.append(el('button', {
        class: `iface-item${active ? ' is-active' : ''}`,
        type: 'button',
        'aria-current': active ? 'true' : null,
        onclick: () => { if (!active) this.loadDetail(s.pdb_id); },
      },
        el('span', { class: 'iface-item-id num', text: s.pdb_id }),
        el('span', { class: 'iface-item-ligand', text: s.ligand }),
        el('span', { class: 'iface-item-meta fine muted num' },
          `${s.method} · ${s.resolution?.toFixed ? s.resolution.toFixed(2) : s.resolution} Å`),
        s.coordinates_cached ? null
          : el('span', { class: 'iface-item-warn fine', text: 'coordinates not cached' })));
    }
    rail.append(el('div', { class: 'config-block' },
      el('span', { class: 'label', text: 'complex' }), list));

    /* -- overlay toggles -- */
    const s = this.detail?.structure || {};
    const toggles = el('div', { class: 'iface-toggles' });
    for (const [key, spec] of this.kinds()) {
      const n = s[spec.profile_field];
      const on = this.enabled.has(key);
      toggles.append(el('label', {
        class: `iface-toggle${n ? '' : ' is-zero'}`,
        title: spec.criterion,
      },
        el('input', {
          type: 'checkbox', checked: on ? 'checked' : null,
          disabled: !n ? 'disabled' : null,
          onchange: (e) => {
            if (e.target.checked) this.enabled.add(key); else this.enabled.delete(key);
            this.refreshOverlays();
          },
        }),
        el('span', { class: 'iface-toggle-rule', style: `background:${KIND_COLOUR[key]}` }),
        el('span', { class: 'iface-toggle-name', text: spec.label }),
        el('span', { class: 'iface-toggle-n num fine', text: n === undefined ? '—' : String(n) }),
        el('span', {
          class: `iface-ev iface-ev-${EVIDENCE_TONE[spec.evidence] || 'stone'}`,
          text: spec.evidence,
        })));
    }
    rail.append(el('div', { class: 'config-block' },
      el('span', { class: 'label', text: 'contact overlays' }),
      toggles,
      el('p', { class: 'fine muted iface-note',
        text: 'Counts are the saved profile’s. Each overlay draws one segment '
          + 'per saved pair; nothing is recomputed here.' })));

    /* -- view scope -- */
    rail.append(el('div', { class: 'config-block' },
      el('span', { class: 'label', text: 'view' }),
      el('div', { class: 'segmented' },
        el('button', {
          type: 'button', 'aria-pressed': String(this.scope === 'interface'),
          text: 'interface', onclick: () => this.setScope('interface'),
        }),
        el('button', {
          type: 'button', 'aria-pressed': String(this.scope === 'receptor'),
          text: 'whole chain', onclick: () => this.setScope('receptor'),
        })),
      el('div', { class: 'row iface-view-actions' },
        el('button', {
          class: 'btn btn-ghost fine', type: 'button', text: 'reset rotation',
          onclick: () => this.viewer?.reset(),
        }),
        this.selected !== null
          ? el('button', {
            class: 'btn btn-ghost fine', type: 'button', text: 'clear residue',
            onclick: () => this.select(null),
          })
          : null)));

    return rail;
  }

  setScope(scope) {
    this.scope = scope;
    $$('.iface-rail .segmented button', this.host).forEach((b) => {
      b.setAttribute('aria-pressed',
        String(b.textContent === (scope === 'receptor' ? 'whole chain' : 'interface')));
    });
    this.viewer?.setScope(scope);
    this.syncScopeNote();
  }

  syncScopeNote() {
    const note = this.host.querySelector('.iface-scope-note');
    if (note) note.textContent = this.scopeNote();
  }

  scopeNote() {
    const g = this.detail?.geometry;
    if (!g) return '';
    const unmapped = (g.receptor_trace || []).filter((t) => t.position === null).length;
    if (this.scope !== 'receptor') {
      return 'Scaled to the interface: the peptide and the receptor residues '
        + 'within 8 Å of it. The C-alpha trace continues past the frame and is '
        + 'cropped; switch to the whole chain to see all of it.';
    }
    const tail = 'Scaled to the 97th percentile radius of the chain, so the '
      + 'furthest few residues fall outside the frame.';
    return unmapped
      ? `Scaled to the whole deposited chain. ${unmapped} of `
        + `${g.receptor_trace.length} residues did not align to the MC4R `
        + 'reference sequence — this entry carries a fusion partner, drawn '
        + `dashed and not named from the reference. ${tail}`
      : 'Scaled to the whole deposited chain. Every residue aligned to the '
        + `MC4R reference sequence. ${tail}`;
  }

  refreshOverlays() {
    this.viewer?.setEnabled(this.enabled);
    const host = this.host.querySelector('.iface-legend-host');
    if (host) {
      clear(host).append(viewerLegend(
        this.kinds(), this.enabled, this.overlayCounts()));
    }
  }

  overlayCounts() {
    const g = this.detail?.geometry;
    if (!g) return {};
    const out = {};
    for (const [k] of this.kinds()) out[k] = (g.interactions?.[k] || []).length;
    return out;
  }

  select(pos) {
    this.selected = pos;
    this.viewer?.setSelected(pos);
    const inspector = this.host.querySelector('.iface-inspector-host');
    if (inspector) clear(inspector).append(this.inspector());
    this.syncContactMapSelection();
    const clearBtn = this.host.querySelector('.iface-view-actions');
    if (clearBtn) {
      const existing = clearBtn.querySelector('[data-role="clear-residue"]');
      if (pos === null && existing) existing.remove();
      if (pos !== null && !existing) {
        clearBtn.append(el('button', {
          class: 'btn btn-ghost fine', type: 'button', text: 'clear residue',
          'data-role': 'clear-residue', onclick: () => this.select(null),
        }));
      }
    }
  }

  syncContactMapSelection() {
    $$('.iface-col', this.host).forEach((c) => {
      const on = Number(c.dataset.position) === this.selected;
      c.classList.toggle('is-selected', on);
      c.setAttribute('aria-pressed', String(on));
    });
  }

  /* ---------------- figure 1: the viewer ---------------- */

  viewerRegion() {
    const g = this.detail.geometry;
    const s = this.detail.structure;

    if (!g) {
      return region('The complex', {
        caption: 'The saved profile and the contact distances below do not need '
          + 'the coordinate file; the drawing does.',
      },
        emptyState('◇', `No coordinates for ${s.pdb_id}`,
          this.detail.geometry_unavailable || 'The coordinate file is absent.'),
        el('div', { class: 'note note-copper' },
          el('p', { text: 'Nothing is drawn in place of the structure. '
            + 'Coordinate files are cached outside version control, so a fresh '
            + 'checkout has none; atlas.ml.receptor_pocket.fetch_structure '
            + 'restores one from the PDB.' })));
    }

    const host = el('div', { class: 'struct-host', tabindex: '0',
      'aria-label': 'Structure viewer. Drag or use the arrow keys to rotate.' });
    const legendHost = el('div', { class: 'iface-legend-host' });

    const body = el('div', {}, host, legendHost,
      el('p', { class: 'fig-caption iface-scope-note' }));

    const fig = figure(1,
      `${s.pdb_id} · ${s.ligand} · ${s.method} ${s.resolution} Å`,
      body,
      this.index.rendering_note);

    const head = fig.querySelector('.figure-head');
    head.append(el('span', { class: 'spacer' }));
    head.append(el('span', {
      class: `iface-ev iface-ev-${g.agrees_with_profile ? 'teal' : 'copper'}`,
      text: g.agrees_with_profile
        ? 'pair counts match the saved profile'
        : 'pair counts disagree with the saved profile',
    }));

    // Held for mountViewer, which runs once the region is in the document.
    this.pending = { host, legendHost, geometry: g };

    return region('The complex', {
      caption: 'Drag to rotate, or focus the figure and use the arrow keys. '
        + 'Click a ringed position to select that pocket residue; the selection '
        + 'carries to the contact map and the inspector below.',
    }, fig);
  }

  /* ---------------- figure 2: the contact map ---------------- */

  contactMapRegion() {
    const positions = this.index.pocket.positions || [];
    const contacts = this.detail.structure.contacts || {};
    const cutoff = this.index.pocket.cutoff_angstrom;
    const mutated = new Set(this.detail.structure.mutated_contact_positions || []);
    const maxSupport = Math.max(1, ...positions.map((p) => p.support));

    const strip = el('div', { class: 'iface-map', role: 'group',
      'aria-label': 'Consensus pocket positions' });
    for (const p of positions) {
      const d = contacts[String(p.position)];
      const touching = d !== undefined && d !== null;
      const col = el('button', {
        class: `iface-col${p.position === this.selected ? ' is-selected' : ''}`,
        type: 'button',
        'aria-pressed': String(p.position === this.selected),
        'data-position': String(p.position),
        'aria-label':
          `${p.residue}${p.position}: supported by ${p.support} of 10 structures; `
          + (touching
            ? `nearest heavy atom ${d.toFixed(2)} Å in this complex`
            : `no contact within ${cutoff} Å in this complex`),
        onclick: () => this.select(p.position === this.selected ? null : p.position),
        onmouseenter: (e) => tip(
          `<strong>${p.residue}${p.position}</strong>`
          + `support ${p.support}/10 structures<br>`
          + (touching ? `nearest heavy atom ${d.toFixed(2)} Å here`
            : `not within ${cutoff} Å here`)
          + (mutated.has(p.position)
            ? '<br>engineered mutation in this construct' : ''),
          e.clientX, e.clientY),
        onmouseleave: hideTip,
      },
        el('span', { class: 'iface-col-support' },
          el('span', {
            class: 'iface-col-fill',
            style: `height:${(p.support / maxSupport) * 100}%`,
          })),
        el('span', {
          class: `iface-col-contact${touching ? ' is-touching' : ''}`,
          // Nearer contacts read darker. The ramp runs over the 2.2–4.5 Å
          // window the cutoff defines, not over this structure's own range,
          // so the shading is comparable between complexes.
          style: touching
            ? `opacity:${(0.25 + 0.75 * Math.max(0, Math.min(1, (cutoff - d) / (cutoff - 2.2)))).toFixed(2)}`
            : null,
        }),
        el('span', { class: 'iface-col-aa', text: p.residue || '·' }),
        el('span', { class: 'iface-col-pos num', text: String(p.position) }),
        mutated.has(p.position)
          ? el('span', { class: 'iface-col-mut', 'aria-hidden': 'true', text: '△' })
          : null);
      strip.append(col);
    }

    const n = Object.keys(contacts).length;
    const body = el('div', {}, strip,
      el('div', { class: 'legend' },
        el('span', { class: 'lg-item' },
          el('span', { class: 'lg-rule', style: 'background:var(--ink-4)' }),
          'bar: structures supporting the position, of 10'),
        el('span', { class: 'lg-item' },
          el('span', { class: 'lg-dot', style: 'background:var(--teal)' }),
          `square: nearest heavy atom in ${this.detail.structure.pdb_id}, darker is nearer`),
        el('span', { class: 'lg-item' }, '△ engineered mutation in this construct')));

    return region('Consensus pocket, and this complex against it', {
      caption: `The ${this.index.pocket.n_positions} positions a peptide `
        + `contacts within ${cutoff} Å in at least `
        + `${this.index.pocket.min_structures} of the ten complexes. This `
        + 'figure needs no coordinate file: both the support counts and the '
        + 'distances are read from the saved pocket definition.',
    },
      figure(2, `${n} of ${this.index.pocket.n_positions} consensus positions `
        + `in contact in ${this.detail.structure.pdb_id}`, body,
      'A contact is a geometric fact about these structures. It is not a '
      + 'measurement of any residue’s energetic contribution to binding, and '
      + 'a position supported by ten structures is not thereby a more '
      + 'important one — setmelanotide appears in three of the ten entries and '
      + 'afamelanotide in two, so the entries are not independent.'));
  }

  /* ---------------- the linked inspector ---------------- */

  inspectorRegion() {
    const host = el('div', { class: 'iface-inspector-host' }, this.inspector());
    return region('Residue inspector', {
      caption: 'Position, contacts, chemistry and provenance for one pocket '
        + 'residue, linked to both figures above.',
    }, host);
  }

  inspector() {
    if (this.selected === null) {
      return emptyState('◇', 'No residue selected',
        'Click a ringed position in the structure, or a column in the contact '
        + 'map, to see its contacts, its side-chain chemistry and where each '
        + 'number came from.');
    }
    const pos = this.selected;
    const entry = (this.index.pocket.positions || [])
      .find((p) => p.position === pos);
    if (!entry) {
      return emptyState('◇', `Position ${pos} is not a consensus pocket position`,
        'Only the consensus positions carry support counts and a transfer.');
    }

    const s = this.detail.structure;
    const g = this.detail.geometry;
    const d = (s.contacts || {})[String(pos)];
    const observed = g
      ? (g.receptor_residues || []).find((r) => r.position === pos)
      : null;
    const wrap = el('div', { class: 'iface-inspector' });

    /* -- identity and geometry -- */
    const head = el('div', { class: 'iface-insp-head' },
      el('p', { class: 'iface-insp-aa display' },
        `${entry.residue || '·'}${entry.position}`),
      el('div', { class: 'iface-insp-facts' },
        fact('support', `${entry.support} / 10`, 'structures in which this '
          + 'position contacts a peptide'),
        fact('here', d === undefined || d === null
          ? 'no contact'
          : `${d.toFixed(2)} Å`, d === undefined || d === null
          ? `not within ${this.index.pocket.cutoff_angstrom} Å in ${s.pdb_id}`
          : `nearest heavy atom to the peptide in ${s.pdb_id}`),
        fact('observed', observed ? observed.observed : '—',
          observed && observed.reference && observed.observed !== observed.reference
            ? 'differs from the reference sequence — an engineered mutation. '
              + 'Geometry is from the structure, identity from UniProt.'
            : 'as deposited in this construct')));
    wrap.append(head);

    /* -- the contacts this residue makes, by type -- */
    wrap.append(el('p', { class: 'label iface-insp-label', text: 'contacts at this position' }));
    if (!g) {
      wrap.append(el('p', { class: 'fine muted',
        text: 'Per-pair detail needs the coordinate file, which is not cached '
          + 'for this entry. The distance above comes from the saved pocket '
          + 'definition and is unaffected.' }));
    } else {
      const rows = [];
      for (const [key, spec] of this.kinds()) {
        const pairs = (g.interactions?.[key] || [])
          .filter((p) => p.receptor_position === pos);
        if (!pairs.length) continue;
        rows.push({ key, spec, pairs });
      }
      if (!rows.length) {
        wrap.append(el('p', { class: 'fine muted',
          text: `No typed interaction at this position in ${s.pdb_id}. The `
            + 'position is in the consensus pocket on proximity, which is a '
            + 'weaker statement than a typed contact.' }));
      } else {
        for (const { key, spec, pairs } of rows) {
          wrap.append(el('details', { class: 'iface-pairs', open: 'open' },
            el('summary', {},
              el('span', { class: 'iface-toggle-rule', style: `background:${KIND_COLOUR[key]}` }),
              el('span', { text: spec.label }),
              el('span', { class: 'num fine', text: `${pairs.length}` }),
              el('span', {
                class: `iface-ev iface-ev-${EVIDENCE_TONE[spec.evidence] || 'stone'}`,
                text: spec.evidence,
              })),
            el('p', { class: 'fine muted', text: spec.criterion }),
            spec.caveat ? el('p', { class: 'fine iface-caveat-inline', text: spec.caveat }) : null,
            pairTable(pairs)));
        }
      }
    }

    /* -- side-chain chemistry -- */
    wrap.append(this.chemistryBlock(entry));

    /* -- transfer to the other receptors -- */
    wrap.append(this.transferBlock(entry));
    return wrap;
  }

  chemistryBlock(entry) {
    const chem = this.index.chemistry || {};
    const res = (chem.by_residue || {})[entry.residue];
    const box = el('div', { class: 'iface-insp-sect' },
      el('p', { class: 'label', text: 'side-chain chemistry' }));
    if (!res) {
      box.append(el('p', { class: 'fine muted',
        text: `No descriptor set for residue ${entry.residue || '—'}.` }));
      return box;
    }
    const prov = chem.descriptor_provenance || {};
    const unavailable = new Set(res.unavailable || []);
    for (const [title, names] of CHEM_GROUPS) {
      const dl = el('dl', { class: 'ledger ledger-tight' });
      for (const n of names) {
        const v = res.descriptors?.[n];
        const p = unavailable.has(n) ? 'unavailable' : (prov[n] || 'computed');
        dl.append(el('dt', { text: titleCase(n) }),
          el('dd', {},
            p === 'unavailable' ? el('span', { class: 'iface-na', text: 'n/a' })
              : fmt(v, 4),
            el('span', { class: `iface-prov iface-prov-${p}`, text: p })));
      }
      box.append(el('p', { class: 'fine muted iface-insp-sub', text: title }), dl);
    }
    box.append(el('div', { class: 'note note-amber' },
      el('p', { text: chem.note || '' }),
      (res.caveats || []).length
        ? el('p', { text: res.caveats.join(' ') }) : null));
    return box;
  }

  transferBlock(entry) {
    const transfer = this.index.pocket.transfer || {};
    const box = el('div', { class: 'iface-insp-sect' },
      el('p', { class: 'label', text: 'this position on the other receptors' }));
    const table = el('table', { class: 'register' },
      el('thead', {}, el('tr', {},
        el('th', { text: 'receptor' }),
        el('th', { text: 'residue here' }),
        el('th', { class: 'num', text: 'pocket identity' }),
        el('th', { text: 'basis' }))),
      el('tbody', {}));
    const body = table.querySelector('tbody');
    const ref = this.index.pocket.reference_receptor;
    body.append(el('tr', {},
      el('td', { text: `${ref} (reference)` }),
      el('td', { class: 'num', text: entry.residue || '—' }),
      el('td', { class: 'num', text: '1.000' }),
      el('td', {}, el('span', { class: 'iface-ev iface-ev-teal', text: 'experimental' }))));
    for (const [name, aa] of Object.entries(entry.transferred || {})) {
      const t = transfer[name] || {};
      body.append(el('tr', {},
        el('td', { text: name }),
        el('td', { class: 'num', text: aa === '-' ? 'gap' : (aa || '—') }),
        el('td', { class: 'num',
          text: t.identity_to_reference === undefined ? '—'
            : t.identity_to_reference.toFixed(3) }),
        el('td', {}, el('span', { class: 'iface-ev iface-ev-copper', text: 'transferred' }))));
    }
    box.append(el('div', { class: 'table-wrap' }, table));
    box.append(el('div', { class: 'note note-copper' },
      el('p', { text: (this.index.evidence_classes?.transferred?.detail) || '' })));
    return box;
  }

  /* ---------------- interface measurements ---------------- */

  measurementRegion() {
    const s = this.detail.structure;
    const sasa = s.sasa || {};
    const geo = s.interface_geometry || {};
    const near = s.nearest_distances || {};
    const sweep = s.cutoff_sensitivity || {};

    const readings = el('div', { class: 'readings' },
      reading('buried surface area', sasa.buried_surface_area_a2, 'Å²',
        'Shrake-Rupley, receptor and peptide chains only'),
      reading('interface area', sasa.interface_area_a2, 'Å²',
        'half the buried surface area'),
      reading('peptide buried', sasa.peptide_fraction_buried, '',
        'fraction of the free peptide’s surface lost on binding'),
      reading('centroid separation', geo.centroid_separation_a, 'Å',
        'receptor interface centroid to peptide centroid'));

    const dl = el('dl', { class: 'ledger' });
    const row = (k, v, unit, how) => {
      dl.append(el('dt', { text: k }),
        el('dd', {}, v === null || v === undefined ? 'none'
          : measure(v, unit), el('span', { class: 'prov', text: how })));
    };
    row('nearest donor–acceptor', near.donor_acceptor_a, 'Å',
      'closest hydrogen-bond candidate. A heavy-atom distance only; no angle '
      + 'was checked, because there are no hydrogens to check one against.');
    row('nearest charged pair', near.charged_pair_a, 'Å',
      'closest oppositely charged group pair. The peptide-side assignment is '
      + 'inferred from geometry.');
    row('nearest cation to ring', near.cation_to_ring_centroid_a, 'Å',
      'closest cationic group to an aromatic ring centroid');
    row('peptide residues', s.n_peptide_residues, '',
      'every residue of the ligand chain, canonical or not');
    row('receptor interface residues', s.n_receptor_interface_residues, '',
      'typed within 8 Å of the peptide');
    row('contacts per peptide residue', geo.contacts_per_peptide_residue, '',
      'total typed contacts divided by peptide length');
    row('peptide rings found', geo.peptide_rings_found, '',
      'inferred from distance-based connectivity, not from residue names');
    row('peptide cationic groups', geo.peptide_cationic_groups, '',
      'inferred: guanidinium and primary amine patterns');

    const sweepBody = el('div', { class: 'iface-sweeps' });
    for (const [field, series] of Object.entries(sweep)) {
      const keys = Object.keys(series);
      const dlx = el('dl', { class: 'ledger ledger-tight' });
      for (const k of keys) {
        dlx.append(el('dt', { class: 'num', text: `${k} Å` }),
          el('dd', { class: 'num', text: String(series[k]) }));
      }
      sweepBody.append(el('div', {},
        el('p', { class: 'fine muted iface-insp-sub', text: titleCase(field) }), dlx));
    }

    return region('Interface measurements', {
      caption: 'Computed from the deposited coordinates of this complex under '
        + 'the criteria recorded with the profile. Geometric consequences of an '
        + 'experimental structure, not measurements of binding.',
    },
      readings,
      el('div', { class: 'split-grid iface-measure-grid' },
        el('div', {}, el('p', { class: 'label', text: 'nearest approaches and geometry' }), dl),
        el('div', {},
          el('p', { class: 'label', text: 'sensitivity to the cutoff' }),
          el('p', { class: 'fine muted',
            text: 'Each criterion is a choice. The count at neighbouring '
              + 'cutoffs is reported so the choice is visible rather than '
              + 'assumed away.' }),
          sweepBody)));
  }

  /* ---------------- the ten complexes ---------------- */

  panelRegion() {
    const kinds = this.kinds();
    const table = el('table', { class: 'register' });
    table.append(el('thead', {}, el('tr', {},
      el('th', { text: 'PDB' }),
      el('th', { text: 'ligand' }),
      el('th', { text: 'method' }),
      el('th', { class: 'num', text: 'Å' }),
      el('th', { class: 'num', text: 'pocket contacts' }),
      ...kinds.map(([, spec]) =>
        el('th', { class: 'num', title: spec.criterion, text: shortKind(spec.label) })),
      el('th', { class: 'num', text: 'buried Å²' }))));
    const body = el('tbody', {});
    for (const s of this.index.structures || []) {
      body.append(el('tr', {
        class: s.pdb_id === this.pdbId ? 'is-selected' : null,
        onclick: () => { if (s.pdb_id !== this.pdbId) this.loadDetail(s.pdb_id); },
      },
        el('td', {}, el('span', { class: 'cell-name num', text: s.pdb_id })),
        el('td', { text: s.ligand }),
        el('td', { class: 'fine muted', text: s.method }),
        el('td', { class: 'num', text: s.resolution?.toFixed ? s.resolution.toFixed(2) : '—' }),
        el('td', { class: 'num', text: String(s.n_pocket_contacts ?? '—') }),
        ...kinds.map(([, spec]) =>
          el('td', { class: 'num', text: String(s[spec.profile_field] ?? '—') })),
        el('td', { class: 'num',
          text: s.sasa?.buried_surface_area_a2
            ? Math.round(s.sasa.buried_surface_area_a2).toLocaleString('en-US') : '—' })));
    }
    table.append(body);

    return region('Across the ten complexes', {
      caption: `${this.index.n_structures} structures covering `
        + `${this.index.n_distinct_ligands} distinct ligands. Setmelanotide `
        + 'appears in three entries and afamelanotide in two, so the rows are '
        + 'not ten independent observations. Select a row to open that complex.',
    }, el('div', { class: 'table-wrap' }, table));
  }

  /* ---------------- caveats ---------------- */

  caveatRegion() {
    const items = [
      ...(this.detail.structure.caveats || []),
      ...(this.detail.structure.pocket_caveats || []),
      ...(this.detail.geometry?.caveats || []),
      ...(this.index.pocket.caveats || []),
    ];
    const seen = new Set();
    const list = el('div', { class: 'caveats' });
    for (const text of items) {
      if (seen.has(text)) continue;
      seen.add(text);
      list.append(el('div', { class: 'caveat' },
        el('span', { class: 'caveat-mark', 'aria-hidden': 'true', text: '※' }),
        el('span', { text })));
    }

    const mp = this.index.model_position || {};
    const gate = mp.result?.criteria || {};
    const gateList = el('div', { class: 'gate' });
    for (const [name, c] of Object.entries(gate)) {
      gateList.append(el('div', { class: `gate-row ${c.met ? 'is-ok' : 'is-no'}` },
        el('span', { class: 'gate-mark', text: c.met ? '✓' : '✕' }),
        el('div', {},
          el('span', { class: 'gate-name', text: titleCase(name) }),
          el('span', { class: 'gate-reason num',
            text: `${fmt(c.actual, 3)} · required ${c.required}` }))));
    }

    return region('What these numbers do not establish', {
      caption: 'Every criterion above is a stated distance or angle threshold, '
        + 'and every structure is one crystal or cryo-EM form of one complex.',
    },
      list,
      el('div', { class: 'note note-copper' },
        el('p', { class: 'label', text: 'the affinity model’s gate' }),
        el('p', { text: mp.statement || '' })),
      gateList,
      el('p', { class: 'fine muted iface-note', text: mp.retrospective_only || '' }));
  }
}

/* ---------------- small parts ---------------- */

function fact(name, value, how) {
  return el('div', { class: 'iface-fact' },
    el('span', { class: 'label', text: name }),
    el('p', { class: 'iface-fact-v num', text: String(value) }),
    el('p', { class: 'fine muted', text: how }));
}

function reading(name, value, unit, how) {
  return el('div', { class: 'reading' },
    label(name),
    el('p', { class: 'reading-value' },
      value === null || value === undefined ? '—' : fmt(value, 4),
      unit ? el('span', { class: 'unit', text: unit }) : null),
    el('p', { class: 'fine muted', text: how }));
}

function pairTable(pairs) {
  const table = el('table', { class: 'register iface-pair-table' });
  const hasAngle = pairs.some((p) => p.interplanar_deg !== undefined);
  table.append(el('thead', {}, el('tr', {},
    el('th', { text: 'receptor' }),
    el('th', { text: 'peptide' }),
    el('th', { class: 'num', text: 'Å' }),
    hasAngle ? el('th', { class: 'num', text: 'deg' }) : null)));
  const body = el('tbody', {});
  for (const p of pairs) {
    const rec = p.receptor_atom || (p.receptor_atoms || []).join(' ');
    const pep = p.peptide_atom || (p.peptide_atoms || []).join(' ');
    body.append(el('tr', {},
      el('td', { class: 'num', text: rec }),
      el('td', { class: 'num', text: pep }),
      el('td', { class: 'num', text: p.distance_a?.toFixed ? p.distance_a.toFixed(2) : '—' }),
      hasAngle
        ? el('td', { class: 'num',
          text: p.interplanar_deg === undefined ? '—' : p.interplanar_deg.toFixed(1) })
        : null));
  }
  table.append(body);
  return el('div', { class: 'table-wrap' }, table);
}

function shortKind(label) {
  return label
    .replace('Hydrogen-bond candidate', 'H-bond cand.')
    .replace('Hydrophobic contact', 'phobic')
    .replace('Aromatic stacking', 'stacked')
    .replace('Aromatic T-shaped', 'T-shaped')
    .replace('Aromatic oblique', 'oblique')
    .replace('Salt bridge', 'salt')
    .replace('Cation-pi', 'cation-π');
}
