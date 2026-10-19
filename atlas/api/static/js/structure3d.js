/* ATLAS — orthographic structure viewer.

   This is not a molecular-graphics package and does not pretend to be one.
   ATLAS loads no external script: the whole frontend is ES modules served
   from the same process, and adding a five-megabyte viewer bundle from a CDN
   would make a page that currently works offline depend on the network to
   draw an experimental structure. So the structure is drawn here, in SVG,
   the same way every other figure in this instrument is drawn.

   What that means for the reader is stated on the figure rather than implied:
   a C-alpha trace, interface side chains and the peptide as stick figures,
   with the typed contacts as dashed segments. There is no surface, no
   electron density, no secondary-structure assignment and no occlusion —
   depth is cued by opacity and line weight alone, so a bond behind another
   shows through it.

   Coordinates arrive already translated to the interface centroid and
   rounded to 0.01 A by atlas.ml.interface_geometry. Nothing here computes a
   distance, an angle or an interaction: every mark corresponds to a record
   the API sent. */

import { svg, el, tip, hideTip } from './ui.js';

/* Element colours. Deliberately the instrument's own accents rather than
   CPK: this palette has to coexist with the rest of the canvas, and the
   figure legend names every colour it uses. */
const ELEMENT = {
  C: 'var(--ink-3)',
  N: 'var(--blue)',
  O: 'var(--copper)',
  S: 'var(--amber)',
  F: 'var(--teal)',
  CL: 'var(--teal)',
  BR: 'var(--teal)',
  I: 'var(--teal)',
};

export const KIND_COLOUR = {
  hbond_candidate: 'var(--blue)',
  salt_bridge: 'var(--copper)',
  hydrophobic: 'var(--stone-ink)',
  aromatic_stacked: 'var(--teal)',
  aromatic_tshaped: 'var(--teal-deep)',
  aromatic_oblique: 'var(--teal)',
  cation_pi: 'var(--amber)',
};

/* A hydrogen-bond candidate is drawn with a longer gap than a salt bridge,
   so the candidate class is legible without colour. */
const KIND_DASH = {
  hbond_candidate: '1 3',
  salt_bridge: '4 2',
  hydrophobic: '1 2',
  aromatic_stacked: '5 2',
  aromatic_tshaped: '5 2',
  aromatic_oblique: '2 2',
  cation_pi: '3 2',
};

/* ---------------- rotation ---------------- */

function rotation(yaw, pitch) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cx = Math.cos(pitch), sx = Math.sin(pitch);
  // Rx(pitch) * Ry(yaw), stored row-major.
  return [
    [cy, 0, sy],
    [sx * sy, cx, -sx * cy],
    [-cx * sy, sx, cx * cy],
  ];
}

function multiply(a, b) {
  const out = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
    }
  }
  return out;
}

function apply(m, v) {
  return [
    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
  ];
}

/** The default view: looking into the pocket, with the peptide nearest.
    Derived from the receptor-to-peptide vector so every complex opens in a
    comparable orientation instead of in whatever frame it was deposited in. */
function defaultOrientation(geometry) {
  const mean = (pts) => {
    if (!pts.length) return [0, 0, 0];
    const s = pts.reduce((a, p) => [a[0] + p[0], a[1] + p[1], a[2] + p[2]], [0, 0, 0]);
    return s.map((v) => v / pts.length);
  };
  const pep = mean((geometry.peptide?.atoms || []).map((a) => a.xyz));
  const rec = mean((geometry.receptor_residues || []).flatMap((r) => r.atoms.map((a) => a.xyz)));
  const v = [pep[0] - rec[0], pep[1] - rec[1], pep[2] - rec[2]];
  const n = Math.hypot(v[0], v[1], v[2]);
  if (!(n > 1e-6)) return { yaw: 0, pitch: 0 };
  const u = v.map((c) => c / n);
  const yaw = -Math.atan2(u[0], u[2]);
  const h = Math.hypot(u[0], u[2]);
  const pitch = Math.atan2(u[1], h);
  return { yaw, pitch };
}

/* ---------------- the viewer ---------------- */

export class StructureViewer {
  /**
   * @param host      element to draw into
   * @param geometry  the `geometry` object from /api/interfaces/<pdb>
   * @param opts.pocketPositions  Set of consensus pocket positions
   * @param opts.support          position -> number of structures supporting it
   * @param opts.onSelect         called with a position when a residue is picked
   */
  constructor(host, geometry, opts = {}) {
    this.host = host;
    this.geometry = geometry;
    this.opts = opts;
    this.enabled = new Set(opts.enabled || []);
    this.selected = opts.selected ?? null;
    this.scope = opts.scope || 'interface';
    const o = defaultOrientation(geometry);
    this.base = rotation(o.yaw, o.pitch);
    this.yaw = 0;
    this.pitch = 0;
    this.install();
  }

  setEnabled(set) { this.enabled = new Set(set); this.draw(); }
  setSelected(pos) { this.selected = pos; this.draw(); }
  setScope(scope) { this.scope = scope; this.draw(); }
  reset() { this.yaw = 0; this.pitch = 0; this.draw(); }

  destroy() {
    if (this._ro) this._ro.disconnect();
    this._ro = null;
  }

  install() {
    const host = this.host;
    let dragging = false;
    let last = null;

    const down = (e) => {
      dragging = true;
      last = [e.clientX, e.clientY];
      host.setPointerCapture?.(e.pointerId);
      host.classList.add('is-rotating');
    };
    const move = (e) => {
      if (!dragging) return;
      const dx = e.clientX - last[0];
      const dy = e.clientY - last[1];
      last = [e.clientX, e.clientY];
      this.yaw += dx * 0.008;
      this.pitch = Math.max(-1.5, Math.min(1.5, this.pitch + dy * 0.008));
      this.draw();
    };
    const up = (e) => {
      dragging = false;
      host.releasePointerCapture?.(e.pointerId);
      host.classList.remove('is-rotating');
    };

    host.addEventListener('pointerdown', down);
    host.addEventListener('pointermove', move);
    host.addEventListener('pointerup', up);
    host.addEventListener('pointercancel', up);
    // Keyboard rotation, so the view is reachable without a pointer.
    host.addEventListener('keydown', (e) => {
      const step = 0.12;
      const map = {
        ArrowLeft: () => { this.yaw -= step; },
        ArrowRight: () => { this.yaw += step; },
        ArrowUp: () => { this.pitch = Math.max(-1.5, this.pitch - step); },
        ArrowDown: () => { this.pitch = Math.min(1.5, this.pitch + step); },
        Home: () => { this.yaw = 0; this.pitch = 0; },
      };
      if (map[e.key]) { e.preventDefault(); map[e.key](); this.draw(); }
    });

    this._ro = new ResizeObserver(() => this.draw());
    this._ro.observe(host);
    this.draw();
  }

  /** Projected screen coordinates and depth for one point.
      The view rotates about the current scope's centre, so switching scope
      recentres the drawing rather than swinging it off the frame. */
  project(xyz, frame) {
    const c = frame.centre;
    const p = apply(frame.m, [xyz[0] - c[0], xyz[1] - c[1], xyz[2] - c[2]]);
    return [frame.cx + p[0] * frame.k, frame.cy - p[1] * frame.k, p[2]];
  }

  /** Centre and radius for each scope, computed once per structure.

      The radius is the 97th percentile rather than the maximum: in the three
      X-ray fusion constructs a single distant residue of the glycogen-synthase
      partner is 90 A out, and scaling to it shrinks the receptor to a smudge.
      The tail is clipped instead, which the caption says. */
  bounds() {
    if (this._bounds) return this._bounds;
    const g = this.geometry;
    const trace = (g.receptor_trace || []).map((t) => t.xyz);
    const centre = trace.length
      ? trace.reduce((a, p) => [a[0] + p[0], a[1] + p[1], a[2] + p[2]], [0, 0, 0])
        .map((v) => v / trace.length)
      : [0, 0, 0];
    const radii = trace
      .map((p) => Math.hypot(p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]))
      .sort((a, b) => a - b);
    const q = radii.length
      ? radii[Math.min(radii.length - 1, Math.floor(radii.length * 0.97))]
      : 20;
    this._bounds = {
      interface: { centre: [0, 0, 0], extent: Math.max(1, g.extent_a || 20) },
      receptor: { centre, extent: Math.max(1, q) },
    };
    return this._bounds;
  }

  frame() {
    const width = Math.max(320, this.host.clientWidth || 640);
    const height = Math.max(300, Math.min(520, Math.round(width * 0.68)));
    const m = multiply(rotation(this.yaw, this.pitch), this.base);
    // The interface scope fits the typed shell and the peptide; the receptor
    // scope fits the whole aligned chain, which for the three fusion
    // constructs is mostly not the receptor.
    const b = this.bounds()[this.scope === 'receptor' ? 'receptor' : 'interface'];
    const pad = 22;
    const k = (Math.min(width, height) / 2 - pad) / b.extent;
    return {
      width, height, m, k, cx: width / 2, cy: height / 2,
      extent: b.extent, centre: b.centre,
    };
  }

  /** Depth cue: 0 at the back, 1 at the front. */
  static depth(z, extent) {
    return Math.max(0, Math.min(1, 0.5 + z / (2 * extent)));
  }

  draw() {
    const g = this.geometry;
    const f = this.frame();
    const host = this.host;
    host.textContent = '';

    const root = svg('svg', {
      width: f.width, height: f.height, viewBox: `0 0 ${f.width} ${f.height}`,
      class: 'struct-svg', role: 'img',
      'aria-label':
        `Orthographic projection of the ${g.pdb_id} ${g.ligand}-MC4R complex: `
        + `receptor C-alpha trace, interface side chains, the peptide ligand `
        + `and ${[...this.enabled].length} enabled contact overlays. `
        + `Rotate with the arrow keys.`,
    });

    const layer = (cls) => {
      const grp = svg('g', { class: cls });
      root.append(grp);
      return grp;
    };
    const gTrace = layer('struct-trace');
    const gShell = layer('struct-shell');
    const gOverlay = layer('struct-overlay');
    const gPeptide = layer('struct-peptide');
    const gPick = layer('struct-pick');

    /* --- receptor C-alpha trace ------------------------- */
    const trace = g.receptor_trace || [];
    let run = [];
    let runMapped = true;
    const flushRun = () => {
      if (run.length > 1) {
        gTrace.append(svg('polyline', {
          points: run.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' '),
          fill: 'none',
          stroke: runMapped ? 'var(--ink-5)' : 'var(--rule)',
          'stroke-width': runMapped ? 1.4 : 1,
          'stroke-dasharray': runMapped ? null : '2 3',
          'stroke-linejoin': 'round',
          opacity: runMapped ? 0.85 : 0.5,
        }));
      }
      run = [];
    };
    for (let i = 0; i < trace.length; i++) {
      const t = trace[i];
      const mapped = t.position !== null && t.position !== undefined;
      const prev = trace[i - 1];
      const broken = prev && (
        Math.hypot(t.xyz[0] - prev.xyz[0], t.xyz[1] - prev.xyz[1], t.xyz[2] - prev.xyz[2]) > 4.5
        || mapped !== runMapped);
      if (broken) flushRun();
      if (!run.length) runMapped = mapped;
      const p = this.project(t.xyz, f);
      run.push(p);
    }
    flushRun();

    /* --- interface side chains -------------------------- */
    const pocket = this.opts.pocketPositions instanceof Set
      ? this.opts.pocketPositions : new Set();
    for (const res of g.receptor_residues || []) {
      const inPocket = res.position !== null && pocket.has(res.position);
      const isSelected = res.position !== null && res.position === this.selected;
      const pts = res.atoms.map((a) => this.project(a.xyz, f));
      const zs = pts.map((p) => p[2]);
      const near = StructureViewer.depth(
        zs.reduce((a, b) => a + b, 0) / Math.max(1, zs.length), f.extent);
      for (const [i, j] of res.bonds || []) {
        if (!pts[i] || !pts[j]) continue;
        gShell.append(svg('line', {
          x1: pts[i][0].toFixed(1), y1: pts[i][1].toFixed(1),
          x2: pts[j][0].toFixed(1), y2: pts[j][1].toFixed(1),
          stroke: isSelected ? 'var(--teal)' : (inPocket ? 'var(--ink-4)' : 'var(--ink-5)'),
          'stroke-width': isSelected ? 2.4 : (inPocket ? 1.3 : 0.8),
          'stroke-linecap': 'round',
          opacity: (isSelected ? 1 : (inPocket ? 0.5 : 0.22)) * (0.45 + 0.55 * near),
        }));
      }
    }

    /* --- contact overlays ------------------------------- */
    // Drawn from the records the API sent, one segment per saved pair.
    for (const kind of Object.keys(KIND_COLOUR)) {
      if (!this.enabled.has(kind)) continue;
      for (const pair of (g.interactions || {})[kind] || []) {
        const a = this.project(pair.a, f);
        const b = this.project(pair.b, f);
        const near = StructureViewer.depth((a[2] + b[2]) / 2, f.extent);
        const dim = this.selected !== null
          && pair.receptor_position !== this.selected;
        gOverlay.append(svg('line', {
          x1: a[0].toFixed(1), y1: a[1].toFixed(1),
          x2: b[0].toFixed(1), y2: b[1].toFixed(1),
          stroke: KIND_COLOUR[kind],
          'stroke-width': dim ? 1 : 1.8,
          'stroke-dasharray': KIND_DASH[kind],
          'stroke-linecap': 'round',
          // Selecting a residue recedes the other contacts rather than
          // hiding them: at a lower floor the figure looked broken rather
          // than focused.
          opacity: (dim ? 0.35 : 0.95) * (0.5 + 0.5 * near),
        }));
      }
    }

    /* --- the peptide ------------------------------------ */
    const pep = g.peptide || { atoms: [], bonds: [] };
    const pp = pep.atoms.map((a) => this.project(a.xyz, f));
    for (const [i, j] of pep.bonds || []) {
      if (!pp[i] || !pp[j]) continue;
      const near = StructureViewer.depth((pp[i][2] + pp[j][2]) / 2, f.extent);
      gPeptide.append(svg('line', {
        x1: pp[i][0].toFixed(1), y1: pp[i][1].toFixed(1),
        x2: pp[j][0].toFixed(1), y2: pp[j][1].toFixed(1),
        stroke: 'var(--ink-2)',
        'stroke-width': 2.2,
        'stroke-linecap': 'round',
        opacity: 0.55 + 0.45 * near,
      }));
    }
    // Heteroatoms only: carbon dots would bury the bonds they sit on.
    pep.atoms.forEach((a, i) => {
      if (a.el === 'C' || !pp[i]) return;
      const near = StructureViewer.depth(pp[i][2], f.extent);
      gPeptide.append(svg('circle', {
        cx: pp[i][0].toFixed(1), cy: pp[i][1].toFixed(1), r: 2.4,
        fill: ELEMENT[a.el?.toUpperCase()] || 'var(--ink-3)',
        opacity: 0.6 + 0.4 * near,
      }));
    });

    /* --- pick targets for the pocket residues ----------- */
    // One target per consensus pocket residue, at its side-chain centroid.
    for (const res of g.receptor_residues || []) {
      if (res.position === null || !pocket.has(res.position)) continue;
      const side = res.atoms.filter((a) => !['N', 'C', 'O'].includes(a.name));
      const atoms = (side.length ? side : res.atoms).map((a) => a.xyz);
      const c = atoms.reduce((s, p) => [s[0] + p[0], s[1] + p[1], s[2] + p[2]], [0, 0, 0])
        .map((v) => v / atoms.length);
      const p = this.project(c, f);
      const isSelected = res.position === this.selected;
      const support = this.opts.support?.[String(res.position)] ?? 0;
      const target = svg('circle', {
        cx: p[0].toFixed(1), cy: p[1].toFixed(1), r: isSelected ? 6 : 4.5,
        class: 'struct-pick-dot',
        fill: isSelected ? 'var(--teal)' : 'var(--surface)',
        stroke: isSelected ? 'var(--teal)' : 'var(--ink-4)',
        'stroke-width': 1.2,
        'fill-opacity': isSelected ? 1 : 0.65,
        tabindex: '0',
        role: 'button',
        'aria-label':
          `Position ${res.position}, ${res.reference || res.observed}, `
          + `observed in ${support} of 10 structures`,
        onclick: (e) => {
          e.stopPropagation();
          this.opts.onSelect?.(res.position === this.selected ? null : res.position);
        },
        onkeydown: (e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            this.opts.onSelect?.(res.position === this.selected ? null : res.position);
          }
        },
        onmouseenter: (e) => tip(
          `<strong>${res.reference || res.observed}${res.position}</strong>`
          + `contacts the peptide in ${support} of 10 structures`
          + (res.reference && res.observed !== res.reference
            ? `<br>observed here as ${res.observed} — engineered mutation` : ''),
          e.clientX, e.clientY),
        onmousemove: (e) => tip(
          `<strong>${res.reference || res.observed}${res.position}</strong>`
          + `contacts the peptide in ${support} of 10 structures`
          + (res.reference && res.observed !== res.reference
            ? `<br>observed here as ${res.observed} — engineered mutation` : ''),
          e.clientX, e.clientY),
        onmouseleave: hideTip,
      });
      gPick.append(target);
    }

    root.addEventListener('click', (e) => {
      if (e.target === root) this.opts.onSelect?.(null);
    });

    host.append(root);
  }
}

/** The legend. Separate from the drawing so a caption can sit between them.
    `kinds` is an ordered array of [key, spec] pairs, not an object: the
    declared order of the interaction kinds is meaningful and object key
    order does not survive JSON serialisation. */
export function viewerLegend(kinds, enabled, counts) {
  const items = [
    el('span', { class: 'lg-item' },
      el('span', { class: 'lg-rule', style: 'background:var(--ink-5)' }),
      'receptor Cα trace'),
    el('span', { class: 'lg-item' },
      el('span', { class: 'lg-rule', style: 'background:var(--ink-2);height:3px' }),
      'peptide ligand'),
    el('span', { class: 'lg-item' },
      el('span', { class: 'lg-dot', style: 'background:var(--blue)' }), 'N'),
    el('span', { class: 'lg-item' },
      el('span', { class: 'lg-dot', style: 'background:var(--copper)' }), 'O'),
    el('span', { class: 'lg-item' },
      el('span', { class: 'lg-dot', style: 'background:var(--amber)' }), 'S'),
  ];
  for (const [key, spec] of kinds || []) {
    if (!enabled.has(key)) continue;
    items.push(el('span', { class: 'lg-item' },
      el('span', {
        class: 'lg-rule',
        style: `background:${KIND_COLOUR[key] || 'var(--ink-3)'}`,
      }),
      `${spec.label.toLowerCase()} (${counts?.[key] ?? 0})`));
  }
  return el('div', { class: 'legend' }, ...items);
}
