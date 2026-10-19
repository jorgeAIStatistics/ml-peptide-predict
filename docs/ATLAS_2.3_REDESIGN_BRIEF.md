# ATLAS 2.3 — Complete Experience Reconstruction

Full frontend redesign directive. Received 2026-10-18, to begin after ATLAS 2.2
(Scientific Interpretation Engine) is committed.

## Verdict on the current interface
Rejected as "a conventional SaaS analytics dashboard with scientific
terminology". Specifically criticised:
dark dashboard appearance · horizontal numbered navigation tabs · large
rectangular containers · generic filter bars · tables as the primary experience ·
repeated boxed layouts · small teal accents · workspaces that look identical ·
too little interactive scientific visualisation.

**Note from implementation:** the reviewer is seeing the dark palette, which the
current build selects automatically from `prefers-color-scheme`. 2.3 requires
light-first as the *default*, not as an option.

## Non-negotiable: all 15 must materially change
global structure · navigation architecture · visual identity · typography ·
spatial composition · information hierarchy · data presentation · compound
exploration · Model Studio · evidence/uncertainty visualisation · interaction
patterns · component architecture · motion · responsive behaviour · the
relationship between analysis, visualisation and interpretation.

## Explicitly prohibited
Dark→light recolour only · recolouring existing panels · same layouts with new
type · everything as rounded white cards · sidebar+cards+charts on every page ·
generic admin dashboard · identical workspaces · fake visualisations · hiding the
old UI behind a landing page · spending the cycle on mockups · stopping after the
Compound Atlas · declaring it done without implementing.

## Creative direction — "ATLAS Scientific Instrument Design System"
A living research atlas, combining: scientific editorial publishing · precision
laboratory instrumentation · interactive biological atlases · high-end analytical
software · modern digital product design.

### Palette (starting tokens, not a finished design)
canvas `#F6F5F0` · surface `#FFFFFF` · stone `#EDECE6` · ink `#182423` ·
secondary `#65716E` · teal `#0E6B63` · copper `#AD6747` · muted blue `#9BBFCA` ·
divider `#DDDCD5` · amber `#B58A48`

Semantic use: teal = active/selected biological relationships; blue = model
context; amber = uncertainty/qualified evidence; copper = emphasis; neutrals =
structure. Never use colour to imply a compound is safe or beneficial.

## New information architecture (replaces the 5 numbered tabs)
Research Home · Molecular Atlas · Systems Explorer · Simulation Studio ·
Interaction Laboratory · Geroscience Atlas · Model Intelligence · Evidence Library

## Workspace requirements
- **Compound Atlas** — a molecular exploration environment; selecting a compound
  opens a research workspace with its identity as centrepiece. Keep the sortable
  table as an *advanced browsing mode*, not the default centrepiece.
- **Systems Explorer** — large interactive biological network canvas on real
  knowledge-graph data. Scientific legibility over visual complexity.
- **Model Studio** — three coordinated zones: instrument-style configuration,
  large simulation canvas, contextual interpretation inspector. Live parameter
  updates with debouncing; inspectable equations.
- **Evidence & Gaps** — a visual evidence landscape, not a matrix of cells.
  Gap inspection explains what is missing and which model depends on it.
- **Model Intelligence** — predicted-vs-observed, residuals, error by molecular
  family, reliability diagrams, applicability maps, version comparison.
- **Geroscience Atlas** — the twelve hallmarks in an original composition, not
  twelve cards in a grid.

## Design system required
Tokens for colour, type, spacing, grid, notation, chart palette, interaction
states, motion, radius, focus, breakpoints, themes. Research-specific components
(Molecular Identity Header, Sequence Annotation Viewer, Scientific Measurement,
Evidence Provenance Inspector, Parameter Control, Prediction Interval Plot,
Uncertainty Explanation, Simulation Timeline, …) rather than business-dashboard
components. Scientific state kept separate from presentation.

## Must be preserved
Backend APIs · scientific modelling · datasets · ML models · evidence-gating
logic · scientific validation · computation · numerical results. The frontend's
current visual architecture is explicitly NOT to be preserved.

## Validation required
Browser screenshots at 1440 / 1280 / 1024 / 768 / 390 / 360 px. Verify search,
filtering, evidence classification, predictions, identical numerical results,
and the existing test suite. Real data only — no mock data.

### Critique questions after each workspace
1. Does this resemble a generic SaaS dashboard?
2. Is scientific information the visual centrepiece?
3. Does the composition feel intentionally designed?
4. Is the information hierarchy obvious?
5. Are interactions coherent and responsive?
6. Are numerical results scientifically interpretable?
7. Is visual complexity justified by research usefulness?
8. Does it look like an original premium scientific instrument?

## Staged plan
0 preserve/audit · 1 design foundation + new shell · 2 Compound Atlas ·
3 Model Studio + Model Lab · 4 Evidence & Gaps · 5 Systems + Geroscience ·
6 motion/spatial coherence · 7 full quality audit.

Commit after each stage. First cycle must deliver a working new shell **and** a
materially redesigned Compound Atlas — not documentation.

## Reference imagery supplied
Four directional references (not layouts to copy): editorial scientific visual
identity; biological information as visual centrepiece (3D molecular render);
precision scientific instrumentation (dense analytical workstation);
publication-grade biological graphics (annotated pathway figure).

---

## Implementation log

### Stage 1–2 complete — commit below
**Delivered:** new design system, new application shell, new navigation
architecture, reconstructed Molecular Atlas, specimen workspace.

**Diagnosis that changed the plan.** The reviewer was seeing the *dark* palette:
the previous build selected it automatically from `prefers-color-scheme`. Light
is now the instrument's defined appearance, set on `<html data-theme="light">`,
with dark available only by explicit toggle and remembered per browser. There is
no `prefers-color-scheme` rule anywhere in the system, and a test asserts that.

**Frontend replaced, not patched.** The previous `app.js`/`app.css` had grown to
3,072 lines through 43 accumulated patch functions. Both deleted. The new
frontend is six ES modules with separated concerns — `ui` (primitives and
notation), `api` (one fetch layer), `figures` (scientific figures), `atlas`,
`workspaces`, `app` (shell and routing) — plus `design.css` (tokens) and
`components.css` (research components).

### Remaining
- **Stage 3** Simulation Studio and Model Intelligence are rebuilt in the new
  language but deserve the linked-view and live-parameter work the brief asks for.
- **Stage 4** Evidence Library is rebuilt as a reason landscape; the coverage
  map and timeline views are not built.
- **Stage 5** Systems Explorer, Geroscience Atlas and Interaction Laboratory are
  navigable with honest unbuilt states naming the evidence each needs. They stay
  empty deliberately: drawing a network or mapping compounds onto hallmarks
  without a knowledge graph would mean inventing relationships.
- **Stage 6** Motion is currently workspace entry, spine expansion and hover
  states. Shared-element transitions between landscape and specimen are not built.
- **Stage 7** Full accessibility and performance audit not yet run.
