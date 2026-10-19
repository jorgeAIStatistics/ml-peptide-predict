# ATLAS — build status

Last updated: 2026-10-18 · 218 tests passing

## What this is

A computational pharmacology research platform for literature analysis,
preclinical computational research and model validation. It is **not** a
clinical tool: it produces no dosing, administration or combination-safety
guidance, and no individual risk estimates.

The guiding constraint is that scientific content is **retrieved from public
databases at runtime**, not written into the codebase. The compound registry
holds naming facts only; all chemistry, development status and literature
counts come from ChEMBL, PubChem, PubMed and ClinicalTrials.gov with the
retrieval URL, timestamp and body hash recorded.

## Working now

### Core (`atlas/core/`)
- **`units.py`** — SI dimensional analysis for quantitative evidence. Composite
  units parse as pharmacologists write them (`mL/min/kg`, `ng*h/mL`). Mass↔molar
  conversion requires an explicit molecular weight and raises rather than
  defaulting, because several registry entries have no known MW.
- **`evidence.py`** — A–E evidence classification, experimental context,
  structured `Abstention`. The schema cannot represent a measured null result
  and an unmeasured endpoint identically, and refuses to label an observation
  class A when it carries no data.

### Data layer (`atlas/sources/`)
Cached, rate-limited HTTP with per-host limits matching published provider
figures, 429 backoff, and body SHA-256 on every fetch. Clients for PubMed
(E-utilities), PubChem (PUG REST), ChEMBL and ClinicalTrials.gov v2. Retraction
status is actually inspected, so `False` is a finding and `None` means unchecked.

### Registry (`atlas/registry/`)
46 entries. Identity resolution scores candidates on **exact name agreement**,
never on search rank, after live testing showed ChEMBL returns the approved
SEMAGLUTIDE record fourth behind three unnamed structure records, and returns
5-aminovaleric acid as the top hit for "5-Amino-1MQ".

Current resolution: 33 resolved · 7 mixtures · 3 unresolved · 2 structurally
ambiguous · 1 supplier-defined.

The name gate automatically rejected GHK-Cu→ibuprofen lysine, IGF-1
LR3→iobenguane, Semax→yunnanin A, SNAP-8→S-nitroso-N-acetylpenicillamine and
melanotan II→afamelanotide. Ambiguity is decided **structurally**, so duplicate
deposits resolve normally while genuinely divergent records block
structure-dependent models — kisspeptin resolves to ~1302 Da in ChEMBL and
~5857 Da in PubChem (kisspeptin-10 vs -54) and is flagged.

### Mechanistic engine (`atlas/mechanistic/`)
One-compartment IV, one-compartment first-order absorption, two-compartment IV,
each paired with its closed-form solution. Mass balance is modelled, not
asserted: the depot drains into both the central compartment (F) and an
unabsorbed sink (1−F), so conservation is testable.

**Verification: 63/63 checks across 9 model/parameter combinations.** Exact
identities checked: `AUC == F·Dose/CL`, `eliminated(∞) == F·Dose`, superposition
under dose scaling, mean steady-state `== F·Dose/(CL·τ)`, conservation, and
convergence under tightening tolerance. Analytical agreement at ~1e-12 relative.

The `ka == ke` Bateman singularity is handled by its finite limit. Flip-flop
kinetics are detected, where the terminal slope reflects absorption rather than
elimination.

### Evidence gate (`atlas/analysis/`)
Gates six analysis kinds on measured evidence. Across the registry:

| analysis | permitted |
|---|---|
| exploratory simulation | 46/46 |
| literature summary | 38/46 |
| mechanism hypothesis | 38/46 |
| structure-based prediction | 32/46 |
| human PK simulation | **0/46** |
| quantitative safety prediction | **0/46** |
| combination effect prediction | **0/46** |

Human PK is 0/46 because it requires source-backed *extracted* parameters, and
none have been extracted yet. That is the truthful state, not a placeholder.

### Web interface (`atlas/api/`)
Flask serves a JSON API and a four-workspace frontend from one process. The
frontend has **no runtime dependencies** — no bundler, no `node_modules`, no
CDN — so it runs offline and starts instantly. Charts are hand-drawn SVG,
re-rendered on container resize so axis type stays at its true size.

```bash
PYTHONPATH=. python3 -m atlas.api.app      # http://127.0.0.1:5055
```

| workspace | what it does |
|---|---|
| Compound Atlas | sortable registry, filters, inspector with per-field provenance |
| Model Studio | runs the real solver; NCA, diagnostics, equations, conservation plot |
| Evidence & Gaps | compound × analysis matrix, every refusal reason on hover/focus |
| Verification | all 63 solver checks with metric against tolerance |

The API reads from the versioned snapshots rather than querying external
databases per request, so responses are fast, reproducible and offline, and
each carries the `snapshot_id` its evidence came from. Simulation is the one
live computation.

**The gate is enforced at the API boundary.** A run attributed to a compound
without source-backed parameters returns class D with the abstention attached,
so the interface cannot present a trajectory as a statement about a compound.
Attributing a run to BPC-157 returns, alongside the chart: *"not a statement
about BPC-157 in humans … Measured evidence: 3 human PK publications, 0 trials
with posted results."*

Accessibility: keyboard-navigable table rows and matrix cells, visible focus
rings, `aria-sort` on sortable headers, a keyboard-resizable inspector divider,
and `prefers-reduced-motion` honoured. Dark mode is a token-level palette, not
an inverted sheet.

## Verification vs validation

This distinction is load-bearing and currently sits on one side of it.

- **Verified** — the solver solves the equations it claims to solve, to machine
  precision, against closed-form solutions and conservation laws.
- **Not validated** — nothing here has been shown to describe a real biological
  system. No model has been fitted to or tested against observed measurements.

A model passing all 63 checks says nothing about any compound's behaviour in an
organism.

## Not built yet

1. **PK parameter extraction** — the gap keeping human PK at 0/46. Needs
   clearance, volume and bioavailability extracted from identified records with
   units, uncertainty and verification against source text.
2. **Trained ML model** — no model trained or evaluated. Requires a legitimate
   dataset with held-out evaluation; no performance figure should appear until
   it is measured.
3. **Knowledge graph** — typed schema and traversal not implemented.
4. **Evidence ledger database** — provenance is per-record; the append-only
   versioned store is not built.
5. **Geroscience module** — hallmark framework not implemented.
6. **Interface gaps** — no multi-compound comparison view, no pathway/network
   visualisation, and the Model Studio cannot yet overlay observed data against
   a simulated curve (which needs item 1).

## Known limitations

- Literature queries use exact phrases, so `CJC-1295 with DAC` returns 0 records
  although literature on "CJC-1295" exists. The DAC and non-DAC entities are
  kept separate deliberately rather than merged, which understates retrievable
  literature for both. Resolving this needs per-record identity adjudication,
  not a looser query.
- Broad names inflate counts: `glutathione` matches ~199k records partly through
  its aliases. Counts measure query recall, not compound-specific evidence.
- `-1` in a literature count means *not measured* (a failed request), never zero.
- Python 3.14 is the only interpreter present; `torch` and `rdkit` have cp314
  candidates but are not installed and are not needed for current functionality.
- Docker daemon is not running; no Postgres/Redis/Neo4j locally. Current code
  needs none of them.

## Reproducing

```bash
python3 -m pytest -q                              # 218 tests
PYTHONPATH=. python3 -m atlas.mechanistic.verify  # 63/63 solver checks
PYTHONPATH=. python3 -m atlas.registry.build      # resolve registry (live)
PYTHONPATH=. python3 -m atlas.analysis.report     # evidence/feasibility sweep
PYTHONPATH=. python3 -m atlas.api.app             # web interface on :5055
```

Append `--offline` to the last two to rebuild from the HTTP cache without
network access.
