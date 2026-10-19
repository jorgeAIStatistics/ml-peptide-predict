# ATLAS — build status

Last updated: 2026-10-19 · 569 tests passing

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

### Trained ML model (`atlas/ml/`)
**Endpoint:** MC4R binding affinity (pKi), from ChEMBL. 1,981 unique molecules
after restricting to Ki from binding assays only — IC50 depends on radioligand
concentration via Cheng-Prusoff and EC50 is functional, so pooling them would
mix physical quantities under one label. Dataset is content-hashed; a tampered
file fails to load.

MC4R was chosen because it is directly relevant to this registry (PT-141,
melanotan I and II act at melanocortin receptors) **and** because ChEMBL holds
both peptide and small-molecule ligands for it, which makes transfer between
modalities measurable rather than assumed.

Held-out results (7 model families × 4 splits):

| split | best model | MAE | R² | Spearman | 90% interval coverage |
|---|---|---|---|---|---|
| random | fingerprint kNN | 0.478 | 0.699 | 0.818 | 0.924 |
| **scaffold** | **fingerprint kNN** | **0.681** | **0.297** | **0.590** | 0.794 |
| peptide holdout | *none beat baseline* | 1.461 | −0.842 | 0.086 | 0.452 |
| small-molecule holdout | combined RF | 0.930 | −0.007 | 0.130 | 0.815 |

Three findings, all measured:

1. **A random split flatters this model by 42%.** It shares 51% of its test
   scaffolds with training; the scaffold split shares none. MAE goes 0.478 →
   0.681 and R² 0.699 → 0.297. The scaffold figure is the one quoted everywhere.
2. **Negative transfer to peptides is real.** Trained on small molecules and
   tested on peptides, *every* model scored worse than predicting the training
   mean (best 1.46 vs 1.31), with rank correlation near zero. The model
   therefore refuses peptide input — which includes PT-141, melanotan I and II.
3. **Conformal guarantees collapse off-distribution.** Coverage at 0.90 nominal
   measures 0.92 (random), 0.79 (scaffold), 0.25–0.46 (modality holdout).
   Intervals are reported with measured coverage, never the nominal level alone.

Model complexity did not pay off: gradient boosting won the random split and
lost out-of-scaffold to a Tanimoto nearest-neighbour regressor, which is what
is deployed.

Inference is gated — peptide-like input, similarity below 0.30 to training, or
an unparseable structure each return a structured abstention instead of a
number. Quality tier is `internally_validated`: held-out on the same source,
never tested against an independent external dataset or prospectively.

```bash
PYTHONPATH=. python3 -m atlas.ml.dataset   # build dataset from ChEMBL
PYTHONPATH=. python3 -m atlas.ml.train     # train, evaluate, write registry
PYTHONPATH=. python3 -m atlas.ml.peptide_train   # peptide experiment
PYTHONPATH=. python3 -m atlas.ml.calibrate_experiment  # calibration comparison
PYTHONPATH=. python3 -m atlas.ml.peptide_v2            # representation comparison
PYTHONPATH=. python3 -m atlas.mechanistic.pk_store     # extract + validate PK params
```

### Peptide ligands — served by measurement, not by a model

The small-molecule model refused peptides, leaving the registry's melanocortin
compounds unserved. Investigating produced a better answer than a model:
**PT-141, melanotan I and melanotan II all have exact structural matches in
ChEMBL with measured MC4R affinities.** They are now answered with the
measurement (evidence class A) rather than a prediction. Returning an inference
for a molecule whose affinity has been measured — and which sits in the model's
own training set — would substitute class C for class A.

| compound | median pKi | reported range | studies |
|---|---|---|---|
| PT-141 | 7.68 | 7.61 – 9.60 (98-fold) | 4 |
| melanotan I | 9.33 | 7.72 – 9.57 (71-fold) | 7 |
| melanotan II | 8.90 | 8.90 – 10.47 (37-fold) | 5 |

All three are **discordant**: independent records disagree by 37- to 98-fold.
The interface reports the median together with the range and an explicit
warning that it summarises disagreeing measurements rather than a settled value.

A peptide-specific model was built and evaluated, and is **not deployed**. On a
Taylor-Butina cluster split it reached MAE 1.208 against a mean-predictor
baseline of 1.324 — 8.8% better, R² 0.073, and 90% intervals covering 57% of
cases. Error did not fall with similarity to training, so no sub-domain is
trustworthy. It is clearly better than applying the small-molecule model (1.675),
which confirms modality-specific modelling is the right direction without making
this model good enough to serve numbers.

Two further findings:

- **Peptide-specific features did not help.** Residue composition, macrocyclisation
  and terminal state were built and validated at **94.8% against HELM sequences**
  (146/154 residue counts over 22 molecules), yet Morgan fingerprints beat them on
  every split, and ridge-on-peptide-features was worse than the mean on the
  cluster split.
- **A scaffold split is misleading for peptides.** It reports zero shared
  scaffolds while leaving 100% of test peptides with a >0.7-Tanimoto training
  neighbour, because changing one residue in a macrocycle changes the whole
  Murcko scaffold. The cluster split drops that to 49%.

### Dataset quality filters

Inspecting the raw ChEMBL query surfaced three defects, all now filtered:

- **67 activities measured mutant MC4R** (alanine scans such as I129A, Y268A)
  and were being pooled into a wild-type affinity label. Removing them moved
  melanotan I's median from 9.20 to 9.33 — the mutant data was biasing it.
- **31 records carry a ChEMBL `data_validity_comment`** ("Outside typical
  range", "Potential transcription error") and are dropped.
- **80 duplicate records** inflated apparent replication, which is why the
  median replicate spread read as exactly zero. Records are deduplicated on
  (molecule, document, rounded value) before aggregation.

### Uncertainty calibration

The previous model's nominal 90% intervals covered 79.4% of held-out molecules.
The cause was structural: the calibration set was drawn from the training pool,
so it was in-distribution, while the test set was scaffold-shifted. Conformal
coverage holds under exchangeability; these were not exchangeable.

Data roles are now strictly separate — **train / select / calibrate / test**,
all Bemis-Murcko scaffold-disjoint, enforced by a constructor that raises on any
overlap. Model selection uses the `select` split and never touches `test`.

| strategy | nominal | empirical | median width |
|---|---|---|---|
| random calibration (previous) | 0.90 | 0.838 | 2.154 |
| **disjoint calibration** | 0.90 | **0.909** | 2.636 |
| mondrian (per-band) | 0.90 | 0.889 | 2.415 |

Subgroup coverage under disjoint calibration:

| subgroup | n | coverage | MAE |
|---|---|---|---|
| similarity 0.0–0.4 | 6 | 0.500 | 1.340 |
| similarity 0.4–0.55 | 22 | 0.909 | 0.496 |
| similarity 0.55–0.7 | 79 | 0.924 | 0.599 |
| similarity 0.7–1.0 | 289 | 0.913 | 0.581 |
| peptide-like | 134 | 0.925 | 0.586 |
| small-molecule-like | 262 | 0.901 | 0.594 |

**Limitations.** Correct coverage cost ~0.5 log units of width: a median
interval of 2.64 log units spans roughly a 430-fold range in Ki. The intervals
are honest and the model is imprecise on unseen chemotypes. The lowest
similarity band held 6 test molecules, so its 0.50 coverage is an observation on
six points, not an estimate. Nominal coverage is **not** guaranteed under
distribution shift and is not claimed.

### Target relevance (A / B / C / D)

A model returns a number for any molecule whose fingerprint it can compute. That
is a statement about chemical similarity, not biology. Asked about **Pinealon**,
the earlier model returned pKi 5.45; 0 of its 18 publications concern MC4R and
its nearest documented active is 0.375 Tanimoto away. **KPV** was likewise given
4.95.

Four classes are now enforced before any number is shown:

- **A — documented engagement.** A measurement exists. Only class A may become a
  knowledge-graph interaction edge.
- **B — prediction within a validated domain.** No measurement, but ≥0.70
  Tanimoto to a documented active, and literature does not contradict it.
- **C — exploratory screening hypothesis.** Inside the applicability domain but
  target relevance unestablished. **No affinity value is published**, because a
  number attached to an unestablished interaction reads as evidence it occurs.
- **D — unsupported.** Outside the domain, or no evidence of any kind.

Literature evidence is MC4R-specific. A broader "melanocortin" term matched 160
of Semax's 214 publications — Semax is an ACTH(4-10) analogue — which would have
been a false positive for *this* target. Ambiguous short aliases are dropped from
literature queries: `MT-2` retrieves metallothionein-2 papers and had inflated
melanotan II's publication count from 625 to 2728.

Current registry outcomes: Pinealon, KPV, Semax, Dihexa and semaglutide all
abstain at class C; GLOW and Amino H2O at class D. No registry compound receives
a model-generated MC4R affinity.

### Peptide representation and endpoint separation

Endpoints are built separately because they measure different quantities:

| endpoint | peptides | molecules | quantity |
|---|---|---|---|
| Ki (binding) | 628 | 1981 | equilibrium binding affinity |
| EC50 (functional) | 551 | 1115 | agonist potency |
| IC50 (binding) | 272 | 688 | displacement potency |
| Kd (binding) | 11 | 11 | direct binding constant |

For the 201 peptides measured both ways, pKi and pEC50 correlate at Pearson
0.657 with the difference having **sd 1.03 log units** — comparable to the whole
model error. Pooling them would inject that as label noise.

Held-out performance on Taylor-Butina cluster splits:

| endpoint | best representation | MAE | R² | Spearman | mean baseline |
|---|---|---|---|---|---|
| Ki binding | multitask fingerprints / RF | 1.121 | 0.270 | 0.561 | 1.324 |
| EC50 functional | combined / RF | 0.878 | 0.180 | 0.531 | 0.992 |

**Peptide-specific descriptors did not help on either endpoint** (Ki 1.335 vs
1.253 for fingerprints; EC50 0.919 vs 0.907), despite being validated at 94.8%
against HELM sequences. **Multitask sharing across the two endpoints did help**
for Ki binding: MAE 1.121 vs 1.253, with R² rising from −0.003 to 0.270. That is
the one representation change that survived an honest split.

### PK parameter extraction and mechanistic validation

The mechanistic engine was verified but never validated: no model had been
fitted to or tested against observed measurements, and human PK simulation was
permitted for **0/46** entries because literature *mentioning* pharmacokinetics
is not the same as holding parameters.

Parameters are now extracted from FDA labelling via openFDA. **Every extracted
number carries the sentence that states it.** Three distinctions are preserved:
point values vs bounds (tesamorelin's bioavailability is "less than 4%", a
censored observation), mean vs median (Tmax is a median), and apparent vs true
parameters (`CL/F` is confounded with bioavailability and is labelled so).

Coverage is honest and small: **13 registry entries attempted, 5 had a label
with a PK section, 1 yielded a usable parameter set.** A compound qualifies only
when the label states clearance, volume of distribution, bioavailability as a
point estimate, Tmax and a dose. Tirzepatide is blocked for want of a Tmax;
tesamorelin for want of clearance; semaglutide for want of bioavailability and
Tmax. None is completed with an assumed value.

**This is where verification becomes validation.** Bremelanotide's label is
over-determined: it reports CL, V and F, and *separately* reports Cmax, AUC and
terminal half-life. The first three parameterise the model; the second three are
independent observations it can be asked to reproduce.

| quantity | predicted | observed | error | within 90% interval |
|---|---|---|---|---|
| AUC | 269.2 | 276 ng·h/mL | **2.5%** | yes |
| terminal half-life | 2.666 | 2.7 h | **1.3%** | yes |
| Cmax | 53.97 | 72.8 ng/mL | 25.9% | yes |

AUC and half-life are reproduced to within 3%. Cmax is underpredicted by 26% at
mean parameters, though the observed value falls inside the interval implied by
the label's own CL ± 1.0 and V ± 5.8. The likely cause is structural: a
one-compartment model with first-order absorption cannot produce a peak as sharp
as the data show. That is reported, not smoothed over.

Human PK simulation is now permitted for **1/46** entries — exactly those with
source-backed parameters — and a simulation attributed to PT-141 returns
evidence class B citing the label, while every other compound still abstains.

**Limitations.** Parameters and the quantities they are checked against come
from the same labelling document, so this is internal consistency plus
structural adequacy, not independent external validation. Agreement does not
establish that the true disposition is one-compartment. Labels report population
means; between-subject variability is represented only by the standard
deviations stated.

### Scientific Interpretation Engine

Every number ATLAS reports now carries an explanation computed from its own
value and from the metadata of whatever produced it. None of it is prewritten
narrative attached after the fact: a pKi of 5.45 is reported as a Ki of 3.55 µM
because 10^−5.45 was evaluated, and a ±1.21 log interval is reported as a
263-fold range because 10^2.42 was evaluated.

Each interpretation answers six questions — what was measured, what the number
means mathematically, what it implies biologically, how uncertain it is, why the
model produced it, and what would confirm or refute it.

The load-bearing field is **what a quantity does not establish**, since most
misreading in pharmacology comes from treating a real number as answering a
question it cannot: affinity as efficacy, exposure as benefit, plasma as tissue,
occupancy as response.

Covered: pKi, Ki, AUC, Cmax, Tmax, half-life, clearance, volume of distribution,
bioavailability, MAE, R², interval coverage, Tanimoto similarity, receptor
occupancy.

Derived quantities are computed, not stated: half-life → elimination rate
constant, time to steady state, and accumulation ratio at a given interval;
AUC + dose → implied clearance; MAE on a log scale → fold-error and its ratio to
the measured experimental noise floor; coverage → signed gap against nominal.

**Cross-endpoint bridge.** Where a measured Cmax and a measured Ki both exist,
the engine computes the exposure-to-affinity ratio and the implied equilibrium
occupancy. For PT-141: peak 71 nM against a measured Ki of 20.9 nM, giving 73%
occupancy at peak — reported with the caveats that make it honest (plasma is not
tissue, total is not free, occupancy is not response).

Interpretations attach to measurements, predictions, simulations and model-card
metrics, and are absent for abstentions, which have no number to explain.

## Verification vs validation

This distinction is load-bearing and currently sits on one side of it.

- **Verified** — the solver solves the equations it claims to solve, to machine
  precision, against closed-form solutions and conservation laws.
- **Partly validated** — one mechanistic model has now been tested against
  observed human data. The bremelanotide parameter set reproduces the AUC and
  terminal half-life its label reports independently, to within 3%, and
  underpredicts Cmax by 26%. Every other compound remains unvalidated for want
  of parameters.

A solver passing all 63 checks says nothing about any compound's behaviour in
an organism.

The ML model is the one component evaluated against real measurements, and it
sits one tier down from externally validated: held-out scaffold-split
performance on ChEMBL data, with no independent external test set and no
prospective evaluation.

## Peptide model: why it is not deployed (ATLAS 2.4 Stage A)

The peptide affinity model remains undeployed, and the Stage A audit
strengthens that decision rather than weakening it. Full findings in
`docs/ATLAS_2.4_STAGE_A_FINDINGS.md`; report in `data/ml/peptide_audit.json`;
reproduce with `python3 -m atlas.ml.peptide_audit`.

The earlier `peptide_like` label counted amide bonds. Replacing it with
backbone tracing (`atlas/ml/peptide_structure.py`, validated against 14 PubChem
structures — all 14 residue counts exact, oxytocin recovered as `CYIQNCPLG`,
somatostatin-14 as `AGCKNFFWKTFTSC`) gives 630 peptides rather than 628, but
disagrees on membership for 44 molecules, 7.0% of the set. 620 molecules trace
only one or two residues and have no sequence to featurise.

Measured on one fixed test set of 141 held-out peptides (Taylor-Butina cluster
split, 489 train):

- Best representation reaches **MAE 1.136, R² 0.123** against a mean-predictor
  baseline of 1.343. The model accounts for ~12% of held-out variance.
- The 630 peptides form only **30 similarity clusters**; the largest holds 53%.
  Effective sample size is nearer 30 than 630, and **two-thirds of the label
  spread lies within series** (SD 0.863 of 1.281).
- **80.5% carry non-canonical residues** across 160 distinct fragments, mostly
  halogenated and naphthyl phenylalanines. A canonical-alphabet protein
  language model cannot represent them.
- Training on 1351 non-peptides instead of 489 peptides is **worse than
  predicting the mean** (R² −0.791); pooling both helps slightly.
- Label noise is **not** the constraint: noise floor MAE 0.245 on the
  replicated test subset, leaving 0.891 log units of headroom.

Residue composition — the representation backbone tracing newly enables —
scores *below* the baseline, because a bag of residue counts cannot
distinguish positional isomers and this population is single-substitution
analogue series. That negative result is asserted by a test.

Stereochemistry is **not** resolved by the tracer: D- versus L-residues are
indistinguishable in the current representation, which matters because D-amino
acid substitution is central to melanocortin peptide design.

### Stage B: representations benchmarked (gate still FAILS)

Full findings in `docs/ATLAS_2.4_STAGE_B_FINDINGS.md`; report in
`data/ml/peptide_benchmark.json`; reproduce with
`python3 -m atlas.ml.peptide_benchmark`.

Evaluation is now split in two, because Stage A found a single aggregate error
conceals two capabilities:

| | Cross-series | Within-series |
|---|---|---|
| Best MAE | 0.877 | 0.513 |
| Best R² | +0.264 | +0.701 |
| Median neighbour similarity | 0.600 | 0.941 |

The peptide-native representation (`positional_fragment_fp`: a Morgan
fingerprint of each side-chain fragment, indexed by position) roughly **doubles
cross-series R² over the whole-molecule fingerprint, +0.131 to +0.264**. The
mechanism is measured: 38.7% of held-out molecules carry a side chain absent
from the training vocabulary, which a one-hot index cannot encode and a
fragment fingerprint can.

**All three representation contrasts reverse between the two evaluations.**
Benchmarking only within-series would have selected one-hot indices, explicit
topology and whole-molecule fingerprints — each the worst choice for
generalisation. Two Stage A recommendations were refuted by measurement:
explicit topology features *hurt* cross-series (0.894 → 1.168 with a forest),
and vocabulary collapse does *not* currently rule out frozen protein language
models, whose ceiling (MAE 0.382) is better than the best model measured (0.877).

Representational ceilings were computed without running a language model, by
grouping molecules on representation key and predicting group means — a hard
lower bound on any model using that representation. Canonical 20-letter
sequences collapse 630 peptides to 205 strings; the worst group is 10 molecules
reading `XRWX` that span 4.01 log units of pKi, a 10,233-fold affinity range
rendered as one string.

**Deployment gate FAILS:** cross-series MAE 0.877 against ≤0.70 required, R²
0.264 against ≥0.30. Within-series beats a series-mean lookup (0.513 vs 0.845)
but its median neighbour similarity of 0.941 makes that interpolation, not
generalisation. The model publishes no numeric affinity estimates.

Stereochemistry remains unresolved: D-Phe⁷ and L-Phe⁷ encode identically,
which in a melanocortin dataset is a material omission and the most likely
source of cheap improvement.

## Not built yet

1. **PK coverage beyond approved products.** Only labelled drugs have a PK
   section, so 45/46 entries still lack parameters. Extending to published
   literature needs numeric extraction from papers, with the same verbatim
   requirement.
2. **External validation of the ML model** — needs an independent affinity set
   for MC4R not drawn from the same ChEMBL extraction.
3. **Order-aware peptide features** — only 28% of peptide entries carry HELM
   notation, so composition is recovered but sequence order is not. Features are
   named for composition and nothing here is a sequence model.
4. **Knowledge graph** — typed schema and traversal not implemented.
5. **Evidence ledger database** — provenance is per-record; the append-only
   versioned store is not built.
6. **Geroscience module** — hallmark framework not implemented.
7. **Interface gaps** — no multi-compound comparison view, no pathway/network
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
- RDKit 2026.09.1 is installed and required by the ML layer. PyTorch is not
  installed and is not needed: no model tried here benefited from it, and the
  best scaffold-split model is a nearest-neighbour regressor.
- The MC4R model covers one target. It says nothing about selectivity over
  MC1R, MC3R or MC5R, and binding affinity is not efficacy, exposure or safety.
- Docker daemon is not running; no Postgres/Redis/Neo4j locally. Current code
  needs none of them.

## Reproducing

```bash
python3 -m pytest -q                              # 569 tests
PYTHONPATH=. python3 -m atlas.mechanistic.verify  # 63/63 solver checks
PYTHONPATH=. python3 -m atlas.registry.build      # resolve registry (live)
PYTHONPATH=. python3 -m atlas.analysis.report     # evidence/feasibility sweep
PYTHONPATH=. python3 -m atlas.api.app             # web interface on :5055
```

Append `--offline` to the last two to rebuild from the HTTP cache without
network access.
