# ATLAS — build status

Last updated: 2026-10-19 · 795 tests passing

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
Flask serves a JSON API and a five-workspace frontend from one process. The
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
| Interface Explorer | ten solved MC4R–peptide complexes; rotatable structure, consensus pocket, per-residue contacts and chemistry |

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
| Best MAE | 0.870 | 0.513 |
| Best R² | +0.276 | +0.701 |
| Median neighbour similarity | 0.600 | 0.941 |

The peptide-native representation (`positional_fragment_fp`: a Morgan
fingerprint of each side-chain fragment, indexed by position) roughly **doubles
cross-series R² over the whole-molecule fingerprint, +0.131 to +0.240**
(random forest, the scale-invariant comparator). The
mechanism is measured: 38.7% of held-out molecules carry a side chain absent
from the training vocabulary, which a one-hot index cannot encode and a
fragment fingerprint can.

**On MC4R, all five representation contrasts reverse between the two
evaluations** (three of three before the scaling correction). Benchmarking only
within-series would have selected one-hot indices, explicit topology and
whole-molecule fingerprints — each the worst choice for generalisation here.
**This tally is MC4R-specific and does not replicate across the family** (see
the Stage 3b section below: MC1R 3/5, MC3R 2/5, MC5R 1/5); what survives is
that the two evaluations can disagree, so both must be reported. Two Stage A
recommendations were refuted by measurement: explicit topology features *hurt*
cross-series (0.894 → 1.168 with a forest), and vocabulary collapse does *not*
currently rule out frozen protein language models, whose ceiling (MAE 0.382) is
better than the best model measured (0.870).

Representational ceilings were computed without running a language model, by
grouping molecules on representation key and predicting group means — a hard
lower bound on any model using that representation. Canonical 20-letter
sequences collapse 630 peptides to 205 strings; the worst group is 10 molecules
reading `XRWX` that span 4.01 log units of pKi, a 10,233-fold affinity range
rendered as one string.

**Deployment gate FAILS:** cross-series MAE 0.870 against ≤0.70 required, R²
0.276 against ≥0.30. Within-series beats a series-mean lookup (0.513 vs 0.845)
but its median neighbour similarity of 0.941 makes that interpolation, not
generalisation. The model publishes no numeric affinity estimates.

Stereochemistry remains unresolved: D-Phe⁷ and L-Phe⁷ encode identically,
which in a melanocortin dataset is a material omission and the most likely
source of cheap improvement.

## ATLAS 2.5: biophysics-aware interaction intelligence

Stage 1 audit and Stage 2 chemistry layer delivered. Full audit in
`docs/ATLAS_2.5_STAGE_1_AUDIT.md`.

**The primary reference studied** is Feng et al., *ACS Omega* 2026;11(10):16501
([DOI](https://doi.org/10.1021/acsomega.5c12511)), which benchmarked 25
algorithms on noncovalent interaction descriptors from 44,848 protein
complexes. Its taxonomy of nine interaction classes and its correlation-based
model de-duplication (drop candidates correlating above 0.75) are adopted. Its
task is **not** adopted: it is binary interaction classification with
ZDOCK-docked decoys as negatives, on a random 80:20 split with no similarity
clustering, using features computed from the bound complex. None of its
performance figures are imported.

**Two measured constraints reorder the brief's milestones:**

1. **ATLAS has exactly one target** (MC4R, `CHEMBL259`). A dual encoder cannot
   be evaluated with zero target variance, so multi-target data must come
   first. The melanocortin family is the natural expansion — MC1R 599, MC3R
   1,634, MC5R 1,387 Ki records, and these peptides are routinely profiled
   across the family because subtype selectivity is the question that matters.
2. **Structural coverage is ~1%.** RCSB holds 13 experimental MC4R structures
   (UniProt P32245), 10 with peptide ligands but only **7 unique peptides**
   (setmelanotide ×3, afamelanotide ×2), at 2.58–3.30 Å. Against 630 peptides
   with measured affinity, interface geometry cannot be a training
   representation; it is reframed as extractor validation and per-ligand
   explanation.

**Stage 2 — `atlas/ml/residue_chemistry.py`.** Residue descriptors computed
from side-chain fragment structure rather than tabulated, because no published
amino-acid scale has a row for 2,4-dichlorophenylalanine. Typed provenance
(COMPUTED / ESTIMATED / UNAVAILABLE; nothing is MEASURED). Ionisation by
Henderson-Hasselbalch at a parameterised pH, so histidine's condition
sensitivity is measurable rather than assumed away.

**Its measured contribution is negative on the harder question**: cross-series
MAE 0.936 against 0.894 for a fingerprint of the same side chains, and adding
it makes that slightly worse (0.908). It beats the mean baseline (1.189) and
one-hot indices (1.009), so it carries real signal, but a side-chain
fingerprint already encodes this chemistry more richly than 22 hand-chosen
descriptors. It is retained for **explanation**, where named descriptors are
interpretable and fingerprint bit 1,473 is not.

### Stage 3: the melanocortin panel (target encoder not established)

Full findings in `docs/ATLAS_2.5_STAGE_3_PANEL.md`; data in
`data/ml/melanocortin_panel.json`; reproduce with
`python3 -m atlas.ml.multitarget` then `python3 -m atlas.ml.panel_benchmark`.

ATLAS is no longer single-target. `build_dataset` takes an endpoint (Ki/Kd/IC50
binding, EC50 functional) and `atlas/ml/multitarget.py` builds one dataset per
(receptor, endpoint) pair. **Endpoints are never merged** — there is
deliberately no accessor returning several, and a test asserts it. The existing
`mc4r_pki.json` is untouched.

**3,499 Ki measurements, 2,066 molecules, 5 receptors** (MC1R Q01726, MC2R
Q01718, MC3R P41968, MC4R P32245, MC5R P33032 — accessions from ChEMBL,
sequences from UniProt).

The premise holds: **789 of 2,066 molecules (38.2%) are measured on 2+
receptors**, 105 on all four. **Median selectivity span 1.31 log units**, max
4.40 — and that median *exceeds* the best cross-series error ATLAS has achieved
(0.870), so a model ignoring which receptor it is asked about carries a
systematic error larger than its own accuracy.

Restricted to peptides, the family adds **only 10 new peptides (630 → 640) but
2.15× the measurements (630 → 1,354)**, because 66.2% of peptides were already
profiled across receptors. This addresses Stage A's ~30-series ceiling from a
new angle: not more chemistry, but a second axis.

**Receptor identity carries signal** — peptide-only MAE 1.125 against 0.989
with a one-hot receptor, a 0.136 gain. But both have **negative R²**: worse
than predicting the test set's own mean. Only the relative claim holds. Panel
numbers are **not comparable** with 2.4 figures — harder split, extra
across-receptor label variance.

**A target encoder is not established.** Leave-one-receptor-out is the only
split that tests one (the held-out one-hot column is necessarily all-zero).
Against the mean baseline, transfer succeeds on MC3R and MC5R and **fails on
MC1R and MC4R**. The sequence encoding beat peptide-only on 4 of 4, but on the
two failures both arms are worse than the mean, so that comparison is the wrong
one. Five receptors is five points in a composition space and four held-out
receptors is four trials.

Leakage guard: a molecule on several receptors contributes several rows, so all
splits move **whole molecules**, with clusters computed per molecule. Four
tests assert no molecule or cluster straddles a split.

MC2R contributes **no peptides at all** (23 molecules, none traced) — the
abstention case the Stage 1 audit anticipated.

### A binding pocket from structures (encoder works, does not help)

Full findings in `docs/ATLAS_2.5_POCKET_ENCODER.md`; definition in
`data/ml/receptor_pocket.json`; reproduce with
`python3 -m atlas.ml.receptor_pocket`.

Whole-sequence composition is a poor description of a binding site, so the
receptor encoder was rebuilt from observed geometry. All ten experimental
MC4R–peptide complexes were parsed; receptor residues within 4.5 Å of any
peptide heavy atom are contacts, and a position joins the consensus pocket if
it contacts in ≥3 structures. **31 positions, 17 of them contacting in all
ten** (25 positions at 4.0 Å, 36 at 5.0 Å, so the cutoff is not doing the
work).

Numbering is never trusted: author numbers match UniProt in the eight cryo-EM
entries but not in the three X-ray fusion constructs, which carry a
glycogen-synthase partner and 12 engineered mutations each. Positions come
from local alignment to P32245 and are verified letter by letter; contacts are
located from the structure and **named from UniProt**, so a mutated construct
residue does not enter the encoding.

Pocket transfer across the family is far more discriminating than whole-sequence
identity: MC3R 25/31 pocket residues identical to MC4R, MC5R 24/31, MC1R 21/31,
MC2R 12/31 — against 50–63% whole-sequence identity.

**The encoder does not improve prediction.** On the cluster split the three
target encodings (one-hot 0.989, sequence 0.992, pocket 0.996) sit within 0.007
log units — that split cannot distinguish them, since four receptors give four
vectors under any encoding. Under leave-one-receptor-out the pocket beats
whole-sequence on 1 of 4 and beats the mean baseline on 2 of 4, the same two
the crude encoding already managed.

**That localises the bottleneck: the limiting factor is the number of targets,
not the quality of their representation.** With four receptors every encoding
is four points and LORO asks a model to reach the fourth from three. The pocket
encoder is kept for interpretability — a SHAP attribution over
`D122 estimated_charge` is readable where fingerprint bit 1,473 is not — and
for the residue-level interface data a structural viewer needs. It should be
re-tested if more receptors become available, not adopted on performance it
has not shown.

**Deferred with a reason:** protein language model embeddings — no PyTorch or
transformers installed.

#### Three groupings, including study grouping

Peptide-cluster (`grouped_cluster_split`), target (`leave_one_receptor_out`)
and study (`document_groups` + `leave_documents_out`). A study group cannot be
one document: molecules are re-measured across papers, so a leak-free group is
a connected component of the (molecule, document) graph. Measured cost of that
closure — **39 documents behind the peptide rows collapse to 17 groups, the
largest holding 43.9% of rows**. A study-grouped holdout therefore cannot be
tuned to a target size, and a result from it rests on 17 independent units,
not 39.

### Stage 3b: Stage A and Stage B re-run per receptor (one finding did not survive)

`atlas/ml/panel_replication.py`; data `data/ml/panel_replication.json`;
reproduce with `python3 -m atlas.ml.panel_replication`. Full tables in
`docs/ATLAS_2.5_STAGE_3_PANEL.md` §8.

Stage A and Stage B were each measured on **one receptor**, so each finding
was a single observation. Re-run per receptor — not pooled, because pooling
would put a molecule's MC1R and MC4R affinities in one label column a median
1.31 log units apart. MC4R re-run through the same adapter reproduces the
canonical numbers exactly as a control, asserted by two tests.

**Stage A replicates.** Series per receptor MC1R 22, MC3R 22, MC4R 30, MC5R
14; effective sample size 0.048–0.146 of nominal. The **~30-series ceiling is
a property of melanocortin peptide chemistry, not of the MC4R literature** —
MC4R, with 2.3× the peptides of its nearest rival, has the most series. The
non-canonical burden is class-wide at **68.4–80.5%**, so canonical-only
representations are inadequate family-wide.

**Stage B's headline finding does NOT replicate.** Stage B found all five
representation contrasts reversing between cross-series and within-series
evaluation. Re-measured: **MC4R 5/5, MC1R 3/5, MC3R 2/5, MC5R 1/5** — all five
reverse on MC4R and nowhere else. The structural point (the two evaluations
can disagree, so one number hides which question was asked) stands; the tally
of five was an n=1 result reported as general. **Nor does the representation
ranking transfer** — four distinct best representations across four receptors,
none winning twice.

**The gate fails on every receptor.** Best cross-series MAE in the family is
0.827 (MC3R) against 0.70 required. The conjunction matters: MC1R's R² of
**+0.397 clears** the R² ≥ 0.30 criterion — better than MC4R's +0.276 — while
its MAE 0.919 does not clear 0.70. A gate on R² alone would have passed it.
The gate was not touched and the peptide model still publishes no numeric
estimates.

The expansion **confirmed Stage A and weakened Stage B**, and improved no
model. That is a negative result and is recorded as one.

### ATLAS 2.5 Stage 2: biophysics-aware representations

Reproduce with `python3 -m atlas.ml.interaction_features`,
`python3 -m atlas.ml.assay_context`, `python3 -m atlas.ml.peptide_benchmark`.

**Typed interaction descriptors** (`atlas/ml/interaction_features.py`). All ten
MC4R peptide complexes now yield hydrogen-bond candidates, salt bridges,
hydrophobic contacts, aromatic stacking classified by interplanar angle,
cation-pi, Shrake-Rupley solvent accessibility and interface geometry, with
every threshold saved alongside. Receptor atoms are typed by name (always
standard residues); peptide atoms by element and distance-inferred bonds,
because these ligands are largely non-canonical — ring detection finds 3-6
rings per peptide from the graph alone.

Measured: buried peptide fraction 0.52-0.74, higher for constrained
heptapeptides than 13-mers. Hydrogen bonds are **candidates**, not bonds: at
2.58-3.30 A there are no hydrogens, so D-H...A angles cannot be checked and
counts nearly double from 3.2 to 3.9 A. **Cation-pi is genuinely absent** —
nearest cationic-group-to-ring-centroid distance 6.8-8.6 A in nine of ten
structures, refuting an expectation that Arg-aromatic contacts would be common.
21LT reports no salt bridge at 4.0 A and two at 4.5 A because its Arg-Asp pair
sits at 4.24 A against 7PIU's 2.54 A — the same ligand either side of a cutoff,
which is why a sweep is saved per structure.

**These are retrospective by construction.** Every descriptor comes from a
solved bound complex and cannot exist for an unseen pair.
`InteractionProfile.vector()` calls a guard that raises rather than degrading
quietly, so bound geometry cannot enter a prospective pathway.

**Assay context** (`atlas/ml/assay_context.py`). 89 binding assays resolved:
4 BAO formats (cell-based 56, single protein 27, membrane 4), 7 cell types, two
confidence scores. 162 molecules span several formats, so this is real
within-molecule variation. pH, temperature and ionic strength are **not** in
ChEMBL's structured fields and are recorded as absent.

**Local sequence context** (`residue_chemistry.local_context`). Neighbour
chemistry per position, wrapping for cyclic backbones and not for linear ones.

**Ablation, cross-series, random forest: 0 of 4 feature groups improved the
positional fragment fingerprint and 4 made it worse** — assay context 0.894 to
0.946, local context to 0.964, residue chemistry to 0.908, topology to 1.168.
Across 2.4 and 2.5 the representation layer has been extended four times and
none improves cross-series generalisation. The best cross-series model remains
`positional_fragment_fp+morgan_fp` at **MAE 0.870, R2 0.276 — unchanged**. On
this dataset the representation is not the binding constraint, which agrees
with Stage A's ~30 effective series and the pocket encoder's null result.

Assay context behaves as its caveat warned: R2 -1.714 cross-series, +0.110
within-series. Assay identity proxies for series, so it is kept as a recorded
covariate, not a predictive feature.

**The deployment gate still fails** and no improvement is claimed.

### Targeting the bottleneck, not the representation

Reproduce with `python3 -m atlas.ml.pairwise` and
`python3 -m atlas.ml.multitask`.

Four representation extensions had failed, and the diagnosis was that the
representation was never the binding constraint: effective sample size is ~30
series, two thirds of label variance is *within* series, and only four
receptors carry peptide data. Two ways at that were tried.

**More data is not available.** Non-human melanocortin receptors hold 325 Ki
binding records against the human family's 7,342 — and a mouse MC4R is a
different protein, so pooling species repeats the mutant-receptor error at
species scale. Censored relations add 50. BindingDB's SOAP endpoint is gone.
Stage A put MAE 0.70 at ~158,000 training peptides. The dataset cannot be
fixed, so the task had to change.

**Within-series pairwise Δ-affinity.** 630 peptides in 30 series give 15,506
ordered pairs, a 25-fold increase, with both orderings included so
antisymmetry is enforced by the data. Splits partition molecules and clusters,
never pairs — a test pair whose members both appear in training pairs would be
almost fully determined.

*Null as a training scheme*: on held-out molecules the pairwise model reaches
MAE 0.579 against 0.541 for simply subtracting two absolute predictions; on
held-out series 1.204 against 1.151. Training on differences buys nothing.

**But the evaluation reframe is the real result.** The same model that fails
its gate for publishing numbers (MAE 0.870, R² 0.276) ranks two analogues of
one series correctly **92.0%** of the time when their true difference exceeds
0.5 log units, on 5,880 held-out pairs whose test member it had never seen.
Across unseen series it ranks at 70.8% — above chance, far weaker. Absolute
error concealed a usable capability because it answered a question nobody asks
of this model.

This licenses a **within-series ranking** claim with its accuracy attached,
bounded to peptides with a close training neighbour. It does **not** license a
numeric affinity estimate and does **not** change the deployment gate.
Ranking and estimating are different claims; only the first now has evidence.

**Family pooling does not buy sample efficiency.** With peptide clusters held
out *globally* — so a held-out peptide is absent from training for every
receptor, and the pooled model cannot see its affinity on a sister receptor —
pooling helps 2 of 4 receptors and hurts 2, mean −0.032 log units. The sign
tracks existing data volume: MC4R (most rows) is the worst hurt at −0.172,
while the data-poor MC1R and MC3R gain. Pooling redistributes rather than
adds.
### Structural interface explorer (Stage 2 priority 10)

`atlas/ml/interface_geometry.py`, `atlas/api/interface_payload.py`,
`GET /api/interfaces` and `GET /api/interfaces/<pdb_id>`, and a fifth
workspace at `#interfaces`.

The typed profiles and the consensus pocket were saved as numbers; this makes
them inspectable. A complex is chosen from the ten, its interface is drawn,
and selecting a pocket residue carries that selection across four views at
once: its place in the structure, the contacts it makes with the peptide by
type, its side-chain chemistry with per-descriptor provenance, and its
transfer to MC1R, MC3R and MC5R.

**The viewer is ATLAS's own, not Mol\*.** Mol\* loads from a CDN and the
frontend currently fetches nothing: making an offline instrument reach the
network to draw an experimental structure, for a five-megabyte third-party
bundle, was not worth it. So the structure is drawn in SVG like every other
figure here — an orthographic projection of the deposited coordinates, with
the C-alpha trace, the interface side chains, the peptide's heavy atoms and
the typed contacts, rotatable by drag or arrow key. It is not a
molecular-graphics package and says so on the figure: no surface, no electron
density, no secondary-structure fitting, and no occlusion, since depth is cued
by opacity alone.

**Nothing is recomputed in the browser.** `interface_geometry` re-enumerates
the pairs behind each saved count using the same typed atom sets, the same
thresholds and the same chain identification that produced
`interaction_profiles.json`, then checks its totals against the published
profile and returns the comparison. All ten agree, and the figure carries the
result as a chip; a disagreement would announce itself rather than quietly
replacing the published number.

**The vocabulary is the deliverable.** Every structure here is experimental,
but the typing laid over it is not uniformly so, and the interface separates
six classes of mark at the point of use rather than in a footnote:
experimental coordinates, a geometric consequence of them, a **candidate**
hydrogen bond (no hydrogens at 2.58-3.30 A, so no D-H...A angle was ever
checked), an **inferred** peptide-side charged group or ring (these ligands
are largely non-canonical, so residue names carry no chemistry), an
illustrative drawing convention, and a position **transferred** by alignment
to a receptor with no peptide-bound structure at all. The affinity model's
failing gate travels with every payload, because a drawn salt bridge is the
thing most likely to be misread as evidence that a peptide binds tightly.

Coordinate files are gitignored, so a fresh checkout cannot draw. That is
reported as an absent input with the command that restores it, and nothing is
drawn in its place; the contact map, the measurements and the profile all
still work, because they never needed the file.

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
python3 -m pytest -q                              # 795 tests
PYTHONPATH=. python3 -m atlas.mechanistic.verify  # 63/63 solver checks
PYTHONPATH=. python3 -m atlas.registry.build      # resolve registry (live)
PYTHONPATH=. python3 -m atlas.analysis.report     # evidence/feasibility sweep
PYTHONPATH=. python3 -m atlas.api.app             # web interface on :5055
```

Append `--offline` to the last two to rebuild from the HTTP cache without
network access.
