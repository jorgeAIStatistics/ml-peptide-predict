# ATLAS 2.5 — Stage 1: scientific and repository audit

**Scope:** study the primary reference and supporting resources, audit what
ATLAS currently represents, establish the data reality, and produce an
implementation plan grounded in the existing code.
**Code touched:** none in this section (Stage 2 deliverable is
`atlas/ml/residue_chemistry.py`).

---

## 1. The primary reference, and what of it transfers

**Feng H, Sun X, Li Q, Zhang S, Xing G, Zhang G, Wang F.** "Machine Learning
Methods for Protein–Protein Interaction Prediction Based on Noncovalent
Interactions." *ACS Omega* 2026;11(10):16501–16509.
[DOI](https://doi.org/10.1021/acsomega.5c12511) · PMID 41867546 · PMC13000789.
Full text retrieved from PubMed Central; the ACS page itself returns 403 to
automated access.

### 1.1 What the paper actually did

- **Source data:** noncovalent interaction descriptors derived from 44,848
  protein–protein complexes from RCSB-PDB.
- **Negative class:** generated computationally. Proteins were docked against
  regions of their partners *distal to the native interface* using **ZDOCK**
  (rigid-body), with **Rosetta PackRotamers** side-chain repacking on a fixed
  backbone outside the original interface "to eliminate native-like contact
  bias."
- **Features:** computed by **ProtInter** over nine interaction classes —
  hydrophobic (HI), disulfide bridges (DB), ionic (IOI), aromatic–aromatic
  (AAI), aromatic–sulfur (ASI), cation–π (CPI), and hydrogen bonding split
  three ways by main/side chain (HBMM, HBMS, HBSS). Eight summary statistics
  per class: count, mean, std, min, 25%, 50%, 75%, max.
- **Filtering:** columns and rows with 75% zeros removed, yielding a
  **63,169 × 57** matrix with 38,542 positive and 24,627 negative samples.
  HI- and AAI-derived features were among those dropped for sparsity.
- **Split:** random **80:20** train/test, then scaled and centred.
- **Expansion:** polynomial feature generation → **1,596** features; SMOTE on
  the training set only.
- **Model search:** 25 algorithms via LazyPredict at default parameters →
  8 passed a 0.800 threshold on ≥2 metrics → models with pairwise correlation
  >0.75 on predicted probabilities removed, leaving ExtraTrees (ETs),
  ExtraTree (ET) and DecisionTree (DT) → random hyperparameter search with
  10-fold CV → SelectFromModel feature selection (487/375/415 features) →
  stacking (logistic-regression meta-learner) and voting ensembles. 12 models
  final.
- **Reported best:** ETsO at accuracy 0.927, sensitivity 0.916, specificity
  0.866, F1 0.922, MCC 0.846, AUC-ROC 0.967; SM_dt415 at AUC-ROC 0.969 and
  AUC-PRC 0.974.
- **Interpretation:** SHAP. Of the top 20 features, all but one
  (`cationpi_min`) were polynomial products. Hydrogen-bond and
  aromatic–sulfur terms were the most frequent across models.

**These performance figures are not imported anywhere into ATLAS**, per the
brief. They describe a different task on different data.

### 1.2 Directly applicable

1. **The noncovalent interaction taxonomy.** Nine classes with eight
   distributional statistics each is a well-shaped feature block, and
   reusable wherever ATLAS has coordinates.
2. **Correlation-based model de-duplication before ensembling.** Dropping
   candidate models whose predicted probabilities correlate above 0.75 is
   cheap and principled — it ensembles genuinely different learners rather
   than near-copies. ATLAS has no equivalent and should adopt it.
3. **Tree ensembles as the right family for interaction features.** Six of
   their eight best were tree-based. This agrees with ATLAS's own Stage B
   measurements and reinforces the choice of random forests as the
   scale-invariant comparator.
4. **Feature selection as evidence, not just compression.** That performance
   barely moved after cutting to ~400 features is informative about how much
   of the representation is doing work.

### 1.3 Applicable only with modification

1. **Interface features require the bound complex.** Every ProtInter
   descriptor is computed from observed bound geometry. For a genuinely unseen
   peptide–target pair that geometry does not exist, so the task as posed is
   *retrospective analysis of known complexes*, not prospective prediction.
   ATLAS must therefore keep any interface-feature model in a separate,
   explicitly structure-conditioned inference mode and must not let it inform
   a de novo prediction path.
2. **Splitting.** A random 80:20 split over PDB entries does not account for
   the PDB's redundancy — multiple entries of the same complex, homologous
   targets, repeated crystal forms. The paper describes no sequence or
   structure similarity clustering. ATLAS already splits by Taylor-Butina
   similarity cluster and must not relax that to match the reference.
3. **SHAP on polynomial products.** With 19 of the top 20 features being
   products of pairs of descriptors, "synergy among noncovalent interactions"
   is a weaker reading than it appears: products of correlated descriptors are
   hard to attribute mechanistically. ATLAS's existing rule stands — feature
   importance is not an energetic contribution.

### 1.4 Not applicable

1. **The task.** Binary interaction classification is not affinity
   regression. The paper's own labels are "affinity samples" and "nonaffinity
   samples", but they are *interacting* and *non-interacting pairs*; no Ki,
   Kd or any measured affinity appears anywhere in the study. ATLAS's endpoint
   is measured pKi.
2. **The negative class.** Negatives are ZDOCK decoys at non-native sites, so
   a classifier can partly succeed by recognising "computationally docked
   decoy" rather than "does not bind." The paper's own SHAP discussion
   supports this concern: it attributes the disulfide-feature importance in
   two models to the negatives showing an "obvious difference … (bimodal
   distribution)" because they were built from chains not at the interface.
   ATLAS's task needs no negatives at all, and fabricating them this way would
   violate the standing rule against treating unobserved pairs as
   non-binding.
3. **Dropping hydrophobic and aromatic–aromatic features.** These were
   removed for 75% sparsity, which the authors list as a limitation
   ("the lack of HI- and AAI-related features made the model … still have
   great room for improvement"). Since hydrophobic burial is a primary
   determinant of peptide binding, a representation that discards it is not a
   model for ATLAS to copy.

### 1.5 Provenance discrepancies worth recording

The 44,848 figure is described three ways within the same paper: "44848 pdb
files" (abstract and methods), "44848 interacting protein pairs"
(introduction), and "44848 protein complexes" (results). The algorithm count
is given as "twenty-five" in the abstract and methods but "26 different ML
algorithms" and "Twenty-six base models" in the results, with 26 rows in
Table 1. These are cited here as the paper states them rather than silently
reconciled.

---

## 2. Supporting resources assessed

Both supporting PMC articles are **review** papers. Neither introduces a
model or dataset, so several questions the brief implies have negative
answers, recorded here rather than papered over.

**Esmaielbeiki R, Krawczyk K, Knapp B, Nebel J-C, Deane CM.** "Progress and
challenges in predicting protein interfaces." (reviewed as PMC4338852)
[DOI](https://doi.org/10.1186/s13015-015-0033-9) — a review of protein–protein
interaction *site* prediction. Reusable conventions: a **15 Å** neighbour
sphere for residue-environment averaging, and **31** interacting atom types
for atomic contact density maps. Its most relevant warning is about label
noise: non-interacting residues "may be mislabelled (since every possible
protein complex is certainly not known)." Its dataset, feature and
performance tables were not recoverable from the retrieved text, so no
numbers from them are cited.

**A comprehensive review of protein-centric predictors for biomolecular
interactions** (PMC11089422) [DOI](https://doi.org/10.1093/bib/bbae162) —
surveys over 160 predictors including nine for protein–peptide interaction.
Useful specifics: PSSM is an **L × 20** matrix, HMM profiles **L × 30**;
evolutionary information is used by **67%** of all predictors; the DBv5
benchmark is **230 non-redundant complexes, resolution better than 3.25 Å,
each sequence over 30 amino acids**. Of the nine protein–peptide predictors,
**six take only the protein as input** and identify interface residues without
reference to a specific peptide; only InterPep and PepNN take both; only CAMP
predicts interacting residues on the *peptide* side. Peptide-side reported
AUCs: PepBCL 84.1% and PepBind 79.3% on TE125, PepNN 83.3%. It also names
over-prediction and cross-prediction metrics (OPR/AUOPC, CPR/AUCPC, AULC)
that suit imbalanced interface tasks.

**The decisive literature finding.** Across both reviews — more than 160
predictors and the whole protein–peptide subfield — there are **zero**
mentions of cyclic peptides, macrocycles, non-canonical residues or D-amino
acids, **no** peptide length definition, and **no** affinity-regression
predictor; all nine protein–peptide methods are residue-level binary
classifiers. ATLAS's actual problem — affinity regression over a population
that is 80.5% non-canonical and 79% conformationally constrained — is
therefore not a problem this literature has addressed. That is a reason to
adapt its feature vocabulary rather than its task framing or its benchmarks.

**PepBDB** (`http://huanglab.phys.hust.edu.cn/pepbdb/`) — curated
peptide–protein complexes from the PDB. Peptide and protein chains are
treated as interacting when their minimum distance is **within 5.0 Å**;
complexes are kept when the peptide has **fewer than 50 amino acids**;
non-interacting chains, ions and irrelevant ligands are stripped. Provides
sequences, resolution, interacting residues and LigPlot+ maps. **No binding
affinity data.** The publication states no licence terms.

**Propedia v2.6** (`https://bioinfo.dcc.ufmg.br/propedia26`) — 93,151 entries
counting duplicate sequences (73,392 peptide–protein, 19,759 multi-protein),
**38,218 unique**. Clusters by peptide sequence, interface structure and
binding site; explicitly aimed at machine-learning use. **No binding affinity
data**, no stated licence, marked "in development."

Both structural resources therefore supply **geometry, not affinity**. Under
the brief's dataset separation rule they belong to the structural-complex
dataset type and may not be merged into the affinity supervision.

---

## 3. What ATLAS currently represents

Audited against the repository at commit `1545d46`.

| Component | Status | Module |
|---|---|---|
| Peptide backbone and residue identity | **Built**, validated on 14 PubChem structures | `ml/peptide_structure.py` |
| Positional, chemistry-aware peptide encoding | **Built**, benchmarked | `ml/peptide_representations.py` |
| Whole-molecule descriptors and fingerprints | Built | `ml/features.py` |
| Similarity-clustered splits | Built (Taylor-Butina) | `ml/splits.py` |
| Conformal uncertainty, Mondrian calibration | Built, coverage 0.909 at nominal 0.90 | `ml/calibration.py` |
| Applicability domain by nearest-neighbour similarity | Built | `ml/features.py` |
| Target relevance gating (classes A–D) | Built | `ml/target_relevance.py` |
| Structured abstention with machine-readable reasons | Built | `core/evidence.py` |
| Mechanistic PK with SI dimensional analysis | Built, 63/63 verification | `mechanistic/` |
| **Residue-level chemistry descriptors** | **Stage 2, this milestone** | `ml/residue_chemistry.py` |
| **Protein target representation** | **Absent** | — |
| **Experimental context (pH, temperature, assay)** | **Absent** | — |
| **3D conformation / interface geometry** | **Absent** | — |
| **Evolutionary features / PLM embeddings** | **Absent** (no PyTorch installed) | — |
| **Receptor state (active/inactive)** | **Absent** | — |

### The binding constraint on a dual encoder

ATLAS's dataset has **exactly one target**: MC4R, `CHEMBL259`, hard-coded in
`atlas/ml/dataset.py:70`. A peptide–target dual encoder cannot be trained or
evaluated on one target, because the target side has **zero variance** — any
target encoder would be a constant, and an ablation removing it would show no
change by construction. Building one now would produce an untestable
component, which the brief's own rule forbids: "Every component must have an
independently testable contribution."

This reorders the milestones. **Stage 3 (multi-target data) must precede
Stage 4 (dual encoder).**

---

## 4. The data reality, measured

### 4.1 The melanocortin family is the natural, and sufficient, expansion

Queried live from the ChEMBL API through ATLAS's own cached client (human,
single-protein targets):

| Receptor | ChEMBL ID | Ki | EC50 | IC50 |
|---|---|---|---|---|
| MC1R (MSH receptor) | CHEMBL3795 | 599 | 974 | 476 |
| MC2R (ACTH receptor) | CHEMBL1965 | 46 | 2 | 1 |
| MC3R | CHEMBL4644 | 1,634 | 678 | 1,420 |
| **MC4R (current)** | **CHEMBL259** | **3,678** | **2,081** | **2,270** |
| MC5R | CHEMBL4608 | 1,387 | 460 | 1,829 |

Expanding to MC1R/MC3R/MC4R/MC5R roughly doubles the available Ki data, but
the quantity is not the point. **Melanocortin peptides are routinely profiled
across the whole receptor family**, because subtype selectivity is the central
pharmacological question for this class. That gives:

- genuine target variance, so a target encoder is testable;
- the same peptide measured against several receptors, so the model must learn
  *selectivity* rather than generic peptide quality;
- a legitimate leave-one-receptor-out evaluation, which is the brief's
  "known peptide family, related target" regime;
- MC2R as a natural hard case at only 46 Ki records, suitable for testing
  abstention rather than prediction.

### 4.2 Structural data exists, but covers about 1% of the affinity set

Queried from RCSB by UniProt accession **P32245** (human MC4R): **13
experimental structures**, of which **10 contain a peptide ligand**,
representing **7 distinct peptides**.

| PDB | Method | Resolution (Å) | Peptide ligand |
|---|---|---|---|
| 7PIU | cryo-EM | 2.58 | setmelanotide |
| 6W25 | X-ray | 2.75 | SHU9119 |
| 7PIV | cryo-EM | 2.86 | NDP-α-MSH (afamelanotide) |
| 8WKY | X-ray | 2.90 | S25 |
| 7AUE | cryo-EM | 2.97 | setmelanotide-type |
| 7F53 | cryo-EM | 3.00 | α-MSH |
| 7F54 | cryo-EM | 3.00 | afamelanotide |
| 7F55 | cryo-EM | 3.10 | bremelanotide |
| 21LT | cryo-EM | 3.11 | setmelanotide |
| 8WKZ | X-ray | 3.30 | S31 |

Three further entries are not peptide complexes: 7F58 (THIQ, a small
molecule), 8QJ2 (nanobody), 9K3K (unliganded).

**Redundancy, as the brief requires separately:** 10 peptide-containing
entries, 7 unique peptides. Setmelanotide appears three times (7AUE, 7PIU,
21LT) and afamelanotide twice (7F54, 7PIV). Any structural analysis must group
by peptide, not by PDB entry.

**Consequence.** 7 distinct peptides with experimental geometry against 630
peptides with measured affinity is roughly **1% coverage**. Interface
geometry therefore **cannot be a training representation** for affinity
regression on this dataset. It can validate that a computed interaction-feature
extractor produces chemically sensible output, and it can support residue-level
*explanation* for the handful of ligands that have structures — including
bremelanotide and afamelanotide, both already in the ATLAS registry with
measured values. Stage 5 must be reframed accordingly.

**A convergence worth noting.** The PDB writes SHU9119's sequence as
`XLDHXRWKX` and S25's as `XLDPXRWGKX`. That is the same canonical-alphabet
collapse Stage B measured independently: the one-letter representation
discards exactly the non-canonical residues that distinguish these ligands.

---

## 5. Implementation plan, grounded in the code

Revised from the brief's ordering on the basis of §3 and §4.

**Stage 2 — residue chemistry layer. Delivered this milestone.**
`atlas/ml/residue_chemistry.py`. Descriptors computed from side-chain fragment
structure rather than tabulated, because no published amino-acid scale has a
row for 2,4-dichlorophenylalanine. Typed provenance throughout. 26 tests.

**Stage 3 — multi-target melanocortin dataset.** Generalise
`atlas/ml/dataset.py` from its hard-coded `CHEMBL259` to a target set, keeping
endpoint-specific supervision separate (Ki, EC50 and IC50 as distinct
datasets, never pooled into one label). Add peptide-cluster, target and
study grouping so the brief's three generalisation regimes can be evaluated
separately. This is the prerequisite for everything below.

**Stage 4 — dual encoder, simple baseline first.** Peptide encoder from the
existing representations, target encoder starting as a one-hot receptor
identity, then receptor sequence features. The one-hot version is the control
that says whether a sequence-based target encoder earns its complexity.

**Stage 5 — interaction geometry, reframed as validation and explanation.**
Compute ProtInter-style contact features on the 10 MC4R peptide complexes.
Use them to validate the extractor and to provide residue-level explanations
for specific ligands. Do **not** use them as affinity-regression features at
1% coverage, and keep them in a separately labelled structure-conditioned
mode.

**Stage 6 — ablations.** One feature group removed at a time, cross-series and
within-series reported separately, forest as the scale-invariant comparator.

**Stage 7 — uncertainty.** Extend the existing conformal machinery to
peptide-family and target novelty; keep the deployment gate unchanged.

**Stage 8 — Interface Laboratory.** Mol*-based viewer over the 10 real
complexes, with experimental versus predicted provenance always visible.

**Deferred with a reason:** protein language model embeddings. PyTorch and
transformers are not installed. Stage B bounded what a frozen canonical-alphabet
embedding can achieve on this data (oracle MAE 0.382, better than the current
best model at 0.870), so the ceiling is not currently the binding constraint
and this is a lower priority than Stage 3.

---

## 6. Stage 2 result: the chemistry layer does not help cross-series

Measured on the Stage B benchmark, forest as comparator, one fixed test set
per evaluation. Reported because the brief requires negative results to be
reported and complexity not to be assumed beneficial.

| Representation | Cross-series MAE | R² | Within-series MAE | R² |
|---|---|---|---|---|
| `positional_fragment_fp` | **0.894** | **+0.240** | 0.613 | +0.585 |
| `residue_chemistry` | 0.936 | +0.156 | **0.602** | +0.597 |
| `residue_chemistry + positional_fragment_fp` | 0.908 | +0.210 | 0.606 | +0.603 |
| *mean baseline* | *1.189* | *−0.276* | *1.040* | *−0.000* |

The hand-engineered biophysical descriptors beat the mean baseline
comfortably and beat a one-hot positional encoding (1.009), so they carry real
signal. But they **do not beat a Morgan fingerprint of the same side chains**
cross-series, and adding them to it makes it slightly worse. Within-series
they are marginally ahead.

The honest reading: a fingerprint of the side-chain fragment already encodes
this chemistry, implicitly and more richly than twenty-two hand-chosen
descriptors. Explicit biophysical featurisation is **not** justified by
generalisation performance on this dataset. It remains justified for
*explanation* — a SHAP attribution over named descriptors such as
`estimated_charge` or `aromatic_rings` is interpretable in a way that
attribution over fingerprint bit 1,473 is not, and the brief's explanation
engine needs that.

---

## 7. A methodological correction to Stage B

While wiring the new representation in, two unfairnesses in the Stage B
benchmark came to light and are now fixed. Both change published numbers.

1. **Ridge was fitted on unscaled features.** Ridge penalises coefficients on
   each feature's own scale, so a representation mixing surface areas near 60
   with charges near 1 was handicapped against one of binary bits.
   `residue_chemistry__ridge` scored R² −3.55 unscaled. A `StandardScaler` now
   precedes ridge. Best cross-series improved from MAE 0.877 / R² 0.264 to
   **0.870 / 0.276**.
2. **Representation contrasts were taken as best-of-model.** That let a
   scaling artefact decide a representation question — and scaling is not
   neutral: standardising *helped* the dense descriptors and *hurt* the sparse
   binary fingerprint, whose rare bits are amplified by division by a tiny
   standard deviation (`positional_fragment_fp__ridge` went from 0.907 to
   1.143). Contrasts are now read off the random forest alone, which is
   invariant to monotone transforms of each feature.

**The Stage B headline survives and strengthens.** With the fairer comparator,
**all five** representation contrasts reverse between cross-series and
within-series evaluation, where three of three did before. The deployment gate
still fails: cross-series MAE 0.870 against ≤0.70 required and R² 0.276
against ≥0.30.

---

## 8. Deliverables status against the brief's §25

Reported now where measurable, and marked outstanding where not yet.

| # | Deliverable | Status |
|---|---|---|
| 1 | ACS methods studied and adapted | §1.2, §1.3 |
| 2 | Methods unsuitable | §1.4 |
| 3 | Peptide–protein structural dataset counts | 13 MC4R entries, 10 peptide complexes |
| 4 | Unique versus duplicated complexes | 10 entries, 7 unique peptides |
| 5 | Independently measured affinity labels | 630 peptides, MC4R pKi |
| 6 | Chemically identified peptide sequences | 630, backbone-traced |
| 7 | Protein targets | **1** today; 4 viable (§4.1) |
| 8 | Quality of structural information | 2.58–3.30 Å, 8 cryo-EM / 2 X-ray |
| 9–13 | Baseline through combined performance | §6, §7 and Stage B findings |
| 14 | Ablation results | Outstanding — Stage 6 |
| 15 | Confidence interval calibration | Unchanged: 0.909 at nominal 0.90 |
| 16 | Applicability-domain diagnostics | Stage B §5 |
| 17 | Computational limitations | §9 |
| 18 | Interactive visualization | Outstanding — Stage 8 |
| 19 | Tests and reproducibility | 595 tests passing |
| 20 | Git commit identifiers | Stage A `dcff98e`, Stage B `1545d46` |

---

## 9. Computational limitations

- **No PyTorch, no transformers, no ESM.** Protein language model embeddings
  cannot be computed here. Stage B bounded them instead of running them, which
  is a stronger claim for the ceiling but no substitute for a measurement.
- **No molecular dynamics, no docking software.** Conformational ensembles and
  docking-derived features are out of reach, and the brief's own instruction
  not to substitute MD for experimental evidence makes this less costly than
  it sounds.
- **Structural coverage is 1%.** Not a tooling limit but a data limit, and the
  main constraint on the geometric parts of the brief.
- **No recorded assay conditions.** pH, temperature and ionic strength are not
  in the ingested ChEMBL fields, so the residue chemistry engine applies one
  assumed pH of 7.4 uniformly and labels every ionisation estimate ESTIMATED.
  Condition sensitivity can be explored by varying the parameter; it cannot be
  fitted against measured conditions that are not present.
- **Stereochemistry still unresolved.** D- and L-residues remain
  indistinguishable in every representation, which matters in a melanocortin
  dataset where D-Phe⁷ is a defining substitution.

---

*Literature in §1 and §2 retrieved from PubMed and PubMed Central. Structural
data from RCSB PDB; activity counts from the ChEMBL API.*
