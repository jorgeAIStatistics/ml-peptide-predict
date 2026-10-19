# ATLAS 2.4 — Stage A: quantitative failure and dataset audit

**Dataset:** `29fe2a249bcd581d` · MC4R binding affinity (pKi, −log₁₀ M) ·
CHEMBL259, *Homo sapiens* · 1981 molecules
**Code:** `atlas/ml/peptide_structure.py`, `atlas/ml/peptide_audit.py`
**Report:** `data/ml/peptide_audit.json` · reproduce with
`python3 -m atlas.ml.peptide_audit`
**Tests:** 52 new (536 total)

The peptide affinity model was not deployed. This stage establishes why, by
measurement. Every number below is produced by the committed code from the
committed dataset.

> **Replicated 2026-10-19 (ATLAS 2.5 Stage 3b).** This audit was measured on
> MC4R alone. Re-run per receptor over the Stage 3 melanocortin panel, its two
> central findings hold across the family: independent analogue series per
> receptor are MC1R 22, MC3R 22, MC4R 30, MC5R 14 (effective sample size
> 0.048–0.146 of nominal), so the **~30-series ceiling is a property of
> melanocortin peptide chemistry rather than of the MC4R literature** — MC4R,
> with 2.3× the peptides of its nearest rival, has the most series. The
> non-canonical residue burden is **68.4–80.5%** across receptors against the
> 80.5% measured here, so it is class-wide. MC2R abstains: 23 molecules and no
> traced peptides. See `docs/ATLAS_2.5_STAGE_3_PANEL.md` §8.1.

---

## 1. The previous peptide classification was a proxy, and it was wrong on 44 molecules

The earlier classification called a molecule `peptide_like` when it had ≥4
amide bonds and MW ≥600 Da. That counts functional groups; it does not
establish that a molecule is a peptide. Stage A replaces it with backbone
tracing: a unit is `N–Cα–C(=O)`, units chain when one unit's carbonyl bonds the
next unit's nitrogen, and a residue count comes from walking that chain.

The tracer was validated against 14 structures retrieved from PubChem by CID,
with residue counts and sequences known independently of this code. All 14
residue counts are exact, and the recovered sequences are correct:

| Molecule | Expected | Traced | Sequence recovered |
|---|---|---|---|
| Oxytocin | 9 | 9 | `CYIQNCPLG` |
| Somatostatin-14 | 14 | 14 | `AGCKNFFWKTFTSC` |
| Angiotensin II | 8 | 8 | `DRVYIHPF` |
| Bradykinin | 9 | 9 | `RPPGFSPFR` |
| Afamelanotide | 13 | 13 | `SYS·X·EHFRWGKPV`, X = norleucine |
| Cyclosporin A | 11 | 11 | cyclic, MeBmt as `*C(O)C(C)CC=CC` |
| Melanotan II | 7 | 7 | cyclic, Asp–Lys lactam |
| Aspirin, sildenafil | 0 | 0 | not peptides |

Applied to the dataset:

| | Previous label | Backbone tracing |
|---|---|---|
| Peptides | 628 | **630** |
| Peptidomimetics | — | 620 |
| Non-peptide small molecules | 1353 | 731 |

The totals nearly agree. **The membership does not.** 21 molecules counted as
peptides are not peptides, and 23 peptides were missed — a symmetric
difference of 44, or **7.0% of the peptide set**. Two classifications agreeing
on a total while disagreeing on membership is exactly the failure mode the
brief warned against, and it is membership that decides what a model is
trained and judged on.

620 molecules trace one or two residues. They contain peptide bonds and may
bind the same receptor, but they have no sequence to featurise. Counting them
as peptides is what made the peptide population look larger than it is.

### Composition of the real peptide population

- **Constrained, not linear.** 415 side-chain bridged, 77 disulfide-containing,
  58 head-to-tail cyclic (these overlap). Only **131 of 630** are
  unconstrained. 5 molecules are classed cyclic solely on having a
  backbone-containing ring ≥12 atoms.
- **Dominated by one length.** 314 of 630 peptides (49.8%) are heptapeptides —
  the melanocortin cyclic-lactam series.
- **Mostly non-natural.** **507 of 630 (80.5%)** carry at least one residue
  outside the twenty canonical side chains, across **160 distinct fragments**.

The commonest non-canonical residues are not exotic biology; they are
medicinal chemistry:

| Side-chain fragment | Count | Identity |
|---|---|---|
| `*Cc1ccc(Cl)cc1` | 237 | 4-chlorophenylalanine |
| `*Cc1ccc(Cl)cc1Cl` | 228 | 2,4-dichlorophenylalanine |
| `*CCC(=O)NCCC` / `*CCCNC(=O)CC` | 151 each | the two halves of the lactam bridge |
| `*CCCC` | 122 | norleucine |
| `*Cc1ccc2ccccc2c1` | 107 | 2-naphthylalanine |

**This is the single most consequential finding for Stage B.** A protein
language model's vocabulary has no token for 4-chloro-, 2,4-dichloro- or
naphthyl-phenylalanine. Frozen ESM-2 or ProtT5 embeddings would have to render
these as `X` or as plain phenylalanine, collapsing precisely the substitutions
this dataset's structure–activity series varies. 80.5% of the population is
affected. Stage B must therefore treat sequence-only protein embeddings as a
hypothesis to be *tested and expected to fail*, not as the default upgrade.

---

## 2. The peptide population is 30 series, not 630 observations

Taylor-Butina clustering at 0.4 Tanimoto distance:

- **30 clusters** among 630 peptides (effective sample size ratio 0.048)
- largest cluster **335 molecules (53.2%)**; top five **77.6%**
- 7 singletons
- **median within-cluster label SD 0.863**, against **1.281 overall**

Two things follow. First, a held-out-cluster split has roughly 30 independent
groups to draw from, so the effective sample size for generalisation is nearer
30 than 630 — and the learning-curve argument in §4 must be read against that.
Second, **two-thirds of the label spread lives inside analogue series**
(0.863/1.281). Knowing which series a molecule belongs to leaves most of the
affinity range unexplained, so the task is intrinsically one of resolving
single-substitution effects, not of recognising chemotypes.

Provenance is correspondingly narrow: the 630 peptides come from **36
documents and 38 assays**, 17.5 molecules per document, with the largest single
document supplying 21% and the top ten supplying 67%. This is a sample of a
few laboratories' series, not a sample of peptide chemistry.

---

## 3. Peptides and non-peptides are far apart — measured by distance, not hulls

Convex-hull membership was deliberately not used: in 2048-bit fingerprint
space it reports "out of domain" for essentially everything and distinguishes
nothing.

| Nearest-neighbour Tanimoto | median |
|---|---|
| peptide → nearest non-peptide | **0.372** |
| peptide → nearest other peptide | **0.934** |
| non-peptide → nearest peptide | 0.224 |
| non-peptide → nearest non-peptide | 0.851 |

**70.0%** of peptides have no non-peptide neighbour above 0.4. On descriptors,
**12 of 15** differ between the populations by more than one pooled standard
deviation — clogp 3.27, qed 2.97, h_donors 2.59, tpsa 2.35, amide_bonds 2.24.
The shift is not confined to molecular size.

This is the quantitative statement of why exchangeability — the assumption
conformal calibration rests on — does not hold across these two populations,
and why group-conditional (Mondrian) calibration was the right correction.

---

## 4. The failure decomposed

All comparisons use one fixed test set: 141 held-out peptides from a
Taylor-Butina cluster split (489 train / 141 test, seed 0), varying
representation, training population and training size one at a time.

### A — Representation mismatch: real, but not the whole story

| Representation + model | MAE | R² |
|---|---|---|
| **fingerprint + ridge** | **1.136** | **+0.123** |
| descriptor + forest | 1.259 | +0.014 |
| fingerprint + forest | 1.274 | −0.044 |
| descriptor+composition + forest | 1.277 | −0.004 |
| *mean-predictor baseline* | *1.343* | *−0.002* |
| composition + forest | 1.357 | −0.203 |
| descriptor + ridge | 1.567 | −0.442 |
| composition + ridge | 2.212 | −4.113 |

The best model beats predicting the mean by **0.206 log units** and reaches
**R² 0.123**. It accounts for about 12% of held-out variance. This is not
marginal underperformance — the model is close to uninformative on this
population, and that is the honest characterisation of the failure.

Sequence composition — the new representation that backbone tracing makes
available — performs **worse than the baseline**. That is a genuine negative
result with a clear cause: a bag of residue counts cannot distinguish
positional isomers, and this population is largely single-substitution
analogue series of cyclic heptapeptides. A test asserts this limitation
explicitly (`test_composition_cannot_distinguish_positional_isomers`).

**Implication:** the representation must encode residue *identity at a
position* and the macrocyclic constraint. Order-free and vocabulary-limited
representations are both ruled out by measurement.

### B — Domain shift: confirmed, and asymmetric

Same 141 test peptides; only the training population changes (descriptors,
random forest):

| Trained on | n | MAE | R² |
|---|---|---|---|
| 489 peptides | 489 | 1.259 | +0.014 |
| 1351 non-peptides | 1351 | **1.734** | **−0.791** |
| both | 1840 | **1.195** | +0.096 |

Training on 2.8× as much non-peptide data gives a model **worse than
predicting the mean** (R² −0.791). Non-peptide MC4R affinity data does not
transfer to peptides. But pooling is not harmful: with peptides anchoring the
fit, adding the small molecules *helps* by 0.064 log units. So the small
molecules carry some usable signal that they cannot deliver on their own.

**Implication:** multitask or peptide-anchored joint training, not
small-molecule pretraining followed by transfer.

### C — Dataset size: contributing, but not a route to a fix

The learning curve falls monotonically and has not saturated:

| Training peptides | 122 | 244 | 367 | 489 |
|---|---|---|---|---|
| MAE | 1.245 | 1.216 | 1.172 | **1.136** |

Fitting MAE against ln(n) gives slope **−0.0773 per e-fold** (fit R² 0.938).
Extrapolating: MAE 0.7 at roughly **158,000** training peptides, MAE 0.5 at
**2.1 million**. Against 630 peptides in the whole of ChEMBL for this target,
quantity is a real contributor that **cannot be the route to a fix**. (This is
an extrapolation orders of magnitude beyond the observed range; it bounds the
hypothesis rather than forecasting an outcome, and the §2 finding that there
are only 30 independent series means even the observed slope likely overstates
what more molecules from the same series would buy.)

### D — Irreducible variability: not the binding constraint

Across all 184 replicated peptides, mean replicate spread is 0.153 log units,
implying a noise floor near **MAE 0.076**. On the 19 replicated molecules
inside the test set the figure is **0.245** — the small-sample estimate is
more pessimistic, and is the one used for the headroom calculation. Either way
the best model sits at **1.136**, leaving **+0.891 log units of headroom**.

Label quality is otherwise good: 29.2% of peptides have more than one record,
only **1.6% (10 molecules)** are discordant by more than one log unit, and the
median replicate spread is 0.0 — most replicated peptides agree exactly, so the
floor is driven by a small number of genuinely disputed molecules (max spread
1.99). Peptide labels span pKi 4.16–10.82 with SD 1.281, wider than the
non-peptide spread (1.160).

**Label noise does not explain the failure.**

---

## 5. Verdict

The failure is not one cause. In order of measured contribution:

1. **The evaluation task is harder than it appeared (§2).** 630 peptides are 30
   series, and two-thirds of the label variance is *within* series. A
   held-out-cluster split demands cross-series generalisation from ~30
   effective examples.
2. **Representation mismatch is real and both candidate fixes are constrained
   (§4A).** Fingerprints reach R² 0.123; residue composition is worse than the
   baseline. 80.5% non-canonical residues rules out vocabulary-limited
   sequence models before they are tried.
3. **Domain shift is confirmed and asymmetric (§4B).** Small-molecule data
   alone is worse than useless (R² −0.791) but helps when pooled.
4. **Dataset size contributes but bounds itself out as a solution (§4C).**
5. **Label noise is not a factor (§4D).** 0.891 log units of headroom.

### What this licenses, and what it forbids

Under the standing constraints, the peptide affinity model **remains
undeployed**, and this audit strengthens rather than weakens that decision: a
model at R² 0.123 must not publish numeric affinity estimates, and must not
propagate them into PK/PD or knowledge-graph edges. The existing
target-relevance gating and measurement-first behaviour stay as they are — for
PT-141, melanotan I and II the registry already has *measured* values, which
outrank any prediction.

### Stage B is now specified by measurement, not preference

- Positional, not order-free, residue encoding — **required** by §4A.
- Chemistry-aware residue vocabulary covering 160 non-canonical fragments —
  **required** by §1; a canonical-alphabet model cannot represent 80.5% of the
  population.
- Frozen ESM-2 / ProtT5 to be benchmarked as a **predicted-to-fail** control,
  with the vocabulary collapse measured explicitly rather than assumed.
- Macrocyclic constraint as an explicit feature — 499 of 630 are constrained.
- Peptide-anchored multitask training over transfer — **indicated** by §4B.
- Evaluation must report cross-series generalisation and within-series
  resolution **separately**; a single aggregate MAE conceals which of the two a
  model actually achieves.
- Any deployment gate must clear MAE 0.245 by a margin that matters, and the
  within-series SD of 0.863 is the number a useful model has to beat.

---

## Known limitations of this audit

- Backbone tracing establishes connectivity, not stereochemistry. D- versus
  L-residues are **not** resolved, and D-amino acid substitution is a central
  design axis in melanocortin peptides (melanotan II carries D-Phe⁷). This is a
  known gap in the current representation, not a solved problem.
- Residue *order* is recovered along the traced chain, but for branched or
  multi-segment molecules the chain reported is the longest one; 
  discrepancies between total backbone units and the longest chain are recorded
  as caveats per molecule.
- Glutathione is correctly traced as two α-linked residues rather than three,
  because its glutamate is joined through the γ-carboxyl. γ- and β-linked
  residues are not counted as α-backbone, by design.
- The noise floor is estimated only from replicated molecules; the 70.8% of
  peptides measured once contribute no estimate.
- Proline is identified by ring closure onto its own backbone nitrogen;
  hydroxyproline and other imino acids are recorded as non-canonical rather
  than guessed at.
