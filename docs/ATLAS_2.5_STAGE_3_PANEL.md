# ATLAS 2.5 — Stage 3: the melanocortin receptor panel

**Panel:** `71e36895a4b10915` · 5 receptors · Ki endpoint
**Code:** `atlas/ml/multitarget.py`, `atlas/ml/panel_benchmark.py`
**Data:** `data/ml/melanocortin_panel.json`, `data/ml/panel_benchmark.json`
**Reproduce:** `python3 -m atlas.ml.multitarget` then
`python3 -m atlas.ml.panel_benchmark`
**Tests:** 32 new (627 total)

Stage 1 established that ATLAS had exactly one target, so a peptide–target
dual encoder could not be evaluated: the target side had zero variance, any
target encoder would be a constant, and an ablation removing it would show
nothing by construction. This stage removes that constraint and then tests
whether the premise holds.

---

## 1. What was built

`atlas/ml/dataset.py` was generalised minimally: `build_dataset` now takes an
`endpoint`, defaulting to `Ki`, with an assay class per endpoint
(`Ki`/`Kd`/`IC50` → binding, `EC50` → functional). The existing
`data/ml/mc4r_pki.json` is untouched and all prior tests continue to pass.

`atlas/ml/multitarget.py` builds a panel on top: one dataset per
(receptor, endpoint) pair, never merged. **There is deliberately no accessor
that returns several endpoints at once** — a pKi is an equilibrium binding
constant and a pEC50 is a functional response that depends on the assay's
signal amplification, and a model trained on their union predicts neither. A
test asserts the absence.

Receptor metadata is fetched with provenance rather than recalled: UniProt
accessions come from ChEMBL's own target records, sequences from UniProt.

| Receptor | ChEMBL | UniProt | Length | Ki molecules |
|---|---|---|---|---|
| MC1R | CHEMBL3795 | Q01726 | 317 | 480 |
| MC2R | CHEMBL1965 | Q01718 | 297 | 23 |
| MC3R | CHEMBL4644 | P41968 | 323 | 609 |
| MC4R | CHEMBL259 | P32245 | 332 | 1,981 |
| MC5R | CHEMBL4608 | P33032 | 325 | 406 |

MC4R's accession independently matches the `P32245` used for the RCSB
structure query in the Stage 1 audit.

**3,499 Ki measurements over 2,066 distinct molecules.**

---

## 2. The premise holds: the family is profiled across receptors

The expansion is only worth making if the same molecules were measured on
several receptors. Otherwise the panel is four disjoint datasets wearing one
name. Measured, not assumed:

| Receptors per molecule | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Molecules | 1,277 | 250 | 434 | 105 |

**789 of 2,066 molecules (38.2%) are measured on two or more receptors**, and
105 on all four major subtypes. Pairwise: MC3R|MC4R 585, MC1R|MC4R 462,
MC4R|MC5R 357, MC3R|MC5R 349, MC1R|MC3R 306, MC1R|MC5R 123.

**Median selectivity span 1.31 log units; maximum 4.40.** A selectivity span
is the range of p-values one molecule shows across the receptors it was
measured on — the quantity a target encoder must predict.

That median is the number that matters most here. **It exceeds the best
cross-series error ATLAS has ever achieved (MAE 0.870).** A model that ignores
which receptor it is being asked about therefore carries a systematic error
larger than its own accuracy — a 20-fold affinity difference treated as noise.

### Restricted to real peptides

| | MC4R only (2.4) | Panel (2.5) |
|---|---|---|
| Peptides | 630 | **640** |
| Peptide measurements | 630 | **1,354** |

The family adds **only 10 new peptides** but **2.15× the measurements**,
because **424 of 640 peptides (66.2%)** were already profiled on more than one
receptor — a far higher rate than the 38.2% across all molecules, since
melanocortin peptides are routinely run against the whole family.

This matters for the specific problem Stage A diagnosed. Stage A found the
effective sample size was ~30 analogue series and that more molecules from the
same series would buy little. The panel does not add series — it adds a second
axis. The same peptide against four receptors is new information
(selectivity), not more of the same chemistry.

Peptide rows per receptor: MC4R 630, MC1R 359, MC3R 269, MC5R 96. **MC2R
contributes no peptides at all** (23 molecules, none tracing as peptides), so
it drops out of the peptide panel automatically — the natural abstention case
the Stage 1 audit anticipated.

---

## 3. The leakage trap, and the guard against it

A molecule measured on MC1R and MC4R contributes two rows. Split those rows
independently and the model is shown the molecule's affinity on one receptor
while being asked for it on another — a median 1.31 log units away, with the
chemistry handed over. That is not a subtle effect at this overlap rate: 66.2%
of peptides would be affected.

Every split here therefore **moves whole molecules**, and cluster labels are
computed on unique molecules before being broadcast to rows. Four tests
assert it: no molecule, and no cluster, straddles a split; all rows of a
multi-receptor peptide land on one side.

---

## 4. Does knowing the receptor help?

Cluster split, whole molecules grouped, all four receptors present on both
sides. Random forest throughout.

811 training rows, 543 test rows, 29 molecule clusters.

| Arm | MAE | R² |
|---|---|---|
| mean baseline | 1.234 | −0.457 |
| peptide only | 1.125 | **−0.291** |
| peptide + receptor one-hot | **0.989** | **−0.086** |
| peptide + receptor sequence | 0.992 | −0.089 |

Receptor identity carries signal: a relative gain of **0.136 log units** from
knowing which receptor is being asked about. The one-hot and sequence arms are
within 0.003 of each other, exactly as §5 predicts — on receptors already seen
the two encodings have identical capacity, so this split cannot tell them
apart.

**But both arms have negative R²**, meaning both are worse than predicting the
test set's own mean, even while beating the training mean's MAE. The only claim
this comparison supports is the relative one — the receptor helps compared to
not knowing it. It does not show the panel model is good.

**These numbers are not comparable with the ATLAS 2.4 figures.** The panel task
is harder in two ways at once: it holds out whole molecule clusters across four
receptors rather than one, and its labels carry the across-receptor variance a
single-target dataset does not have. A panel MAE and an MC4R-only MAE measure
different problems, and putting them in one table would be the error this note
exists to prevent.

---

## 5. Can a target encoder reach an unseen receptor? Not established

Leave-one-receptor-out is the **only** split that tests a target encoder at
all. Under any split keeping all receptors in training, a one-hot encoding and
a sequence encoding have identical capacity — five receptors give five distinct
vectors either way — so the cluster split above cannot distinguish them even in
principle. Under LORO the held-out receptor's one-hot column is necessarily
all-zero (verified per arm in the saved report), while its sequence features
are real numbers.

| Held out | Rows | Baseline | Peptide only | + one-hot | + sequence | Verdict |
|---|---|---|---|---|---|---|
| MC1R | 359 | 1.223 | 1.447 | 1.418 | 1.425 | **fails** |
| MC3R | 269 | 1.171 | 0.771 | 0.697 | **0.694** | passes |
| MC4R | 630 | 1.100 | 1.192 | 1.187 | 1.176 | **fails** |
| MC5R | 96 | 0.973 | 0.619 | 0.669 | **0.615** | passes |

The sequence encoding beat the peptide-only arm on **4 of 4** held-out
receptors. **That is the wrong comparison.** Against the mean baseline — the
reference that matters — transfer succeeds on **2 of 4** (MC3R, MC5R) and
**fails on 2 of 4** (MC1R, MC4R), where every arm is worse than predicting the
mean. Being the better of two inadequate models is not a result.

**A target encoder is not established.** Two reasons to hold that loosely in
both directions:

1. With five receptors a sequence encoding is five points in a composition
   space. This measures whether their geometry supports reaching a sixth — not
   whether sequence features carry more information than one-hot, which on
   receptors already seen they demonstrably do not.
2. Four held-out receptors is four trials. No seed-variability or
   encoding-variant study was run.

The panel makes the question askable, which one target did not. It does not
answer it.

---

## 6. Generalisation regimes, available but not yet benchmarked

`generalisation_regimes()` returns the three index sets the brief
distinguishes, so a later stage cannot accidentally report an easier regime's
result as the harder one:

- **known target, new peptide family** — unseen chemistry, seen receptors;
- **known peptide family, related target** — unseen receptor, chemistry seen on
  other receptors;
- **new peptide family and new target** — neither, and typically much the
  smallest, since it needs molecules both on the held-out receptor and in
  clusters absent from everything else.

Tests assert the first two are disjoint and that each carries its description.
Benchmarking across all three is Stage 6 work.

---

## 7. What has not changed

- **The deployment gate still fails.** Nothing here touched
  `atlas/ml/peptide_benchmark.py`'s `GATE`, and the MC4R model's cross-series
  MAE 0.870 / R² 0.276 is unchanged. The peptide affinity model publishes no
  numeric estimates.
- **Stage A's conclusions stand.** The panel's peptide population is 640
  against 630, so the structural census, the 30-series finding and the
  non-canonical fraction are all materially unchanged.
- **Endpoints remain unpooled.** Only Ki was built. EC50 and IC50 are
  available through the same code path as separate datasets and have not been
  fetched.
- **Stereochemistry is still unresolved**, now across four receptors rather
  than one.

---

## 8. Honest limitations

- **MC2R is effectively absent** from the peptide panel: 23 molecules, no
  traced peptides. Four receptors, not five, carry peptide data.
- **MC5R is thin** at 96 peptide rows, so its LORO result rests on a small
  test set.
- **The sequence encoding is crude** — amino-acid composition and length over
  the whole receptor, not the binding pocket. A pocket-restricted encoding
  would be more defensible and is not yet built, because identifying the
  pocket requires the structural work reframed in Stage 1 §4.2.
- **No assay conditions.** pH, temperature and ionic strength are still absent
  from the ingested fields, so the experimental-context encoder the brief asks
  for has nothing to encode.
- **One seed** (0) throughout.
- **Cross-receptor rows share molecules under LORO by design.** The held-out
  receptor is unseen, which is the point, but a molecule's affinity on other
  receptors has been seen. That is recorded in a test rather than described as
  novelty.
