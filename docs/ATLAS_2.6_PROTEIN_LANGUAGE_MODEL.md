# ATLAS 2.6 — Protein language model embeddings: measured, no improvement, gate unchanged

**Code:** `atlas/ml/plm_embeddings.py` · **Data:** `data/ml/plm_embeddings.npz`, `data/ml/peptide_benchmark.json`
**Reproduce:** `python3 -m atlas.ml.plm_embeddings` then `python3 -m atlas.ml.peptide_benchmark`
**Tests:** 14 new (886 total) · **Dependencies added:** torch 2.14.1, transformers 5.20.2
(tokenizers 0.23.3, huggingface_hub 2.3.0), the first time this project has
installed either

Five representation extensions had already failed to improve cross-series
generalisation — explicit topology, residue chemistry, local sequence context,
assay context, backbone stereochemistry. Every one of them rearranged
information already present in the same 630 molecules. A pretrained protein
language model is the only lever that imports information from **outside** the
dataset, so it was the one untested extension whose failure would mean
something different from the other five.

It fails too. **Cross-series MAE goes from 0.894 to 1.093** when ESM-2
embeddings are added to the positional fragment fingerprint, and the embedding
alone scores 1.107 against a mean-predictor baseline of 1.189. The deployment
gate is **unchanged and still fails**: cross-series MAE 0.870 against ≤0.70
required, R² 0.276 against ≥0.30, both identical to the last digit of the
previous run. That makes **0 of 6**.

---

## 1. Why this was worth running, given five prior nulls

Two arguments, and they point in opposite directions, which is why the question
needed a measurement rather than a prediction.

**For:** ESM-2 650M was trained on UniRef50 — tens of millions of sequences,
none of them these peptides. Every prior extension was a different arrangement
of the same 630 molecules, so a null result from one of them says nothing about
whether external information would help. This arm is the only one that tests
that.

**Against, and specific:** the model is handed X's. 80.5% of these peptides
carry residues outside the canonical twenty, across 160 distinct side chains,
and `canonical_sequence()` in `atlas/ml/peptide_representations.py` renders
every one of them as `X` because no twenty-letter alphabet can name a
4-chloro-phenylalanine. ESM-2's vocabulary contains `X` as a single unknown
token, so a chlorophenylalanine, a 2-naphthylalanine and an Aib are the same
input symbol.

What that costs, measured rather than asserted:

| | distinct keys for 630 molecules | molecules sharing a key | oracle MAE |
|---|---|---|---|
| canonical 20-letter sequence | **205** | 77.8% | **0.382** |
| chemistry-aware positional | 449 | 40.6% | 0.186 |

The worst single collision: 10 molecules all read as `XRWX` and span pKi 5.42
to 9.43 — a 4.01 log unit, ~10,000-fold range of affinity rendered as one
string. A frozen embedding is a deterministic function of that string, so those
10 molecules receive **identical vectors** and no model built on them can
separate the 10 however good the embedding is.

**But the ceiling leaves room.** Oracle MAE 0.382 is far below the 0.870 the
best real model achieves. A sequence-only representation therefore had ample
headroom, and a null result here *cannot* be explained by the collapse. That
is what makes the experiment informative either way, and it is the reason the
result below is a finding about transfer rather than a restatement of the
ceiling.

## 2. What was computed, and what was frozen

ESM-2 650M (`facebook/esm2_t33_650M_UR50D`), weights never updated, mean-pooled
over residue token positions to a 1280-dimensional vector per sequence. The
flanking `<cls>` and `<eos>` tokens are excluded from the mean: ESM-2 is trained
with a masked-language objective and no sentence-level head, so its first token
carries no pooled summary the way a fine-tuned BERT's does.

The 630 peptides collapse to **205 distinct canonical sequences**, so 205
vectors are computed and cached in `data/ml/plm_embeddings.npz` (977 KB,
committed). The cache records the model name, the pooling rule and the torch,
transformers and numpy versions that produced it, so a stored vector cannot be
silently attributed to a different model.

**On leakage.** The language model never sees a label, and each sequence is
embedded independently of every other — no batch statistics, no cross-row
normalisation. Computing all 205 vectors in one pass is therefore not a
train/test boundary crossing, and a test asserts the property directly:
reordering the molecules permutes the rows and changes nothing else. The only
fitted objects are the downstream ridge and forest, which the existing
benchmark fits on training rows alone exactly as it does for every other
representation.

A sequence absent from the cache **raises** rather than becoming a zero row.
Zero-filling would make the arm score badly for a reason unrelated to the
language model, and the benchmark would report that as a measurement. (The one
place this fires in practice is `panel_replication`, which injects another
receptor's peptides; there the arm is skipped and logged, not zero-filled.)

## 3. Three arms, not one

The honest experiment separates what the PLM sees from the chemistry it cannot
see:

1. **`esm2_embedding`** — the embedding of `canonical_sequence(a)`. What a
   frozen PLM actually gets, X's included.
2. **`positional_fragment_fp+esm2_embedding`** — the same, concatenated with
   the side-chain fragment fingerprint. The chemistry the PLM cannot see, added
   back.
3. **`positional_fragment_fp`** — the existing baseline to beat.

All three run through the existing machinery in `atlas/ml/peptide_benchmark.py`:
the same Butina clusters, the same cross-series and within-series splits, the
same `CONTRAST_MODEL = "forest"` rule for contrasts. No new evaluation was
invented, because the comparison is only meaningful against the existing one.

**Verified like-for-like.** Re-running the benchmark with the new arms moved
**zero** pre-existing metrics: every representational ceiling is byte-identical,
every previously scored model has the same MAE and R² to all digits, and the
gate's three criteria are identical to the last decimal place. Only the four
new entries were added.

## 4. Results

### Cross-series — whole analogue series held out (439 train / 191 test)

Random forest, the contrast model:

| arm | MAE | R² | Spearman |
|---|---|---|---|
| ESM-2 embedding alone | 1.107 | −0.151 | 0.115 |
| ESM-2 + fragment fingerprint | 1.093 | −0.116 | 0.141 |
| **fragment fingerprint (baseline)** | **0.894** | **+0.240** | **0.491** |
| mean predictor | 1.189 | — | — |

Both PLM arms have **negative R²** — worse than predicting the training mean —
and the embedding alone has a Spearman of 0.115, which is close to no ordering
at all. Adding the fragment fingerprint back recovers almost nothing: 1.093
against the 0.894 the fingerprint reaches by itself.

Ridge, for completeness:

| arm | MAE | R² |
|---|---|---|
| ESM-2 embedding alone | 1.430 | −0.809 |
| ESM-2 + fragment fingerprint | 1.009 | +0.051 |
| fragment fingerprint | 1.143 | −0.262 |
| **best model overall** (`positional_fragment_fp+morgan_fp__ridge`) | **0.870** | **+0.276** |

### Within-series — members held out of series the model otherwise sees (474 / 156)

| arm | MAE | R² | Spearman |
|---|---|---|---|
| ESM-2 embedding alone | 0.630 | +0.557 | 0.740 |
| ESM-2 + fragment fingerprint | 0.613 | +0.589 | 0.764 |
| fragment fingerprint | 0.613 | +0.585 | 0.735 |
| series-mean lookup | 0.845 | — | — |
| best model overall (`positional_onehot+topology__forest`) | 0.520 | +0.679 | — |

Within-series, the PLM arm and the baseline are **the same number to three
decimals** (0.6128 against 0.6130), and both are beaten by the existing best at
0.520. The embedding alone does beat a series-mean lookup (0.630 against
0.845), but that is a weaker result than it looks: a Butina cluster can contain
several distinct canonical sequences, so an embedding that is constant within a
sequence group still subdivides a cluster. That is finer-grained lookup, not
resolution of a substitution — and within-series performance is interpolation
among near-duplicates in any case (median nearest-neighbour similarity 0.941,
100% of held-out molecules above 0.7 Tanimoto).

### The contrast table

Read off the random forest alone, per the existing `CONTRAST_MODEL` rule:

| contrast | cross-series | within-series |
|---|---|---|
| `esm2_embedding_vs_fragment_fp` | 1.107 vs **0.894** — fingerprint better | 0.630 vs **0.613** — fingerprint better |
| `esm2_embedding_added_to_fragment_fp` | 1.093 vs **0.894** — worse with it | **0.613** vs 0.613 — no change |

And the full ablation, cross-series, one feature group at a time added to the
fragment fingerprint: **0 of 6 improved it, 6 made it worse.** Topology 0.894 →
1.168; ESM-2 0.894 → 1.093; local context → 0.964; assay context → 0.946;
stereochemistry → 0.912; residue chemistry → 0.908.

## 5. The gate does not change

| criterion | required | actual | met |
|---|---|---|---|
| cross-series MAE | ≤ 0.70 | **0.870** | no |
| cross-series R² | ≥ 0.30 | **0.276** | no |
| within-series beats series-mean | < 0.845 | 0.520 | yes |

**FAIL**, as before, with the same two criteria unmet and the same best model
(`positional_fragment_fp+morgan_fp__ridge`). The affinity model still publishes
no numeric estimates anywhere in the product, and nothing in the prediction
console, the ranking service or the frontend changed as a result of this work.
The ranking gate is untouched: it was measured separately and still passes
within-series at 93.0%.

## 6. One number moved favourably, and it is not an improvement

Under **ridge**, adding the embedding to the fragment fingerprint lowers
cross-series MAE from 1.143 to 1.009 and lifts R² from −0.262 to +0.051. It is
recorded here rather than omitted, and it does not support a claim of
improvement, for two reasons.

First, this is precisely the artefact the project's `CONTRAST_MODEL = "forest"`
rule exists to exclude. Ridge penalises coefficients on the features' own
scale, and standardising a sparse binary fingerprint amplifies its rare bits by
dividing by a tiny standard deviation; adding 1280 dense, well-conditioned
columns changes the effective regularisation of the whole design matrix. That
is a difference between two *scalings*, not between two representations, which
is why representation questions in this project are decided on a forest — a
forest is invariant to monotone transforms of each feature.

Second, and decisively: 1.009 is worse than the 0.894 the fragment fingerprint
reaches on a forest without any embedding, and far worse than the 0.870 of the
best existing model. The best cross-series model is unchanged and so is the
gate. Nothing about this number licenses a claim of improvement on these
splits.

## 7. Why it failed — two explanations, one survives

**Not the representational ceiling.** The canonical sequence's oracle MAE of
0.382 sits far below the 0.870 the best real model reaches. There was room to
improve and the arm declined it. A test asserts this relationship directly, so
if a future dataset rebuild ever pushed the ceiling above the achieved error,
the test would fail and this argument would have to be rewritten rather than
quietly inherited.

**The vocabulary loss is the plausible mechanism.** The 449 chemically distinct
peptides the fragment encoding resolves arrive at the language model as 205
inputs, with 77.8% of molecules sharing their input with another. A model
pretrained on natural proteins has no representation for a
4-chloro-phenylalanine because UniRef contains none, and this dataset's entire
design — single noncanonical substitutions within analogue series — lives in
exactly the space the pretraining corpus does not cover. Pretraining breadth
does not transfer to a chemotype the pretraining alphabet cannot write down.

This is the sixth consecutive null on the representation layer and the first
that cannot be answered by adding more chemistry to the feature set. It points
back where the Stage A diagnosis already pointed: **sample structure.** 630
molecules collapse to 30 analogue series (effective sample ratio 0.048), one
series is 53% of the set, four receptors carry peptide data, and the fitted
learning curve puts MAE 0.70 at roughly 158,000 peptides.

## 8. What this does not show

The claim is narrow on purpose. This measured one thing and the limits are
worth stating, because each is a different experiment rather than a reason to
doubt this one.

- **Frozen embeddings only.** No fine-tuning was attempted. Fine-tuning a
  650M-parameter model on 439 training molecules in ~30 series is not something
  this dataset supports, and the overfitting risk would make a positive result
  harder to believe than this negative one.
- **One model size.** ESM-2 650M, the size published frozen-embedding work
  uses. 150M, 3B and 15B were not run.
- **One pooling rule.** Mean over residue tokens from the final layer.
  Per-residue outputs aligned to the positional slots, and intermediate layers
  (which transfer better than the last for some tasks), were not tested.
- **Ligand only.** The MC4R receptor sequence was not embedded or used to
  condition the ligand representation.
- **No published benchmark number is imported here.** Every figure in this
  document comes from ATLAS's own splits. ESM-2's performance on other tasks is
  not evidence about this one, in either direction.

What would change the conclusion is not a better representation. It is more
series — and that was probed and ruled out for this receptor: non-human
melanocortin receptors add 4% more records.

## 9. What was left installed

torch 2.14.1 and transformers 5.20.2 are now installed, reversing a deliberate
earlier choice. The embedding cache is committed, so **neither is needed to
reproduce the benchmark or to run the test suite** — `atlas/ml/plm_embeddings.py`
imports torch only inside the function that builds the cache, and the test
suite reads the cache. Rebuilding the cache needs both, plus a 2.5 GB model
download, and is a deliberate act.

The arm degrades honestly rather than failing: if the cache is absent, the
benchmark skips the PLM arm, and the running count of measured extensions
drops from six to five rather than claiming a measurement that did not happen.
