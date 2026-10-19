# ATLAS 2.4 — Stage B: peptide-native representations, benchmarked

**Dataset:** `29fe2a249bcd581d` · MC4R pKi · 630 structurally-confirmed peptides
**Code:** `atlas/ml/peptide_representations.py`, `atlas/ml/peptide_benchmark.py`
**Report:** `data/ml/peptide_benchmark.json` · reproduce with
`python3 -m atlas.ml.peptide_benchmark`
**Tests:** 33 new (569 at the time; 595 after ATLAS 2.5 Stage 2)

> **Two corrections, applied 2026-10-19.** Read them before the findings
> below; the second withdraws this document's original headline.
>
> **(a) Method, corrected.** Two unfairnesses were found while adding the
> ATLAS 2.5 residue-chemistry representation: ridge was fitted on unscaled
> features, and representation contrasts were taken as best-of-model, which
> let a scaling artefact decide a representation question. Contrasts are now
> read off the random forest, which is invariant to monotone feature
> transforms. All numbers below are the corrected ones. Detail in
> `docs/ATLAS_2.5_STAGE_1_AUDIT.md` §7.
>
> **(b) Scope, corrected — the headline does not generalise.** Under the
> fairer comparator the tally rose from three of three contrasts reversing to
> five of five, and this document originally called that a strengthening. It
> was not: everything here was measured on **MC4R alone**, and re-running per
> receptor over the Stage 3 panel gives **MC4R 5/5, MC1R 3/5, MC3R 2/5, MC5R
> 1/5**. All five reverse on MC4R and on no other receptor, so the *direction*
> of a contrast is partly a property of this receptor's data. The
> representation *ranking* does not transfer either — four distinct best
> representations across four receptors, none winning twice, and
> `residue_chemistry`, negative cross-series here, is part of the winner on
> MC5R. What survives is the structural point: the two evaluations can
> disagree, so one number hides which question was asked. The tally of five
> was an n=1 result reported as general and must not be cited as a fact about
> peptide representations. Detail in `docs/ATLAS_2.5_STAGE_3_PANEL.md` §8.3.

---

## 1. The evaluation was split in two, and that changed every conclusion

Stage A found that a single aggregate error conceals two different
capabilities. Both are now reported:

- **Cross-series** — whole Taylor-Butina analogue series held out (439 train /
  191 test). Median nearest-training-neighbour similarity **0.600**; only
  26.7% of held-out molecules have a neighbour above 0.7.
- **Within-series** — members held out of series the model otherwise sees (474
  train / 156 test, clusters of ≥4 only). Median neighbour similarity
  **0.941**; **100%** have a neighbour above 0.7.

| | Cross-series | Within-series |
|---|---|---|
| Best MAE | **0.870** | **0.513** |
| Best R² | **+0.276** | **+0.701** |
| Best Spearman ρ | +0.518 | +0.825 |
| Mean-predictor baseline | 1.189 | 1.040 |
| Series-mean lookup | — | 0.845 |

The best model in both cases is `positional_fragment_fp + morgan_fp` with
ridge regression.

### On MC4R, every representation choice reverses

All five design contrasts flip sign between the two evaluations **on this
receptor**. Correction (b) above is essential context: only MC4R behaves this
way, and the per-receptor tallies are 5/5 here against 3/5, 2/5 and 1/5
elsewhere.

| Contrast (random forest) | Cross-series winner | Within-series winner |
|---|---|---|
| fragment fingerprint vs one-hot | **fragment** (0.894 vs 1.009) | one-hot (0.589 vs 0.613) |
| peptide-native vs whole-molecule FP | **fragment** (0.894 vs 0.905) | morgan (0.548 vs 0.613) |
| adding explicit topology | **without** (0.894 vs 1.168) | with (0.546 vs 0.613) |
| residue chemistry vs fragment FP | **fragment** (0.894 vs 0.936) | chemistry (0.602 vs 0.613) |
| adding residue chemistry | **without** (0.894 vs 0.908) | with (0.606 vs 0.613) |

A programme that benchmarked only within-series — the evaluation most readily
reported — would have selected one-hot residue indices, explicit topology
features, whole-molecule fingerprints and hand-engineered biophysical
descriptors. Each is the choice that generalises
*worst* to an unseen series — on this receptor. That vindicates Stage A's
requirement to report the two evaluations separately, which is the part that
held up when the family was measured. What did not hold up is the strength of
the effect: the contrasts reverse far less often on MC1R, MC3R and MC5R, so
this is a reason to report both evaluations rather than evidence that any
particular representation choice is the right one.

---

## 2. Representational ceilings, computed without a protein language model

A frozen sequence embedding is a deterministic function of the sequence
string. Two molecules with the same string receive **identical** embeddings, so
no model built on them can separate the two. Grouping molecules by
representation key and predicting each group's mean therefore gives a hard
lower bound on the error of *any* model using that representation — a stronger
statement than benchmarking one model, and one that needs no weights
downloaded. (PyTorch is not installed in this environment, which made the
stronger argument the necessary one.)

| Representation | Distinct keys | Colliding | Oracle MAE |
|---|---|---|---|
| canonical sequence, length only | 23 | 98.7% | 0.907 |
| **canonical sequence, 20-letter** | **205** | **77.8%** | **0.382** |
| canonical composition (order-free) | 185 | 81.7% | 0.418 |
| chemistry-aware composition (order-free) | 412 | 48.9% | 0.212 |
| **chemistry-aware positional** | **449** | **40.6%** | **0.186** |

The collapse is real and large. The worst single collision: **10 molecules all
read as `XRWX`** yet span pKi **5.42 to 9.43** — 4.01 log units, a
**10,233-fold** difference in affinity rendered as one string. A chemistry-aware
encoding distinguishes 7 of those 10.

Order matters more than chemistry: making the chemistry-aware encoding
order-free costs 0.186 → 0.212, confirming by construction why Stage A's
composition representation scored below the mean baseline.

### This corrects a Stage A prediction

Stage A predicted that vocabulary collapse would make frozen ESM-2 or ProtT5
embeddings fail, and recommended benchmarking them as a *predicted-to-fail*
control. The collapse is confirmed — but its floor of **MAE 0.382 is better
than the best model measured here (0.870)**. At current performance the
vocabulary collapse is therefore **not yet the binding constraint**; it would
become binding only for a model already below 0.382.

The honest conclusion is the opposite of Stage A's expectation:
**learnability, not representational capacity, limits every representation
tried.** A frozen sequence embedding is not ruled out by its ceiling. It is
simply not the bottleneck yet, and should be revisited if and when a model
approaches MAE 0.4.

---

## 3. Where the peptide-native representation genuinely wins

On the harder cross-series question the designed representation does what it
was designed to do.

**Full cross-series ranking:**

| Representation + model | MAE | R² | ρ |
|---|---|---|---|
| positional_fragment_fp + morgan_fp, ridge | **0.870** | **+0.276** | +0.518 |
| positional_fragment_fp + morgan_fp, forest | 0.880 | +0.263 | +0.478 |
| positional_fragment_fp, forest | 0.894 | +0.240 | +0.491 |
| positional_fragment_fp + topology, ridge | 0.905 | **+0.272** | +0.533 |
| morgan_fp, forest | 0.905 | +0.131 | +0.430 |
| positional_fragment_fp, ridge | 0.907 | +0.252 | **+0.573** |
| positional_onehot, forest | 1.009 | +0.014 | +0.431 |
| positional_onehot, ridge | 1.042 | −0.048 | +0.278 |
| *mean baseline* | *1.189* | *−0.276* | — |
| topology, ridge | 1.214 | −0.485 | +0.040 |
| topology, forest | 1.508 | −1.021 | −0.031 |
| morgan_fp, ridge | 1.580 | −1.464 | +0.127 |

Against the whole-molecule fingerprint on the same test set, the peptide-native
representation roughly **doubles cross-series R², from +0.131 to +0.264**. That
is a real improvement on Stage A's position and the main deliverable of this
stage.

### The mechanism is measured, not assumed

The fragment-fingerprint encoding beats one-hot cross-series by 0.116 log units
because **38.7% of held-out molecules carry a side chain absent from the
119-entry training vocabulary** (104 positions in total). A one-hot index
encodes those as all-zero — it has nothing to say about a residue it has never
seen. A fingerprint of the side-chain fragment places an unseen
3-chlorophenylalanine near a seen 4-chlorophenylalanine. Tests assert both
halves of this: `test_onehot_is_blind_to_an_unseen_side_chain` and
`test_fragment_fingerprint_places_unseen_near_seen`.

Within-series, where only 14.7% of held-out molecules carry an unseen side
chain, one-hot wins instead — exactly as the mechanism predicts.

---

## 4. Where Stage A was wrong: explicit topology hurts

Stage A listed "macrocyclic constraint as an explicit feature — **required**",
reasoning that 499 of 630 peptides are constrained. Measured cross-series, it
is **not supported**:

- `positional_fragment_fp` forest: **0.894** → with topology: **1.168**
- topology alone: 1.214 (ridge), 1.508 (forest) — both worse than the 1.189
  mean baseline

Residue count, molecular weight and ring size let a tree split on *series
identity*, which is precisely what does not transfer to an unseen series. The
constraint is better carried implicitly by the bridge chemistry already present
in the fragment encoding, where the lactam halves appear as their own side-chain
fragments.

Within-series, topology helps (0.612 → 0.546) — because there, splitting on
series identity is rewarded rather than punished. The same feature block is
useful or harmful depending on the question, which is the §1 reversal again.

---

## 5. Within-series success is interpolation, and is reported as such

R² 0.701 is a strong number and would be easy to over-read. Its diagnostics do
not support a generalisation claim: median neighbour similarity is **0.941**
and **every** held-out molecule has a training neighbour above 0.7 Tanimoto.
That even `topology` alone reaches R² 0.619 within-series confirms it — twelve
crude features cannot resolve substitution chemistry, but near-identical
analogues share both topology and label.

What the result does establish: the model beats a series-mean lookup
(0.513 against 0.845), so it resolves *some* substitution effect among close
analogues. This is the lead-optimisation regime — ranking variants within a
known series — and it is genuinely useful. It is not evidence that the model
can score a peptide from a series it has not seen.

---

## 6. Deployment gate: FAIL

Thresholds set from Stage A's measurements — label noise floor MAE 0.245,
mean-predictor baseline 1.343 — not from convention.

| Criterion | Required | Actual | Met |
|---|---|---|---|
| cross-series MAE | ≤ 0.70 | 0.870 | **no** |
| cross-series R² | ≥ 0.30 | 0.276 | **no** |
| within-series beats series-mean lookup | < 0.845 | 0.513 | yes |

**The peptide affinity model remains undeployed.** It must not publish numeric
affinity estimates, and must not propagate them into pharmacokinetic models or
knowledge-graph edges. The existing behaviour is unchanged: measurement
outranks prediction, and PT-141, melanotan I and melanotan II already have
measured values in the registry.

Two criteria are close to met rather than far from it — R² 0.276 against 0.30,
MAE 0.870 against 0.70 — which is a material change from Stage A's R² 0.123.
The gap is no longer a gulf, but a gate that is nearly met is not met.

---

## 7. What Stage C should and should not do

Supported by measurement:

- **Keep the fragment-fingerprint positional encoding.** It is the only
  representation that improves cross-series generalisation, and the mechanism
  is understood.
- **Drop the explicit topology block** from cross-series models. Retain it only
  for an explicitly within-series lead-optimisation mode, if one is built.
- **Report both evaluations always.** This is the recommendation that
  survived replication across the family: five of five conclusions reverse on
  MC4R and between one and three of five elsewhere, so a single number can
  select the wrong architecture. Do **not** carry a specific contrast's
  direction to another receptor — the ranking does not transfer.
- **Target the remaining gap with learnability, not representation.** The
  ceiling for the current encoding is MAE 0.186 and the model sits at 0.870, so
  there is a large gap that is not representational. Stage A's finding that the
  effective sample size is ~30 series points at the likely cause.
- **A within-series deployment mode is defensible before a general one.** It
  passes its own criterion, its applicability domain is checkable at prediction
  time (nearest-neighbour similarity above ~0.85 to a training series), and it
  matches a real use case. It would still publish no numbers outside that
  domain.

Not supported:

- Frozen protein language model embeddings are **not** ruled out by vocabulary
  collapse, contrary to Stage A. Their ceiling (0.382) is better than current
  performance (0.870). They remain untested here for want of PyTorch, and that is a
  gap rather than a conclusion.
- Adding more small-molecule data. Stage A measured non-peptide-only training
  at R² −0.791.

---

## Known limitations

- **Stereochemistry is still unresolved.** The tracer does not distinguish D-
  from L-residues, so `positional_fragment_fp` encodes D-Phe⁷ and L-Phe⁷
  identically. In a melanocortin dataset this is a material omission: D-amino
  acid substitution is a central design axis, and some of the unexplained
  within-series variance is certainly stereochemical. This is the single most
  likely source of cheap improvement and remains unaddressed.
- No protein language model was run; §2 bounds them rather than measuring them.
- Head-to-tail cyclic peptides (58 of 630) have their rotation canonicalised
  lexicographically. This makes position indices comparable but the choice of
  origin is still arbitrary with respect to biology.
- The cross-series and within-series splits use one seed (0), and no
  seed-variability study was run. The §1 reversal is large relative to the
  contrasts it reverses, but the per-receptor replication shows the effect is
  receptor-dependent, so single-seed single-receptor results of this kind
  should be treated as provisional until repeated.
- `positional_fragment_fp + topology` with ridge attains the highest
  cross-series R² (+0.272) while having worse MAE (0.905) than the selected
  model. Selection was on MAE; a different criterion would have chosen
  differently, and the difference is within what one seed can resolve.
