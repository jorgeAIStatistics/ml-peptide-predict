# ATLAS 2.7 — Should the target scope expand? A feasibility census

**Study:** 22 agents, 10 target families, every count independently re-derived
**Verdict:** no-go · **Data:** external (ChEMBL API), see provenance warning below
**Reproduce:** the repo-side numbers only — see §0

---

## 0. Provenance, before anything else

This document is **not** like the other findings docs in `docs/`. Those report
numbers a committed script regenerates. This one mixes two kinds of claim and
the distinction governs how much weight each can carry:

- **Repo-side numbers** (learning-curve slope, pooling results, LORO, gate
  figures, cluster counts) come from committed artefacts in `data/ml/` and are
  reproducible with the commands those docs give. They were re-verified against
  the files before this document was written.
- **ChEMBL census numbers** (every per-family molecule and cluster count) come
  from live API queries run by 20 agents on 2026-10-24. **No committed script
  regenerates them.** They were each re-derived by a second, independent agent,
  and §2 separates the ones that reproduced from the ones that did not. Treat
  an unreproduced count as unestablished, not as a smaller number.

Three further cautions that belong at the top rather than the bottom:

1. The census used a **looser peptide filter** than ATLAS's own
   (`MW >= 500` against the repo's `PEPTIDE_MIN_MW = 600` plus backbone
   tracing). Where both were run the looser ruler overcounted by **16%**.
   ATLAS's actual tracer was never run on any new family.
2. **No new family has a leak-free study-group count.** MC4R's own collapses
   from 39 documents to 17 groups. Cluster counts below are therefore a
   ceiling on series supply, plausibly halved.
3. The calibration figure given to the census agents was **wrong**, and seven
   of ten caught it: `data/ml/mc4r_pki.json` records `raw_activities: 2746`
   and `unique_molecules: 1981`, so 1,981 is the molecule count and 630 is the
   peptide subset of it — not "1,981 records over 630 peptides". The error was
   in the study brief, not in the repo, and did not propagate into any count.

---

## 1. HEADLINE

**No-go.** Even granting every peptide the census found, and treating each one as if it were an MC4R peptide, the repo's own learning-curve slope (−0.0773 MAE per e-fold, fit R² 0.938, `docs/ATLAS_2.4_STAGE_A_FINDINGS.md:212`) projects cross-series MAE **0.707 against the 0.70 requirement** — and the only existing measurement of what pooling other receptors actually does to MC4R is **−0.172 log units** (`data/ml/multitask_benchmark.json`, `per_receptor/MC4R/pooled_gain_mae = −0.17195`), which is larger than the entire **0.163** projected gain.

---

## 2. WHAT THE CENSUS FOUND

Ten families censused, each independently re-counted by a second agent. I verified the repo-side numbers in this memo directly (`atlas/ml/features.py`, `data/ml/multitask_benchmark.json`, `data/ml/panel_replication.json`, `data/ml/ranking_gate.json`, `docs/ATLAS_2.4_STAGE_A_FINDINGS.md`, `docs/ATLAS_2.5_POCKET_ENCODER.md`, `docs/ATLAS_2.6_PROTEIN_LANGUAGE_MODEL.md`). I did **not** re-run any ChEMBL query; everything database-side is the census's word plus its verifier's.

### 2a. Counts that reproduced exactly under independent re-derivation

| Family | Headline peptide-like (MW≥500 proxy) | Comparable-to-MC4R figure | Butina clusters @0.4 (largest share) | One SAR campaign? |
|---|---|---|---|---|
| Opioid | **962** | 804 (Ki/Kd only) | 123 (11%) | Many — 212 documents, 416 assays |
| Somatostatin | **614** | not recomputed Ki-only | 48 (50.3%) | Many — 56 docs; the 50% cluster itself spans 43 docs over 22 years |
| NPY (Y1/Y2/Y4/Y5) | **467** | 433 (corrected floor) | 34 (18.6%) | Many — 46 docs, 1992–2024 |
| Vasopressin/oxytocin | **320** | 231 (Ki only) | 43 (35%) | Many — 45 docs, 1979–2024; V1b alone *is* one campaign (51 mols, 67% in one cluster) |
| Angiotensin/bradykinin | **274** | 163 (strict backbone) | 19 (29%) | Many docs, but only **two** parent backbones (DRVYIHPF, RPPGFSPFR) |
| Incretin | **212** | 212 | 7 (90%) | **One backbone family.** Its `one_sar_campaign: true` flag is literally false (20 docs); the defensible statement is "one backbone family, 7–31 series depending on cutoff" |
| **Subtotal** | **2,849** | | **274 clusters** | |

### 2b. Counts that did NOT reproduce — treat as unestablished

| Family | Headline count | What the verifier found | Clusters (largest) |
|---|---|---|---|
| Tachykinin/neurotensin | 727 | All 16 headline counts matched **exactly**; failure is in secondary claims — the "no molecule lost to censoring" caveat is withdrawn (517 molecules, 74 peptide-like, are censored-only) and NK3, not NK1, is the target that collapses under Ki-only (113→15). Ki-only 465; strict 370; both 209 | 95 (20.9%) |
| Chemokine | 423 | **The weakest set.** The "strict backbone = 358" cross-check is nested inside the loose proxy, not independent, so the honest range is **350–410**, not 358–423. "~59% one lineage" is ~46%. Its own `one_sar_campaign: true` flag contradicts its own prose and its verifier. Its MC4R comparator did not reproduce | 20 strict (41%) |
| Ghrelin/motilin/calcitonin | 550 | Every family-specific number matched exactly; the MC4R baseline was never re-derived and one MW is misstated. Peptide count is **definition-dependent across 118–694**; CALCRL is ~62 real peptides, not 274 | 56 (25%) |
| Non-GPCR control | 985 | Per-target census exact; four secondary figures wrong (6.1%→6.5%, 193 not 221 docs, branch split 388/597 not 253/732) and, decisively, a **ruler mismatch** — see below | 221 (3.6%) |
| **Subtotal** | **2,685** | | **392 clusters** |

### 2c. Totals and one systematic correction

- **All ten families: 5,534 peptide-like molecules.** Peptide-GPCR families only (excluding the non-GPCR control): **4,549**. From families whose counts reproduced: **2,849**.
- **Every one of these numbers is measured on a looser ruler than ATLAS's own.** `atlas/ml/features.py` sets `PEPTIDE_MIN_MW = 600.0` and `PEPTIDE_MIN_AMIDES = 4`; the census used MW≥500 throughout. The one family where both rulers were run lost **16%** (985 → 823 molecules, 221 → 186 clusters). So the ten-family total is high by roughly that order against the repo's own definition. Named false positives sit inside the headline counts: ritonavir and saquinavir in four separate families, 211 gepant small molecules inside CALCRL's 274, edoxaban in the control set.
- Nobody ran ATLAS's actual backbone tracer on any new family.

---

## 3. DOES IT RELIEVE THE BOTTLENECK

**On the counting question: yes, and by a lot. On the question that matters: no.**

The brief assumed this estimate would rest on qualitative diversity evidence. It does not, and that is worth correcting: RDKit was installed locally, so **10 of 10 census agents ran the actual ATLAS clustering recipe** (Morgan r=2/2048, Taylor-Butina at 0.4 Tanimoto distance), and 9 of 10 calibrated it against `data/ml/mc4r_pki.json` first, reproducing MC4R's published figures (630 → 636 peptides, 30 → 34 clusters, 53% → 52.7% largest share). The series estimate is therefore measured on the same ruler as the comparator.

**Series supply, against the current 30** (`data/ml/panel_replication.json`, MC4R `n_series = 30`, `effective_sample_size_ratio = 0.0476`):

- As measured, peptide-GPCR families: **445 clusters**
- Conservative reading (tachykinin under the tight peptide filter = 44; chemokine by document reading = ~7 programmes; ghrelin by document count = 19 campaigns of ≥10): **~344**
- So roughly **344–445 new series, an 11×–15× increase** over 30. Chemotype novelty is real and measured: 93% of opioid peptides sit beyond 0.4 Tanimoto distance from *every* MC4R molecule (median nearest-neighbour 0.447); pooling somatostatin with MC4R gives 82 clusters of which exactly **1** is mixed; the non-GPCR control's 985 share only 3 clusters with MC4R's 636.

**This is the honest pro-expansion finding, and it directly contradicts the within-family precedent.** Four melanocortin receptors added 10 peptides (630→640) and *lost* a series (30→29, `data/ml/multitask_benchmark.json`: `n_peptides 640`, `n_clusters 29`). Other families do not behave that way — they bring their own chemistry.

**Uncertainty in the 344–445, stated honestly:**

1. **No new family has a leak-free study-group count.** This is the real limit on the estimate. The number that governs a cross-series split is leak-free study groups, not Butina clusters, and MC4R's own ratio is **39 documents → 17 groups, largest holding 43.9% of rows** (`panel_replication.json/study_grouping`). Nothing equivalent was computed for any of the ten. Treat 344–445 as a **ceiling**, plausibly halved.
2. **Peptide definition is unstable.** The ghrelin family yields 118 / 550 / 694 molecules over the *same* molecules depending on filter. Ruler change costs 16% of molecules and 16% of clusters where measured.
3. **Threshold sensitivity.** Somatostatin's dominant-cluster share swings 26.4% (cutoff 0.3) → 72.6% (0.5).
4. **Fingerprint saturation** understates series for large-peptide families (NPY median MW 1,814; incretin 3,395; vs MC4R 1,068) — this one runs *in favour* of expansion.
5. **Cluster ≠ programme, in both directions.** Documents corroborate clustering in opioid (212), tachykinin (100) and ghrelin (57, 19 with ≥10); they contradict it in incretin (7 clusters vs 20 documents) and in ghrelin's top cluster (116 of 138 molecules from a single patent).
6. **Four of ten families' counts are unestablished** (§2b).

**Now the arithmetic that ends it.** Using the repo's fitted slope as a direction, as the doc instructs:

- Gap to close: 0.8698 − 0.70 = **0.170 log units** (`panel_replication.json` MC4R `cross_series_mae = 0.86976`, `cross_series_r2 = 0.27605`).
- Required increase in training peptides: exp(0.170 / 0.0773) = **9.0×**.
- Available from the census: 630 → 5,179 peptides = **8.2×** → 0.0773 × ln(8.2) = **0.163** → projected MAE **0.707**. Fails.
- Verified counts only (630 → 3,479 = 5.5×): **0.738**. Fails.
- Under ATLAS's own MW≥600 ruler (−16%): **0.719**. Fails.
- Everything including the non-GPCR control (630 → 6,164 = 9.8×): **0.694** — passes by 0.006, and only by pooling thrombin inhibitors, cathepsin warheads and MHC 9-mer epitopes into an MC4R pKi regression. That family's own census calls this training across three different physical mechanisms under one label. It is not an option.
- For reference, the curve's own anchored projection: MAE 0.70 at ~158,000 training peptides. The census supplies 4,549. **Thirty-five-fold short.**

The conclusion is robust to every reading of the count. And all of it assumes an out-of-family peptide counts as an MC4R peptide — which brings us to §4.

---

## 4. THE TRANSFER PROBLEM

**Bluntly: the census is counting the wrong thing.** It measured whether the molecules exist. They do. The binding constraint is whether affinity information crosses a target boundary, and that has been measured four separate ways in this repo. All four are negative, at distances far smaller than any of these families.

1. **Family pooling makes the richest receptor worse.** On identical test rows with peptide clusters held out globally, pooling four melanocortin receptors moved MC4R from MAE 0.9022 to **1.0741** — `pooled_gain_mae = −0.1720`, R² 0.0163 → −0.2798 (verified in `data/ml/multitask_benchmark.json`). Mean gain across four receptors: **−0.0323**; helps 2, hurts 2. The repo's reading: pooling redistributes rather than adds.
2. **Three receptors cannot predict the fourth, half the time.** Leave-one-receptor-out fails against the held-out receptor's own mean baseline on **2 of 4** — MC1R (1.418 vs baseline 1.223) and MC4R itself (1.187 vs 1.100). And LORO is *easier* than cross-family transfer, not harder: 424 of 640 panel peptides (66.2%) are profiled on more than one receptor, so the chemistry was already seen on sister receptors. The repo's verdatim finding: **"A target encoder is not established."**
3. **The similarity gradient points the wrong way.** Transfer passed where the held-out receptor's binding pocket most resembled a training receptor — MC3R at 25/31 consensus positions (+0.434) and MC5R at 24/31 (+0.359) — and failed at 21/31 (MC1R, −0.204) (`docs/ATLAS_2.5_POCKET_ENCODER.md`). MC2R, the nearest thing to an out-of-family melanocortin receptor, sits at **12/31** and contributes **zero peptides**, so it cannot even be tested. Opioid, somatostatin, tachykinin and chemokine receptors are all further from MC4R than MC1R, the one that already failed.
4. **Crossing a chemotype boundary at a *fixed* target already fails.** Training on 1,351 non-peptide MC4R molecules and testing on 141 peptides gives MAE 1.734, **R² −0.791** — worse than predicting the mean, at the same receptor.

Add the representation record: **six consecutive extensions, zero improvements, six regressions** (`docs/ATLAS_2.6_PROTEIN_LANGUAGE_MODEL.md`: topology 0.894→1.168, ESM-2 650M 0.894→1.093, local context →0.964, assay context →0.946). A structural pocket encoder built from all ten experimental MC4R complexes — 31 consensus positions, 17 contacting in all ten, 682 features — does not outperform crude whole-sequence composition.

If a model trained on three melanocortin receptors cannot beat the mean on the fourth, when two thirds of the molecules are shared, then adding receptors whose pockets share 12 of 31 positions or fewer is very unlikely to help. **Nothing in the census changes this, because the census never tested it.** The `new_peptide_family_and_new_target` regime — the exact regime a cross-family expansion would be in — is still referenced only by `tests/test_multitarget.py`. No benchmark calls it. No result file contains it.

Honest counterweight: the transfer evidence is also **weak** — one seed (0), four held-out receptors, four trials, MC5R's result resting on 31 test rows. The repo itself calls the pocket-similarity pattern "a pattern worth noting, not evidence." Weak-negative is not proof the expansion fails. But weak-negative plus six representation nulls plus a learning-curve shortfall is not a basis for spending.

---

## 5. WHAT AN HONEST EXPANDED MODEL WOULD PREDICT

It would answer: *given an arbitrary peptide and an arbitrary peptide-binding GPCR from the training panel, what is pKi?* That is a **selectivity and triage** question. It is not the MC4R potency question the product was built around, and answering it well would not license a published MC4R affinity number.

What the current four-receptor version of exactly that model does: every panel arm has **negative R²** (−0.291, −0.086, −0.089, −0.092) — every arm worse than predicting the test set's own mean. The three competing receptor encodings land within **0.007 log units** of each other (0.9887 / 0.9922 / 0.9960), which a four-receptor split cannot distinguish even in principle. The apparent +0.136 "receptor one-hot" gain is the gain from telling a model which of four receptors a row belongs to, on a task that only exists because receptors were added.

There **is** one genuinely open question the expansion would answer, and it is the repo's own stated motive: *can a target encoder be established at all when N ≫ 4?* Counting receptors with ≥10 peptide-like molecules in the census — opioid 4, somatostatin 5, incretin 3, NPY 4, vasopressin/oxytocin 4, tachykinin/NTS 5, chemokine 2, angiotensin/bradykinin 4, ghrelin family 4 — the expansion takes the encoding problem from **4 points to ~39**. That is a real 10× in the dimension the repo says is limiting: *"The limiting factor is the number of targets, not the quality of their representation."*

But that is a different claim from "expansion fixes cross-series affinity error," and §3 and §4 measure the second one as false. A defensible target encoder would tell you which receptors a peptide hits. It would not move MC4R's 0.870.

---

## 6. COST AND THE ALTERNATIVE

### Expansion cost — the census was the cheapest part

Each item below has a measured size; the calendar figure is my judgement, not a measurement.

- **Re-census all ten families under ATLAS's own ruler** (MW≥600 + backbone tracer). The census used MW≥500 plus amide proxies everywhere, with named false positives inside the headline counts.
- **Re-verify the four families whose counts did not reproduce**, and fix chemokine's nested strict filter and its contradicted campaign flag.
- **Harmonise endpoints to Ki** (`atlas/ml/dataset.py`, Cheng-Prusoff grounds). Measured cuts: opioid 962→804, vasopressin/oxytocin 320→231, tachykinin 727→465, somatostatin records 4,198→1,783 (−57.5%), incretin loses 1,351 GLP-1R functional EC50 records plus 621 more inside binding assays.
- **Build leak-free study groups per family.** Not computed for any. MC4R's own collapse is 39 documents → 17 groups, largest 43.9% of rows.
- **Hygiene filtering** the repo already applies to MC4R: 90 mutant records in tachykinin, 163 in chemokine, 26 in angiotensin/bradykinin; `potential_duplicate` on 772 chemokine, 313 somatostatin, 193 vasopressin/oxytocin records.
- **Target encodings for ~39 receptors.** The 31-position pocket consensus came from ten experimental MC4R complexes. Nothing comparable exists for the new receptors, so either alignment transfer — which already failed to beat crude composition — or new structure curation per receptor.
- The benchmark itself is the cheap part: `atlas/ml/multitask.py` and `atlas/ml/panel_benchmark.py` exist.

Weeks of data-layer work per family, before any model runs, to test a mechanism that four prior measurements say does not work.

### The alternative — measured, and already in the repo

- **Fix the single-seed / four-trial fragility first.** One seed (0) throughout; four held-out receptors = four trials. `atlas/ml/ranking_gate.py` already does clustered bootstrap over 150 groups, so the machinery exists. A 2-of-4 vs 3-of-4 pass rate on one seed is not a result either way. Nearly free, and a precondition for anything else here.
- **Affinity bands.** Already measured, already discriminating: within-series pairwise accuracy by true gap is 61.0% at 0–0.5 log units (n=1,714), **83.8%** at 0.5–1 (1,296), **93.7%** at 1–2 (1,710), **98.6%** above 2 (1,160) (`data/ml/ranking_gate.json`). A banded claim is a reformulation, not a data-acquisition programme.
- **Applicability-domain-restricted answering.** Also already built: 2,730 of 5,880 within-series pairs served, 4,312 of 8,496 cross-series.
- **Extend the within-series ranking claim to MC1R/MC3R/MC5R.** The decisive asymmetry is already measured in `panel_replication.json`: within-series MAE is uniformly good across the family — MC1R 0.530, MC3R 0.518, MC4R 0.520, MC5R 0.622 — while cross-series uniformly fails the gate on all four (best anywhere: MC3R 0.827). The product's one passing gate is within-series ranking at **93.0% [90.6%, 95.1%]** on 5,880 held-out pairs, 92.0% over all pairs above 0.5 log units, Spearman 0.679. Cross-series ranking fails at 68.5% [66.1%, 70.8%], Spearman 0.157.
- **This needs no new data at all.** The repo's single positive result came from changing the question, not the corpus. Reframing beat all six representation extensions and both data extensions tried so far.

---

## 7. RECOMMENDATION

**No-go on cross-family scope expansion as a route to the affinity gate. Reformulate instead: affinity bands plus applicability-domain gating, and extend the within-series ranking claim to MC1R, MC3R and MC5R, where the within-series MAEs are already measured and already comparable to MC4R's.**

Do not treat that as a belief. Gate it on one measurement that costs less than this memo's argument is worth.

**First concrete step.** Run the seed sweep and molecule-level bootstrap on the existing MC4R cross-series number to establish the noise band (the panel has never had one). Then rerun `atlas/ml/multitask.py`'s global-cluster-holdout design with **one** out-of-family receptor set added to the pooled arm, scored on the same 184 MC4R test rows. Use opioid: 804 Ki/Kd peptide-like molecules verified exactly by two independent agents, 123 Butina clusters with the largest at 11%, zero molecule-ID overlap with all 1,981 MC4R molecules, 93% beyond 0.4 Tanimoto distance from every MC4R molecule. That is the best case the census found. Cost: one dataset build plus one existing benchmark.

**What would falsify the no-go.** Pooled MC4R cross-series MAE beating **0.870** by more than the measured seed band — call it 0.80 or better. Then the within-family pooling result does not generalise outward, scope expansion deserves funding, and this memo is wrong.

**What would close the question.** Pooled MC4R MAE at or above 0.870, which is what the melanocortin pooling result (−0.172 at the closest possible receptor distance) predicts. Then the ~30-series ceiling is confirmed as a property of the prediction target rather than of the corpus, and no amount of other-family data is the route to a published MC4R affinity number.

**One thing worth funding separately, and honestly labelled.** If the target-encoder question is worth answering on its own merits — can an encoder be established at N≫4, taking it from 4 points to ~39 — that is a legitimate research programme with a pre-registered mean baseline and the untouched `new_peptide_family_and_new_target` regime. Fund it as *that*, scoped and reported per held-out receptor. Do not fund it as a fix for cross-series affinity error.

---

## 8. WHAT THIS STUDY DID NOT COVER

**Families not censused.** Ten were chosen. **No numbers exist for any other peptide-GPCR family** — not a count, not a cluster, not an availability claim. Unsurveyed, with nothing measured either way: GnRH, CRF/urocortin, secretin/VIP/PACAP, gastrin/CCK, orexin, kisspeptin, galanin, apelin, relaxin, urotensin, neuromedin, prokineticin, formyl-peptide and protease-activated receptors. Anyone quoting a size for those is guessing.

**Counts not reproduced.** Four of ten: tachykinin/neurotensin, chemokine, ghrelin/motilin/calcitonin, non-GPCR control (§2b). For three, the primary per-target counts matched exactly and the failure was in secondary claims. Chemokine is the substantive one and its peptide count should travel as 350–410. Separately, two families' structured `one_sar_campaign` flags are unreliable: incretin's `true` is literally false (20 documents), and chemokine's `true` is contradicted by its own prose and its verifier.

**What the API could not answer.** No clustering (all done locally with RDKit). Censoring is structurally invisible under the protocol: ChEMBL computes `pchembl_value` only for `standard_relation = '='`, so a naive census reports zero censored records; every family had to re-query, and the costs are real (945 records in tachykinin with 517 molecules censored-only, 4,471 in the non-GPCR control, 1,854 in somatostatin, 842 in vasopressin/oxytocin). `molecule_type` is unusable as a peptide flag (DAMGO types "Small molecule"; 97 of 358 in chemokine; 121 of 2,384 in the ghrelin family; all GIPR and GLP-2R peptides type "Unknown"). All 423 chemokine peptides carry no `pref_name` and no synonyms at all, so every lead attribution there is title-level inference. Motixafortide, balixafortide and CTCE-9908 exist in ChEMBL with zero activity records.

**Excluded by protocol, and the cost is uneven.** Functional (EC50) assays were excluded, which hides most incretin pharmacology (GLP-1R: 1,351 functional EC50 records against 290 qualifying binding records) and nearly all PTH (PTH1R 42 records / 13 peptides; PTH2R a verified zero). Integrin peptide data — 234 peptide-like molecules, 54 clusters, cilengitide and eptifibatide — sits on a `PROTEIN COMPLEX` target excluded by the single-protein rule; the amylin and CGRP complexes would add 115 molecules, 40 peptide-like. Non-human orthologues were never queried for any family.

**Not done at all.** No family was benchmarked. No transfer was measured. The `new_peptide_family_and_new_target` regime has still never been run. No family has a leak-free study-group count. ATLAS's actual backbone tracer was never run on any new family. The prior art it all rests on is one seed and four trials.

**Process risk.** Shared-scratchpad cross-contamination was detected in at least four census sessions — neurotensin data surfacing in the incretin run, mu-opioid data overwriting the angiotensin run's input, the tachykinin analysis file overwritten mid-run, a chemokine script shadowing the ghrelin run's write. Each session reports re-running in isolation and I have no way to confirm their final files after the fact. That is part of why the reproduced / unreproduced split in §2 matters more than the totals do.

**Finally, correct the brief.** Seven of ten census agents independently caught the same error: `data/ml/mc4r_pki.json` reports `raw_activities = 2746` and `unique_molecules = 1981`, so "1,981 records over 630 peptides" conflates records with molecules — 1,981 is the molecule count and 630 is the peptide subset of it. The calibration figure that every cross-family comparison is measured against was itself garbled.