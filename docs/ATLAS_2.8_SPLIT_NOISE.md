# ATLAS 2.8 — The noise band under every number we published

**Code:** `atlas/ml/split_noise.py` · **Data:** `data/ml/split_noise.json`
**Reproduce:** `python3 -m atlas.ml.split_noise`
**Tests:** 21 new (907 total)

Every cross-series comparison in this project was drawn with **one** split,
seed 0. Six representation extensions were reported against it, each with a
number. This measures what those numbers were worth.

**The conclusion survives. Most of the individual numbers do not.**

---

## 1. The split moves the answer by more than any extension did

Twelve seeds, random forest, reference representation `positional_fragment_fp`:

| | cross-series MAE |
|---|---|
| mean across 12 splits | **0.942** |
| standard deviation | 0.087 |
| range | 0.819 – 1.157 |
| **seed 0, the published number** | **0.894** |

The spread from the choice of split alone is **0.338 log units**. The largest
effect any extension was credited with is topology's 0.274. Seed 0 is a
favourable draw for the reference arm — 0.048 below the twelve-seed mean,
which flatters every contrast measured against it.

## 2. The split is not merely noisy — it is structurally unstable

This was not previously visible, and it is the more serious finding.
`_cross_series_split` assigns **whole clusters** until it reaches a 25% test
target. The largest Butina cluster holds **53%** of the peptide set. A seed
that draws it first overshoots on the first iteration:

| | test fraction | training molecules |
|---|---|---|
| seeds 0,1,2,5,6,9,10 | 25–30% | 439–472 |
| seeds 3,4,7,8,11 | 56–71% | 184–280 |

Five of twelve seeds train on fewer than 300 molecules and test on most of the
set. Those are a different experiment, not the evaluation the deployment gate
is defined on. **Nothing here changes the splitter** — doing so would silently
move every published number — but the realised fraction is now recorded, and
contrasts are reported on both strata.

## 3. Paired contrasts, and why stratifying matters

A contrast is measured as `mae_extension(s) − mae_reference(s)` on the **same**
split s, then summarised across splits. Pairing removes the split difficulty
both arms share. Positive = the extension is worse.

| Extension | seed 0 (published) | all 12 seeds | **on-target 7 seeds** | helped on |
|---|---|---|---|---|
| topology | **+0.274** | +0.054 ± 0.125 | **+0.117 ± 0.132** | 2 of 7 |
| esm2_embedding | +0.199 | +0.067 ± 0.100 | **+0.101 ± 0.106** | 2 of 7 |
| local_context | +0.070 | +0.018 ± 0.063 | **+0.048 ± 0.063** | 2 of 7 |
| assay_context | +0.052 | +0.017 ± 0.037 | **+0.035 ± 0.038** | 2 of 7 |
| stereo | +0.018 | +0.022 ± 0.037 | **+0.035 ± 0.045** | 1 of 7 |
| residue_chemistry | +0.015 | +0.003 ± 0.088 | **+0.027 ± 0.109** | 3 of 7 |

**Not one of the six clears twice its own spread.** Every extension this
project reported on had a single-seed difference that the split alone could
have produced.

Topology is the sharpest case. Stage B reported it as a dramatic failure —
cross-series MAE 0.894 → 1.168 — and drew a mechanistic conclusion from the
size of that gap. The paired effect is +0.117 ± 0.132, with topology *helping*
on 2 of 7 comparable splits. **Seed 0 overstated it by 2.3×.** The direction
holds; the magnitude was never measured.

### The two strata disagree, and the disagreement is informative

Over all twelve seeds the extensions look nearly harmless: topology helps on
7 of 12, residue chemistry on 8 of 12. Restricted to splits that met their
target, all six have a *larger* adverse effect. The reason is mechanical — a
seed holding out 71% of the set leaves ~200 training molecules, and at that
size every arm collapses toward the baseline together, compressing the
differences between them. Pooling degenerate splits in **understates** the
contrast rather than adding evidence, so the on-target subset is primary.

## 4. What this does and does not change

**Unchanged — and now better supported:**

- **0 of 6 extensions improve cross-series generalisation** on either
  stratum. Reported as failures on one split; still failures on twelve. That
  is the stronger statement, and it is the claim the project's strategy rests
  on.
- The deployment gate still fails. Nothing here touches 0.870 / 0.276.
- The within-series ranking gate is unaffected: it already used a clustered
  bootstrap over 150 molecule groups, which is why it was the one result with
  an honest interval.

**Changed:**

- **Any sentence quoting one of these differences as a quantity was quoting
  noise.** The Stage B topology finding is the clearest example and its
  wording has been corrected in `docs/ATLAS_2.4_STAGE_B_FINDINGS.md`.
- A single-seed cross-series MAE is not a measurement of a representation. It
  is one draw from a distribution 0.338 log units wide.
- Future comparisons should report the paired statistic over the on-target
  seeds, not a single number.

## 5. Limitations

- Twelve seeds reshuffle **one fixed set of 630 molecules**. They are not
  independent samples of peptide chemistry, so the spread is a lower bound on
  real-world variation and no confidence interval is quoted. The "twice its
  own spread" test is a screen, not a p-value.
- The sweep covers the **cross-series** split only. Within-series numbers
  have not been given the same treatment, and the ranking gate's bootstrap is
  a different statistic over a different object.
- Only the random forest is swept. Ridge results are unexamined, though the
  benchmark already restricts contrasts to the forest for a separate reason.
- Seven on-target seeds is few. The stratification is justified mechanically
  rather than statistically.
