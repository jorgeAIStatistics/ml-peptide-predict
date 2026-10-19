# ATLAS 2.5 — A binding pocket from experimental structures

**Code:** `atlas/ml/receptor_pocket.py` · **Data:** `data/ml/receptor_pocket.json`
**Reproduce:** `python3 -m atlas.ml.receptor_pocket` then
`python3 -m atlas.ml.panel_benchmark`
**Tests:** 24 new (651 total)

Stage 3 encoded each receptor by the amino-acid composition of its whole
sequence and could not establish a target encoder. That encoding is a poor
description of a binding site: an MC4R is ~330 residues, a peptide touches a
few dozen, and the rest — transmembrane packing, intracellular loops, the
G-protein interface — contributes most of a composition vector while being
irrelevant to affinity. This replaces it with a pocket read off observed
geometry.

**The encoder works, is better justified, and does not improve prediction.**
That negative result is the most useful thing here, because it localises the
bottleneck.

---

## 1. The pocket, derived from ten complexes

All ten experimental MC4R–peptide complexes were used. Receptor residues with
any heavy atom within **4.5 Å** of any peptide heavy atom are contacts; a
position joins the consensus pocket if it contacts in **≥3** structures.

| PDB | Ligand | Method | Res (Å) | Peptide res | Contacts | Letter mismatches |
|---|---|---|---|---|---|---|
| 7PIU | setmelanotide | cryo-EM | 2.58 | 8 | 25 | 0 |
| 6W25 | SHU9119 | X-ray | 2.75 | 9 | 29 | 12 |
| 7PIV | NDP-α-MSH | cryo-EM | 2.86 | 13 | 26 | 0 |
| 8WKY | S25 | X-ray | 2.90 | 10 | 27 | 12 |
| 7AUE | setmelanotide-type | cryo-EM | 2.97 | 9 | 23 | 0 |
| 7F53 | α-MSH | cryo-EM | 3.00 | 13 | 31 | 0 |
| 7F54 | afamelanotide | cryo-EM | 3.00 | 13 | 28 | 0 |
| 7F55 | bremelanotide | cryo-EM | 3.10 | 7 | 21 | 0 |
| 21LT | setmelanotide | cryo-EM | 3.11 | 8 | 25 | 0 |
| 8WKZ | S31 | X-ray | 3.30 | 10 | 28 | 12 |

**31 consensus positions.** Cutoff sensitivity: 25 positions at 4.0 Å, 31 at
4.5 Å, 36 at 5.0 Å — stable enough that the choice is not doing the work.

**17 of the 31 contact a peptide in all ten structures**: E100, I104, D122,
D126, I129, I185, S188, V193, I194, L197, F261, H264, L265, Y268, F284, N285,
L288. The full set with support counts, in P32245 numbering:

```
Q43(9)  F51(8)  N97(6)  E100(10) T101(7) V103(6) I104(10) T118(4) V119(3)
I121(3) D122(10) N123(7) D126(10) I129(10) C130(8) L133(9) F184(6) I185(10)
S188(10) V193(10) I194(10) L197(10) W258(3) F261(10) H264(10) L265(10)
Y268(10) M281(6) F284(10) N285(10) L288(10)
```

The highly supported set is chemically coherent for a melanocortin receptor —
an acidic pair (D122, D126) alongside an aromatic cluster (F261, H264, Y268,
F284) — but that coherence is an observation about these structures, not a
citation, and no literature pocket definition was used to derive or filter it.

### Three points of discipline

**Numbering was never trusted.** Author residue numbers agree with UniProt
positions in the eight cryo-EM entries but not in the three X-ray entries,
whose chains carry a glycogen-synthase fusion partner. Every position is
established by local alignment to P32245 and verified by checking the residue
letter: the cryo-EM entries map 100% of observed residues with every letter
matching; the fusion constructs map only their MC4R portion (276 of 467) with
**12 letter mismatches each** — the engineered crystallisation mutations.

**Identity comes from UniProt, geometry from the structure.** Since those 12
positions carry non-wild-type residues, contacts are *located* from
coordinates and then *named* from the reference sequence, with mismatches
counted per structure rather than silently absorbed.

**A contact is not an energetic contribution.** This is a geometric fact about
ten structures. A test asserts the caveat survives in the saved definition.

---

## 2. Transfer across the family

Pocket positions are carried onto the other receptors by global alignment of
P32245 to each UniProt sequence. A position with no aligned counterpart is
recorded as a gap, never guessed.

| Receptor | Whole-sequence identity | Pocket residues identical to MC4R | Gaps |
|---|---|---|---|
| MC3R | 59.8% | **25 / 31** | 0 |
| MC5R | 63.0% | **24 / 31** | 0 |
| MC1R | 49.4% | **21 / 31** | 1 |
| MC2R | 49.8% | **12 / 31** | 0 |

Pairwise pocket identity: MC3R–MC4R 81%, MC4R–MC5R 77%, MC3R–MC5R 71%,
MC1R–MC3R 68%, MC1R–MC4R 68%, MC1R–MC5R 65%.

MC2R is the clear outlier at 12 of 31 — consistent with its binding ACTH
rather than the α-MSH pharmacophore, and with its contributing no peptides at
all to the panel.

The encoder itself gives each consensus position the residue-chemistry
descriptor vector of whatever residue that receptor carries there (31 × 22 =
682 features), so two receptors differing at one pocket position differ in
exactly that block.

---

## 3. The benchmark: no improvement

Same panel, same splits, same downstream random forest as Stage 3.

### Cluster split — all receptors present on both sides

| Arm | MAE | R² |
|---|---|---|
| peptide + receptor one-hot | **0.989** | −0.086 |
| peptide + receptor sequence | 0.992 | −0.089 |
| peptide + receptor **pocket** | 0.996 | −0.092 |
| peptide only | 1.125 | −0.291 |
| mean baseline | 1.234 | −0.457 |

The three target encodings sit within **0.007 log units** of each other. This
split cannot distinguish them even in principle: four receptors give four
distinct vectors under any encoding, so they have identical capacity here.

### Leave-one-receptor-out — the only split that tests a target encoder

| Held out | Rows | Baseline | Peptide only | + one-hot | + sequence | + pocket |
|---|---|---|---|---|---|---|
| MC1R | 359 | 1.223 | 1.447 | 1.418 | 1.425 | 1.427 |
| MC3R | 269 | 1.171 | 0.771 | 0.697 | **0.694** | 0.738 |
| MC4R | 630 | 1.100 | 1.192 | 1.187 | 1.176 | 1.182 |
| MC5R | 96 | 0.973 | 0.619 | 0.669 | 0.615 | **0.614** |

The pocket encoder beats whole-sequence composition on **1 of 4** held-out
receptors (MC5R, by 0.001) and beats the mean baseline on **2 of 4** — the
same two the whole-sequence encoding already succeeded on. On MC1R and MC4R
every arm is worse than predicting the mean.

**The structurally principled encoder does not outperform the crude one.**

---

## 4. What the negative result tells us

It localises the bottleneck, which is worth more than a small win would have
been.

With four receptors carrying peptide data, **every** target encoding is four
points in some space, and leave-one-receptor-out asks a model to reach the
fourth from three. No amount of chemical detail in describing those points
overcomes having three of them. The limiting factor is **the number of
targets, not the quality of their representation.**

A supporting observation, offered as an observation and not a correlation:
transfer succeeds where the held-out receptor's pocket most resembles a
training receptor's.

| Held out | Nearest training pocket | Gain over baseline | Outcome |
|---|---|---|---|
| MC3R | MC4R at 25/31 | +0.434 | passes |
| MC5R | MC4R at 24/31 | +0.359 | passes |
| MC1R | MC3R at 21/31 | −0.204 | **fails** |
| MC4R | MC3R at 25/31 | −0.082 | **fails** |

Three of four fit — the least similar receptor fails, the two most similar
pass. **MC4R breaks the pattern**, and the obvious confound is that it carries
630 of 1,354 rows: holding it out removes 47% of the data and the most
chemically diverse portion. With four trials this is a pattern worth noting,
not evidence.

---

## 5. What the pocket is nonetheless good for

The encoder earns its place for reasons other than the benchmark.

- **Interpretability.** A SHAP attribution over `D122 estimated_charge` or
  `F261 aromatic_rings` is readable; one over fingerprint bit 1,473 is not.
  This is the same argument that kept the residue-chemistry engine after it
  too failed to improve prediction, and it is the input the brief's
  explanation engine needs.
- **Residue-level interface data for the Interface Laboratory.** Which
  receptor residues contact which ligand, with per-structure distances and
  provenance, is exactly what a structural viewer must show — and it is now
  derived rather than illustrative.
- **It is the right encoder to re-test** if more receptors become available.
  It should be preferred on interpretability and retested on performance, not
  adopted on performance it has not shown.

---

## 6. Bugs found and fixed

Two were caught by writing the tests, and both would have corrupted the
pocket silently.

1. **`Counter.update(dict)` adds values rather than counting keys.** Fed
   `{position: distance}`, it summed distances, so "support" reached 41 from
   ten structures and the `min_structures` filter admitted nearly anything
   contacting even once. The first pocket had 36 positions for the wrong
   reason. Now `update(contacts.keys())`, and a test asserts support never
   exceeds the number of structures.
2. **Biopython alignment indices are numpy int64**, which
   `json.dumps(default=float)` wrote as `43.0`. Residue positions were being
   saved as floats and every integer lookup into `support` failed with
   `KeyError: '43.0'`. All positions are now cast at source.

A third was caught during exploration: filtering the peptide chain by
standard-amino-acid membership dropped non-canonical residues, giving six of
setmelanotide's eight. Since the non-canonical residues are the chemically
interesting ones, the peptide chain now contributes every non-water residue
whatever it is called. A test pins the known peptide lengths.

---

## 7. Limitations

- **No experimental structure of MC1R, MC3R or MC5R with a peptide was used.**
  Their pockets are inferred by alignment, which assumes the homologous
  position lines the homologous pocket.
- **Ten entries are not ten independent observations.** Setmelanotide appears
  three times and afamelanotide twice; seven distinct ligands.
- **One conformational state.** All ten are agonist-bound active-state
  structures, so the pocket is the active-state pocket. Nothing here describes
  the inactive receptor.
- **No side-chain flexibility.** Contacts come from single deposited
  coordinate sets, with no rotamer or ensemble treatment.
- **The 12 mutated positions per fusion construct** are named from UniProt,
  but their *geometry* is still that of the mutant.
- **Contacts are undirected and untyped.** This module reports proximity, not
  hydrogen bonds, salt bridges or aromatic stacking. Typing them is the
  interaction-feature work the Stage 1 audit deferred.
- **The deployment gate is untouched and still fails.** Nothing here changes
  what the peptide model may publish.
