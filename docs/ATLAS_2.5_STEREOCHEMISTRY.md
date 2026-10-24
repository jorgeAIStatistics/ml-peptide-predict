# ATLAS 2.5 — Backbone configuration, and what it is worth

**Code:** `atlas/ml/stereochemistry.py` · **Data:** `data/ml/peptide_benchmark.json`
**Reproduce:** `python3 -m atlas.ml.peptide_benchmark`
**Tests:** 34 new (854 total)

Every representation ATLAS had built was stereo-blind. The tracer identified a
residue by the constitution of its side chain, the fragment fingerprint was
generated with `includeChirality=False`, and the whole-molecule Morgan
fingerprint likewise — so D-phenylalanine and L-phenylalanine produced the same
key, the same column and the same bits. Stage A, Stage B, the Stage 1 audit and
the Stage 3 panel each flagged this as the outstanding gap. This resolves it.

**The result is split, and the split is the point.** Configuration is now
assigned, it provably separates molecules the old representation merged, and
adding it to the model **does not improve generalisation** — the fifth
extension in a row that does not. What it does change is what the deployed
service is allowed to claim.

---

## 1. Assignment: not from CIP labels

The obvious implementation — read RDKit's R/S label, map it to D or L — is
wrong. L-amino acids are (S) at the alpha carbon **except cysteine**, which is
(R): sulfur outranks the carboxyl carbon, so the identical spatial arrangement
receives the opposite CIP label. A CIP mapping needs a per-residue exception
table, and for the 160 noncanonical side chains in this dataset there is no
table to consult.

What is used instead is the configuration *relative to the backbone*, which is
what D and L mean: the arrangement of (N, C=O, side chain, H) about the alpha
carbon, compared against L-alanine. One chiral substructure query states that
arrangement, and RDKit's matcher handles the permutation bookkeeping for
whatever atom order the depositor happened to write.

```
ALPHA_L_QUERY = [NX3][C@@H]([!#1])[CX3]=[OX1]
ALPHA_D_QUERY = [NX3][C@H]([!#1])[CX3]=[OX1]
```

`[!#1]` stands for any heavy side-chain atom, which is what makes the test
residue-independent: proline's ring carbon, cysteine's thiomethyl and a
naphthylalanine's benzylic carbon all satisfy it.

Five outcomes are reported, not two. A position with two alpha hydrogens is
`achiral` (glycine), one with none is `alpha_disubstituted` (Aib), and a
stereocentre the record leaves undefined is `unspecified` — **not** assumed to
be L. That last case covers 7 of 5409 alpha carbons; filling them in would be
inventing a structural fact.

Configurations are read from deposited structures. They are a property of the
record, not a measurement ATLAS performed.

### Validation

Nine reference standards in PubChem's canonical orientation, including the
three that a naive implementation gets wrong:

| Standard | Expected | Why it is in the battery |
|---|---|---|
| L-alanine / D-alanine | L / D | the calibration pair |
| L-cysteine / D-cysteine | L / D | CIP (R) where the others are (S) |
| L-threonine | L | a second stereocentre in the side chain |
| L-proline | L | side chain closes onto the backbone nitrogen |
| L-tryptophan, L-serine, L-aspartate | L | ordinary cases |
| glycine | achiral | no stereocentre to report |
| Aib | alpha_disubstituted | a stereocentre D/L does not describe |

A further test renumbers each standard's atoms five ways and requires the same
answer, since the property being bought by matching a chiral query rather than
computing a parity by hand is exactly that the depositor's atom ordering does
not reach the result.

On real peptides, α-MSH (`SYSMEHFRWGKPV`) comes back all-L with Gly¹⁰ achiral,
and NDP-α-MSH (`SYSXEHFRWGKPV`, Nle⁴ tracing as X) comes back all-L **except
position 7, which is D** — the substitution that defines the series this entire
dataset is built from.

---

## 2. Census: the gap was not small

630 peptides, 5409 traced alpha carbons.

| Configuration | Alpha carbons | Share |
|---|---|---|
| L | 4476 | 82.8% |
| **D** | **724** | **13.4%** |
| achiral (glycine-like) | 193 | 3.6% |
| alpha-disubstituted | 9 | 0.2% |
| unspecified | 7 | 0.1% |

**539 of 630 peptides (85.6%) carry at least one D residue.** Every one of them
was, until now, encoded identically to its all-L counterpart.

---

## 3. What stereo-blindness cost

### The representational ceiling

Grouping molecules by representation key and predicting each group's mean gives
a hard lower bound on any model using that key.

| Key | Distinct keys | Colliding | Oracle MAE |
|---|---|---|---|
| canonical 20-letter sequence | 205 | 77.8% | 0.382 |
| chemistry-aware positional | 449 | 40.6% | 0.186 |
| **+ backbone configuration** | **490** | **31.4%** | **0.128** |

**31 groups covering 134 molecules (21.3% of the set)** are merged by a
constitutional key and separated by a configuration-aware one. Inside those
groups the measured pKi range is a median **1.30** and a maximum **3.88** log
units.

### The deployed feature rows

Stronger than the grouping above: molecules with *identical* constitution,
found by comparing stereo-stripped canonical SMILES.

- **25 stereoisomer groups, 54 molecules (8.6% of the peptide set), 33 pairs.**
- **All 33 received byte-identical rows** of
  `positional_fragment_fp+morgan_fp`, the matrix the ranking model is fitted
  on. All 33 are separated by the stereo block.
- 17 of them differ by **≥0.5 log units** in measured pKi (median 1.47,
  maximum 3.60). Mean |ΔpKi| across all 33 is 0.945.

The widest pair, CHEMBL263878 against CHEMBL439128, is a 13-residue
disulfide-constrained peptide differing at a single position — D-Arg against
L-Arg — with measured pKi 5.00 and 8.60. A **4000-fold** affinity difference
that the model was structurally incapable of seeing.

---

## 4. Does it improve prediction? No.

Random forest, both evaluations, same splits as Stage B.

| Contrast | Cross-series | Within-series |
|---|---|---|
| `+stereo` vs `positional_fragment_fp` | 0.912 vs **0.894** (worse) | **0.608** vs 0.613 (−0.005) |
| stereo-aware one-hot vs blind one-hot | 1.168 vs **1.009** (worse) | 0.590 vs **0.589** (−) |
| `stereo` block alone | 1.097 (worse than the 1.189 baseline) | 0.765 |

Cross-series error gets slightly **worse**. The within-series gain of 0.005 log
units is far too small to claim. The deployment gate is unchanged: cross-series
MAE **0.870** against ≤0.70 and R² **0.276** against ≥0.30, still **FAIL**, with
the same best representation as before.

This is now **0 of 5**. Explicit topology, residue chemistry, local sequence
context, assay context and backbone stereochemistry have each been added to the
representation and none improves cross-series generalisation. The agreement
with Stage A's ~30 effective series and with the pocket encoder's null result is
hard to read any other way: on this dataset the representation is not the
binding constraint, and the fifth negative result is more informative than the
first.

**The conclusion is not that configuration is unimportant.** It is that the
error does not live where the representation is. A 3.60 log-unit difference the
model cannot see is a defect even when fixing it moves no metric — because the
metric was never measuring that pair.

---

## 5. What changed in the product

Since adding configuration to the features buys no accuracy, it is **not**
deployed into the model. What is deployed is a refusal.

The ranking service measured pair similarity on a chirality-free Morgan
fingerprint, which gives a stereoisomeric pair a Tanimoto of exactly **1.0**.
It read that as identity and told the user:

> *The two structures are the same molecule, so there is nothing to order.*

For CHEMBL263878 and CHEMBL439128 — measured pKi 5.00 and 8.60 — that statement
was false. Two different peptides were reported to the user as one.

Three changes:

1. Identity is now decided on canonical SMILES, not on a fingerprint proxy.
2. A pair with one constitution and different configurations is recognised as
   **stereoisomers**, and refused with that reason and the measured spread it
   is avoiding.
3. The refusal applies **only when a model estimate is involved**.
   Chirality-blindness matters when the model supplies a value; two molecules
   with measured affinities are compared as measurements, and that pair —
   including the 3.60 log-unit one — is still ranked.

The remedy text states that a stereochemistry-aware model is something ATLAS
*has measured and does not have*, rather than implying an available upgrade.
Claiming otherwise would promise an improvement section 4 shows was not
demonstrated.

No frontend change was needed: the console's existing abstention path renders
the new reason and its remedies, verified in the browser.

---

## 6. What remains unresolved

- **Side-chain internal stereochemistry.** Only the alpha carbon is assigned.
  Threonine and allo-threonine, isoleucine and allo-isoleucine still produce
  the same key: the side-chain fragment is rebuilt atom by atom without chiral
  tags, and the fragment fingerprint is generated with chirality off. Alpha
  carbons were fixed first because they carry the D/L axis the melanocortin
  series is designed on. **The side-chain case has since been measured —
  section 7 — and left in place.**
- **Configuration is not conformation.** Knowing a residue is D says nothing
  about the backbone geometry it induces. Nothing here is a structural
  prediction.
- **The affinity gate still fails**, unchanged by this work. The peptide model
  publishes no numeric affinity estimates.

---

## 7. Side-chain stereochemistry: measured, then left alone

**Code:** `atlas/ml/side_chain_stereochemistry.py` ·
**Reproduce:** `python3 -m atlas.ml.side_chain_stereochemistry` ·
**Tests:** 18 new (872 total)

Section 6 recorded the side-chain loss as unmeasured and said it should not be
assumed small. It is now measured, and on this dataset it is small: real at the
residue level, and it separates **no two molecules**. Neither mechanism was
changed.

### How the fragment is cut

Not by rebuilding it. The rebuild in `_side_chain_code` reorders a
stereocentre's neighbours, so a chiral tag copied across is wrong as often as
right, and correcting it is the hand-rolled parity arithmetic section 1 refuses
to do. The census instead cuts every bond leaving the traced side chain with
`Chem.FragmentOnBonds`, which substitutes a dummy for the severed partner and
keeps each stereocentre's neighbour count and ordering intact. RDKit assigns
the tags; the census only compares them, each fragment's stereo SMILES against
the same fragment stripped. Calibration on reference standards, with their CIP
labels:

| Standard | Fragment |
|---|---|
| L-threonine (2S,3R) | `*[C@@H](C)O` |
| allo-L-threonine (2S,3S) | `*[C@H](C)O` |
| **D-threonine (2R,3S)** | **`*[C@H](C)O`** |
| L-isoleucine (2S,3S) | `*[C@@H](C)CC` |
| allo-L-isoleucine (2S,3R) | `*[C@H](C)CC` |

The bolded row is the case a naive fix gets wrong. Inverting the alpha carbon
inverts the beta carbon with it, so D-threonine's fragment is allo-L-threonine's
fragment. A residue key built from the stereo-aware fragment *alone* would merge
them; the fragment together with the alpha-carbon assignment does not. Any fix
has to combine the two axes, not replace one with the other.

### Census: 630 peptides, 5298 traced residues

| | Residues |
|---|---|
| carrying a side-chain stereocentre | **156** (2.9%), in 99 molecules (15.7%) |
| carrying one the record leaves undefined | 19 |
| of the 156, expressible in a cut fragment | 115 |
| of the 156, **only** a stereocentre in the whole molecule | **41** |

Three further residues carry no stereocentre but do carry a side-chain double
bond's geometry, which the flat key discards on the same mechanism. The 41 are
bridged side chains, cut at both ends: `*CC(O)C*` has two branches
that are identical inside the fragment and distinct outside it. No
fragment-local residue key can carry that configuration, stereo-aware or not —
which is a limit on the proposed fix, not on the census, and is why the decisive
measurement below compares whole molecules.

Of 169 distinct flat fragments, **2** occur in more than one configuration:
threonine's `*C(C)O` and isoleucine's `*C(C)CC`. Taking the deployed
alpha-carbon assignment into account leaves **one** ambiguous
(configuration, fragment) pair:

| Residue | Configuration | Count |
|---|---|---|
| isoleucine | L (2S,3S) | 53 |
| isoleucine | D (2R,3R) | 2 |
| threonine | L (2S,3R) | 45 |
| threonine | D (2R,3R), allo | 2 |
| **threonine** | **L (2S,3S), allo** | **7** |

The two D rows cost nothing, for two different reasons. D-isoleucine is the
exact mirror image of L-isoleucine, so its beta centre is inverted *with* the
alpha carbon and the deployed key — `D` plus the flat fragment — already implies
it. The two D threonines are (2R,3R), which is D-*allo*-threonine rather than
D-threonine's (2R,3S); that key is unambiguous only because the set happens to
contain no D-threonine to contrast it with, which is a fact about the data and
not about the key. **7 residues of 5298 (0.13%)
carry a configuration the deployed key cannot imply** — threonine deposited as
(2S,3S) at position 9 of one 14-residue traced series (`ICHFRWRCTASIPP`), in 7
molecules spanning pKi 4.60 to 6.13. All 7 members of that traced series carry
it and the set holds no (2S,3R) counterpart, so the dataset offers no contrast
either way, and whether this is a deliberate allo-threonine or a depositor's
transcription of threonine cannot be settled from the records. The census counts
what the records say.

### What it costs: nothing measurable

| Test | Result |
|---|---|
| stereoisomeric pairs (one constitution, two structures) | 33, in 25 constitutional groups |
| separated by the alpha-carbon key | **33** |
| needing side-chain configuration | **0** |
| groups a flat fragment key merges and a stereo-aware one splits | **0** |
| representational ceiling, with and without | 490 keys / oracle MAE 0.128, unchanged |

The 25 groups and 33 pairs are the same ones section 3 found. Every one of them
is separated by the alpha-carbon assignment already deployed. Not one pair in
the set is a side-chain stereoisomer, so making the fragment rebuild and the
fingerprint generator configuration-aware would change no key, no bit and no
row — and therefore could not change a metric. The fix was not made.

This is not the sixth null result in the series; it is weaker than that. The
previous five were features that were built, fitted and measured not to help.
This one was measured not to *apply*: there is nothing in this dataset for the
feature to be right about. A single threonine/allo-threonine contrast pair would
change the decision, and `tests/test_side_chain_stereochemistry.py` fails if one
appears.

### What the census turned up on the way

Two pairs of molecules receive byte-identical `positional_fragment_fp+morgan_fp`
rows and are separated by *neither* stereochemical axis, because they differ
constitutionally in places the representation cannot see:

- **CHEMBL5169353 / CHEMBL5201799** (pKi 6.70, 6.58) — a 25-residue pair with
  Lys and Glu swapped between positions 14 and 16. The fragment fingerprint
  indexes 8 N-terminal and 3 C-terminal slots, so in a 25-mer both positions
  fall outside every slot, and a positional isomer is invisible to the
  whole-molecule fingerprint as well.
- **CHEMBL5192562 / CHEMBL5206293** (pKi 7.34, 7.33) — identical residues and
  identical configurations, different disulfide pairing: Cys5–Cys18/Cys7–Cys9
  against Cys5–Cys9/Cys7–Cys18.

Both differences are worth 0.12 and 0.01 log units respectively, so neither is
an error worth chasing here. They are recorded because they are where the
deployed representation's remaining blindness actually is — mid-chain
substitution in long peptides and bridge topology — and not where this
milestone went looking.
