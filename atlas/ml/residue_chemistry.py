"""ATLAS 2.5 Stage 2: a chemistry-aware residue descriptor engine.

The design follows from a Stage A measurement rather than from convention.
80.5% of the peptides in this dataset carry at least one residue outside the
canonical twenty, across 160 distinct side chains — mostly halogenated and
naphthyl phenylalanines. A lookup table of published amino-acid property
scales therefore describes a minority of the data: those scales exist for the
twenty canonical residues and have no row for 2,4-dichlorophenylalanine.

So every descriptor here is **computed from the side-chain fragment structure**
rather than looked up. That is what makes the engine cover non-canonical
residues at all, and it is the property the tests assert.

Each descriptor carries its provenance, because the brief requires that
measured, calculated, estimated and unavailable be distinguished rather than
blended:

* ``COMPUTED`` — derived deterministically from the molecular graph by RDKit.
  Reproducible, and a property of the structure as drawn.
* ``ESTIMATED`` — depends on an assumption beyond the graph, stated explicitly.
  Ionisation state is the main case: it needs a pH and reference pKa values,
  and is a modelling choice rather than a measurement.
* ``UNAVAILABLE`` — not derivable for this residue. Glycine has no side chain
  to describe; proline's is fused to its own backbone nitrogen.

No descriptor here is ``MEASURED``. Nothing in this module is an experimental
observation, and labelling a Crippen logP contribution as a measured
hydrophobicity would be exactly the conflation the platform exists to avoid.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

from atlas.ml.peptide_structure import CANONICAL_SIDE_CHAINS, PeptideAnalysis

RDLogger.DisableLog("rdApp.*")

__all__ = [
    "Provenance", "ResidueProperties", "describe_residue", "DESCRIPTOR_NAMES",
    "DESCRIPTOR_PROVENANCE", "residue_property_matrix", "positional_chemistry",
    "ASSUMED_PH", "REFERENCE_PKA", "ionised_fraction", "provenance_summary",
    "local_context",
]


class Provenance(str, enum.Enum):
    MEASURED = "measured"
    COMPUTED = "computed"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"


#: The pH every ionisation estimate in this module assumes. Binding data in
#: the dataset comes from assays whose pH is not recorded in ChEMBL's
#: aggregated fields, so this is an assumption applied uniformly, not a
#: condition read from the experiments.
ASSUMED_PH = 7.4

#: Reference pKa values for the ionisable groups detected below, used only to
#: decide whether a group is predominantly charged at ``ASSUMED_PH``. They are
#: conventional approximate values for side-chain groups in peptides, not
#: measurements on these molecules, and the resulting charge is reported as
#: ESTIMATED. A substituted analogue's true pKa will differ: a
#: 4-nitrophenol-containing residue is far more acidic than tyrosine, and this
#: table does not know that.
REFERENCE_PKA: dict[str, float] = {
    "carboxylic_acid": 4.0,
    "primary_aliphatic_amine": 10.5,
    "guanidine": 12.5,
    "imidazole": 6.0,
    "phenol": 10.0,
    "thiol": 8.3,
}

# --- ionisable group patterns ------------------------------------------------
# Matched on the capped side-chain fragment, so backbone termini never count.
_GROUPS: dict[str, Chem.Mol] = {
    "carboxylic_acid": Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]"),
    "primary_aliphatic_amine": Chem.MolFromSmarts(
        "[NX3;H2;!$(N[CX3]=[O,N,S]);!$(N=*);!$(Na)][CX4]"),
    "guanidine": Chem.MolFromSmarts("[NX3][CX3](=[NX2,NX3+])[NX3]"),
    "imidazole": Chem.MolFromSmarts("c1c[nH0,nH1]c[nH0,nH1]1"),
    "phenol": Chem.MolFromSmarts("[OX2H1]c"),
    "thiol": Chem.MolFromSmarts("[SX2H1]"),
}
_AROMATIC_RING = Chem.MolFromSmarts("a1aaaaa1")
_HALOGEN = Chem.MolFromSmarts("[F,Cl,Br,I]")


@dataclass
class ResidueProperties:
    """Computed properties of one residue's side chain."""

    key: str                      # one-letter code, or fragment SMILES
    is_canonical: bool
    has_side_chain: bool

    # size and shape
    heavy_atoms: float = 0.0
    side_chain_volume: float = 0.0        # Labute approximate surface area
    molar_refractivity: float = 0.0
    rotatable_bonds: float = 0.0
    branching_index: float = 0.0          # fraction of carbons with >2 C neighbours

    # polarity and hydrophobicity
    crippen_logp: float = 0.0
    tpsa: float = 0.0
    h_bond_donors: float = 0.0
    h_bond_acceptors: float = 0.0

    # aromaticity
    aromatic_rings: float = 0.0
    aromatic_atoms: float = 0.0

    # heteroatoms
    halogens: float = 0.0
    sulfur_atoms: float = 0.0
    nitrogen_atoms: float = 0.0
    oxygen_atoms: float = 0.0

    # ionisation, at ASSUMED_PH and under REFERENCE_PKA (ESTIMATED)
    formal_charge_graph: float = 0.0
    estimated_charge: float = 0.0
    is_acidic: float = 0.0
    is_basic: float = 0.0
    is_ionisable: float = 0.0

    # conformational notes
    backbone_constrained: float = 0.0     # proline-like: side chain closes on N
    glycine_like: float = 0.0             # no side chain at all

    provenance: dict[str, Provenance] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)

    def vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in DESCRIPTOR_NAMES], dtype=float)


#: Descriptor order used by every matrix this module builds.
DESCRIPTOR_NAMES: tuple[str, ...] = (
    "heavy_atoms", "side_chain_volume", "molar_refractivity", "rotatable_bonds",
    "branching_index", "crippen_logp", "tpsa", "h_bond_donors",
    "h_bond_acceptors", "aromatic_rings", "aromatic_atoms", "halogens",
    "sulfur_atoms", "nitrogen_atoms", "oxygen_atoms", "formal_charge_graph",
    "estimated_charge", "is_acidic", "is_basic", "is_ionisable",
    "backbone_constrained", "glycine_like",
)

#: Provenance per descriptor, as the brief requires.
DESCRIPTOR_PROVENANCE: dict[str, Provenance] = {
    **{n: Provenance.COMPUTED for n in DESCRIPTOR_NAMES},
    # These four depend on ASSUMED_PH and REFERENCE_PKA, not on the graph.
    "estimated_charge": Provenance.ESTIMATED,
    "is_acidic": Provenance.ESTIMATED,
    "is_basic": Provenance.ESTIMATED,
    "is_ionisable": Provenance.ESTIMATED,
}

_CANONICAL_KEYS = set(CANONICAL_SIDE_CHAINS) | {"G", "P"}


def _capped_fragment(key: str) -> Chem.Mol | None:
    """The side chain as an isolated molecule, capped with hydrogen.

    The fragment arrives with a dummy atom marking where it met the alpha
    carbon. Replacing that dummy with hydrogen rather than a methyl keeps the
    descriptor a property of the side chain alone: alanine's side chain becomes
    methane, which is the convention this module documents and tests.
    """
    if key in ("G", "P"):
        return None
    smi = key if key.startswith("*") else "*" + CANONICAL_SIDE_CHAINS.get(key, "")
    if smi == "*":
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(1)
    try:
        out = rw.GetMol()
        Chem.SanitizeMol(out)
        return Chem.RemoveHs(out)
    except Exception:
        return None


_ACIDIC_GROUPS = ("carboxylic_acid", "phenol", "thiol")


def ionised_fraction(pka: float, ph: float, *, acidic: bool) -> float:
    """Henderson-Hasselbalch fraction of a group bearing charge at ``ph``.

    An acid is charged when deprotonated, a base when protonated, so the two
    cases are mirror images:

        acid:  f = 1 / (1 + 10^(pKa - pH))
        base:  f = 1 / (1 + 10^(pH - pKa))

    This replaced a three-bucket rule that counted anything within one pH unit
    of its pKa as half charged. That rule put cysteine at -0.5 under an assumed
    pH of 7.4 and a thiol pKa of 8.3, where the correct figure is about -0.11;
    at 0.9 units from the pKa a group is not half ionised.
    """
    exponent = (pka - ph) if acidic else (ph - pka)
    # Guard the exponential against overflow for groups far from their pKa.
    if exponent > 12:
        return 0.0
    if exponent < -12:
        return 1.0
    return 1.0 / (1.0 + 10.0 ** exponent)


def _estimate_charge(
    mol: Chem.Mol, ph: float
) -> tuple[float, float, float, list[str]]:
    """Net side-chain charge at ``ph``, with its assumptions recorded.

    The charge is a sum of Henderson-Hasselbalch fractions over the ionisable
    groups found in the fragment. It is an estimate: the reference pKa values
    are conventional values for unsubstituted side chains, and substitution
    shifts them. It is labelled ESTIMATED for that reason.
    """
    acidic = basic = 0.0
    notes: list[str] = []
    for name, patt in _GROUPS.items():
        if patt is None:
            continue
        n = len(mol.GetSubstructMatches(patt))
        if not n:
            continue
        pka = REFERENCE_PKA[name]
        is_acid = name in _ACIDIC_GROUPS
        frac = ionised_fraction(pka, ph, acidic=is_acid)
        if is_acid:
            acidic += frac * n
        else:
            basic += frac * n
        if 0.02 < frac < 0.98:
            notes.append(
                f"{name} (reference pKa {pka}) is only {frac:.0%} ionised at pH "
                f"{ph}; its charge contribution is partial and sensitive to the "
                "assumed pH."
            )
    return basic - acidic, acidic, basic, notes


@lru_cache(maxsize=8192)
def describe_residue(key: str, ph: float = ASSUMED_PH) -> ResidueProperties:
    """Compute every descriptor for one residue key.

    ``key`` is either a canonical one-letter code or a side-chain fragment
    SMILES as produced by :mod:`atlas.ml.peptide_structure`. ``ph`` affects
    only the ionisation estimate and is a parameter rather than a constant so
    that condition sensitivity can be measured rather than assumed away.

    Note on the zeros: glycine and proline have no isolable side chain, so
    their structural descriptors are zero *and* their provenance is
    UNAVAILABLE. A model reading the raw vector cannot see provenance, so the
    ``glycine_like`` and ``backbone_constrained`` flags are included in the
    vector precisely so those zeros remain distinguishable from a real zero.
    """
    is_canon = key in _CANONICAL_KEYS
    props = ResidueProperties(
        key=key, is_canonical=is_canon, has_side_chain=key not in ("G",),
        provenance=dict(DESCRIPTOR_PROVENANCE),
    )

    if key == "G":
        props.glycine_like = 1.0
        props.has_side_chain = False
        for n in DESCRIPTOR_NAMES:
            if n != "glycine_like":
                props.provenance[n] = Provenance.UNAVAILABLE
        props.caveats.append(
            "Glycine has no side chain, so side-chain descriptors are "
            "unavailable rather than zero. Its distinctive backbone "
            "flexibility is a property of the backbone, not of a side chain, "
            "and is not represented here."
        )
        return props

    if key == "P":
        props.backbone_constrained = 1.0
        for n in DESCRIPTOR_NAMES:
            if n != "backbone_constrained":
                props.provenance[n] = Provenance.UNAVAILABLE
        props.caveats.append(
            "Proline's side chain closes onto its own backbone nitrogen, so it "
            "cannot be isolated as a free fragment. Only its backbone "
            "constraint is represented."
        )
        return props

    mol = _capped_fragment(key)
    if mol is None:
        for n in DESCRIPTOR_NAMES:
            props.provenance[n] = Provenance.UNAVAILABLE
        props.caveats.append(f"side chain {key!r} could not be built as a fragment")
        return props

    carbons = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
    branched = sum(
        1 for a in carbons
        if sum(1 for nb in a.GetNeighbors() if nb.GetAtomicNum() == 6) > 2
    )

    props.heavy_atoms = float(mol.GetNumHeavyAtoms())
    props.side_chain_volume = float(rdMolDescriptors.CalcLabuteASA(mol))
    props.molar_refractivity = float(Crippen.MolMR(mol))
    props.rotatable_bonds = float(Lipinski.NumRotatableBonds(mol))
    props.branching_index = branched / len(carbons) if carbons else 0.0
    props.crippen_logp = float(Crippen.MolLogP(mol))
    props.tpsa = float(Descriptors.TPSA(mol))
    props.h_bond_donors = float(Lipinski.NumHDonors(mol))
    props.h_bond_acceptors = float(Lipinski.NumHAcceptors(mol))
    props.aromatic_rings = float(rdMolDescriptors.CalcNumAromaticRings(mol))
    props.aromatic_atoms = float(sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()))
    props.halogens = float(len(mol.GetSubstructMatches(_HALOGEN)))
    props.sulfur_atoms = float(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16))
    props.nitrogen_atoms = float(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7))
    props.oxygen_atoms = float(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8))
    props.formal_charge_graph = float(Chem.GetFormalCharge(mol))

    charge, acidic, basic, notes = _estimate_charge(mol, ph)
    props.estimated_charge = charge
    props.is_acidic = 1.0 if acidic > 0 else 0.0
    props.is_basic = 1.0 if basic > 0 else 0.0
    props.is_ionisable = 1.0 if (acidic or basic) else 0.0
    props.caveats.extend(notes)

    if not is_canon:
        props.caveats.append(
            "A non-canonical residue. Every descriptor above is computed from "
            "its structure; no tabulated amino-acid scale covers it, and the "
            "reference pKa values used for its ionisation estimate were not "
            "measured on this side chain."
        )
    return props


def residue_property_matrix(keys: Sequence[str]) -> np.ndarray:
    """One row of descriptors per residue key."""
    if not keys:
        return np.zeros((0, len(DESCRIPTOR_NAMES)))
    return np.vstack([describe_residue(k).vector() for k in keys])


def positional_chemistry(
    analyses: Sequence[PeptideAnalysis], *, n_term: int = 8, c_term: int = 3,
) -> np.ndarray:
    """Residue chemistry indexed by position, plus whole-peptide aggregates.

    Positional because Stage A measured that order-free composition scores
    below a mean-predictor baseline; aggregates as well because peptides here
    range from 3 to 34 residues and the slots alone would ignore the middle of
    the long ones.
    """
    from atlas.ml.peptide_representations import oriented_keys

    n_desc = len(DESCRIPTOR_NAMES)
    width = (n_term + c_term) * n_desc + 2 * n_desc + 1
    out = np.zeros((len(analyses), width), dtype=float)

    for row, a in enumerate(analyses):
        keys = oriented_keys(a)
        slots = (list(keys[:n_term]) + [""] * max(0, n_term - len(keys)))[:n_term]
        tail = (list(keys[-c_term:]) + [""] * max(0, c_term - len(keys)))[-c_term:]
        for i, k in enumerate(slots + tail):
            if k:
                out[row, i * n_desc:(i + 1) * n_desc] = describe_residue(k).vector()
        base = (n_term + c_term) * n_desc
        if keys:
            m = residue_property_matrix(keys)
            out[row, base:base + n_desc] = m.sum(axis=0)
            out[row, base + n_desc:base + 2 * n_desc] = m.mean(axis=0)
            out[row, base + 2 * n_desc] = len(keys)
    return out


def local_context(
    analyses: Sequence[PeptideAnalysis], *, window: int = 1,
    n_term: int = 8, c_term: int = 3,
) -> np.ndarray:
    """Chemistry of each position's sequence neighbours.

    The descriptors above treat a residue in isolation, but a residue's
    behaviour depends on what flanks it: a charged side chain beside an
    aromatic one is not the same environment as one beside a second charge.
    This adds, per position, the mean descriptor vector of the residues within
    ``window`` positions either side, excluding the residue itself.

    Neighbours **wrap** for a head-to-tail cyclic backbone, where position 1
    genuinely adjoins position n, and do not wrap for a linear or side-chain
    bridged one, where the termini are real. 58 of 630 peptides are cyclic.
    """
    from atlas.ml.peptide_representations import oriented_keys

    n_desc = len(DESCRIPTOR_NAMES)
    out = np.zeros((len(analyses), (n_term + c_term) * n_desc), dtype=float)

    for row, a in enumerate(analyses):
        keys = oriented_keys(a)
        if not keys:
            continue
        vectors = [describe_residue(k).vector() for k in keys]
        cyclic = a.is_cyclic_backbone
        n = len(keys)

        def neighbourhood(i: int) -> np.ndarray:
            picks = []
            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                j = i + offset
                if cyclic:
                    picks.append(vectors[j % n])
                elif 0 <= j < n:
                    picks.append(vectors[j])
            return (np.mean(picks, axis=0) if picks
                    else np.zeros(n_desc, dtype=float))

        slots = list(range(min(n_term, n)))
        tail = list(range(max(0, n - c_term), n))
        for slot, i in enumerate(slots):
            out[row, slot * n_desc:(slot + 1) * n_desc] = neighbourhood(i)
        for k, i in enumerate(tail):
            slot = n_term + (c_term - len(tail)) + k
            out[row, slot * n_desc:(slot + 1) * n_desc] = neighbourhood(i)
    return out


def provenance_summary(analyses: Iterable[PeptideAnalysis]) -> dict[str, object]:
    """What fraction of residues each provenance class actually covers.

    Reported rather than assumed: the engine's value depends on how much of
    the data it can describe, and for non-canonical residues that is the whole
    point.
    """
    import collections

    from atlas.ml.peptide_representations import oriented_keys

    counts: collections.Counter[str] = collections.Counter()
    canonical = noncanonical = unavailable = 0
    for a in analyses:
        for k in oriented_keys(a):
            p = describe_residue(k)
            counts[k] += 1
            if p.is_canonical:
                canonical += 1
            else:
                noncanonical += 1
            if all(v is Provenance.UNAVAILABLE
                   for n, v in p.provenance.items()
                   if n not in ("glycine_like", "backbone_constrained")):
                unavailable += 1
    total = canonical + noncanonical
    return {
        "n_residue_positions": total,
        "n_distinct_residues": len(counts),
        "canonical_positions": canonical,
        "noncanonical_positions": noncanonical,
        "noncanonical_fraction": noncanonical / total if total else 0.0,
        "positions_without_describable_side_chain": unavailable,
        "assumed_ph": ASSUMED_PH,
        "reference_pka": dict(REFERENCE_PKA),
        "note": (
            "Descriptors are computed from structure, so non-canonical "
            "residues are covered. No descriptor here is a measurement, and "
            "ionisation is estimated under one assumed pH applied uniformly "
            "because the assays' pH is not recorded in the ingested fields."
        ),
    }
