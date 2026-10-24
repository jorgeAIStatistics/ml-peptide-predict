"""Trace peptide backbones to establish what a molecule actually is.

The previous classification called a molecule "peptide-like" when it had at
least four amide bonds and weighed at least 600 Da. That is a proxy, and the
2.4 brief is right to refuse it: a bis-amide macrocycle with four amide groups
satisfies it without being a peptide, and a chemically modified tetrapeptide
can fail it.

This module instead walks the molecular graph. A backbone unit is
``N–Cα–C(=O)``; units are chained when one unit's carbonyl carbon bonds to the
next unit's nitrogen. From that chain the module reports residue count,
topology (linear, head-to-tail cyclic, or side-chain bridged), disulfide
bridges, terminal state, and how many residues carry a recognisable canonical
side chain.

Configuration at each alpha carbon is read alongside the chain, by
``atlas.ml.stereochemistry``: tracing the graph alone would make D-Phe and
L-Phe the same residue, and 85.6% of the peptides in this dataset carry a D
residue.

What it cannot do is also recorded. Backbone tracing establishes connectivity
and alpha-carbon configuration, not stereochemistry *within* a side chain --
threonine's beta carbon and allo-threonine's still produce the same key -- nor
residue order confidence for unusual chemistry, and a molecule can be a
genuine peptidomimetic while tracing only two units.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Iterable

from functools import lru_cache

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

from atlas.ml.stereochemistry import Configuration, alpha_configurations

RDLogger.DisableLog("rdApp.*")

_TAUTOMERS = rdMolStandardize.TautomerEnumerator()


@lru_cache(maxsize=4096)
def _tautomer_key(smiles: str) -> str:
    """Canonical SMILES after tautomer normalisation.

    Guanidines and imidazoles are written with the double bond on either
    nitrogen depending on the depositor, so arginine and histidine match in one
    record and fail in the next unless tautomers are normalised first. Observed
    directly: arginine matched in the afamelanotide record and failed in the
    leuprolide record, the same residue written two ways.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        return Chem.MolToSmiles(_TAUTOMERS.Canonicalize(mol))
    except Exception:
        return Chem.MolToSmiles(mol)

__all__ = [
    "PeptideClass",
    "PeptideAnalysis",
    "analyse_peptide",
    "classify_molecule",
    "CANONICAL_SIDE_CHAINS",
    "Configuration",
]


class PeptideClass(str, enum.Enum):
    """What a molecule is, established from its graph rather than assumed."""

    CANONICAL_LINEAR = "canonical_linear_peptide"
    MODIFIED_LINEAR = "modified_linear_peptide"
    CYCLIC = "cyclic_peptide"
    DISULFIDE_CONSTRAINED = "disulfide_constrained_peptide"
    NONCANONICAL = "noncanonical_residue_peptide"
    PEPTIDOMIMETIC = "peptidomimetic"
    SMALL_MOLECULE = "non_peptide_small_molecule"
    UNPARSEABLE = "unparseable"

    @property
    def is_peptide(self) -> bool:
        """Whether a residue chain long enough to be a peptide was traced.

        A peptidomimetic is deliberately excluded. It contains peptide bonds
        and may bind the same receptor, but it has no sequence to featurise,
        so counting it as a peptide is what inflated the original figure.
        """
        return self in (
            PeptideClass.CANONICAL_LINEAR,
            PeptideClass.MODIFIED_LINEAR,
            PeptideClass.CYCLIC,
            PeptideClass.DISULFIDE_CONSTRAINED,
            PeptideClass.NONCANONICAL,
        )

    @property
    def is_peptidomimetic(self) -> bool:
        return self is PeptideClass.PEPTIDOMIMETIC

    @property
    def has_sequence(self) -> bool:
        """Whether a residue sequence can meaningfully be written out."""
        return self.is_peptide


#: Canonical side chains as SMILES, matched against the fragment hanging off
#: each traced alpha carbon. Glycine has no side chain and is detected by its
#: absence; alanine is a lone methyl.
CANONICAL_SIDE_CHAINS: dict[str, str] = {
    "A": "C",
    "R": "CCCNC(N)=N",
    "N": "CC(N)=O",
    "D": "CC(O)=O",
    "C": "CS",
    "E": "CCC(O)=O",
    "Q": "CCC(N)=O",
    "H": "Cc1cnc[nH]1",
    "I": "C(C)CC",
    "L": "CC(C)C",
    "K": "CCCCN",
    "M": "CCSC",
    "F": "Cc1ccccc1",
    "S": "CO",
    "T": "C(C)O",
    "W": "Cc1c[nH]c2ccccc12",
    "Y": "Cc1ccc(O)cc1",
    "V": "C(C)C",
}

#: The comparison keys carry a dummy atom where the side chain met the alpha
#: carbon. The marker has to be a dummy rather than a methyl: capped with a
#: methyl, leucine (CH2-CH(CH3)2) and isoleucine (CH(CH3)-CH2CH3) are both
#: 2-methylbutane, and the attachment point that distinguishes them is lost.
_SIDE_CHAIN_KEYS = {
    code: _tautomer_key("*" + smi)
    for code, smi in CANONICAL_SIDE_CHAINS.items()
    if Chem.MolFromSmiles("*" + smi) is not None
}

#: Proline's side chain closes onto its own backbone nitrogen, so it is
#: identified by that ring closure rather than by a free fragment: cut at the
#: nitrogen, its side chain is indistinguishable from norvaline's n-propyl.
_PROLINE_RING_CARBONS = 3

#: A backbone unit: an amine nitrogen, an alpha carbon, and a carbonyl carbon.
#:
#: The nitrogen must reach the carbonyl *through* the alpha carbon. That is what
#: separates backbone from side chain: asparagine's side-chain nitrogen bonds its
#: carbonyl directly, with no intervening sp3 carbon, so it never matches. An
#: earlier version of this pattern excluded nitrogens attached to a carbonyl,
#: which was self-defeating — every backbone nitrogen but the N-terminal one
#: carries the preceding residue's carbonyl, so no chain could ever link.
_UNIT = Chem.MolFromSmarts("[NX3;!$([NX3]=*);!$([nX3])][CX4][CX3](=[OX1])")
_DISULFIDE = Chem.MolFromSmarts("[SX2][SX2]")
_C_TERM_ACID = Chem.MolFromSmarts("[CX3](=[OX1])[OX2H1]")
_C_TERM_AMIDE = Chem.MolFromSmarts("[CX4][CX3](=[OX1])[NX3H2]")
_N_TERM_FREE = Chem.MolFromSmarts("[NX3H2][CX4][CX3]=[OX1]")
_N_TERM_ACYL = Chem.MolFromSmarts("[CX4,c][CX3](=[OX1])[NX3H1][CX4][CX3]=[OX1]")


@dataclass
class PeptideAnalysis:
    """What the graph walk established about one molecule."""

    smiles: str
    peptide_class: PeptideClass
    n_backbone_units: int = 0
    n_residues_in_longest_chain: int = 0
    is_cyclic_backbone: bool = False
    n_side_chain_bridges: int = 0
    n_disulfides: int = 0
    n_canonical_side_chains: int = 0
    n_unrecognised_side_chains: int = 0
    sequence: str | None = None          # one-letter where recognised, X otherwise
    #: Per-position residue identity along the traced chain: the one-letter
    #: code where canonical, the side-chain fragment SMILES where not.
    residue_keys: list[str] = field(default_factory=list)
    #: Configuration at each position's alpha carbon, aligned to
    #: ``residue_keys``: "L", "D", "achiral", "alpha_disubstituted" or
    #: "unspecified". Read from the deposited structure, not measured here.
    residue_configs: list[str] = field(default_factory=list)
    #: Atom index of each position's alpha carbon, so a caller can go back to
    #: the molecule -- the interaction-feature code needs this to say which
    #: residue a contact belongs to.
    alpha_carbon_indices: list[int] = field(default_factory=list)
    has_free_n_terminus: bool = False
    has_acylated_n_terminus: bool = False
    has_c_terminal_amide: bool = False
    has_c_terminal_acid: bool = False
    largest_ring: int = 0
    #: Largest ring containing at least two backbone atoms. Classification uses
    #: this rather than ``largest_ring``, so that a macrolactone or fused
    #: aromatic system elsewhere in the molecule cannot make it a cyclic peptide.
    largest_backbone_ring: int = 0
    molecular_weight: float = 0.0
    #: Canonical SMILES of side-chain fragments that matched no canonical
    #: residue, so that "noncanonical" can be described rather than asserted.
    unmatched_fragments: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    @property
    def canonical_fraction(self) -> float:
        total = self.n_canonical_side_chains + self.n_unrecognised_side_chains
        return self.n_canonical_side_chains / total if total else 0.0

    @property
    def is_peptide(self) -> bool:
        return self.peptide_class.is_peptide

    @property
    def n_d_residues(self) -> int:
        return sum(1 for c in self.residue_configs
                   if c == Configuration.D.value)

    @property
    def n_unspecified_configurations(self) -> int:
        return sum(1 for c in self.residue_configs
                   if c == Configuration.UNSPECIFIED.value)

    def summary(self) -> str:
        return (
            f"{self.peptide_class.value}: {self.n_backbone_units} backbone units, "
            f"{self.n_canonical_side_chains} canonical side chains"
            + (f", {self.n_disulfides} disulfide" if self.n_disulfides else "")
            + (", cyclic backbone" if self.is_cyclic_backbone else "")
        )


def _side_chain_code(
    mol: Chem.Mol, ca_idx: int, n_idx: int, c_idx: int,
    backbone: set[int], disulfide_bonds: set[frozenset[int]],
) -> tuple[str, bool, str | None]:
    """Identify the fragment hanging off an alpha carbon.

    Returns a one-letter code — canonical residue, 'G' where the alpha carbon
    carries no side chain, 'X' where the fragment is something else — and
    whether the side chain bridges to another residue's backbone.

    The walk is blocked at *every* backbone atom in the molecule, not merely
    this residue's, and is cut at disulfide bonds. Without those two cuts a
    cystine side chain walks across the bridge into the far half of the
    molecule and nothing matches.
    """
    ca = mol.GetAtomWithIdx(ca_idx)
    branch_roots = [
        nb.GetIdx() for nb in ca.GetNeighbors()
        if nb.GetIdx() not in (n_idx, c_idx) and nb.GetAtomicNum() > 1
    ]
    if not branch_roots:
        return "G", False, None  # glycine: no side chain at all

    own = {n_idx, c_idx, ca_idx}
    blocked = (backbone | own) - set(branch_roots)
    collected: set[int] = set()
    bridges = False
    reaches_own_n = False
    stack = list(branch_roots)
    while stack:
        idx = stack.pop()
        if idx in collected:
            continue
        collected.add(idx)
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            j = nb.GetIdx()
            if j in collected:
                continue
            if frozenset((idx, j)) in disulfide_bonds:
                continue  # the bridge is reported separately, not walked
            if j in blocked:
                # Stepping back onto this residue's own backbone is simply the
                # bond the side chain hangs from, not a bridge. Only reaching
                # *another* residue's backbone makes the molecule bridged.
                if j not in own:
                    bridges = True
                elif j == n_idx:
                    reaches_own_n = True
                continue
            stack.append(j)
    if not collected:
        return "G", False, None

    if reaches_own_n:
        # A cyclic imino acid. Proline is the three-carbon case; anything else
        # (hydroxyproline, azetidine and pipecolic analogues) is not asserted.
        carbons = sum(1 for i in collected if mol.GetAtomWithIdx(i).GetAtomicNum() == 6)
        return (("P" if len(collected) == carbons == _PROLINE_RING_CARBONS else "X"),
                bridges, None)

    # Build the fragment with a dummy atom where it met the alpha carbon, so
    # its canonical SMILES matches the reference side chains above.
    try:
        frag = Chem.RWMol()
        mapping: dict[int, int] = {}
        for idx in sorted(collected):
            a = mol.GetAtomWithIdx(idx)
            new = Chem.Atom(a.GetAtomicNum())
            new.SetFormalCharge(a.GetFormalCharge())
            new.SetIsAromatic(a.GetIsAromatic())
            # The explicit hydrogen count must be carried across or aromatic
            # nitrogens lose their NH and sanitisation reads a different ring:
            # histidine's imidazole and tryptophan's indole both depend on it.
            new.SetNumExplicitHs(a.GetNumExplicitHs())
            new.SetNoImplicit(a.GetNoImplicit())
            mapping[idx] = frag.AddAtom(new)
        stub = frag.AddAtom(Chem.Atom(0))  # dummy marks the attachment point
        for idx in sorted(collected):
            for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
                j = nb.GetIdx()
                if j in collected and j > idx:
                    b = mol.GetBondBetweenAtoms(idx, j)
                    frag.AddBond(mapping[idx], mapping[j], b.GetBondType())
            if idx in branch_roots:
                frag.AddBond(mapping[idx], stub, Chem.BondType.SINGLE)
        out = frag.GetMol()
        Chem.SanitizeMol(out)
        key = _tautomer_key(Chem.MolToSmiles(out))
    except Exception:
        return "X", bridges, None

    for code, ref in _SIDE_CHAIN_KEYS.items():
        if key == ref:
            return code, bridges, None
    # An unmatched fragment is returned so the audit can report what the
    # noncanonical residues actually are, rather than counting them as X.
    return "X", bridges, key


def analyse_peptide(smiles: str) -> PeptideAnalysis:
    """Walk a molecule's graph and report what it is."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return PeptideAnalysis(smiles=smiles, peptide_class=PeptideClass.UNPARSEABLE,
                               caveats=["structure could not be parsed"])

    from rdkit.Chem import Descriptors
    mw = float(Descriptors.MolWt(mol))
    rings = mol.GetRingInfo().AtomRings()
    largest_ring = max((len(r) for r in rings), default=0)
    n_disulfide = len(mol.GetSubstructMatches(_DISULFIDE))

    matches = mol.GetSubstructMatches(_UNIT)
    # One alpha carbon belongs to one unit; de-duplicate on it.
    units: dict[int, tuple[int, int, int]] = {}
    for n_idx, ca_idx, c_idx, _o in matches:
        units.setdefault(ca_idx, (n_idx, ca_idx, c_idx))

    analysis = PeptideAnalysis(
        smiles=smiles, peptide_class=PeptideClass.SMALL_MOLECULE,
        n_backbone_units=len(units), n_disulfides=n_disulfide,
        largest_ring=largest_ring, molecular_weight=mw,
    )

    if not units:
        return analysis

    # Chain units: unit i precedes unit j when i's carbonyl bonds j's nitrogen.
    by_n = {u[0]: ca for ca, u in units.items()}
    nxt: dict[int, int] = {}
    prev: dict[int, int] = {}
    for ca, (n_idx, _ca, c_idx) in units.items():
        for nb in mol.GetAtomWithIdx(c_idx).GetNeighbors():
            if nb.GetIdx() in by_n and by_n[nb.GetIdx()] != ca:
                nxt[ca] = by_n[nb.GetIdx()]
                prev[by_n[nb.GetIdx()]] = ca

    # Longest chain, and whether the backbone closes on itself.
    starts = [ca for ca in units if ca not in prev]
    cyclic = not starts and bool(nxt)
    longest: list[int] = []
    for start in (starts or list(units)[:1]):
        walk, seen = [], set()
        cur = start
        while cur is not None and cur not in seen:
            seen.add(cur)
            walk.append(cur)
            cur = nxt.get(cur)
        if len(walk) > len(longest):
            longest = walk

    backbone = {i for u in units.values() for i in u}
    ss_bonds = {
        frozenset(m) for m in mol.GetSubstructMatches(_DISULFIDE)
    }
    results = [
        _side_chain_code(mol, ca, units[ca][0], units[ca][2], backbone, ss_bonds)
        for ca in longest
    ]
    codes = [r[0] for r in results]
    n_bridges = sum(1 for r in results if r[1])
    n_canonical = sum(1 for c in codes if c != "X")
    n_unknown = sum(1 for c in codes if c == "X")
    analysis.unmatched_fragments = [r[2] for r in results if r[2]]
    # Per-position residue identity: the one-letter code where canonical, the
    # side-chain fragment SMILES otherwise. This is what a chemistry-aware
    # representation needs and what a one-letter sequence throws away -- the
    # sequence string renders 4-chlorophenylalanine and 2-naphthylalanine
    # identically as X.
    analysis.residue_keys = [
        (code if code != "X" else (frag or "X"))
        for code, _bridge, frag in results
    ]
    # Configuration is read per position along the same walk, so the two lists
    # are aligned by construction rather than by a later zip.
    configs = alpha_configurations(mol, longest)
    analysis.alpha_carbon_indices = list(longest)
    analysis.residue_configs = [configs[ca].value for ca in longest]

    analysis.n_residues_in_longest_chain = len(longest)
    analysis.is_cyclic_backbone = cyclic
    analysis.n_side_chain_bridges = n_bridges
    analysis.largest_backbone_ring = max(
        (len(r) for r in rings if len(set(r) & backbone) >= 2), default=0
    )
    analysis.n_canonical_side_chains = n_canonical
    analysis.n_unrecognised_side_chains = n_unknown
    analysis.sequence = "".join(codes) if codes else None
    analysis.has_free_n_terminus = mol.HasSubstructMatch(_N_TERM_FREE)
    analysis.has_acylated_n_terminus = mol.HasSubstructMatch(_N_TERM_ACYL)
    analysis.has_c_terminal_amide = mol.HasSubstructMatch(_C_TERM_AMIDE)
    analysis.has_c_terminal_acid = mol.HasSubstructMatch(_C_TERM_ACID)
    analysis.peptide_class = classify_molecule(analysis)

    if analysis.n_backbone_units > len(longest) and not cyclic:
        analysis.caveats.append(
            f"{analysis.n_backbone_units} backbone units were found but the "
            f"longest traceable chain is {len(longest)}; the molecule may be "
            "branched or contain separate peptide segments."
        )
    if analysis.n_unspecified_configurations:
        analysis.caveats.append(
            f"{analysis.n_unspecified_configurations} alpha carbon(s) are "
            "stereocentres the deposited structure does not define; they are "
            "recorded as unspecified rather than assumed to be L."
        )
    if n_unknown:
        analysis.caveats.append(
            f"{n_unknown} of {len(codes)} side chains did not match a canonical "
            "residue and are recorded as X."
        )
    return analysis


#: At least this many chained residues before a molecule is called a peptide
#: rather than a small molecule that happens to contain amide bonds.
MIN_PEPTIDE_RESIDUES = 3

#: A backbone-containing ring of at least this size is a macrocyclisation
#: rather than a residue's own ring (proline's is 5, tryptophan's fused rings
#: 5 and 6).
MACROCYCLE_RING_SIZE = 12


def classify_molecule(a: PeptideAnalysis) -> PeptideClass:
    """Assign a class from the traced structure."""
    n = a.n_residues_in_longest_chain
    if n == 0:
        return PeptideClass.SMALL_MOLECULE
    if n < MIN_PEPTIDE_RESIDUES:
        # Two chained residues with substantial other chemistry is the shape of
        # a peptidomimetic rather than a peptide.
        return (PeptideClass.PEPTIDOMIMETIC if a.molecular_weight >= 400
                else PeptideClass.SMALL_MOLECULE)
    if a.n_disulfides:
        return PeptideClass.DISULFIDE_CONSTRAINED
    if (a.is_cyclic_backbone or a.n_side_chain_bridges
            or a.largest_backbone_ring >= MACROCYCLE_RING_SIZE):
        return PeptideClass.CYCLIC
    if a.n_unrecognised_side_chains:
        return PeptideClass.NONCANONICAL
    if a.has_acylated_n_terminus or a.has_c_terminal_amide:
        return PeptideClass.MODIFIED_LINEAR
    return PeptideClass.CANONICAL_LINEAR


def analyse_many(smiles_list: Iterable[str]) -> list[PeptideAnalysis]:
    return [analyse_peptide(s) for s in smiles_list]
