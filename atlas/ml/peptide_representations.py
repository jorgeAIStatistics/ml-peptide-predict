"""ATLAS 2.4 Stage B: peptide representations, built to Stage A's specification.

Stage A measured three constraints and this module is written to them:

* **Positional, not order-free.** Residue composition scored *below* a
  mean-predictor baseline, because a bag of residue counts cannot distinguish
  positional isomers and this population is single-substitution analogue
  series. Every representation here indexes residues by position.
* **Chemistry-aware vocabulary.** 80.5% of the peptides carry residues outside
  the canonical twenty, across 160 distinct side chains, mostly halogenated
  and naphthyl phenylalanines. A twenty-letter alphabet cannot name them.
* **Macrocyclic constraint explicit.** Only 131 of 630 peptides are
  unconstrained.

Two families are provided, and the difference between them is the point.
A one-hot encoding over the observed vocabulary can only recognise side chains
it has already seen; a fingerprint *of the side chain fragment* places an
unseen 3-chlorophenylalanine near a seen 4-chlorophenylalanine. Since
cross-series evaluation holds out whole analogue series, and series differ
precisely by which exotic side chain they carry, the two should behave very
differently — which is a measurement, not an assumption.

Head-to-tail cyclic peptides have no first residue; their rotation is
canonicalised here (lexicographically smallest key tuple) so that position
indices mean something. 58 of 630 are affected.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from atlas.ml.peptide_structure import CANONICAL_SIDE_CHAINS, PeptideAnalysis
from atlas.ml.stereochemistry import Configuration, stereo_residue_keys

RDLogger.DisableLog("rdApp.*")

__all__ = [
    "canonical_sequence", "residue_vocabulary", "oriented_keys",
    "oriented_configs", "stereo_oriented_keys", "stereo_features",
    "positional_onehot", "positional_fragment_fingerprint",
    "topology_features", "RepresentationSet", "build_representations",
    "N_TERM_SLOTS", "C_TERM_SLOTS", "FRAGMENT_BITS", "STEREO_FEATURE_NAMES",
]

#: Residues indexed from the N-terminus. Eight covers the dominant
#: heptapeptide series (314 of 630 peptides) in full.
N_TERM_SLOTS = 8
#: Residues indexed back from the C-terminus, so peptides of different length
#: still align at the end that is often amidated or acylated.
C_TERM_SLOTS = 3
#: Bits per side-chain fragment fingerprint.
FRAGMENT_BITS = 64

_CANONICAL = set(CANONICAL_SIDE_CHAINS) | {"G", "P"}
_PAD = "-"

_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=FRAGMENT_BITS, includeChirality=False
)


def canonical_sequence(a: PeptideAnalysis) -> str:
    """The sequence as a canonical-alphabet model would see it.

    Every non-canonical residue becomes ``X``. This is the representation a
    frozen protein language model is handed, and the function exists so the
    information it destroys can be measured rather than argued about.
    """
    if not a.residue_keys:
        return ""
    return "".join(k if k in _CANONICAL else "X" for k in a.residue_keys)


def _configs(a: PeptideAnalysis) -> list[str]:
    """Configurations padded to the residue count.

    An analysis constructed by hand, or produced before this field existed,
    carries no configurations; those positions read as unspecified rather than
    being defaulted to L.
    """
    cfgs = list(a.residue_configs)
    if len(cfgs) == len(a.residue_keys):
        return cfgs
    return cfgs[:len(a.residue_keys)] + [Configuration.UNSPECIFIED.value] * (
        len(a.residue_keys) - len(cfgs))


def _rotation(a: PeptideAnalysis) -> int:
    """Offset of the canonical first position for a cyclic backbone.

    One offset is computed and shared by every representation, so that slot 3
    of the fragment fingerprint and slot 3 of the stereo block describe the
    same residue. Choosing the rotation separately per representation would
    silently misalign them whenever stereochemistry broke a tie the
    constitution alone did not.
    """
    keys = a.residue_keys
    if not a.is_cyclic_backbone or len(keys) < 2:
        return 0
    cfgs = _configs(a)
    n = len(keys)
    return min(range(n), key=lambda i: (tuple(keys[i:] + keys[:i]),
                                        tuple(cfgs[i:] + cfgs[:i])))


def oriented_keys(a: PeptideAnalysis) -> list[str]:
    """Residue keys with a well-defined first position.

    A head-to-tail cyclic backbone has no N-terminus, so the tracer's starting
    residue is an artefact of atom ordering. Rotating to the lexicographically
    smallest tuple makes position 3 mean the same thing across the series.
    Linear and side-chain-bridged backbones are returned unchanged, since
    their N-terminus is real.
    """
    keys = list(a.residue_keys)
    i = _rotation(a)
    return keys[i:] + keys[:i]


def oriented_configs(a: PeptideAnalysis) -> list[str]:
    """Alpha-carbon configurations under the same rotation as the keys."""
    cfgs = _configs(a)
    i = _rotation(a)
    return cfgs[i:] + cfgs[:i]


def stereo_oriented_keys(a: PeptideAnalysis) -> list[str]:
    """Residue keys that distinguish D from L.

    This is the key the stereo-blind representations do not have: with it,
    D-Phe and L-Phe are different tokens, and the 137 molecules that share a
    constitutional key with a stereoisomer stop sharing a representation.
    """
    return stereo_residue_keys(oriented_keys(a), oriented_configs(a))


def residue_vocabulary(
    analyses: Iterable[PeptideAnalysis], *, stereo: bool = False
) -> dict[str, int]:
    """Index every side chain actually observed, canonical or not.

    Built from the training molecules only, when used in a model: a vocabulary
    fitted on the test set would leak. With ``stereo=True`` a D residue and
    its L counterpart receive separate indices.
    """
    keyer = stereo_oriented_keys if stereo else oriented_keys
    counts: collections.Counter[str] = collections.Counter()
    for a in analyses:
        counts.update(keyer(a))
    vocab = {_PAD: 0}
    for key, _n in counts.most_common():
        vocab.setdefault(key, len(vocab))
    return vocab


def _slots(keys: Sequence[str]) -> list[str]:
    """N-terminal and C-terminal slots, padded where the peptide is short."""
    n_part = list(keys[:N_TERM_SLOTS]) + [_PAD] * max(0, N_TERM_SLOTS - len(keys))
    c_part = list(keys[-C_TERM_SLOTS:]) + [_PAD] * max(0, C_TERM_SLOTS - len(keys))
    return n_part[:N_TERM_SLOTS] + c_part[-C_TERM_SLOTS:]


def positional_onehot(
    analyses: Sequence[PeptideAnalysis], vocab: dict[str, int],
    *, stereo: bool = False,
) -> np.ndarray:
    """One-hot residue identity per position. Memorises; cannot extrapolate.

    A side chain absent from ``vocab`` contributes an all-zero block, which is
    the honest behaviour: this representation has nothing to say about a
    residue it has never seen. ``stereo`` must match how ``vocab`` was built,
    or every key misses and the matrix is all zeros.
    """
    keyer = stereo_oriented_keys if stereo else oriented_keys
    width = len(vocab)
    n_slots = N_TERM_SLOTS + C_TERM_SLOTS
    out = np.zeros((len(analyses), n_slots * width), dtype=float)
    for row, a in enumerate(analyses):
        for slot, key in enumerate(_slots(keyer(a))):
            idx = vocab.get(key)
            if idx is not None:
                out[row, slot * width + idx] = 1.0
    return out


def _fragment_bits(key: str) -> np.ndarray:
    """Fingerprint one side chain, by fragment structure where available."""
    if key == _PAD:
        return np.zeros(FRAGMENT_BITS, dtype=float)
    smi = key if key.startswith("*") else ("*" + CANONICAL_SIDE_CHAINS.get(key, ""))
    if key in ("G", "P"):
        # Glycine has no side chain and proline's closes onto the backbone, so
        # neither has a free fragment. They are given reserved bits instead of
        # being silently encoded as padding.
        v = np.zeros(FRAGMENT_BITS, dtype=float)
        v[0 if key == "G" else 1] = 1.0
        return v
    mol = Chem.MolFromSmiles(smi) if smi != "*" else None
    if mol is None:
        return np.zeros(FRAGMENT_BITS, dtype=float)
    return np.asarray(_FP_GEN.GetFingerprintAsNumPy(mol), dtype=float)


def positional_fragment_fingerprint(
    analyses: Sequence[PeptideAnalysis],
) -> np.ndarray:
    """Morgan fingerprint of each side chain, concatenated by position.

    Unlike the one-hot encoding this generalises across side chains: an unseen
    3-chlorophenylalanine shares most of its bits with a seen
    4-chlorophenylalanine. That is what cross-series evaluation demands, since
    held-out series differ by exactly these substitutions.
    """
    cache: dict[str, np.ndarray] = {}
    n_slots = N_TERM_SLOTS + C_TERM_SLOTS
    out = np.zeros((len(analyses), n_slots * FRAGMENT_BITS), dtype=float)
    for row, a in enumerate(analyses):
        for slot, key in enumerate(_slots(oriented_keys(a))):
            if key not in cache:
                cache[key] = _fragment_bits(key)
            out[row, slot * FRAGMENT_BITS:(slot + 1) * FRAGMENT_BITS] = cache[key]
    return out


#: Column names of ``stereo_features``, in order.
STEREO_FEATURE_NAMES = (
    [f"slot{i}_is_D" for i in range(N_TERM_SLOTS + C_TERM_SLOTS)]
    + [f"slot{i}_is_unspecified" for i in range(N_TERM_SLOTS + C_TERM_SLOTS)]
    + [f"slot{i}_is_alpha_disubstituted"
       for i in range(N_TERM_SLOTS + C_TERM_SLOTS)]
    + ["n_d_residues", "d_fraction", "n_unspecified",
       "n_alpha_disubstituted", "all_l"]
)


def stereo_features(analyses: Sequence[PeptideAnalysis]) -> np.ndarray:
    """Backbone configuration as a positional feature block.

    L is the implicit default and gets no column of its own: what the block
    marks is each position's *departure* from it. A padded slot is therefore
    all zeros like an L residue, which is correct for this block's purpose --
    the fragment fingerprint it accompanies already distinguishes a padded
    slot from an occupied one, so spending three more columns to repeat that
    would add collinear features rather than information.

    ``unspecified`` is kept separate from both D and L rather than being
    folded into either. It covers 7 of 5409 alpha carbons in the MC4R set, and
    merging it into L would record a structural fact the deposited record does
    not state.
    """
    n_slots = N_TERM_SLOTS + C_TERM_SLOTS
    out = np.zeros((len(analyses), len(STEREO_FEATURE_NAMES)), dtype=float)
    for row, a in enumerate(analyses):
        cfgs = _slots(oriented_configs(a))
        for slot, cfg in enumerate(cfgs):
            if cfg == Configuration.D.value:
                out[row, slot] = 1.0
            elif cfg == Configuration.UNSPECIFIED.value:
                out[row, n_slots + slot] = 1.0
            elif cfg == Configuration.ALPHA_DISUBSTITUTED.value:
                out[row, 2 * n_slots + slot] = 1.0
        # Molecule-level counts run over every residue, not just the indexed
        # slots: a fifteen-residue peptide's D residue at position 11 falls
        # outside the eight N-terminal and three C-terminal slots entirely.
        full = oriented_configs(a)
        n_d = sum(1 for c in full if c == Configuration.D.value)
        n_un = sum(1 for c in full if c == Configuration.UNSPECIFIED.value)
        n_aa = sum(1 for c in full
                   if c == Configuration.ALPHA_DISUBSTITUTED.value)
        base = 3 * n_slots
        out[row, base] = n_d
        out[row, base + 1] = n_d / len(full) if full else 0.0
        out[row, base + 2] = n_un
        out[row, base + 3] = n_aa
        out[row, base + 4] = float(bool(full) and n_d == n_un == n_aa == 0)
    return out


def topology_features(analyses: Sequence[PeptideAnalysis]) -> np.ndarray:
    """The macrocyclic constraint, made explicit as Stage A required."""
    rows = []
    for a in analyses:
        rows.append([
            a.n_residues_in_longest_chain,
            float(a.is_cyclic_backbone),
            a.n_side_chain_bridges,
            a.n_disulfides,
            a.largest_backbone_ring,
            float(a.has_c_terminal_amide),
            float(a.has_c_terminal_acid),
            float(a.has_acylated_n_terminus),
            float(a.has_free_n_terminus),
            a.n_unrecognised_side_chains,
            a.canonical_fraction,
            a.molecular_weight / 1000.0,
        ])
    return np.asarray(rows, dtype=float)


@dataclass
class RepresentationSet:
    """Named feature matrices over one aligned list of molecules."""

    names: list[str]
    matrices: dict[str, np.ndarray]
    vocabulary_size: int
    n_molecules: int
    stereo_vocabulary_size: int = 0

    def __getitem__(self, name: str) -> np.ndarray:
        return self.matrices[name]


def build_representations(
    analyses: Sequence[PeptideAnalysis],
    *,
    vocabulary: dict[str, int] | None = None,
    stereo_vocabulary: dict[str, int] | None = None,
    extra: dict[str, np.ndarray] | None = None,
) -> RepresentationSet:
    """Assemble every representation to be benchmarked.

    ``vocabulary`` should be fitted on training molecules only; passing None
    fits it on everything given, which is correct for a descriptive audit and
    wrong for a model comparison.
    """
    vocab = vocabulary if vocabulary is not None else residue_vocabulary(analyses)
    svocab = (stereo_vocabulary if stereo_vocabulary is not None
              else residue_vocabulary(analyses, stereo=True))
    topo = topology_features(analyses)
    onehot = positional_onehot(analyses, vocab)
    onehot_stereo = positional_onehot(analyses, svocab, stereo=True)
    stereo = stereo_features(analyses)
    fragfp = positional_fragment_fingerprint(analyses)

    # Imported here rather than at module scope: residue_chemistry needs
    # oriented_keys from this module, and a top-level import either way round
    # would be circular.
    from atlas.ml.residue_chemistry import local_context, positional_chemistry

    chem = positional_chemistry(analyses)
    context = local_context(analyses)

    mats: dict[str, np.ndarray] = {
        "topology": topo,
        "positional_onehot": onehot,
        "positional_fragment_fp": fragfp,
        "residue_chemistry": chem,
        "stereo": stereo,
        "positional_onehot_stereo": onehot_stereo,
        "positional_fragment_fp+stereo": np.hstack([fragfp, stereo]),
        "positional_onehot+topology": np.hstack([onehot, topo]),
        "positional_fragment_fp+topology": np.hstack([fragfp, topo]),
        "residue_chemistry+positional_fragment_fp": np.hstack([chem, fragfp]),
        "residue_chemistry+local_context": np.hstack([chem, context]),
        "positional_fragment_fp+local_context": np.hstack([fragfp, context]),
    }
    for name, mat in (extra or {}).items():
        mats[name] = mat
        mats[f"positional_fragment_fp+{name}"] = np.hstack([fragfp, mat])

    return RepresentationSet(
        names=list(mats), matrices=mats,
        vocabulary_size=len(vocab), n_molecules=len(analyses),
        stereo_vocabulary_size=len(svocab),
    )
