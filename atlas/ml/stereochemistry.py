"""Backbone configuration at the alpha carbon: the D/L distinction.

Every representation ATLAS has built so far is stereo-blind. The tracer
identifies a residue by the constitution of its side-chain fragment, so
D-phenylalanine and L-phenylalanine produce the same key, the same one-hot
column and the same fragment fingerprint. In a melanocortin dataset that is
not a minor omission: D-Phe(7) is the defining substitution of NDP-alpha-MSH
and of the cyclic analogue series derived from it, and this module's census
finds a D residue in 85.6% of the peptides.

**Why not CIP.** The obvious route -- read RDKit's R/S label and map it to D
or L -- is wrong, and wrong in a way that is easy to miss. L-amino acids are
(S) at the alpha carbon except cysteine, which is (R): sulfur outranks the
carboxyl carbon, so the same spatial arrangement receives the opposite CIP
label. A CIP-based mapping needs a per-residue exception table, and for the
160 noncanonical side chains in this dataset there is no table to consult.

What is used instead is the configuration *relative to the backbone*, which is
what D and L mean: the arrangement of (N, C(=O), side chain, H) around the
alpha carbon, compared against L-alanine. A chiral substructure query states
that arrangement once, and RDKit's own matcher handles the permutation
bookkeeping for whatever order the depositor happened to write the atoms in.
That definition extends to any side chain, canonical or not, and returns L for
L-cysteine as it should. Verified against nine reference standards in
``tests/test_stereochemistry.py``, cysteine, threonine and proline among them.

The configurations this module reports are read from the deposited structure.
They are a property of the record, not a measurement ATLAS performed, and
where a record leaves a stereocentre undefined that is reported as
``UNSPECIFIED`` rather than assumed to be L.
"""

from __future__ import annotations

import collections
import enum
from typing import Iterable, Sequence

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

__all__ = [
    "Configuration", "ALPHA_L_QUERY", "ALPHA_D_QUERY",
    "alpha_configurations", "configuration_at", "stereo_residue_keys",
    "configuration_census",
    "D_PREFIX", "UNSPECIFIED_PREFIX",
]


class Configuration(str, enum.Enum):
    """The configuration of one traced alpha carbon."""

    L = "L"
    D = "D"
    #: Two hydrogens on the alpha carbon, so there is no stereocentre to
    #: report. Glycine, and the glycine-like positions of its analogues.
    ACHIRAL = "achiral"
    #: No alpha hydrogen: an alpha,alpha-disubstituted residue such as Aib.
    #: It may still be a stereocentre, but not one D and L describe, so it is
    #: named for what it is rather than forced into the binary.
    ALPHA_DISUBSTITUTED = "alpha_disubstituted"
    #: A stereocentre the deposited structure does not define. Not assumed to
    #: be L: 7 of 5409 alpha carbons in the MC4R set are in this state, and
    #: filling them in would be inventing a structural fact.
    UNSPECIFIED = "unspecified"

    @property
    def is_defined(self) -> bool:
        return self in (Configuration.L, Configuration.D)


#: The L arrangement of (N, H, side chain, C=O) about the alpha carbon, and its
#: mirror image. ``[!#1]`` stands for any heavy side-chain atom, which is what
#: makes the test independent of the residue: proline's ring carbon, cysteine's
#: thiomethyl and a naphthylalanine's benzylic carbon all satisfy it. Matching
#: with ``useChirality=True`` compares the arrangement, not the atom order.
ALPHA_L_QUERY = Chem.MolFromSmarts("[NX3][C@@H]([!#1])[CX3]=[OX1]")
ALPHA_D_QUERY = Chem.MolFromSmarts("[NX3][C@H]([!#1])[CX3]=[OX1]")

#: Prefixes for the residue tokens below. L is left unmarked, so a stereo-aware
#: vocabulary differs from the stereo-blind one only where the configuration
#: departs from the biological default -- which keeps the vocabulary close to
#: its current size instead of doubling it.
D_PREFIX = "D-"
UNSPECIFIED_PREFIX = "?-"
_DISUBSTITUTED_PREFIX = "aa-"


def alpha_configurations(
    mol: Chem.Mol, alpha_indices: Iterable[int]
) -> dict[int, Configuration]:
    """Configuration of each given alpha carbon, keyed by atom index.

    The two chiral queries are run once over the whole molecule; an alpha
    carbon matching neither is classified from its hydrogen count, which
    separates "no stereocentre here" from "a stereocentre nobody specified".
    """
    l_set = {m[1] for m in mol.GetSubstructMatches(ALPHA_L_QUERY, useChirality=True)}
    d_set = {m[1] for m in mol.GetSubstructMatches(ALPHA_D_QUERY, useChirality=True)}
    out: dict[int, Configuration] = {}
    for idx in alpha_indices:
        if idx in l_set:
            out[idx] = Configuration.L
        elif idx in d_set:
            out[idx] = Configuration.D
        else:
            n_h = mol.GetAtomWithIdx(idx).GetTotalNumHs()
            out[idx] = (
                Configuration.ACHIRAL if n_h >= 2
                else Configuration.ALPHA_DISUBSTITUTED if n_h == 0
                else Configuration.UNSPECIFIED
            )
    return out


def configuration_at(mol: Chem.Mol, alpha_index: int) -> Configuration:
    """Convenience wrapper for a single alpha carbon."""
    return alpha_configurations(mol, [alpha_index])[alpha_index]


def stereo_residue_keys(
    residue_keys: Sequence[str], configurations: Sequence[str]
) -> list[str]:
    """Residue keys carrying their configuration, for use as feature keys.

    ``D-F`` and ``F`` are different tokens; an achiral position and an L
    position keep the bare key. Lengths must agree -- a mismatch means the
    chain walk and the stereo assignment disagree about how many residues
    there are, and silently zipping to the shorter of the two would attach
    configurations to the wrong positions.
    """
    if len(residue_keys) != len(configurations):
        raise ValueError(
            f"{len(residue_keys)} residue keys against "
            f"{len(configurations)} configurations"
        )
    prefix = {
        Configuration.D.value: D_PREFIX,
        Configuration.UNSPECIFIED.value: UNSPECIFIED_PREFIX,
        Configuration.ALPHA_DISUBSTITUTED.value: _DISUBSTITUTED_PREFIX,
    }
    return [prefix.get(str(cfg), "") + key
            for key, cfg in zip(residue_keys, configurations)]


def configuration_census(
    configuration_lists: Iterable[Sequence[str]],
) -> dict[str, object]:
    """Counts over many molecules' configuration lists, for the audit."""
    per_residue: collections.Counter[str] = collections.Counter()
    n_molecules = 0
    with_d = 0
    with_unspecified = 0
    for cfgs in configuration_lists:
        n_molecules += 1
        per_residue.update(cfgs)
        if Configuration.D.value in cfgs:
            with_d += 1
        if Configuration.UNSPECIFIED.value in cfgs:
            with_unspecified += 1
    total = sum(per_residue.values())
    return {
        "n_molecules": n_molecules,
        "n_alpha_carbons": total,
        "per_residue_counts": dict(per_residue.most_common()),
        "per_residue_fractions": {
            k: v / total for k, v in per_residue.most_common()
        } if total else {},
        "molecules_with_a_d_residue": with_d,
        "molecules_with_a_d_residue_fraction": (
            with_d / n_molecules if n_molecules else 0.0),
        "molecules_with_an_unspecified_centre": with_unspecified,
        "provenance": (
            "Configurations are read from the deposited structures, not "
            "determined experimentally by ATLAS."
        ),
    }
