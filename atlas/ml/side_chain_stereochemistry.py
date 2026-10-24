"""Stereochemistry *inside* the side chain: the census, before any fix.

ATLAS 2.5 assigned configuration at the alpha carbon and left the side chain
flat. Two mechanisms do it: ``_side_chain_code`` rebuilds the side-chain
fragment atom by atom into a fresh ``RWMol`` and never copies the chiral tags,
and the fragment fingerprint is generated with ``includeChirality=False``. So
threonine and allo-threonine, isoleucine and allo-isoleucine, and any
noncanonical side chain carrying a stereocentre produce the same residue key
and the same bits.

The ATLAS 2.5 write-up recorded that loss as *unmeasured and not to be assumed
small*. This module measures it, and the measurement is the deliverable: on
the MC4R set the loss is real at the residue level and separates **no two
molecules**, so neither mechanism is changed. Five representation extensions
in a row have failed to improve cross-series generalisation; a sixth was not
going to be adopted on the strength of a loss that collides with nothing.

**How the fragment is cut.** Not by rebuilding it. A rebuild reorders a
stereocentre's neighbours, so a chiral tag copied across is then wrong as
often as right, and the bookkeeping to correct it is exactly the kind of
hand-rolled parity arithmetic ``stereochemistry.py`` refuses to do. Instead
every bond leaving the traced side chain is cut with
:func:`rdkit.Chem.FragmentOnBonds`, which substitutes a dummy atom for the
severed partner and therefore keeps each stereocentre's neighbour count and
ordering intact. RDKit assigns the tags; this module only compares them.

**The instrument is deliberately finer than the deployed key.** Its fragments
carry a dummy at every cut, so a bridged cysteine (``*CS*``) differs from a
free one (``*CS``) where the deployed key merges them, and proline is a cut
fragment rather than the reserved letter ``P``. That only helps: the
comparison here is each fragment's stereo SMILES against *the same fragment
with stereochemistry stripped*, so a finer partition gives configuration more
chances to split a group, not fewer. Zero splits on the finer partition is a
stronger statement than zero on the coarser one.

**What a cut fragment cannot see, and what is measured instead.** A
stereocentre can be a stereocentre only in the whole molecule's context: a
bridged side chain cut at both ends, ``*CC(O)C*``, has two branches that are
identical inside the fragment and distinct outside it, so the tag is dropped
when the fragment is canonicalised -- correctly, since the fragment alone does
not have a stereocentre there. 41 of the MC4R set's 156 side-chain
stereocentres are of this kind, and **no** fragment-local residue key can
express them, configuration-aware or not. They are covered by the second
measurement here, which compares whole molecules: a pair of molecules with one
constitution, one alpha-carbon assignment and two canonical SMILES differs in
side-chain configuration *somewhere*, however that configuration is carried.
That measurement is immune to the cut.

**Reproduce:** ``python3 -m atlas.ml.side_chain_stereochemistry``
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from rdkit import Chem, RDLogger

from atlas.ml.peptide_structure import PeptideAnalysis, side_chain_atoms
from atlas.ml.stereochemistry import Configuration

RDLogger.DisableLog("rdApp.*")

__all__ = [
    "SideChainFragment", "side_chain_fragments", "side_chain_collisions",
    "side_chain_stereoisomers", "side_chain_census", "main",
]


@dataclass(frozen=True)
class SideChainFragment:
    """One residue's side chain, with and without its stereochemistry.

    ``stereo_smiles`` is what a configuration-aware residue key would use and
    ``flat_smiles`` is what the deployed key uses. They differ exactly where
    the deployed key loses something.
    """

    #: Canonical SMILES of the cut fragment, chiral tags kept. Several
    #: fragments joined by "." where one alpha carbon carries two side chains
    #: (an alpha,alpha-disubstituted residue), and "" for glycine.
    stereo_smiles: str
    #: The same fragment with stereochemistry removed.
    flat_smiles: str
    #: Stereocentres among the side-chain atoms of the *intact* molecule,
    #: counted independently of the cut so that a tag the fragment loses is
    #: still visible in the census.
    n_specified_centres: int = 0
    n_unspecified_centres: int = 0

    @property
    def carries_configuration(self) -> bool:
        """Whether cutting with stereochemistry says more than cutting flat."""
        return self.stereo_smiles != self.flat_smiles

    @property
    def is_empty(self) -> bool:
        return not self.flat_smiles


#: Returned where the cut did not yield a sanitisable fragment. Counted in the
#: census rather than silently folded in with the glycines.
UNFRAGMENTED = SideChainFragment(stereo_smiles="?", flat_smiles="?")


def side_chain_fragments(mol: Chem.Mol) -> dict[int, SideChainFragment]:
    """Cut every traced side chain out of a molecule, keyed by alpha carbon.

    Lines up against ``PeptideAnalysis.alpha_carbon_indices``.
    """
    walks = side_chain_atoms(mol)
    centres = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False)
    specified = {i for i, tag in centres if tag != "?"}
    unspecified = {i for i, tag in centres if tag == "?"}
    n_atoms = mol.GetNumAtoms()

    out: dict[int, SideChainFragment] = {}
    for ca, walk in walks.items():
        if not walk.atoms:
            out[ca] = SideChainFragment("", "")        # glycine
            continue
        n_spec = len(walk.atoms & specified)
        n_unspec = len(walk.atoms & unspecified)
        cuts = sorted({
            mol.GetBondBetweenAtoms(i, nb.GetIdx()).GetIdx()
            for i in walk.atoms
            for nb in mol.GetAtomWithIdx(i).GetNeighbors()
            if nb.GetIdx() not in walk.atoms
        })
        try:
            # The bond to the alpha carbon always crosses the boundary, so
            # there is always at least one cut to make.
            cut = Chem.FragmentOnBonds(
                mol, cuts, addDummies=True, dummyLabels=[(0, 0)] * len(cuts))
            mapping: list[tuple[int, ...]] = []
            pieces = Chem.GetMolFrags(cut, asMols=True, sanitizeFrags=True,
                                      fragsMolAtomMapping=mapping)
        except Exception:
            out[ca] = UNFRAGMENTED
            continue
        ours, covered = [], set()
        for piece, atoms in zip(pieces, mapping):
            real = {a for a in atoms if a < n_atoms}
            if real & walk.atoms:
                ours.append(piece)
                covered |= real
        if not ours or covered != set(walk.atoms):
            # Cutting every boundary bond should isolate exactly these atoms;
            # if it did not, the fragment is not the side chain and saying so
            # is better than keying on whatever came back.
            out[ca] = UNFRAGMENTED
            continue
        stereo = ".".join(sorted(Chem.MolToSmiles(p) for p in ours))
        flats = []
        for piece in ours:
            bare = Chem.Mol(piece)
            Chem.RemoveStereochemistry(bare)
            flats.append(Chem.MolToSmiles(bare))
        out[ca] = SideChainFragment(stereo, ".".join(sorted(flats)),
                                    n_spec, n_unspec)
    return out


def _oriented(
    analysis: PeptideAnalysis, fragments: dict[int, SideChainFragment]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """(flat fragments, stereo fragments, alpha configurations), aligned.

    A head-to-tail cyclic backbone has no first residue, so its rotation is
    canonicalised the way ``peptide_representations._rotation`` does it --
    lexicographically smallest, on the *flat* key, and the resulting offset is
    applied to all three lists. Choosing the offset on the stereo key instead
    would let configuration rotate one molecule differently from its own flat
    twin, which would manufacture a difference out of the orientation.
    """
    pairs = [fragments.get(ca, UNFRAGMENTED)
             for ca in analysis.alpha_carbon_indices]
    flat = [f.flat_smiles for f in pairs]
    stereo = [f.stereo_smiles for f in pairs]
    cfgs = list(analysis.residue_configs[:len(pairs)])
    cfgs += [Configuration.UNSPECIFIED.value] * (len(pairs) - len(cfgs))
    i = 0
    if analysis.is_cyclic_backbone and len(flat) > 1:
        n = len(flat)
        i = min(range(n), key=lambda k: (tuple(flat[k:] + flat[:k]),
                                         tuple(cfgs[k:] + cfgs[:k])))
    rot = lambda xs: tuple(xs[i:] + xs[:i])                 # noqa: E731
    return rot(flat), rot(stereo), rot(cfgs)


def side_chain_collisions(
    analyses: Sequence[PeptideAnalysis], y: np.ndarray
) -> dict[str, Any]:
    """Groups a flat side-chain key merges that a stereo-aware one separates.

    The same comparison ``peptide_benchmark.representational_ceilings`` makes
    for the alpha carbon, on the side-chain axis: both keys carry the alpha
    configuration and the cut fragments, and they differ only in whether the
    fragments keep their chiral tags. A non-empty result would be a label
    spread no model on the deployed representation can avoid.
    """
    blind: dict[tuple, list[int]] = collections.defaultdict(list)
    aware: list[tuple] = []
    for i, a in enumerate(analyses):
        mol = Chem.MolFromSmiles(a.smiles)
        flat, stereo, cfgs = _oriented(a, side_chain_fragments(mol)
                                       if mol is not None else {})
        blind[(flat, cfgs)].append(i)
        aware.append((stereo, cfgs))

    groups, members_total, ranges, examples = 0, 0, [], []
    for members in blind.values():
        if len(members) < 2 or len({aware[i] for i in members}) < 2:
            continue
        groups += 1
        members_total += len(members)
        idx = np.asarray(members)
        ranges.append(float(y[idx].max() - y[idx].min()))
        examples.append({"members": members,
                         "pki": [float(v) for v in y[idx]]})
    return {
        "n_molecules": int(len(analyses)),
        "groups_merged_by_side_chain_blindness": groups,
        "molecules_affected": members_total,
        "molecules_affected_fraction": float(
            members_total / max(len(analyses), 1)),
        "median_pki_range_within_those_groups": (
            float(np.median(ranges)) if ranges else 0.0),
        "max_pki_range_within_those_groups": (
            float(max(ranges)) if ranges else 0.0),
        "examples": examples[:10],
        "note": (
            "Molecules that the deployed flat side-chain key merges and a "
            "configuration-aware side-chain key separates. Zero means the "
            "loss costs this dataset nothing, not that the key is complete."
        ),
    }


def side_chain_stereoisomers(
    analyses: Sequence[PeptideAnalysis],
    y: np.ndarray,
    ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Pairs that differ in configuration *outside* the alpha carbons.

    Molecules are grouped by their stereo-stripped canonical SMILES, which is
    one constitution exactly; inside a group a pair is separated by the
    deployed key when ``stereo_oriented_keys`` differs, and otherwise its two
    canonical SMILES differ somewhere ATLAS does not look -- a side-chain
    stereocentre, or a double bond's geometry.

    This is the measurement that decides whether the loss costs anything,
    because it reads whole molecules rather than cut fragments and so sees the
    context-dependent centres a fragment key cannot carry. ATLAS 2.5 found 25
    such groups and 33 pairs on the alpha-carbon axis; the question here is how
    many of those 33 the alpha-carbon assignment leaves merged.
    """
    import itertools

    from atlas.ml.peptide_representations import stereo_oriented_keys

    groups: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for i, a in enumerate(analyses):
        mol = Chem.MolFromSmiles(a.smiles)
        if mol is None:
            continue
        flat = Chem.Mol(mol)
        Chem.RemoveStereochemistry(flat)
        # One row per distinct canonical SMILES: a duplicated molecule is not
        # a stereoisomeric pair.
        groups[Chem.MolToSmiles(flat)].setdefault(Chem.MolToSmiles(mol), i)

    n_groups = n_pairs = separated = 0
    unseparated: list[dict[str, Any]] = []
    for members in groups.values():
        idx = list(members.values())
        if len(idx) < 2:
            continue
        n_groups += 1
        for i, j in itertools.combinations(idx, 2):
            n_pairs += 1
            if (tuple(stereo_oriented_keys(analyses[i]))
                    != tuple(stereo_oriented_keys(analyses[j]))):
                separated += 1
                continue
            unseparated.append({
                "molecules": [(ids[i] if ids is not None else i),
                              (ids[j] if ids is not None else j)],
                "pki": [float(y[i]), float(y[j])],
                "abs_delta_pki": float(abs(y[i] - y[j])),
            })

    deltas = [p["abs_delta_pki"] for p in unseparated]
    return {
        "constitutional_groups_with_more_than_one_configuration": n_groups,
        "stereoisomeric_pairs": n_pairs,
        "pairs_separated_by_the_alpha_carbon_key": separated,
        "pairs_needing_side_chain_configuration": len(unseparated),
        "pairs": unseparated[:25],
        "median_abs_delta_pki": float(np.median(deltas)) if deltas else 0.0,
        "max_abs_delta_pki": float(max(deltas)) if deltas else 0.0,
        "pairs_above_half_a_log_unit": int(sum(1 for d in deltas if d >= 0.5)),
        "note": (
            "A pair counted here has one constitution, one alpha-carbon "
            "assignment and two structures, so only side-chain "
            "stereochemistry (or double-bond geometry) tells it apart. Its "
            "measured pKi difference is error the deployed representation "
            "cannot avoid."
        ),
    }


def side_chain_census(
    analyses: Sequence[PeptideAnalysis],
    y: np.ndarray | None = None,
    ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Count the side-chain configuration the deployed key cannot express.

    Three questions, in order of how much they matter:

    1. How many residues carry a side-chain stereocentre at all.
    2. How many of those the deployed key cannot reconstruct. A key is only
       *ambiguous* when one (alpha configuration, flat fragment) pair occurs
       in two side-chain configurations. A D residue's fragment differing from
       its L counterpart's is not ambiguity: inverting the alpha carbon of
       isoleucine inverts its beta carbon with it, so the deployed key already
       implies the fragment.
    3. Whether any two molecules are separated by that ambiguity, which is
       the only thing a model could learn from.
    """
    per_flat: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter)
    per_alpha_flat: dict[tuple[str, str], collections.Counter] = (
        collections.defaultdict(collections.Counter))
    where: dict[tuple[str, str, str], list[dict[str, Any]]] = (
        collections.defaultdict(list))

    n_residues = n_spec = n_unspec = n_unfragmented = 0
    n_carrying = n_context_dependent = 0
    mols_with_centre = 0
    for mi, a in enumerate(analyses):
        mol = Chem.MolFromSmiles(a.smiles)
        frags = side_chain_fragments(mol) if mol is not None else {}
        hit = False
        for pos, ca in enumerate(a.alpha_carbon_indices):
            f = frags.get(ca, UNFRAGMENTED)
            n_residues += 1
            if f is UNFRAGMENTED:
                n_unfragmented += 1
                continue
            n_spec += bool(f.n_specified_centres)
            n_unspec += bool(f.n_unspecified_centres)
            n_carrying += bool(f.carries_configuration)
            n_context_dependent += bool(
                f.n_specified_centres and not f.carries_configuration)
            hit = hit or bool(f.n_specified_centres)
            cfg = (a.residue_configs[pos]
                   if pos < len(a.residue_configs)
                   else Configuration.UNSPECIFIED.value)
            per_flat[f.flat_smiles][f.stereo_smiles] += 1
            per_alpha_flat[(cfg, f.flat_smiles)][f.stereo_smiles] += 1
            where[(cfg, f.flat_smiles, f.stereo_smiles)].append({
                "molecule": (ids[mi] if ids is not None else mi),
                "position": pos + 1,
                "residue_key": (a.residue_keys[pos]
                                if pos < len(a.residue_keys) else "?"),
                "pki": (float(y[mi]) if y is not None else None),
            })
        mols_with_centre += hit

    ambiguous = []
    for (cfg, flat), stereo_counts in sorted(per_alpha_flat.items()):
        if len(stereo_counts) < 2:
            continue
        minority = sorted(stereo_counts.items(), key=lambda kv: kv[1])[:-1]
        ambiguous.append({
            "alpha_configuration": cfg,
            "flat_fragment": flat,
            "configurations": dict(stereo_counts.most_common()),
            "n_residues": int(sum(stereo_counts.values())),
            "n_residues_in_the_minority_configuration": int(
                sum(n for _s, n in minority)),
            "minority_carriers": [
                c for stereo, _n in minority
                for c in where[(cfg, flat, stereo)]
            ],
        })

    out: dict[str, Any] = {
        "n_molecules": int(len(analyses)),
        "n_residues_traced": n_residues,
        "residues_with_a_side_chain_stereocentre": n_spec,
        "residues_with_an_unspecified_side_chain_stereocentre": n_unspec,
        "residues_whose_cut_fragment_keeps_stereochemistry": n_carrying,
        "residues_whose_stereocentre_is_context_dependent": n_context_dependent,
        "molecules_with_a_side_chain_stereocentre": mols_with_centre,
        "residues_the_cut_could_not_isolate": n_unfragmented,
        "distinct_flat_fragments": len(per_flat),
        "flat_fragments_seen_in_more_than_one_configuration": sum(
            1 for c in per_flat.values() if len(c) > 1),
        "ambiguous_keys": ambiguous,
        "residues_on_an_ambiguous_key": int(
            sum(k["n_residues"] for k in ambiguous)),
        "residues_the_alpha_axis_cannot_imply": int(
            sum(k["n_residues_in_the_minority_configuration"]
                for k in ambiguous)),
        "provenance": (
            "Configurations are read from the deposited structures, not "
            "determined experimentally by ATLAS. A minority configuration may "
            "equally be a depositor's transcription of the majority one; this "
            "census counts what the records say."
        ),
    }
    if y is not None:
        out["stereoisomers"] = side_chain_stereoisomers(analyses, y, ids)
        out["collisions"] = side_chain_collisions(analyses, y)
    return out


def _load() -> tuple[list[PeptideAnalysis], np.ndarray, list[str]]:
    from atlas.ml.dataset import load_dataset
    from atlas.ml.features import featurize
    from atlas.ml.peptide_structure import analyse_peptide

    fs = featurize(load_dataset().records)
    analyses = [analyse_peptide(s) for s in fs.smiles]
    keep = np.array([a.is_peptide for a in analyses])
    return ([a for a, k in zip(analyses, keep) if k], fs.y[keep],
            [i for i, k in zip(fs.ids, keep) if k])


def main() -> int:
    analyses, y, ids = _load()
    c = side_chain_census(analyses, y, ids)
    print(f"{c['n_molecules']} peptides, {c['n_residues_traced']} residues "
          f"({c['residues_the_cut_could_not_isolate']} not isolable)")
    print(f"  side-chain stereocentres          : "
          f"{c['residues_with_a_side_chain_stereocentre']} residues in "
          f"{c['molecules_with_a_side_chain_stereocentre']} molecules "
          f"({c['residues_with_an_unspecified_side_chain_stereocentre']} "
          f"unspecified)")
    kept = (c["residues_with_a_side_chain_stereocentre"]
            - c["residues_whose_stereocentre_is_context_dependent"])
    print(f"  of those, carried by a cut fragment: {kept}; "
          f"{c['residues_whose_stereocentre_is_context_dependent']} are "
          f"stereocentres only in the whole molecule, which no fragment key "
          f"can carry")
    print(f"  fragments keeping any stereochemistry: "
          f"{c['residues_whose_cut_fragment_keeps_stereochemistry']} "
          f"(the {kept} above plus side-chain double bonds)")
    print(f"  distinct flat fragments           : "
          f"{c['distinct_flat_fragments']}, of which "
          f"{c['flat_fragments_seen_in_more_than_one_configuration']} occur "
          f"in more than one configuration")
    print(f"  ambiguous (alpha, fragment) keys  : {len(c['ambiguous_keys'])}, "
          f"covering {c['residues_on_an_ambiguous_key']} residues; "
          f"{c['residues_the_alpha_axis_cannot_imply']} of them carry a "
          f"configuration the alpha axis cannot imply")
    for k in c["ambiguous_keys"]:
        print(f"    alpha {k['alpha_configuration']} {k['flat_fragment']}: "
              f"{k['configurations']}")
        for carrier in k["minority_carriers"]:
            print(f"      {carrier['molecule']} position "
                  f"{carrier['position']} ({carrier['residue_key']}), pKi "
                  f"{carrier['pki']:.2f}")
    si = c["stereoisomers"]
    print(f"  stereoisomeric pairs              : "
          f"{si['stereoisomeric_pairs']} in "
          f"{si['constitutional_groups_with_more_than_one_configuration']} "
          f"constitutional groups; "
          f"{si['pairs_separated_by_the_alpha_carbon_key']} separated by the "
          f"alpha-carbon key, "
          f"{si['pairs_needing_side_chain_configuration']} needing "
          f"side-chain configuration")
    for pair in si["pairs"]:
        print(f"    {pair['molecules'][0]} vs {pair['molecules'][1]}: pKi "
              f"{pair['pki'][0]:.2f} / {pair['pki'][1]:.2f} "
              f"(|d| {pair['abs_delta_pki']:.2f})")
    if si["pairs_needing_side_chain_configuration"]:
        print(f"    median |dpKi| {si['median_abs_delta_pki']:.2f}, max "
              f"{si['max_abs_delta_pki']:.2f}, "
              f"{si['pairs_above_half_a_log_unit']} at or above 0.5 log units")
    col = c["collisions"]
    print(f"  groups merged by side-chain stereo-blindness: "
          f"{col['groups_merged_by_side_chain_blindness']} "
          f"({col['molecules_affected']} molecules)")
    if (col["groups_merged_by_side_chain_blindness"]
            or si["pairs_needing_side_chain_configuration"]):
        for e in col["examples"]:
            print(f"    {e['members']} pKi {e['pki']}")
        print("  The loss separates molecules: the fragment rebuild and the "
              "fingerprint generator should be made configuration-aware.")
    else:
        print("  The loss separates no two molecules in this dataset. Making "
              "the key configuration-aware would change no row, so it is not "
              "done on the strength of this census.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
