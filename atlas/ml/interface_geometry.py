"""ATLAS 2.5 Stage 2: renderable geometry for one solved interface.

:mod:`atlas.ml.interaction_features` types the interactions at each MC4R
peptide interface and saves their *counts*; :mod:`atlas.ml.receptor_pocket`
saves, per structure, the minimum heavy-atom distance from each pocket
position to the peptide. Neither saves the coordinates, because a model does
not need them. An interface explorer does: to draw a contact you need the two
atoms it runs between, and to highlight a residue you need where it is.

This module reads the cached mmCIF for one complex and emits the per-pair
detail and the coordinates behind the saved counts, in a form a browser can
draw directly.

Three properties make it safe to show:

* **It recomputes nothing new.** Chain identification, atom typing, the
  distance criteria and the pair enumeration are the same code and the same
  constants that produced ``interaction_profiles.json``. Every count it
  derives is checked against the saved profile and the comparison is returned
  in :attr:`InterfaceGeometry.agreement`, so a drawing that disagrees with the
  published profile announces itself rather than quietly replacing it.
* **It keeps the candidate/inferred distinction.** Hydrogen bonds are emitted
  under the key ``hbond_candidate`` and carry
  :data:`HBOND_CANDIDATE_CAVEAT`. These structures are 2.58-3.30 A and hold no
  hydrogens, so no D-H...A angle was ever checked. Peptide-side charged groups
  and rings are geometric inferences, and the pairs that rest on them carry
  ``inferred_peptide_chemistry``.
* **It is retrospective only.** Like the profiles, every number here comes from
  a solved complex and does not exist for an unsolved pair.
  :func:`atlas.ml.interaction_features.assert_retrospective_only` governs its
  mode as it governs theirs.

Coordinates are translated so the interface centroid is the origin and rounded
to 0.01 A. The translation is a rendering convenience; the rounding is two
orders of magnitude finer than the resolution of any of these structures. No
rotation and no scaling is applied, and the translation is reported in
:attr:`InterfaceGeometry.centroid` so the deposited frame can be recovered.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from atlas.ml.interaction_features import (
    _AROM_MAX, _CATION_PI_MAX, _HBOND_MAX, _PHOBIC_MAX, _SALT_MAX,
    CRITERIA, InferenceMode, _group_min_distance, _infer_bonds,
    _interplanar_angle, _pairs_within, _ring_frame, _type_peptide,
    _type_receptor, assert_retrospective_only,
)
from atlas.ml.receptor_pocket import (
    MC4R_PEPTIDE_COMPLEXES, PEPTIDE_MAX_RES, PEPTIDE_MIN_RES, STRUCTURE_CACHE,
    _aligner, _parse_structure, _THREE_TO_ONE,
)

logger = logging.getLogger(__name__)

__all__ = [
    "InterfaceGeometry", "InterfaceGeometryUnavailable", "INTERACTION_KINDS",
    "HBOND_CANDIDATE_CAVEAT", "INFERRED_CHEMISTRY_CAVEAT",
    "geometry_for_structure", "available_structures",
]

#: Radius within which a receptor residue is typed, matching
#: :func:`atlas.ml.interaction_features.profile_structure`. Residues outside it
#: cannot reach the peptide under any of the criteria above, so excluding them
#: changes no count; it only keeps the payload to the part of the receptor the
#: peptide is near.
INTERFACE_SHELL_A = 8.0

#: Coordinate precision in the emitted payload, in Angstrom.
COORD_DECIMALS = 2

HBOND_CANDIDATE_CAVEAT = (
    "Candidate, not an established hydrogen bond. These structures carry no "
    "hydrogens, so the donor-hydrogen-acceptor angle was never evaluated and "
    "a heavy-atom distance alone over-counts."
)

INFERRED_CHEMISTRY_CAVEAT = (
    "The peptide-side group in this pair is inferred from element and "
    "distance-inferred connectivity, not read from a residue name. These "
    "ligands are largely non-canonical, so names carry no chemistry."
)

#: Every interaction kind this module emits, with what the drawn mark means.
#: ``evidence`` is the honest status of the *typing*, not of the structure:
#: every structure here is experimental, and these are its geometry.
INTERACTION_KINDS: dict[str, dict[str, Any]] = {
    "hbond_candidate": {
        "label": "Hydrogen-bond candidate",
        "criterion": f"donor and acceptor heavy atoms within {_HBOND_MAX} A",
        "evidence": "candidate",
        "caveat": HBOND_CANDIDATE_CAVEAT,
        "profile_field": "hbond_candidates",
    },
    "salt_bridge": {
        "label": "Salt bridge",
        "criterion": f"oppositely charged groups within {_SALT_MAX} A",
        "evidence": "geometric",
        "caveat": INFERRED_CHEMISTRY_CAVEAT,
        "profile_field": "salt_bridges",
    },
    "hydrophobic": {
        "label": "Hydrophobic contact",
        "criterion": f"apolar carbons within {_PHOBIC_MAX} A",
        "evidence": "geometric",
        "caveat": None,
        "profile_field": "hydrophobic_contacts",
    },
    "aromatic_stacked": {
        "label": "Aromatic stacking",
        "criterion": (
            f"ring centroids within {_AROM_MAX} A, interplanar angle at most "
            f"{CRITERIA['aromatic_stacked_max_deg']} deg"),
        "evidence": "geometric",
        "caveat": INFERRED_CHEMISTRY_CAVEAT,
        "profile_field": "aromatic_stacked",
    },
    "aromatic_tshaped": {
        "label": "Aromatic T-shaped",
        "criterion": (
            f"ring centroids within {_AROM_MAX} A, interplanar angle at least "
            f"{CRITERIA['aromatic_tshaped_min_deg']} deg"),
        "evidence": "geometric",
        "caveat": INFERRED_CHEMISTRY_CAVEAT,
        "profile_field": "aromatic_tshaped",
    },
    "aromatic_oblique": {
        "label": "Aromatic oblique",
        "criterion": "ring centroids within 6.0 A, interplanar angle between",
        "evidence": "geometric",
        "caveat": INFERRED_CHEMISTRY_CAVEAT,
        "profile_field": "aromatic_oblique",
    },
    "cation_pi": {
        "label": "Cation-pi",
        "criterion": f"cationic group within {_CATION_PI_MAX} A of a ring centroid",
        "evidence": "geometric",
        "caveat": INFERRED_CHEMISTRY_CAVEAT,
        "profile_field": "cation_pi",
    },
}


class InterfaceGeometryUnavailable(RuntimeError):
    """Raised when the coordinates for a structure are not on disk.

    Coordinate files are gitignored, so a fresh checkout has none. That is a
    recoverable state, not an error in the data: ``fetch_structure`` restores
    the file. The explorer reports it as an absent input rather than drawing
    something else and calling it the structure.
    """


@dataclass
class InterfaceGeometry:
    """Drawable geometry and per-pair detail for one solved complex."""

    pdb_id: str
    ligand: str
    method: str
    resolution: float | None
    receptor_chain: str
    peptide_chain: str
    mode: str = InferenceMode.RETROSPECTIVE_STRUCTURE.value

    #: Translation applied to every coordinate, in the deposited frame.
    centroid: list[float] = field(default_factory=list)
    #: Largest distance from the origin after translation, for view fitting.
    extent_a: float = 0.0

    #: C-alpha trace of the receptor residues that aligned to the reference.
    receptor_trace: list[dict[str, Any]] = field(default_factory=list)
    #: Heavy atoms of the receptor residues inside the interface shell.
    receptor_residues: list[dict[str, Any]] = field(default_factory=list)
    #: The peptide ligand: residues, heavy atoms and inferred bonds.
    peptide: dict[str, Any] = field(default_factory=dict)

    #: Per-pair detail keyed by :data:`INTERACTION_KINDS`.
    interactions: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    #: Derived count against saved profile count, per kind.
    agreement: dict[str, dict[str, Any]] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)

    @property
    def agrees_with_profile(self) -> bool:
        return all(a["agrees"] for a in self.agreement.values())


def available_structures() -> dict[str, bool]:
    """Which complexes have coordinates cached, in the catalogue's order."""
    return {
        pdb_id: (STRUCTURE_CACHE / f"{pdb_id}.cif").exists()
        for pdb_id in MC4R_PEPTIDE_COMPLEXES
    }


def _round(vec: Sequence[float]) -> list[float]:
    return [round(float(v), COORD_DECIMALS) for v in vec]


def _auth_from_label(label: str) -> int | None:
    """Author residue number out of a receptor atom label like ``D122:OD2``.

    Receptor labels are built from the author numbering, which agrees with
    UniProt in the cryo-EM entries and not in the X-ray fusion constructs. The
    number is therefore only an index into the alignment map, never a position
    in its own right.
    """
    head = label.split(":", 1)[0]
    digits = "".join(c for c in head if c.isdigit())
    return int(digits) if digits else None


def _identify_chains(model, reference_sequence: str):
    """The receptor and peptide chains of one model.

    Deliberately the same procedure as
    :func:`atlas.ml.interaction_features.profile_structure`: the receptor is
    the chain scoring highest against the reference sequence, the peptide the
    shortest polymer in the permitted size range. Identifying them any other
    way could pair this module's coordinates with a different chain than the
    saved profile's counts, which is the one inconsistency the agreement check
    could not detect.
    """
    local = _aligner("local")
    receptor = None
    best = 0.0
    sizes: list[tuple[str, int]] = []
    for chain in model:
        residues = [r for r in chain if r.id[0] != "W"]
        standard = [r for r in residues if r.get_resname() in _THREE_TO_ONE]
        if len(standard) > 5:
            score = local.score(
                reference_sequence,
                "".join(_THREE_TO_ONE[r.get_resname()] for r in standard))
            if score > best:
                receptor, best = chain, score
        sizes.append((chain.id, len(residues)))
    candidates = [
        c for c in sizes
        if PEPTIDE_MIN_RES <= c[1] <= PEPTIDE_MAX_RES
        and (receptor is None or c[0] != receptor.id)
    ]
    if receptor is None or not candidates:
        raise ValueError("cannot identify receptor and peptide chains")
    peptide = model[min(candidates, key=lambda c: c[1])[0]]
    return receptor, peptide, local


def _position_map(residues: Sequence[Any], reference_sequence: str, local) -> dict[int, int]:
    """Author residue number to 1-based position in the reference sequence.

    The same local alignment :mod:`atlas.ml.receptor_pocket` uses, so a
    position drawn here is the position the pocket's support counts are keyed
    by. Biopython returns numpy integers; they are cast, because a position
    serialised as ``43.0`` fails every integer lookup downstream.
    """
    observed = "".join(_THREE_TO_ONE[r.get_resname()] for r in residues)
    alignment = local.align(reference_sequence, observed)[0]
    out: dict[int, int] = {}
    for (ref_start, ref_end), (obs_start, _obs_end) in zip(*alignment.aligned):
        for k in range(int(ref_end) - int(ref_start)):
            out[int(residues[int(obs_start) + k].id[1])] = int(ref_start) + k + 1
    return out


def _pair(
    rec, pep, i: int, j: int, distance: float, position: dict[int, int],
    shift: np.ndarray,
) -> dict[str, Any]:
    """One receptor-atom to peptide-atom pair, as a drawable segment."""
    auth = _auth_from_label(rec.labels[i])
    return {
        "receptor_atom": rec.labels[i],
        "peptide_atom": pep.labels[j],
        "receptor_auth": auth,
        "receptor_position": position.get(auth) if auth is not None else None,
        "distance_a": round(float(distance), 2),
        "a": _round(rec.coords[i] - shift),
        "b": _round(pep.coords[j] - shift),
    }


def _group_pair(
    rec, pep, group_rec: Sequence[int], group_pep: Sequence[int],
    distance: float, kind: str, position: dict[int, int], shift: np.ndarray,
) -> dict[str, Any]:
    """One charged-group to charged-group pair, drawn centroid to centroid."""
    auths = {a for a in (_auth_from_label(rec.labels[i]) for i in group_rec)
             if a is not None}
    rec_centre = rec.coords[list(group_rec)].mean(axis=0)
    pep_centre = pep.coords[list(group_pep)].mean(axis=0)
    return {
        "kind": kind,
        "receptor_atoms": [rec.labels[i] for i in group_rec],
        "peptide_atoms": [pep.labels[j] for j in group_pep],
        "receptor_auth": min(auths) if auths else None,
        "receptor_position": (
            position.get(min(auths)) if auths else None),
        "distance_a": round(float(distance), 2),
        "a": _round(rec_centre - shift),
        "b": _round(pep_centre - shift),
    }


def _ring_pair(
    rec, pep, rec_ring: Sequence[int], pep_ring: Sequence[int],
    rec_centroid: np.ndarray, pep_centroid: np.ndarray, angle: float,
    position: dict[int, int], shift: np.ndarray,
) -> dict[str, Any]:
    """One ring-to-ring pair, drawn centroid to centroid with its angle."""
    auths = {a for a in (_auth_from_label(rec.labels[i]) for i in rec_ring)
             if a is not None}
    return {
        "receptor_atoms": [rec.labels[i] for i in rec_ring],
        "peptide_atoms": [pep.labels[j] for j in pep_ring],
        "receptor_auth": min(auths) if auths else None,
        "receptor_position": position.get(min(auths)) if auths else None,
        "distance_a": round(float(np.linalg.norm(rec_centroid - pep_centroid)), 2),
        "interplanar_deg": round(float(angle), 1),
        "a": _round(rec_centroid - shift),
        "b": _round(pep_centroid - shift),
    }


def _typed_interactions(rec, pep, position: dict[int, int], shift: np.ndarray):
    """Enumerate every pair behind the saved counts, by kind.

    The enumeration mirrors ``profile_structure`` statement for statement. It
    is written out again rather than imported because that function returns
    totals and discards the pairs; the criteria and the typed atom sets it uses
    are imported, so there is one definition of each threshold.
    """
    empty = np.zeros((0, 3))
    out: dict[str, list[dict[str, Any]]] = {k: [] for k in INTERACTION_KINDS}

    # Hydrogen-bond candidates, both directions. The donor/acceptor index
    # lists are views into the atom set, so a pair index must be mapped back
    # through them before it names an atom.
    for rec_idx, pep_idx in ((rec.donors, pep.acceptors), (rec.acceptors, pep.donors)):
        pairs = _pairs_within(
            rec.coords[rec_idx] if rec_idx else empty,
            pep.coords[pep_idx] if pep_idx else empty, _HBOND_MAX)
        for i, j, d in pairs:
            out["hbond_candidate"].append(
                _pair(rec, pep, rec_idx[i], pep_idx[j], d, position, shift))

    # Salt bridges, charged group to oppositely charged group.
    for groups_a, groups_b, sign in ((rec.anionic, pep.cationic, "receptor_anionic"),
                                     (rec.cationic, pep.anionic, "receptor_cationic")):
        for ga, gb in itertools.product(groups_a, groups_b):
            d = _group_min_distance(rec.coords, ga, pep.coords, gb)
            if d <= _SALT_MAX:
                out["salt_bridge"].append(
                    _group_pair(rec, pep, ga, gb, d, sign, position, shift))

    # Hydrophobic contacts.
    pairs = _pairs_within(
        rec.coords[rec.apolar_carbons] if rec.apolar_carbons else empty,
        pep.coords[pep.apolar_carbons] if pep.apolar_carbons else empty, _PHOBIC_MAX)
    for i, j, d in pairs:
        out["hydrophobic"].append(_pair(
            rec, pep, rec.apolar_carbons[i], pep.apolar_carbons[j], d, position, shift))

    # Aromatic interactions, classified by interplanar angle.
    rec_frames = [(_ring_frame(rec.coords, r), r) for r in rec.rings]
    pep_frames = [(_ring_frame(pep.coords, r), r) for r in pep.rings]
    for ((c1, n1), r1), ((c2, n2), r2) in itertools.product(rec_frames, pep_frames):
        if float(np.linalg.norm(c1 - c2)) > _AROM_MAX:
            continue
        angle = _interplanar_angle(n1, n2)
        if angle <= CRITERIA["aromatic_stacked_max_deg"]:
            kind = "aromatic_stacked"
        elif angle >= CRITERIA["aromatic_tshaped_min_deg"]:
            kind = "aromatic_tshaped"
        else:
            kind = "aromatic_oblique"
        out[kind].append(
            _ring_pair(rec, pep, r1, r2, c1, c2, angle, position, shift))

    # Cation-pi, in both directions. The second direction puts the cationic
    # group on the peptide, so the drawn segment runs peptide-to-receptor and
    # its endpoints are swapped back to keep ``a`` on the receptor side.
    for cationic_on_receptor in (True, False):
        groups = rec.cationic if cationic_on_receptor else pep.cationic
        coords = rec.coords if cationic_on_receptor else pep.coords
        frames = pep_frames if cationic_on_receptor else rec_frames
        for group, (((centroid, _n)), ring) in itertools.product(groups, frames):
            pts = coords[list(group)]
            d = float(np.sqrt(((pts - centroid) ** 2).sum(-1)).min())
            if d > _CATION_PI_MAX:
                continue
            group_centre = pts.mean(axis=0)
            if cationic_on_receptor:
                auths = {a for a in (_auth_from_label(rec.labels[i]) for i in group)
                         if a is not None}
                entry = {
                    "receptor_atoms": [rec.labels[i] for i in group],
                    "peptide_atoms": [pep.labels[j] for j in ring],
                    "receptor_auth": min(auths) if auths else None,
                    "receptor_position": position.get(min(auths)) if auths else None,
                    "a": _round(group_centre - shift),
                    "b": _round(centroid - shift),
                }
            else:
                auths = {a for a in (_auth_from_label(rec.labels[i]) for i in ring)
                         if a is not None}
                entry = {
                    "receptor_atoms": [rec.labels[i] for i in ring],
                    "peptide_atoms": [pep.labels[j] for j in group],
                    "receptor_auth": min(auths) if auths else None,
                    "receptor_position": position.get(min(auths)) if auths else None,
                    "a": _round(centroid - shift),
                    "b": _round(group_centre - shift),
                }
            entry["distance_a"] = round(d, 2)
            entry["cation_on"] = "receptor" if cationic_on_receptor else "peptide"
            out["cation_pi"].append(entry)

    return out


def _peptide_payload(pep, pep_res: Sequence[Any], shift: np.ndarray) -> dict[str, Any]:
    """The peptide as atoms, inferred bonds and residue spans.

    Atom order is ``_type_peptide``'s own, so an interaction's peptide index
    and this list's index are the same thing. Bonds are the distance-inferred
    connectivity the typing already relies on, at the same 1.95 A cutoff; they
    are a drawing aid, not a statement about bond orders.
    """
    atoms = [
        {"name": pep.labels[i].split(":", 1)[1], "el": pep.elements[i],
         "xyz": _round(pep.coords[i] - shift)}
        for i in range(len(pep.coords))
    ]
    neigh = _infer_bonds(pep.coords, pep.elements) if len(pep.coords) else []
    bonds = sorted({(i, j) for i, ns in enumerate(neigh) for j in ns if i < j})

    residues: list[dict[str, Any]] = []
    cursor = 0
    for res in pep_res:
        heavy = [a for a in res if a.element != "H"]
        residues.append({
            "name": res.get_resname(),
            "auth": int(res.id[1]),
            "atom_from": cursor,
            "atom_to": cursor + len(heavy),
            "is_canonical": res.get_resname() in _THREE_TO_ONE,
        })
        cursor += len(heavy)
    return {
        "atoms": atoms,
        "bonds": [[int(i), int(j)] for i, j in bonds],
        "residues": residues,
        "rings": [[int(i) for i in r] for r in pep.rings],
        "cationic_groups": [[int(i) for i in g] for g in pep.cationic],
        "anionic_groups": [[int(i) for i in g] for g in pep.anionic],
    }


def geometry_for_structure(
    pdb_id: str,
    reference_sequence: str,
    *,
    profile: dict[str, Any] | None = None,
    path: Path | None = None,
) -> InterfaceGeometry:
    """Read one cached complex and emit its drawable interface.

    ``profile`` is the saved entry from ``interaction_profiles.json`` for this
    structure. It is not used to compute anything: it is the published result
    the derived counts are checked against.
    """
    assert_retrospective_only(InferenceMode.RETROSPECTIVE_STRUCTURE)
    path = path or (STRUCTURE_CACHE / f"{pdb_id}.cif")
    if not path.exists() or path.stat().st_size == 0:
        raise InterfaceGeometryUnavailable(
            f"{pdb_id}: no coordinate file at {path}. Coordinate files are "
            "cached outside version control; atlas.ml.receptor_pocket."
            "fetch_structure re-downloads one."
        )

    model = _parse_structure(path)
    receptor, peptide, local = _identify_chains(model, reference_sequence)

    pep_res = [r for r in peptide if r.id[0] != "W"]
    rec_res = [r for r in receptor if r.get_resname() in _THREE_TO_ONE]
    position = _position_map(rec_res, reference_sequence, local)

    pep_coords = np.array([a.coord for r in pep_res for a in r if a.element != "H"])
    if not len(pep_coords):
        raise ValueError(f"{pdb_id}: peptide chain has no heavy atoms")

    near: list[Any] = []
    for res in rec_res:
        coords = np.array([a.coord for a in res if a.element != "H"])
        if not len(coords):
            continue
        d = np.sqrt(((coords[:, None, :] - pep_coords[None, :, :]) ** 2).sum(-1)).min()
        if d <= INTERFACE_SHELL_A:
            near.append(res)

    rec = _type_receptor(near)
    pep = _type_peptide(pep_res)

    shift = np.concatenate([rec.coords, pep.coords]).mean(axis=0)
    meta = MC4R_PEPTIDE_COMPLEXES.get(pdb_id, {})

    geometry = InterfaceGeometry(
        pdb_id=pdb_id,
        ligand=meta.get("ligand", "?"),
        method=meta.get("method", "?"),
        resolution=meta.get("resolution"),
        receptor_chain=receptor.id,
        peptide_chain=peptide.id,
        centroid=_round(shift),
    )

    # The C-alpha trace covers the whole aligned chain, not just the shell, so
    # the drawn receptor reads as a receptor rather than as a cloud of pocket
    # residues. Residues that did not align carry a null position: they are
    # the fusion partner, and naming them from the reference would be wrong.
    for res in rec_res:
        ca = res["CA"] if "CA" in res else None
        if ca is None:
            continue
        auth = int(res.id[1])
        geometry.receptor_trace.append({
            "auth": auth,
            "position": position.get(auth),
            "observed": _THREE_TO_ONE[res.get_resname()],
            "xyz": _round(ca.coord - shift),
        })

    for res in near:
        auth = int(res.id[1])
        pos = position.get(auth)
        heavy = [a for a in res if a.element != "H"]
        coords = np.array([a.coord for a in heavy]) if heavy else np.zeros((0, 3))
        # Bonds within the residue, at the same inferred-connectivity cutoff the
        # typing uses. They exist so a side chain can be drawn as a stick figure
        # rather than a dot cloud; no bond order is claimed.
        neigh = _infer_bonds(coords, [a.element for a in heavy]) if len(coords) else []
        geometry.receptor_residues.append({
            "auth": auth,
            "position": pos,
            "observed": _THREE_TO_ONE[res.get_resname()],
            "reference": (
                reference_sequence[pos - 1] if pos is not None
                and pos - 1 < len(reference_sequence) else None),
            "atoms": [
                {"name": a.get_id(), "el": a.element, "xyz": _round(a.coord - shift)}
                for a in heavy
            ],
            "bonds": sorted(
                [int(i), int(j)] for i, ns in enumerate(neigh) for j in ns if i < j),
        })

    geometry.peptide = _peptide_payload(pep, pep_res, shift)
    geometry.interactions = _typed_interactions(rec, pep, position, shift)

    everything = np.concatenate([rec.coords, pep.coords]) - shift
    geometry.extent_a = round(float(np.sqrt((everything ** 2).sum(-1)).max()), 2)

    saved = profile or {}
    for kind, spec in INTERACTION_KINDS.items():
        derived = len(geometry.interactions[kind])
        published = saved.get(spec["profile_field"])
        geometry.agreement[kind] = {
            "derived": derived,
            "published": published,
            "agrees": published is None or int(published) == derived,
        }

    # Caveats here are about this extraction. The candidate and inferred
    # qualifications belong to individual marks and travel with them, on the
    # ``caveat`` field of each entry in INTERACTION_KINDS, so that a reader
    # meets them where the mark is shown rather than in a list at the bottom.
    geometry.caveats = [
        f"Coordinates are translated by the interface centroid "
        f"{geometry.centroid} and rounded to {COORD_DECIMALS} decimal places. "
        "No rotation or scaling is applied.",
        "Each interaction kind carries its own qualification: hydrogen bonds "
        "are candidates, and peptide-side charged groups and rings are "
        "inferred from geometry. See the criterion attached to each kind.",
    ]
    mismatched = [k for k, a in geometry.agreement.items() if not a["agrees"]]
    if mismatched:
        geometry.caveats.append(
            "Derived pair counts disagree with the saved profile for: "
            + ", ".join(mismatched)
            + ". The saved profile is the published result; treat the drawing "
            "as unverified until the disagreement is explained."
        )
        logger.warning("%s: geometry disagrees with profile for %s",
                       pdb_id, ", ".join(mismatched))
    return geometry
