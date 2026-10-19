"""ATLAS 2.5 Stage 2: typed noncovalent interaction descriptors.

The pocket module records which receptor residues sit within 4.5 A of a bound
peptide. That is bare proximity: it does not say whether a contact is a
hydrogen bond, a salt bridge, an aromatic stack or simple apolar packing, and
those have different chemistry. This module types them, and adds solvent
accessibility and interface geometry.

**These features are retrospective by construction.** Every one is computed
from an experimentally determined bound complex. For a peptide-target pair
with no solved structure they do not exist, and a model trained on them could
not be applied prospectively. The brief is explicit that geometry-based
retrospective analysis must not leak into prediction for unseen pairs, so the
output type carries that restriction in its own right: see
:class:`InteractionProfile.mode` and :func:`assert_retrospective_only`.

Criteria are conventional and are recorded with each profile rather than left
implicit:

* **Hydrogen bond** — donor and acceptor heavy atoms within 3.5 A. The
  structures here are 2.58-3.30 A and carry **no hydrogens**, so the usual
  D-H...A angular criterion cannot be evaluated. A heavy-atom distance alone
  over-counts: it admits donor-acceptor pairs whose geometry forbids a bond.
  The count is therefore reported as ``hbond_candidates``, not hydrogen bonds.
* **Salt bridge** — a charged group's atoms within 4.0 A of an oppositely
  charged group's.
* **Hydrophobic contact** — apolar carbons (carbon bonded to no N or O) within
  4.5 A.
* **Aromatic** — ring centroids within 6.0 A, classified by interplanar angle:
  stacked below 30 degrees, T-shaped above 60, oblique between. Cation-pi is a
  cationic group within 6.0 A of a ring centroid.
* **Buried surface area** — Shrake-Rupley, as
  ``(SASA_receptor + SASA_peptide - SASA_complex) / 2``. The complex is the
  receptor and peptide chains only: a G protein or nanobody elsewhere in the
  deposit is not part of this interface and including it would inflate the
  figure.

Atom typing differs by side, deliberately. Receptor residues are always
standard amino acids here, so their charged groups and rings are identified by
atom name, which is exact. Peptide residues are frequently **not** standard --
norleucine, chlorinated phenylalanines, lactam bridges -- so names carry no
information, and their chemistry is inferred from element and
distance-inferred connectivity instead. That is the whole reason this module
cannot simply use a residue lookup table.
"""

from __future__ import annotations

import copy
import enum
import itertools
import json
import logging
import math
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from atlas.core.evidence import utc_now
from atlas.ml.dataset import DATA_DIR
from atlas.ml.receptor_pocket import (
    MC4R_PEPTIDE_COMPLEXES, _THREE_TO_ONE, fetch_structure,
)

logger = logging.getLogger(__name__)

__all__ = [
    "INTERACTION_PATH", "CRITERIA", "InferenceMode", "InteractionProfile",
    "profile_structure", "build_interaction_profiles", "load_profiles",
    "assert_retrospective_only", "ProspectiveLeakError",
]

INTERACTION_PATH = DATA_DIR / "interaction_profiles.json"

#: Every geometric threshold, in one place and saved with the output.
CRITERIA: dict[str, float | str] = {
    "hbond_donor_acceptor_max_a": 3.5,
    "hbond_angular_criterion": (
        "not applied: these structures carry no hydrogens, so D-H...A angles "
        "cannot be evaluated and the count is of candidates"
    ),
    "salt_bridge_max_a": 4.0,
    "hydrophobic_max_a": 4.5,
    "aromatic_centroid_max_a": 6.0,
    "aromatic_stacked_max_deg": 30.0,
    "aromatic_tshaped_min_deg": 60.0,
    "cation_pi_max_a": 6.0,
    "covalent_bond_max_a": 1.95,
    "sasa_algorithm": "Shrake-Rupley (Biopython), probe 1.40 A",
}

_HBOND_MAX = 3.5
_SALT_MAX = 4.0
_PHOBIC_MAX = 4.5
_AROM_MAX = 6.0
_CATION_PI_MAX = 6.0
_BOND_MAX = 1.95

#: Receptor side: exact, because every receptor residue here is standard.
_ANIONIC_ATOMS = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2")}
_CATIONIC_ATOMS = {
    ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"), ("LYS", "NZ"),
    ("HIS", "ND1"), ("HIS", "NE2"),
}
_RECEPTOR_RINGS: dict[str, tuple[tuple[str, ...], ...]] = {
    "PHE": (("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),),
    "TYR": (("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),),
    "TRP": (("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
            ("CG", "CD1", "CD2", "NE1", "CE2")),
    "HIS": (("CG", "ND1", "CD2", "CE1", "NE2"),),
}


class InferenceMode(str, enum.Enum):
    """Which features a prediction is allowed to use."""

    #: Sequence and chemistry only. Available for any peptide-target pair.
    PROSPECTIVE = "prospective"
    #: Adds features computed from an experimentally determined bound complex.
    #: Only meaningful where such a structure exists, and never a prediction
    #: about an unseen pair.
    RETROSPECTIVE_STRUCTURE = "retrospective_structure"


class ProspectiveLeakError(RuntimeError):
    """Raised when structure-derived features reach a prospective pathway."""


def assert_retrospective_only(mode: InferenceMode | str) -> None:
    """Guard every consumer of these features must pass through.

    Calling this with ``PROSPECTIVE`` raises rather than degrading quietly.
    A silent fallback is what would let a bound-structure feature into a
    de novo prediction and make the evaluation meaningless.
    """
    resolved = InferenceMode(mode)
    if resolved is not InferenceMode.RETROSPECTIVE_STRUCTURE:
        raise ProspectiveLeakError(
            "Interaction descriptors are computed from an experimentally "
            "determined bound complex. They cannot exist for an unseen "
            f"peptide-target pair, so mode {resolved.value!r} may not use "
            "them. Use InferenceMode.RETROSPECTIVE_STRUCTURE for analysis of "
            "a solved complex, and sequence or chemistry features for "
            "prediction."
        )


# ------------------------------------------------------------ atom typing ----

def _infer_bonds(coords: np.ndarray, elements: Sequence[str]) -> list[set[int]]:
    """Neighbour lists from interatomic distance.

    Needed because the peptide ligands are largely non-canonical, so atom
    names identify nothing and Biopython supplies no connectivity. A heavy-atom
    pair within 1.95 A is bonded; that admits S-S at ~2.05 A only marginally,
    and disulfides are reported separately by the structure module anyway.
    """
    n = len(coords)
    neigh: list[set[int]] = [set() for _ in range(n)]
    if n == 0:
        return neigh
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    for i, j in zip(*np.where((d > 0.4) & (d <= _BOND_MAX))):
        if i < j:
            neigh[int(i)].add(int(j))
            neigh[int(j)].add(int(i))
    return neigh


@dataclass
class _AtomSet:
    """Typed atoms of one molecular side of the interface."""

    coords: np.ndarray
    elements: list[str]
    labels: list[str]                       # human-readable residue+atom
    donors: list[int] = field(default_factory=list)
    acceptors: list[int] = field(default_factory=list)
    anionic: list[list[int]] = field(default_factory=list)   # groups
    cationic: list[list[int]] = field(default_factory=list)
    apolar_carbons: list[int] = field(default_factory=list)
    rings: list[list[int]] = field(default_factory=list)


def _type_receptor(residues: Sequence[Any]) -> _AtomSet:
    """Type receptor atoms by residue and atom name."""
    coords, elements, labels = [], [], []
    donors, acceptors, apolar = [], [], []
    anionic: list[list[int]] = []
    cationic: list[list[int]] = []
    rings: list[list[int]] = []

    for res in residues:
        name = res.get_resname()
        index_of: dict[str, int] = {}
        for atom in res:
            if atom.element == "H":
                continue
            idx = len(coords)
            coords.append(atom.coord)
            elements.append(atom.element)
            labels.append(f"{_THREE_TO_ONE.get(name, name)}{res.id[1]}:{atom.get_id()}")
            index_of[atom.get_id()] = idx
            if atom.element == "N":
                donors.append(idx)
                acceptors.append(idx)
            elif atom.element == "O":
                acceptors.append(idx)
                donors.append(idx)       # hydroxyls and waters donate too
            elif atom.element == "S":
                acceptors.append(idx)
            elif atom.element == "C":
                apolar.append(idx)

        grp = [index_of[a] for (r, a) in _ANIONIC_ATOMS
               if r == name and a in index_of]
        if grp:
            anionic.append(grp)
        grp = [index_of[a] for (r, a) in _CATIONIC_ATOMS
               if r == name and a in index_of]
        if grp:
            cationic.append(grp)
        for ring_names in _RECEPTOR_RINGS.get(name, ()):
            ring = [index_of[a] for a in ring_names if a in index_of]
            if len(ring) >= 5:
                rings.append(ring)

    # A carbon bonded to N or O is not apolar. Receptor backbone carbonyl and
    # alpha carbons would otherwise count as hydrophobic contact partners.
    arr = np.array(coords) if coords else np.zeros((0, 3))
    neigh = _infer_bonds(arr, elements)
    apolar = [
        i for i in apolar
        if not any(elements[j] in ("N", "O") for j in neigh[i])
    ]
    return _AtomSet(arr, elements, labels, donors, acceptors,
                    anionic, cationic, apolar, rings)


def _planar_ring(coords: np.ndarray, ring: Sequence[int], tol: float = 0.35) -> bool:
    """Whether a candidate cycle is planar enough to be aromatic."""
    pts = coords[list(ring)]
    centred = pts - pts.mean(axis=0)
    # Smallest singular value is the out-of-plane spread.
    return float(np.linalg.svd(centred, compute_uv=False)[-1]) <= tol * len(ring) ** 0.5


def _find_rings(neigh: Sequence[set[int]], coords: np.ndarray) -> list[list[int]]:
    """Planar 5- and 6-membered carbon/nitrogen cycles, by graph search.

    Used on the peptide side, where residues are often non-canonical and no
    atom-name table applies. A chlorinated phenylalanine's ring must be found
    from the graph or it is invisible.
    """
    rings: list[list[int]] = []
    seen: set[frozenset[int]] = set()
    for start in range(len(neigh)):
        # Depth-limited walk looking for cycles of length 5 or 6.
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if len(path) > 6:
                continue
            for nxt in neigh[node]:
                if nxt == start and len(path) in (5, 6):
                    key = frozenset(path)
                    if key not in seen and _planar_ring(coords, path):
                        seen.add(key)
                        rings.append(list(path))
                elif nxt not in path and len(path) < 6:
                    stack.append((nxt, path + [nxt]))
    return rings


def _type_peptide(residues: Sequence[Any]) -> _AtomSet:
    """Type peptide atoms from element and inferred connectivity."""
    coords, elements, labels = [], [], []
    for res in residues:
        for atom in res:
            if atom.element == "H":
                continue
            coords.append(atom.coord)
            elements.append(atom.element)
            labels.append(f"{res.get_resname()}{res.id[1]}:{atom.get_id()}")

    arr = np.array(coords) if coords else np.zeros((0, 3))
    neigh = _infer_bonds(arr, elements)

    donors, acceptors, apolar = [], [], []
    anionic: list[list[int]] = []
    cationic: list[list[int]] = []
    for i, el in enumerate(elements):
        if el == "N":
            donors.append(i)
            acceptors.append(i)
        elif el == "O":
            donors.append(i)
            acceptors.append(i)
        elif el == "S":
            acceptors.append(i)
        elif el == "C" and not any(elements[j] in ("N", "O") for j in neigh[i]):
            apolar.append(i)

    for i, el in enumerate(elements):
        if el != "C":
            continue
        oxygens = [j for j in neigh[i] if elements[j] == "O"]
        nitrogens = [j for j in neigh[i] if elements[j] == "N"]
        # Carboxylate: carbon with two terminal oxygens.
        if len(oxygens) == 2 and all(len(neigh[o]) == 1 for o in oxygens):
            anionic.append([i, *oxygens])
        # Guanidinium: carbon with three nitrogens.
        elif len(nitrogens) == 3:
            cationic.append([i, *nitrogens])
    # Nitrogens already claimed by a guanidinium must not also count as
    # primary amines. Arginine's NH1 and NH2 each have exactly one heavy
    # neighbour, so without this exclusion one Arg yields three cationic
    # groups and its single salt bridge to an aspartate is counted three
    # times -- which is what the first run of this module did.
    claimed = {i for group in cationic for i in group}
    for i, el in enumerate(elements):
        # Primary amine nitrogen: one heavy neighbour, an sp3 carbon bearing no
        # oxygen, which excludes amide nitrogens.
        if el == "N" and i not in claimed and len(neigh[i]) == 1:
            (j,) = tuple(neigh[i])
            if elements[j] == "C" and not any(elements[k] == "O" for k in neigh[j]):
                cationic.append([i])

    rings = _find_rings(neigh, arr) if len(arr) else []
    return _AtomSet(arr, elements, labels, donors, acceptors,
                    anionic, cationic, apolar, rings)


# ----------------------------------------------------------- interactions ----

def _pairs_within(a: np.ndarray, b: np.ndarray, cutoff: float) -> list[tuple[int, int, float]]:
    if not len(a) or not len(b):
        return []
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    return [(int(i), int(j), float(d[i, j])) for i, j in zip(*np.where(d <= cutoff))]


def _group_min_distance(coords_a, group_a, coords_b, group_b) -> float:
    pa, pb = coords_a[list(group_a)], coords_b[list(group_b)]
    return float(np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1)).min())


def _count_hbonds(rec: "_AtomSet", pep: "_AtomSet", cutoff: float) -> int:
    empty = np.zeros((0, 3))
    return (len(_pairs_within(
                rec.coords[rec.donors] if rec.donors else empty,
                pep.coords[pep.acceptors] if pep.acceptors else empty, cutoff))
            + len(_pairs_within(
                rec.coords[rec.acceptors] if rec.acceptors else empty,
                pep.coords[pep.donors] if pep.donors else empty, cutoff)))


def _count_salt_bridges(rec: "_AtomSet", pep: "_AtomSet", cutoff: float) -> int:
    n = 0
    for groups_a, groups_b in ((rec.anionic, pep.cationic),
                               (rec.cationic, pep.anionic)):
        for ga, gb in itertools.product(groups_a, groups_b):
            if _group_min_distance(rec.coords, ga, pep.coords, gb) <= cutoff:
                n += 1
    return n


def _ring_frame(coords: np.ndarray, ring: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    pts = coords[list(ring)]
    centroid = pts.mean(axis=0)
    _u, _s, vh = np.linalg.svd(pts - centroid)
    return centroid, vh[-1]           # normal is the least-variance direction


def _interplanar_angle(n1: np.ndarray, n2: np.ndarray) -> float:
    cosine = abs(float(np.dot(n1, n2)) / (np.linalg.norm(n1) * np.linalg.norm(n2)))
    return math.degrees(math.acos(min(1.0, max(0.0, cosine))))


# ------------------------------------------------------------------ SASA ----

def _buried_surface_area(model, receptor_id: str, peptide_id: str) -> dict[str, float]:
    """Shrake-Rupley buried surface area for this interface alone.

    The complex is receptor plus peptide only. A G protein or nanobody in the
    same deposit is not part of this interface, and including it would inflate
    the buried area with a surface the peptide never touches.
    """
    from Bio.PDB.SASA import ShrakeRupley

    sr = ShrakeRupley()

    def per_chain(chain_ids: Iterable[str]) -> dict[str, float]:
        """SASA summed per chain, for a sub-structure of the given chains."""
        wanted = set(chain_ids)
        sub = copy.deepcopy(model)
        for chain in list(sub):
            if chain.id not in wanted:
                sub.detach_child(chain.id)
        for chain in sub:
            for res in list(chain):
                if res.id[0] == "W":
                    chain.detach_child(res.id)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # The entity is a Model, so "S" is above it; atom level
            # is also what the sums below read.
            sr.compute(sub, level="A")
        return {
            chain.id: float(sum(
                a.sasa for a in chain.get_atoms() if hasattr(a, "sasa")))
            for chain in sub
        }

    in_complex = per_chain([receptor_id, peptide_id])
    receptor_alone = per_chain([receptor_id])[receptor_id]
    peptide_alone = per_chain([peptide_id])[peptide_id]

    # Each side's buried area is what it loses on complexation. The total
    # interface area is their sum; the conventional single figure is half of
    # it. Dividing the *total* by one side's surface is what produced a
    # "fraction buried" above 1.
    buried_peptide = peptide_alone - in_complex[peptide_id]
    buried_receptor = receptor_alone - in_complex[receptor_id]
    return {
        "sasa_receptor_alone_a2": receptor_alone,
        "sasa_peptide_alone_a2": peptide_alone,
        "sasa_peptide_in_complex_a2": in_complex[peptide_id],
        "buried_peptide_a2": buried_peptide,
        "buried_receptor_a2": buried_receptor,
        "interface_area_a2": (buried_peptide + buried_receptor) / 2.0,
        "buried_surface_area_a2": buried_peptide + buried_receptor,
        "peptide_fraction_buried": (
            buried_peptide / peptide_alone if peptide_alone > 0 else 0.0),
    }


# --------------------------------------------------------------- profiles ----

@dataclass
class InteractionProfile:
    """Typed interactions at one experimentally determined interface."""

    pdb_id: str
    ligand: str
    method: str
    resolution: float
    receptor_chain: str
    peptide_chain: str
    mode: str = InferenceMode.RETROSPECTIVE_STRUCTURE.value

    n_peptide_residues: int = 0
    n_receptor_interface_residues: int = 0

    hbond_candidates: int = 0
    hbond_min_distance_a: float | None = None
    salt_bridges: int = 0
    salt_bridge_details: list[dict[str, Any]] = field(default_factory=list)
    hydrophobic_contacts: int = 0
    aromatic_stacked: int = 0
    aromatic_tshaped: int = 0
    aromatic_oblique: int = 0
    cation_pi: int = 0

    sasa: dict[str, float] = field(default_factory=dict)
    interface_geometry: dict[str, float] = field(default_factory=dict)
    #: Counts at alternative thresholds. Recorded because setmelanotide's
    #: Arg-Asp pair sits at 2.54 A in 7PIU and 4.24 A in 21LT -- the same
    #: ligand on either side of a 4.0 A cutoff -- so a single threshold would
    #: report a salt bridge in one deposit and none in the other.
    cutoff_sensitivity: dict[str, dict[str, int]] = field(default_factory=dict)
    nearest_distances: dict[str, float | None] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)

    @property
    def total_typed_contacts(self) -> int:
        return (self.hbond_candidates + self.salt_bridges
                + self.hydrophobic_contacts + self.aromatic_stacked
                + self.aromatic_tshaped + self.aromatic_oblique + self.cation_pi)

    def vector(self) -> np.ndarray:
        """Feature vector, usable only in retrospective mode."""
        assert_retrospective_only(self.mode)
        g = self.interface_geometry
        return np.array([
            self.hbond_candidates, self.salt_bridges, self.hydrophobic_contacts,
            self.aromatic_stacked, self.aromatic_tshaped, self.aromatic_oblique,
            self.cation_pi, self.n_receptor_interface_residues,
            self.sasa.get("buried_surface_area_a2", 0.0),
            self.sasa.get("peptide_fraction_buried", 0.0),
            g.get("contacts_per_peptide_residue", 0.0),
            g.get("centroid_separation_a", 0.0),
        ], dtype=float)


def profile_structure(path: Path, reference_sequence: str) -> InteractionProfile:
    """Type every interaction at one peptide-receptor interface."""
    from atlas.ml.receptor_pocket import _parse_structure, _aligner

    model = _parse_structure(path)
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
        if 3 <= c[1] <= 25 and (receptor is None or c[0] != receptor.id)
    ]
    if receptor is None or not candidates:
        raise ValueError(f"{path.stem}: cannot identify receptor and peptide chains")
    peptide = model[min(candidates, key=lambda c: c[1])[0]]

    pep_res = [r for r in peptide if r.id[0] != "W"]
    rec_res = [r for r in receptor if r.get_resname() in _THREE_TO_ONE]

    # Restrict the receptor to residues near the peptide before typing, so the
    # whole 300-residue chain is not cross-multiplied against the ligand.
    pep_coords = np.array([a.coord for r in pep_res for a in r if a.element != "H"])
    near: list[Any] = []
    for res in rec_res:
        coords = np.array([a.coord for a in res if a.element != "H"])
        if not len(coords):
            continue
        d = np.sqrt(((coords[:, None, :] - pep_coords[None, :, :]) ** 2).sum(-1)).min()
        if d <= 8.0:
            near.append(res)

    rec = _type_receptor(near)
    pep = _type_peptide(pep_res)

    meta = MC4R_PEPTIDE_COMPLEXES.get(path.stem, {})
    profile = InteractionProfile(
        pdb_id=path.stem, ligand=meta.get("ligand", "?"),
        method=meta.get("method", "?"), resolution=meta.get("resolution", float("nan")),
        receptor_chain=receptor.id, peptide_chain=peptide.id,
        n_peptide_residues=len(pep_res),
        n_receptor_interface_residues=len(near),
    )

    # Hydrogen-bond candidates, both directions.
    hb = (_pairs_within(rec.coords[rec.donors] if rec.donors else np.zeros((0, 3)),
                        pep.coords[pep.acceptors] if pep.acceptors else np.zeros((0, 3)),
                        _HBOND_MAX)
          + _pairs_within(rec.coords[rec.acceptors] if rec.acceptors else np.zeros((0, 3)),
                          pep.coords[pep.donors] if pep.donors else np.zeros((0, 3)),
                          _HBOND_MAX))
    profile.hbond_candidates = len(hb)
    profile.hbond_min_distance_a = min((d for _i, _j, d in hb), default=None)

    # Salt bridges, charged group to oppositely charged group.
    for groups_a, groups_b, sign in ((rec.anionic, pep.cationic, "receptor_anionic"),
                                     (rec.cationic, pep.anionic, "receptor_cationic")):
        for ga, gb in itertools.product(groups_a, groups_b):
            d = _group_min_distance(rec.coords, ga, pep.coords, gb)
            if d <= _SALT_MAX:
                profile.salt_bridges += 1
                profile.salt_bridge_details.append({
                    "kind": sign,
                    "receptor_atoms": [rec.labels[i] for i in ga],
                    "peptide_atoms": [pep.labels[i] for i in gb],
                    "distance_a": round(d, 2),
                })

    # Hydrophobic contacts.
    profile.hydrophobic_contacts = len(_pairs_within(
        rec.coords[rec.apolar_carbons] if rec.apolar_carbons else np.zeros((0, 3)),
        pep.coords[pep.apolar_carbons] if pep.apolar_carbons else np.zeros((0, 3)),
        _PHOBIC_MAX))

    # Aromatic interactions and cation-pi.
    rec_frames = [_ring_frame(rec.coords, r) for r in rec.rings]
    pep_frames = [_ring_frame(pep.coords, r) for r in pep.rings]
    for (c1, n1), (c2, n2) in itertools.product(rec_frames, pep_frames):
        if float(np.linalg.norm(c1 - c2)) > _AROM_MAX:
            continue
        angle = _interplanar_angle(n1, n2)
        if angle <= CRITERIA["aromatic_stacked_max_deg"]:
            profile.aromatic_stacked += 1
        elif angle >= CRITERIA["aromatic_tshaped_min_deg"]:
            profile.aromatic_tshaped += 1
        else:
            profile.aromatic_oblique += 1
    for groups, frames in ((rec.cationic, pep_frames), (pep.cationic, rec_frames)):
        coords = rec.coords if groups is rec.cationic else pep.coords
        for group, (centroid, _n) in itertools.product(groups, frames):
            pts = coords[list(group)]
            if float(np.sqrt(((pts - centroid) ** 2).sum(-1)).min()) <= _CATION_PI_MAX:
                profile.cation_pi += 1

    profile.cutoff_sensitivity = {
        "salt_bridges": {
            f"{c}": _count_salt_bridges(rec, pep, c) for c in (4.0, 4.5, 5.0)},
        "hbond_candidates": {
            f"{c}": _count_hbonds(rec, pep, c) for c in (3.2, 3.5, 3.9)},
    }
    # The nearest approach of each interaction class, so an absent class can be
    # told apart from a marginal one.
    empty = np.zeros((0, 3))
    anionic_cationic = [
        _group_min_distance(rec.coords, ga, pep.coords, gb)
        for ga, gb in itertools.product(rec.anionic, pep.cationic)
    ] + [
        _group_min_distance(rec.coords, ga, pep.coords, gb)
        for ga, gb in itertools.product(rec.cationic, pep.anionic)
    ]
    cation_ring = [
        float(np.sqrt(((coords[list(group)] - centroid) ** 2).sum(-1)).min())
        for groups, frames, coords in (
            (rec.cationic, pep_frames, rec.coords),
            (pep.cationic, rec_frames, pep.coords))
        for group, (centroid, _n) in itertools.product(groups, frames)
    ]
    profile.nearest_distances = {
        "charged_pair_a": min(anionic_cationic, default=None),
        "cation_to_ring_centroid_a": min(cation_ring, default=None),
        "donor_acceptor_a": profile.hbond_min_distance_a,
    }

    profile.sasa = _buried_surface_area(model, receptor.id, peptide.id)

    rec_centroid = rec.coords.mean(axis=0) if len(rec.coords) else np.zeros(3)
    pep_centroid = pep.coords.mean(axis=0) if len(pep.coords) else np.zeros(3)
    profile.interface_geometry = {
        "contacts_per_peptide_residue": (
            profile.total_typed_contacts / max(len(pep_res), 1)),
        "centroid_separation_a": float(np.linalg.norm(rec_centroid - pep_centroid)),
        "peptide_heavy_atoms": float(len(pep.coords)),
        "peptide_rings_found": float(len(pep.rings)),
        "peptide_cationic_groups": float(len(pep.cationic)),
        "peptide_anionic_groups": float(len(pep.anionic)),
    }
    profile.caveats.append(
        "Hydrogen bonds are reported as candidates: at 2.58-3.30 A these "
        "structures carry no hydrogens, so the donor-hydrogen-acceptor angle "
        "cannot be checked and a heavy-atom distance alone over-counts."
    )
    profile.caveats.append(
        "Peptide chemistry is typed from element and distance-inferred bonds, "
        "not from residue names, because these ligands are largely "
        "non-canonical. Charged-group and ring assignments on the peptide side "
        "are therefore inferences from geometry."
    )
    return profile


def build_interaction_profiles(
    *, receptors: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Profile every available MC4R peptide complex."""
    if receptors is None:
        from atlas.ml.multitarget import load_panel
        receptors = load_panel().receptors
    reference = receptors["MC4R"]["sequence"]

    profiles: list[InteractionProfile] = []
    for pdb_id in MC4R_PEPTIDE_COMPLEXES:
        try:
            profiles.append(profile_structure(fetch_structure(pdb_id), reference))
        except Exception as exc:
            logger.warning("%s skipped: %s", pdb_id, exc)

    report = {
        "mode": InferenceMode.RETROSPECTIVE_STRUCTURE.value,
        "criteria": CRITERIA,
        "n_structures": len(profiles),
        "n_distinct_ligands": len({p.ligand for p in profiles}),
        "profiles": [asdict(p) for p in profiles],
        "built_at": utc_now().isoformat(),
        "restriction": (
            "Every descriptor here is computed from an experimentally "
            "determined bound complex and does not exist for an unsolved "
            "peptide-target pair. These features must not enter a prospective "
            "prediction pathway; assert_retrospective_only() enforces it."
        ),
    }
    INTERACTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    INTERACTION_PATH.write_text(json.dumps(report, indent=2, default=float))
    return report


def load_profiles(path: Path = INTERACTION_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = build_interaction_profiles()
    print(f"mode: {report['mode']}")
    print(f"{report['n_structures']} structures, "
          f"{report['n_distinct_ligands']} distinct ligands\n")
    head = (f"{'pdb':6s}{'ligand':22s}{'res':>5s}{'pepres':>7s}{'hb':>5s}"
            f"{'salt':>6s}{'phob':>6s}{'stack':>6s}{'Tshp':>6s}{'cat-pi':>7s}"
            f"{'BSA':>8s}{'%bur':>6s}")
    print(head)
    for p in report["profiles"]:
        print(f"{p['pdb_id']:6s}{p['ligand'][:21]:22s}{p['resolution']:5.2f}"
              f"{p['n_peptide_residues']:7d}{p['hbond_candidates']:5d}"
              f"{p['salt_bridges']:6d}{p['hydrophobic_contacts']:6d}"
              f"{p['aromatic_stacked']:6d}{p['aromatic_tshaped']:6d}"
              f"{p['cation_pi']:7d}"
              f"{p['sasa']['buried_surface_area_a2']:8.0f}"
              f"{p['sasa']['peptide_fraction_buried']:6.2f}")
    print(f"\nSaved to {INTERACTION_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
