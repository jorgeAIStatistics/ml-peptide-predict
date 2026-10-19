"""ATLAS 2.5: a binding pocket defined from experimental structures.

Stage 3 encoded each receptor by the amino-acid composition of its whole
sequence, and could not establish a target encoder. Whole-sequence composition
is a poor description of a binding site: an MC4R is roughly 330 residues, of
which a peptide touches a few dozen, and the rest — transmembrane packing,
intracellular loops, the G-protein interface — is irrelevant to affinity while
contributing most of the signal in a composition vector.

This module replaces that with a pocket defined from observed geometry. Ten
experimental MC4R–peptide complexes are available (Stage 1 §4.2); the residues
that actually contact a bound peptide are read off their coordinates, pooled
into a consensus across structures, mapped to UniProt numbering, and
transferred to the other melanocortin receptors by alignment.

Three points of discipline:

* **Numbering is never trusted.** Author residue numbers agree with UniProt
  positions in the eight cryo-EM entries but not in the three X-ray fusion
  constructs, whose chains carry a glycogen-synthase partner. Every position
  is established by local alignment to the UniProt sequence and verified by
  checking the residue letter.
* **Identity comes from UniProt, geometry from the structure.** The
  crystallisation constructs carry engineered mutations, so the residue
  observed at a contact position is not always the wild-type residue. Contacts
  are located from the structure and then *named* from UniProt, with the
  mismatches counted and reported.
* **A contact is not an energetic contribution.** This module reports which
  residues are close enough to interact in an observed complex. That is a
  geometric fact about ten structures, not a measurement of any residue's
  contribution to binding, and nothing downstream may treat it as one.
"""

from __future__ import annotations

import collections
import json
import logging
import warnings
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from atlas.core.evidence import utc_now
from atlas.ml.dataset import DATA_DIR
from atlas.sources.base import CachedHttpClient

logger = logging.getLogger(__name__)

__all__ = [
    "MC4R_PEPTIDE_COMPLEXES", "POCKET_PATH", "STRUCTURE_CACHE",
    "CONTACT_CUTOFF_A", "ContactSet", "PocketDefinition", "build_pocket",
    "load_pocket", "pocket_residues_for_receptor", "pocket_encoding",
]

POCKET_PATH = DATA_DIR / "receptor_pocket.json"
#: Downloaded coordinate files are large and re-derivable, so they are cached
#: outside version control alongside the HTTP cache.
STRUCTURE_CACHE = DATA_DIR.parents[0] / "cache" / "structures"
RCSB_CIF = "https://files.rcsb.org/download/{pdb_id}.cif"

#: Heavy-atom distance within which two residues are called in contact.
#: PepBDB uses 5.0 A to decide whether two *chains* interact; for residue-level
#: contacts 4.5 A is the conventional heavy-atom cutoff. The sensitivity of
#: the pocket to this choice is measured and reported rather than assumed away.
CONTACT_CUTOFF_A = 4.5
CUTOFF_SWEEP = (4.0, 4.5, 5.0)

#: Minimum number of structures a residue must contact a peptide in before it
#: joins the consensus pocket. One structure is one crystal form; agreement
#: across independent complexes is what makes the pocket a property of the
#: receptor rather than of one ligand.
MIN_STRUCTURES = 3

#: The ten experimental MC4R complexes carrying a peptide ligand, from the
#: Stage 1 RCSB query on UniProt P32245. Ligand names are as RCSB records them.
MC4R_PEPTIDE_COMPLEXES: dict[str, dict[str, Any]] = {
    "7PIU": {"ligand": "setmelanotide", "method": "cryo-EM", "resolution": 2.58},
    "6W25": {"ligand": "SHU9119", "method": "X-ray", "resolution": 2.75},
    "7PIV": {"ligand": "NDP-alpha-MSH", "method": "cryo-EM", "resolution": 2.86},
    "8WKY": {"ligand": "S25", "method": "X-ray", "resolution": 2.90},
    "7AUE": {"ligand": "setmelanotide-type", "method": "cryo-EM", "resolution": 2.97},
    "7F53": {"ligand": "alpha-MSH", "method": "cryo-EM", "resolution": 3.00},
    "7F54": {"ligand": "afamelanotide", "method": "cryo-EM", "resolution": 3.00},
    "7F55": {"ligand": "bremelanotide", "method": "cryo-EM", "resolution": 3.10},
    "21LT": {"ligand": "setmelanotide", "method": "cryo-EM", "resolution": 3.11},
    "8WKZ": {"ligand": "S31", "method": "X-ray", "resolution": 3.30},
}

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

#: A peptide chain has at least this many and at most this many residues. The
#: upper bound separates a peptide ligand from a nanobody or G-protein subunit.
PEPTIDE_MIN_RES, PEPTIDE_MAX_RES = 3, 25


def _aligner(mode: str):
    from Bio.Align import PairwiseAligner, substitution_matrices

    al = PairwiseAligner(mode=mode)
    al.substitution_matrix = substitution_matrices.load("BLOSUM62")
    al.open_gap_score, al.extend_gap_score = -11, -1
    return al


def fetch_structure(pdb_id: str, http: CachedHttpClient | None = None) -> Path:
    """Download one mmCIF into the structure cache, with provenance."""
    STRUCTURE_CACHE.mkdir(parents=True, exist_ok=True)
    path = STRUCTURE_CACHE / f"{pdb_id}.cif"
    if path.exists() and path.stat().st_size > 0:
        return path
    http = http or CachedHttpClient(ttl=timedelta(days=365))
    fetch = http.get(RCSB_CIF.format(pdb_id=pdb_id), accept="text/plain")
    path.write_bytes(fetch.body)
    return path


@dataclass
class ContactSet:
    """Receptor residues contacting the peptide in one structure."""

    pdb_id: str
    receptor_chain: str
    peptide_chain: str
    n_peptide_residues: int
    n_receptor_residues_observed: int
    n_receptor_residues_mapped: int
    letter_mismatches: int
    contacts: dict[int, float] = field(default_factory=dict)   # uniprot pos -> min dist
    mutated_contact_positions: list[int] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)


def _parse_structure(path: Path):
    from Bio.PDB import MMCIFParser

    with warnings.catch_warnings():
        # Biopython warns about discontinuous chains and unknown residues in
        # engineered constructs; those are expected here and handled below.
        warnings.simplefilter("ignore")
        structure = MMCIFParser(QUIET=True).get_structure(path.stem, str(path))
    return next(iter(structure))


def contacts_for_structure(
    path: Path, reference_sequence: str, *, cutoff: float = CONTACT_CUTOFF_A,
) -> ContactSet:
    """Find receptor residues within ``cutoff`` of any peptide heavy atom.

    The receptor chain is identified by local alignment to the reference
    sequence, not by chain label, and the peptide chain as the shortest polymer
    in the permitted size range. Every residue of the peptide chain is used,
    whatever it is called: the non-canonical residues are the chemically
    interesting ones, and filtering by standard-amino-acid membership silently
    dropped two of setmelanotide's eight.
    """
    model = _parse_structure(path)
    local = _aligner("local")

    receptor = None
    best_score = 0.0
    sizes: list[tuple[str, int]] = []
    for chain in model:
        residues = [r for r in chain if r.id[0] != "W"]
        standard = [r for r in residues if r.get_resname() in _THREE_TO_ONE]
        if len(standard) > 5:
            score = local.score(
                reference_sequence,
                "".join(_THREE_TO_ONE[r.get_resname()] for r in standard),
            )
            if score > best_score:
                receptor, best_score = chain, score
        sizes.append((chain.id, len(residues)))

    candidates = [
        c for c in sizes
        if PEPTIDE_MIN_RES <= c[1] <= PEPTIDE_MAX_RES
        and (receptor is None or c[0] != receptor.id)
    ]
    if receptor is None or not candidates:
        raise ValueError(
            f"{path.stem}: could not identify receptor and peptide chains "
            f"(chain sizes {sizes})"
        )
    peptide = model[min(candidates, key=lambda c: c[1])[0]]

    standard = [r for r in receptor if r.get_resname() in _THREE_TO_ONE]
    observed = "".join(_THREE_TO_ONE[r.get_resname()] for r in standard)
    alignment = local.align(reference_sequence, observed)[0]

    # auth_seq_id -> 1-based position in the reference sequence.
    # Biopython's alignment indices are numpy integers; cast them, or a
    # residue position is serialised as 43.0 and every integer lookup fails.
    position: dict[int, int] = {}
    for (ref_start, ref_end), (obs_start, _obs_end) in zip(*alignment.aligned):
        for k in range(int(ref_end) - int(ref_start)):
            position[int(standard[int(obs_start) + k].id[1])] = int(ref_start) + k + 1

    mismatched = {
        r.id[1] for r in standard
        if r.id[1] in position
        and reference_sequence[position[r.id[1]] - 1] != _THREE_TO_ONE[r.get_resname()]
    }

    peptide_atoms = np.array([
        a.coord for r in peptide if r.id[0] != "W"
        for a in r if a.element != "H"
    ])
    if not len(peptide_atoms):
        raise ValueError(f"{path.stem}: peptide chain {peptide.id} has no heavy atoms")

    contacts: dict[int, float] = {}
    for residue in standard:
        pos = position.get(residue.id[1])
        if pos is None:
            continue
        coords = np.array([a.coord for a in residue if a.element != "H"])
        if not len(coords):
            continue
        d = float(np.sqrt(
            ((coords[:, None, :] - peptide_atoms[None, :, :]) ** 2).sum(-1)
        ).min())
        if d <= cutoff:
            contacts[pos] = min(d, contacts.get(pos, np.inf))

    cs = ContactSet(
        pdb_id=path.stem, receptor_chain=receptor.id, peptide_chain=peptide.id,
        n_peptide_residues=sum(1 for r in peptide if r.id[0] != "W"),
        n_receptor_residues_observed=len(standard),
        n_receptor_residues_mapped=len(position),
        letter_mismatches=len(mismatched),
        contacts={int(k): float(v) for k, v in sorted(contacts.items())},
        mutated_contact_positions=sorted(
            int(position[a]) for a in mismatched if position[a] in contacts),
    )
    if cs.n_receptor_residues_mapped < cs.n_receptor_residues_observed:
        cs.caveats.append(
            f"{cs.n_receptor_residues_observed - cs.n_receptor_residues_mapped} "
            "observed residues did not align to the reference sequence; this "
            "chain carries a fusion partner."
        )
    if cs.mutated_contact_positions:
        cs.caveats.append(
            f"{len(cs.mutated_contact_positions)} contact positions carry a "
            "residue differing from the reference sequence, i.e. an engineered "
            "mutation. Geometry is taken from the structure; identity is taken "
            "from the reference."
        )
    return cs


@dataclass
class PocketDefinition:
    """A consensus binding pocket, in reference-sequence coordinates."""

    reference_receptor: str
    reference_accession: str
    cutoff_angstrom: float
    min_structures: int
    positions: list[int]                       # consensus pocket, sorted
    support: dict[str, int]                    # position -> structures contacting
    per_structure: list[dict[str, Any]]
    transferred: dict[str, dict[str, Any]]     # receptor -> mapping + residues
    cutoff_sensitivity: dict[str, int]
    built_at: str = ""
    caveats: list[str] = field(default_factory=list)

    def save(self, path: Path = POCKET_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=float))
        return path


def _transfer(
    reference: str, reference_seq: str, positions: Sequence[int],
    receptors: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Carry pocket positions onto the other receptors by global alignment.

    The melanocortin receptors are homologous, so a position that lines the
    MC4R pocket has a counterpart in MC1R, MC3R and MC5R. A position with no
    aligned counterpart is recorded as a gap rather than guessed at.
    """
    glob = _aligner("global")
    out: dict[str, dict[str, Any]] = {}
    for name, meta in receptors.items():
        seq = meta.get("sequence")
        if not seq:
            out[name] = {"residues": None, "note": "no sequence available"}
            continue
        if name == reference:
            out[name] = {
                "residues": [reference_seq[p - 1] for p in positions],
                "positions": [int(p) for p in positions],
                "identity_to_reference": 1.0, "n_gaps": 0,
                "n_identical_pocket_residues": len(positions),
            }
            continue
        aln = glob.align(reference_seq, seq)[0]
        ref_to_other: dict[int, int] = {}
        for (rs, re_), (os_, _oe) in zip(*aln.aligned):
            for k in range(int(re_) - int(rs)):
                ref_to_other[int(rs) + k + 1] = int(os_) + k + 1
        residues, mapped = [], []
        for p in positions:
            q = ref_to_other.get(p)
            residues.append(seq[q - 1] if q else "-")
            mapped.append(int(q) if q else None)
        aligned_len = sum(e - s for s, e in aln.aligned[0])
        identical = sum(
            1 for (rs, re_), (os_, _oe) in zip(*aln.aligned)
            for k in range(re_ - rs)
            if reference_seq[rs + k] == seq[os_ + k]
        )
        out[name] = {
            "residues": residues,
            "positions": mapped,
            "identity_to_reference": identical / max(aligned_len, 1),
            "n_gaps": sum(1 for r in residues if r == "-"),
            "n_identical_pocket_residues": sum(
                1 for r, p in zip(residues, positions)
                if r == reference_seq[p - 1]),
        }
    return out


def build_pocket(
    *,
    receptors: dict[str, dict[str, Any]] | None = None,
    cutoff: float = CONTACT_CUTOFF_A,
    min_structures: int = MIN_STRUCTURES,
    http: CachedHttpClient | None = None,
) -> PocketDefinition:
    """Derive the consensus pocket from the MC4R peptide complexes."""
    if receptors is None:
        from atlas.ml.multitarget import load_panel
        receptors = load_panel().receptors

    mc4r = receptors["MC4R"]
    reference_seq = mc4r["sequence"]

    sets: list[ContactSet] = []
    for pdb_id in MC4R_PEPTIDE_COMPLEXES:
        path = fetch_structure(pdb_id, http)
        try:
            sets.append(contacts_for_structure(path, reference_seq, cutoff=cutoff))
        except Exception as exc:
            logger.warning("%s skipped: %s", pdb_id, exc)

    # Counter.update on a dict adds its *values*, so passing {position:
    # distance} would sum distances instead of counting structures. The keys
    # are what is being counted.
    support = collections.Counter()
    for cs in sets:
        support.update(cs.contacts.keys())
    positions = sorted(int(p) for p, n in support.items() if n >= min_structures)

    sweep: dict[str, int] = {}
    for c in CUTOFF_SWEEP:
        if c == cutoff:
            sweep[str(c)] = len(positions)
            continue
        alt = collections.Counter()
        for pdb_id in MC4R_PEPTIDE_COMPLEXES:
            try:
                alt.update(contacts_for_structure(
                    fetch_structure(pdb_id, http), reference_seq,
                    cutoff=c).contacts.keys())
            except Exception:
                continue
        sweep[str(c)] = sum(1 for _p, n in alt.items() if n >= min_structures)

    definition = PocketDefinition(
        reference_receptor="MC4R",
        reference_accession=(mc4r.get("uniprot_accessions") or ["?"])[0],
        cutoff_angstrom=cutoff, min_structures=min_structures,
        positions=positions,
        support={str(int(p)): int(support[p]) for p in positions},
        per_structure=[
            {**asdict(cs), "contacts": {str(k): v for k, v in cs.contacts.items()},
             **MC4R_PEPTIDE_COMPLEXES.get(cs.pdb_id, {})}
            for cs in sets
        ],
        transferred=_transfer("MC4R", reference_seq, positions, receptors),
        cutoff_sensitivity=sweep,
        built_at=utc_now().isoformat(),
    )
    definition.caveats.extend([
        f"Derived from {len(sets)} experimental complexes covering "
        f"{len({v['ligand'] for k, v in MC4R_PEPTIDE_COMPLEXES.items() if k in {s.pdb_id for s in sets}})} "
        "distinct ligands. Setmelanotide appears in three entries and "
        "afamelanotide in two, so entries are not independent observations.",
        "A contact is a geometric fact about these structures. It is not a "
        "measurement of any residue's energetic contribution to binding, and "
        "must not be reported as one.",
        "Positions are transferred to the other receptors by sequence "
        "alignment, which assumes the homologous position lines the "
        "homologous pocket. No experimental structure of MC1R, MC3R or MC5R "
        "with a peptide was used.",
    ])
    return definition


def load_pocket(path: Path = POCKET_PATH) -> PocketDefinition:
    payload = json.loads(path.read_text())
    return PocketDefinition(**payload)


# ------------------------------------------------------------- encoding ----

def pocket_residues_for_receptor(
    pocket: PocketDefinition, receptor: str
) -> list[str] | None:
    entry = pocket.transferred.get(receptor) or {}
    return entry.get("residues")


def pocket_encoding(
    receptor_names: Sequence[str], pocket: PocketDefinition,
) -> np.ndarray:
    """Chemistry of each pocket position, per receptor.

    Each consensus position contributes the residue-chemistry descriptor vector
    of whatever residue that receptor carries there, so two receptors differing
    at one pocket position differ in exactly that block. A gap contributes
    zeros.

    This is the encoder Stage 3's whole-sequence composition should be compared
    against: same receptors, same downstream model, a target description
    restricted to the residues a peptide actually touches.
    """
    from atlas.ml.residue_chemistry import DESCRIPTOR_NAMES, describe_residue

    width = len(pocket.positions) * len(DESCRIPTOR_NAMES)
    cache: dict[str, np.ndarray] = {}
    for name in set(receptor_names):
        residues = pocket_residues_for_receptor(pocket, name)
        if not residues:
            cache[name] = np.zeros(width)
            continue
        blocks = [
            describe_residue(r).vector() if r != "-"
            else np.zeros(len(DESCRIPTOR_NAMES))
            for r in residues
        ]
        cache[name] = np.concatenate(blocks)

    return np.vstack([cache[n] for n in receptor_names])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pocket = build_pocket()
    path = pocket.save()

    print(f"reference      {pocket.reference_receptor} ({pocket.reference_accession})")
    print(f"cutoff         {pocket.cutoff_angstrom} A, consensus in "
          f">= {pocket.min_structures} structures")
    print(f"pocket size    {len(pocket.positions)} positions")
    print(f"cutoff sweep   {pocket.cutoff_sensitivity}")
    print("\nper structure:")
    for s in pocket.per_structure:
        print(f"  {s['pdb_id']}  {s.get('ligand','?'):20s} {s.get('method','?'):9s} "
              f"{s['n_peptide_residues']:2d} pep res, "
              f"{len(s['contacts']):3d} contacts, "
              f"{s['letter_mismatches']:2d} letter mismatches")
    print("\nconsensus positions and support:")
    ref = None
    from atlas.ml.multitarget import load_panel
    ref = load_panel().receptors["MC4R"]["sequence"]
    line = "  "
    for p in pocket.positions:
        line += f"{ref[p-1]}{p}({pocket.support[str(p)]}) "
        if len(line) > 90:
            print(line); line = "  "
    if line.strip():
        print(line)
    print("\ntransfer to the family:")
    for name, t in pocket.transferred.items():
        if t.get("residues") is None:
            print(f"  {name}: {t.get('note')}"); continue
        print(f"  {name}: identity {t['identity_to_reference']:.1%}, "
              f"{t.get('n_identical_pocket_residues', 0)}/{len(pocket.positions)} "
              f"pocket residues identical to MC4R, {t['n_gaps']} gaps")
        print(f"       {''.join(t['residues'])}")
    print(f"\nSaved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
