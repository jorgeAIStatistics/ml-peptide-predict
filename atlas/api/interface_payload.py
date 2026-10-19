"""Composed payloads for the structural interface explorer.

The explorer needs three things joined that are saved separately: the typed
interaction profiles (``interaction_profiles.json``), the consensus pocket with
its per-structure contacts and its transfer to the other melanocortin
receptors (``receptor_pocket.json``), and the residue chemistry of whatever
residue each receptor carries at each pocket position.

Joining them here rather than in the browser keeps one property that matters:
nothing the interface draws is computed client-side, so what is drawn is what
was saved, and the vocabulary that separates a measurement from a candidate
from an inference is set once, in Python, where the modules that made the
distinction live.

Two statements are attached to every payload rather than written into the
frontend:

* **Every structure here is experimental.** The ten complexes are deposited
  cryo-EM and X-ray structures. What is *not* experimental is some of the
  typing laid over them: hydrogen bonds are candidates because no hydrogens
  are present to check an angle against, and the peptide-side charged groups
  and rings are inferred from geometry because these ligands are largely
  non-canonical.
* **The peptide affinity model publishes no numeric estimate.** It fails its
  deployment gate. An interface explorer sits next to that model and must not
  be read as its output, so the gate result travels with the payload.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "interface_index", "interface_detail", "STRUCTURE_EVIDENCE",
    "RENDERING_NOTE",
]

#: What kind of thing each layer of the drawing is. The explorer labels every
#: layer with one of these, because "experimental structure", "a geometric
#: consequence of it" and "an illustration of it" are three different claims.
STRUCTURE_EVIDENCE: dict[str, dict[str, str]] = {
    "experimental": {
        "label": "Experimental",
        "detail": (
            "Atomic coordinates as deposited, from a cryo-EM map or X-ray "
            "diffraction. The receptor trace, the peptide and every distance "
            "drawn between them are read from the deposited model."
        ),
    },
    "geometric": {
        "label": "Geometric consequence",
        "detail": (
            "A distance or angle computed from those coordinates under a "
            "stated cutoff. It is a fact about the structure, not a "
            "measurement of any residue's contribution to binding."
        ),
    },
    "candidate": {
        "label": "Candidate",
        "detail": (
            "Satisfies the distance criterion but not a full geometric test. "
            "Hydrogen bonds are the case here: these structures carry no "
            "hydrogens, so no donor-hydrogen-acceptor angle was ever checked."
        ),
    },
    "inferred": {
        "label": "Inferred chemistry",
        "detail": (
            "The functional group behind this mark was assigned from element "
            "and distance-inferred connectivity, not read from a residue "
            "name. Required on the peptide side, whose residues are largely "
            "non-canonical."
        ),
    },
    "illustrative": {
        "label": "Illustrative",
        "detail": (
            "A drawing convention with no structural content: the schematic "
            "contact map's layout, the ribbon's thickness, the colour of a "
            "mark. Nothing in this class carries data beyond what the layers "
            "above already state."
        ),
    },
    "transferred": {
        "label": "Transferred by alignment",
        "detail": (
            "A pocket position carried from MC4R to another receptor by "
            "sequence alignment. No experimental structure of MC1R, MC3R or "
            "MC5R with a peptide exists, so these positions are homology "
            "assignments, not observations."
        ),
    },
}

RENDERING_NOTE = (
    "The viewer is ATLAS's own orthographic projection of the deposited "
    "coordinates, drawn in SVG. It is not a molecular-graphics package: there "
    "is no surface, no electron density and no secondary-structure fitting. "
    "It draws the C-alpha trace, the interface side chains, the peptide's "
    "heavy atoms with distance-inferred bonds, and the typed contacts between "
    "them."
)


def _data_file(name: str) -> dict[str, Any]:
    from atlas.ml.dataset import DATA_DIR
    import json

    path = DATA_DIR / name
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def _profiles() -> dict[str, Any]:
    return _data_file("interaction_profiles.json")


@lru_cache(maxsize=1)
def _pocket() -> dict[str, Any]:
    return _data_file("receptor_pocket.json")


@lru_cache(maxsize=1)
def _reference_sequence() -> str | None:
    """The MC4R sequence the pocket positions are numbered against.

    Read from the saved melanocortin panel, so the explorer works offline and
    against the same snapshot the pocket was built from.
    """
    try:
        from atlas.ml.multitarget import load_panel

        return load_panel().receptors["MC4R"]["sequence"]
    except Exception as exc:  # pragma: no cover - absent or unreadable panel
        logger.warning("MC4R reference sequence unavailable: %s", exc)
        return None


@lru_cache(maxsize=1)
def _pocket_chemistry() -> dict[str, Any]:
    """Residue chemistry at each consensus pocket position, per receptor.

    The same descriptor engine the pocket encoder uses, so a value shown here
    is the value a model would have been given. Provenance is carried per
    descriptor: none of these is a measurement, four of them depend on an
    assumed pH, and glycine and proline have no isolable side chain at all.
    """
    from atlas.ml.residue_chemistry import (
        ASSUMED_PH, DESCRIPTOR_NAMES, DESCRIPTOR_PROVENANCE, describe_residue,
    )

    pocket = _pocket()
    transferred = pocket.get("transferred") or {}
    residues: dict[str, list[str]] = {
        name: entry.get("residues") or []
        for name, entry in transferred.items()
    }

    keys = {r for rs in residues.values() for r in rs if r and r != "-"}
    by_residue: dict[str, dict[str, Any]] = {}
    for key in sorted(keys):
        props = describe_residue(key)
        by_residue[key] = {
            "key": key,
            "is_canonical": props.is_canonical,
            "has_side_chain": props.has_side_chain,
            "descriptors": {n: round(float(getattr(props, n)), 4)
                            for n in DESCRIPTOR_NAMES},
            "unavailable": sorted(
                n for n, p in props.provenance.items() if p.value == "unavailable"),
            "caveats": list(props.caveats),
        }

    return {
        "assumed_ph": ASSUMED_PH,
        "descriptor_names": list(DESCRIPTOR_NAMES),
        "descriptor_provenance": {n: p.value for n, p in DESCRIPTOR_PROVENANCE.items()},
        "by_residue": by_residue,
        "note": (
            "Every descriptor is computed from the side-chain fragment rather "
            "than looked up from a published amino-acid scale, because 80.5% "
            "of the peptides in this dataset carry a residue outside the "
            "canonical twenty and those scales have no row for them. None is "
            "a measurement. The four ionisation descriptors are estimates "
            f"that assume pH {ASSUMED_PH} and reference pKa values."
        ),
    }


def _pocket_positions() -> list[dict[str, Any]]:
    """The consensus pocket, one entry per position, with its support."""
    pocket = _pocket()
    positions = pocket.get("positions") or []
    support = pocket.get("support") or {}
    transferred = pocket.get("transferred") or {}
    reference = pocket.get("reference_receptor", "MC4R")

    index = {name: (entry.get("residues") or [])
             for name, entry in transferred.items()}
    ref_residues = index.get(reference, [])

    out: list[dict[str, Any]] = []
    for i, pos in enumerate(positions):
        ref_aa = ref_residues[i] if i < len(ref_residues) else None
        others = {
            name: (residues[i] if i < len(residues) else None)
            for name, residues in index.items() if name != reference
        }
        out.append({
            "position": int(pos),
            "residue": ref_aa,
            "support": int(support.get(str(int(pos)), 0)),
            "transferred": others,
            "conserved_in": sorted(
                name for name, aa in others.items() if aa and aa == ref_aa),
            "differs_in": sorted(
                name for name, aa in others.items()
                if aa and aa != ref_aa and aa != "-"),
            "gap_in": sorted(name for name, aa in others.items() if aa == "-"),
        })
    return out


def _model_position() -> dict[str, Any]:
    """Where the affinity model stands, carried with every interface payload.

    The explorer shows a binding interface. The obvious misreading is that it
    therefore predicts binding, so the gate result is attached rather than
    left for the reader to go and find.
    """
    from atlas.ml.peptide_benchmark import GATE

    benchmark = _data_file("peptide_benchmark.json")
    gate = benchmark.get("gate") or {}
    return {
        "publishes_numeric_affinity": False,
        "thresholds": dict(GATE),
        "result": gate,
        "statement": (
            "The peptide affinity model fails its deployment gate and serves "
            "no numeric affinity estimate. Nothing in this explorer is a "
            "prediction: it describes ten solved complexes. A contact, a "
            "buried area or a salt bridge drawn here says nothing about how "
            "tightly any peptide binds, and the explorer offers no affinity "
            "for the peptides it draws."
        ),
        "retrospective_only": (
            "These descriptors exist only where a bound structure was solved. "
            "They cannot be computed for an unsolved pair and are barred from "
            "the prediction pathway by assert_retrospective_only()."
        ),
    }


@lru_cache(maxsize=1)
def interface_index() -> dict[str, Any]:
    """Everything the explorer needs before a structure is chosen."""
    from atlas.ml.interface_geometry import INTERACTION_KINDS, available_structures

    profiles = _profiles()
    pocket = _pocket()
    if not profiles or not pocket:
        return {
            "available": False,
            "reason": (
                "interaction_profiles.json or receptor_pocket.json is not "
                "present in this snapshot. Both are written by "
                "atlas.ml.interaction_features and atlas.ml.receptor_pocket."
            ),
        }

    cached = available_structures()
    contacts_by_pdb = {
        entry.get("pdb_id"): entry
        for entry in (pocket.get("per_structure") or [])
    }

    structures: list[dict[str, Any]] = []
    for profile in profiles.get("profiles") or []:
        pdb_id = profile.get("pdb_id")
        pocket_entry = contacts_by_pdb.get(pdb_id) or {}
        structures.append({
            **profile,
            "coordinates_cached": bool(cached.get(pdb_id)),
            "contacts": pocket_entry.get("contacts") or {},
            "n_pocket_contacts": len(pocket_entry.get("contacts") or {}),
            "mutated_contact_positions": pocket_entry.get(
                "mutated_contact_positions") or [],
            "pocket_caveats": pocket_entry.get("caveats") or [],
            "n_receptor_residues_observed": pocket_entry.get(
                "n_receptor_residues_observed"),
            "n_receptor_residues_mapped": pocket_entry.get(
                "n_receptor_residues_mapped"),
        })

    reference = _reference_sequence()
    return {
        "available": True,
        "mode": profiles.get("mode"),
        "restriction": profiles.get("restriction"),
        "criteria": profiles.get("criteria") or {},
        "built_at": profiles.get("built_at"),
        "n_structures": profiles.get("n_structures"),
        "n_distinct_ligands": profiles.get("n_distinct_ligands"),
        "interaction_kinds": INTERACTION_KINDS,
        # Flask sorts object keys when it serialises, so the declared order of
        # INTERACTION_KINDS does not survive the wire. The explorer presents
        # the kinds strongest-evidence-first, which is a deliberate order and
        # not alphabetical, so it is sent explicitly.
        "interaction_kind_order": list(INTERACTION_KINDS),
        "evidence_classes": STRUCTURE_EVIDENCE,
        "rendering_note": RENDERING_NOTE,
        "structures": structures,
        "pocket": {
            "reference_receptor": pocket.get("reference_receptor"),
            "reference_accession": pocket.get("reference_accession"),
            "reference_length": len(reference) if reference else None,
            "cutoff_angstrom": pocket.get("cutoff_angstrom"),
            "min_structures": pocket.get("min_structures"),
            "cutoff_sensitivity": pocket.get("cutoff_sensitivity") or {},
            "built_at": pocket.get("built_at"),
            "caveats": pocket.get("caveats") or [],
            "n_positions": len(pocket.get("positions") or []),
            "positions": _pocket_positions(),
            "transfer": {
                name: {
                    k: v for k, v in entry.items()
                    if k in ("identity_to_reference", "n_gaps",
                             "n_identical_pocket_residues")
                }
                for name, entry in (pocket.get("transferred") or {}).items()
            },
        },
        "chemistry": _pocket_chemistry(),
        "model_position": _model_position(),
    }


@lru_cache(maxsize=16)
def interface_detail(pdb_id: str) -> dict[str, Any]:
    """One structure: its saved profile, its contacts and its geometry.

    Geometry is read from the cached coordinate file. That file is gitignored,
    so its absence is normal in a fresh checkout and is reported as an absent
    input with the command that restores it -- never filled in with something
    else.
    """
    from atlas.ml.interface_geometry import (
        InterfaceGeometryUnavailable, geometry_for_structure,
    )

    index = interface_index()
    if not index.get("available"):
        return {"available": False, "reason": index.get("reason")}

    summary = next(
        (s for s in index["structures"] if s.get("pdb_id") == pdb_id), None)
    if summary is None:
        return {
            "available": False,
            "reason": f"{pdb_id} is not one of the profiled complexes.",
        }

    payload: dict[str, Any] = {
        "available": True,
        "structure": summary,
        "rendering_note": RENDERING_NOTE,
        "evidence_classes": STRUCTURE_EVIDENCE,
        "model_position": index["model_position"],
    }

    reference = _reference_sequence()
    if reference is None:
        payload["geometry"] = None
        payload["geometry_unavailable"] = (
            "The MC4R reference sequence is unavailable, so structure "
            "positions cannot be mapped to pocket positions. The saved "
            "profile and contact distances below do not depend on it."
        )
        return payload

    try:
        geometry = geometry_for_structure(
            pdb_id, reference, profile=summary)
    except InterfaceGeometryUnavailable as exc:
        payload["geometry"] = None
        payload["geometry_unavailable"] = str(exc)
        return payload
    except Exception as exc:  # pragma: no cover - malformed coordinate file
        logger.warning("geometry for %s failed: %s", pdb_id, exc)
        payload["geometry"] = None
        payload["geometry_unavailable"] = (
            f"The cached coordinate file for {pdb_id} could not be read: {exc}"
        )
        return payload

    payload["geometry"] = asdict(geometry)
    payload["geometry"]["agrees_with_profile"] = geometry.agrees_with_profile
    return payload
