"""Multi-endpoint MC4R datasets, kept deliberately separate.

ChEMBL holds four activity types for this target. They are different physical
quantities and are built as different datasets:

===============  ===========  =========================================
endpoint         peptides     what it measures
===============  ===========  =========================================
Ki (binding)         628      equilibrium binding affinity
EC50 (functional)    551      agonist potency in a functional assay
IC50 (binding)       272      displacement, concentration-dependent
Kd (binding)          11      direct binding constant
===============  ===========  =========================================

Merging them is not a matter of taste. For the 201 peptides measured both ways,
pKi and pEC50 correlate at Pearson r 0.657, with a mean difference of -0.311
log units and a standard deviation of **1.03 log units** -- about the size of
the whole model error. Pooling them would inject that spread as label noise.

The overlap is nonetheless large enough that multitask sharing is worth testing
rather than assuming, which :mod:`atlas.ml.peptide_v2` does.
"""

from __future__ import annotations

import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from rdkit import Chem, RDLogger

from atlas.ml.features import Modality, classify_modality
from atlas.sources.base import CachedHttpClient

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)

__all__ = ["Endpoint", "ENDPOINTS", "EndpointDataset", "load_endpoint"]

API = "https://www.ebi.ac.uk/chembl/api/data"
TARGET = "CHEMBL259"
PKI_MIN, PKI_MAX = 3.0, 12.5


@dataclass(frozen=True)
class Endpoint:
    """One activity type and assay format, with what it actually measures."""

    key: str
    standard_type: str
    assay_type: str          # "B" binding, "F" functional
    quantity: str
    interpretation: str

    @property
    def label(self) -> str:
        fmt = {"B": "binding", "F": "functional"}[self.assay_type]
        return f"{self.standard_type} ({fmt})"


ENDPOINTS: dict[str, Endpoint] = {
    "ki_binding": Endpoint(
        "ki_binding", "Ki", "B", "equilibrium binding affinity",
        "How tightly the molecule binds the receptor. Says nothing about "
        "whether binding produces a response.",
    ),
    "ec50_functional": Endpoint(
        "ec50_functional", "EC50", "F", "agonist potency",
        "The concentration producing half-maximal response in a functional "
        "assay. Depends on receptor reserve and assay system, and is not a "
        "binding constant.",
    ),
    "ic50_binding": Endpoint(
        "ic50_binding", "IC50", "B", "displacement potency",
        "Concentration displacing half the radioligand. Depends on radioligand "
        "concentration through the Cheng-Prusoff relation.",
    ),
}


@dataclass
class EndpointDataset:
    endpoint: Endpoint
    values: dict[str, float]            # molecule id -> median p-value
    smiles: dict[str, str]
    modality: dict[str, Modality]
    n_raw: int
    n_rejected: dict[str, int]

    def subset(self, modality: Modality) -> dict[str, float]:
        return {k: v for k, v in self.values.items() if self.modality.get(k) is modality}

    @property
    def n_peptide(self) -> int:
        return sum(1 for m in self.modality.values() if m is Modality.PEPTIDE_LIKE)

    def summary(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint.key,
            "label": self.endpoint.label,
            "quantity": self.endpoint.quantity,
            "n_molecules": len(self.values),
            "n_peptide_like": self.n_peptide,
            "n_small_molecule_like": len(self.values) - self.n_peptide,
            "n_raw_activities": self.n_raw,
            "rejected": self.n_rejected,
        }


def load_endpoint(
    key: str, *, http: CachedHttpClient | None = None
) -> EndpointDataset:
    """Fetch and aggregate one endpoint, applying the same quality filters."""
    endpoint = ENDPOINTS[key]
    http = http or CachedHttpClient(ttl=timedelta(days=30))

    raw: list[dict[str, Any]] = []
    offset = 0
    while True:
        payload, _ = http.get_json(f"{API}/activity", {
            "target_chembl_id": TARGET,
            "standard_type": endpoint.standard_type,
            "standard_units": "nM",
            "standard_relation": "=",
            "assay_type": endpoint.assay_type,
            "format": "json", "limit": 1000, "offset": offset,
        })
        acts = payload.get("activities", [])
        if not acts:
            break
        raw.extend(acts)
        offset += 1000
        if offset >= payload.get("page_meta", {}).get("total_count", 0):
            break

    rejected = {"no_pchembl": 0, "mutant": 0, "validity_flag": 0,
                "no_smiles": 0, "duplicate": 0, "out_of_range": 0}
    grouped: dict[str, list[float]] = defaultdict(list)
    smiles: dict[str, str] = {}
    seen: set[tuple] = set()

    for act in raw:
        if act.get("assay_variant_mutation"):
            rejected["mutant"] += 1
            continue
        if act.get("data_validity_comment"):
            rejected["validity_flag"] += 1
            continue
        if not act.get("pchembl_value"):
            rejected["no_pchembl"] += 1
            continue
        if not act.get("canonical_smiles"):
            rejected["no_smiles"] += 1
            continue
        value = float(act["pchembl_value"])
        if not (PKI_MIN <= value <= PKI_MAX):
            rejected["out_of_range"] += 1
            continue
        mol_id = act["molecule_chembl_id"]
        fingerprint = (mol_id, act.get("document_chembl_id") or "", round(value, 2))
        if fingerprint in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(fingerprint)
        grouped[mol_id].append(value)
        smiles[mol_id] = act["canonical_smiles"]

    values = {k: float(statistics.median(v)) for k, v in grouped.items()}
    modality: dict[str, Modality] = {}
    for mol_id, smi in smiles.items():
        mol = Chem.MolFromSmiles(smi)
        modality[mol_id] = (
            classify_modality(mol) if mol is not None else Modality.UNPARSEABLE
        )

    return EndpointDataset(
        endpoint=endpoint, values=values, smiles=smiles, modality=modality,
        n_raw=len(raw), n_rejected=rejected,
    )
