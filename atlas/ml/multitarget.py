"""ATLAS 2.5 Stage 3: the melanocortin receptor panel.

A peptide-target dual encoder cannot be evaluated on one target. With MC4R
alone the target side has zero variance: any target encoder is a constant, and
an ablation removing it shows no change by construction. The Stage 1 audit
established the family is available — MC1R, MC3R, MC4R and MC5R all carry
substantial Ki data in ChEMBL — and, more to the point, that melanocortin
peptides are *routinely profiled across the whole family*, because subtype
selectivity is the central pharmacological question for this class.

That cross-receptor profiling is what makes selectivity learnable, so this
module measures it rather than assuming it: the overlap audit reports how many
molecules are actually measured against two or more receptors. If that number
were small, the panel would be four separate datasets wearing one name, and
the dual encoder would still not be evaluable.

Endpoints are kept strictly apart. A pKi and a pEC50 are different physical
quantities — one an equilibrium binding constant, the other a functional
response that depends on the assay's signal amplification — and a model
trained on their union predicts neither. The panel therefore holds one bundle
per (receptor, endpoint) pair and offers no method that merges them.
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from atlas.core.evidence import utc_now
from atlas.ml.dataset import (
    DATA_DIR, ENDPOINT_ASSAY_TYPE, LICENSE_NOTE, ActivityRecord, build_dataset,
)
from atlas.sources.base import CachedHttpClient

logger = logging.getLogger(__name__)

__all__ = [
    "MELANOCORTIN_RECEPTORS", "PANEL_PATH", "PanelRow", "TargetPanel",
    "build_panel", "load_panel", "leave_one_receptor_out",
    "generalisation_regimes",
]

PANEL_PATH = DATA_DIR / "melanocortin_panel.json"
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"

#: The human melanocortin receptor family. ChEMBL identifiers and preferred
#: names are as ChEMBL reports them; MC1R and MC2R are named in ChEMBL after
#: their endogenous ligands rather than by receptor number, which is why the
#: short names are recorded separately.
MELANOCORTIN_RECEPTORS: dict[str, dict[str, str]] = {
    "MC1R": {
        "target_chembl_id": "CHEMBL3795",
        "pref_name": "Melanocyte-stimulating hormone receptor",
        "organism": "Homo sapiens", "short_name": "MC1R",
    },
    "MC2R": {
        "target_chembl_id": "CHEMBL1965",
        "pref_name": "Adrenocorticotropic hormone receptor",
        "organism": "Homo sapiens", "short_name": "MC2R",
    },
    "MC3R": {
        "target_chembl_id": "CHEMBL4644",
        "pref_name": "Melanocortin receptor 3",
        "organism": "Homo sapiens", "short_name": "MC3R",
    },
    "MC4R": {
        "target_chembl_id": "CHEMBL259",
        "pref_name": "Melanocortin receptor 4",
        "organism": "Homo sapiens", "short_name": "MC4R",
    },
    "MC5R": {
        "target_chembl_id": "CHEMBL4608",
        "pref_name": "Melanocortin receptor 5",
        "organism": "Homo sapiens", "short_name": "MC5R",
    },
}


@dataclass(frozen=True, slots=True)
class PanelRow:
    """One molecule's aggregated activity on one receptor for one endpoint."""

    molecule_chembl_id: str
    smiles: str
    receptor: str
    target_chembl_id: str
    endpoint: str
    value: float                 # p-scale, -log10 M
    value_min: float
    value_max: float
    n_replicates: int
    replicate_spread: float
    document_ids: tuple[str, ...]
    assay_ids: tuple[str, ...]

    @property
    def is_discordant(self) -> bool:
        return self.replicate_spread > 1.0


def _receptor_sequences(http: CachedHttpClient) -> dict[str, dict[str, Any]]:
    """Fetch each receptor's UniProt accession and sequence, with provenance.

    The accession comes from ChEMBL's own target record rather than from
    recall, and the sequence from UniProt. A target encoder in Stage 4 needs
    these; fetching them here keeps the panel self-contained and makes the
    provenance checkable.
    """
    out: dict[str, dict[str, Any]] = {}
    for short, target in MELANOCORTIN_RECEPTORS.items():
        meta: dict[str, Any] = {**target}
        try:
            payload, fetch = http.get_json(
                f"{CHEMBL_API}/target/{target['target_chembl_id']}",
                {"format": "json"},
            )
            accessions = [
                c.get("accession")
                for c in payload.get("target_components", [])
                if c.get("accession")
            ]
            meta["uniprot_accessions"] = accessions
            meta["chembl_target_url"] = fetch.url
            if accessions:
                seq_fetch = http.get(
                    UNIPROT_FASTA.format(acc=accessions[0]), accept="text/plain")
                text = seq_fetch.body.decode("utf-8", errors="replace")
                lines = [ln for ln in text.splitlines() if ln and not ln.startswith(">")]
                meta["sequence"] = "".join(lines)
                meta["sequence_length"] = len(meta["sequence"])
                meta["sequence_url"] = seq_fetch.url
        except Exception as exc:                       # pragma: no cover - network
            logger.warning("receptor metadata for %s unavailable: %s", short, exc)
            meta["sequence"] = None
            meta["metadata_error"] = str(exc)
        out[short] = meta
    return out


def _overlap_audit(rows: Sequence[PanelRow], endpoint: str) -> dict[str, Any]:
    """How much cross-receptor profiling the panel actually contains.

    Reported rather than assumed. Selectivity is only learnable from molecules
    measured on more than one receptor; without them the panel is several
    disjoint datasets and a target encoder has nothing to generalise over.
    """
    subset = [r for r in rows if r.endpoint == endpoint]
    by_mol: dict[str, set[str]] = collections.defaultdict(set)
    for r in subset:
        by_mol[r.molecule_chembl_id].add(r.receptor)

    breadth = collections.Counter(len(v) for v in by_mol.values())
    multi = {m: v for m, v in by_mol.items() if len(v) > 1}

    # Within molecules measured on several receptors, how far apart are the
    # values? This is the selectivity signal a model would have to predict.
    spans: list[float] = []
    by_key = {(r.molecule_chembl_id, r.receptor): r.value for r in subset}
    for mol, recs in multi.items():
        vals = [by_key[(mol, rec)] for rec in recs]
        spans.append(max(vals) - min(vals))

    pair_counts = collections.Counter()
    for recs in multi.values():
        for a in sorted(recs):
            for b in sorted(recs):
                if a < b:
                    pair_counts[f"{a}|{b}"] += 1

    return {
        "endpoint": endpoint,
        "n_measurements": len(subset),
        "n_distinct_molecules": len(by_mol),
        "receptors_per_molecule": dict(sorted(breadth.items())),
        "n_molecules_on_multiple_receptors": len(multi),
        "fraction_on_multiple_receptors": len(multi) / max(len(by_mol), 1),
        "median_selectivity_span_log": float(np.median(spans)) if spans else None,
        "max_selectivity_span_log": float(max(spans)) if spans else None,
        "receptor_pair_counts": dict(pair_counts.most_common()),
        "note": (
            "A selectivity span is the range of p-values one molecule shows "
            "across the receptors it was measured on. It is the quantity a "
            "target encoder must predict; a panel of unrelated molecules per "
            "receptor would leave it undefined."
        ),
    }


@dataclass
class TargetPanel:
    """Activities for several receptors, with endpoints kept separate."""

    rows: list[PanelRow]
    receptors: dict[str, dict[str, Any]]
    endpoints: list[str]
    counts: dict[str, Any]
    overlap: dict[str, Any]
    license_note: str
    built_at: str
    source_urls: list[str] = field(default_factory=list)

    @property
    def version(self) -> str:
        body = [
            [r.molecule_chembl_id, r.receptor, r.endpoint, round(r.value, 4)]
            for r in sorted(self.rows, key=lambda r: (
                r.endpoint, r.receptor, r.molecule_chembl_id))
        ]
        blob = json.dumps(body, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def for_endpoint(self, endpoint: str) -> list[PanelRow]:
        """Rows for one endpoint. There is deliberately no method for several."""
        if endpoint not in self.endpoints:
            raise KeyError(
                f"endpoint {endpoint!r} is not in this panel; it holds "
                f"{self.endpoints}"
            )
        return [r for r in self.rows if r.endpoint == endpoint]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "receptors": self.receptors,
            "endpoints": self.endpoints,
            "counts": self.counts,
            "overlap": self.overlap,
            "license_note": self.license_note,
            "built_at": self.built_at,
            "source_urls": self.source_urls,
            "rows": [
                {
                    "molecule_chembl_id": r.molecule_chembl_id,
                    "smiles": r.smiles, "receptor": r.receptor,
                    "target_chembl_id": r.target_chembl_id,
                    "endpoint": r.endpoint, "value": r.value,
                    "value_min": r.value_min, "value_max": r.value_max,
                    "n_replicates": r.n_replicates,
                    "replicate_spread": r.replicate_spread,
                    "document_ids": list(r.document_ids),
                    "assay_ids": list(r.assay_ids),
                } for r in self.rows
            ],
        }

    def save(self, path: Path = PANEL_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


def build_panel(
    *,
    receptors: Iterable[str] = tuple(MELANOCORTIN_RECEPTORS),
    endpoints: Iterable[str] = ("Ki",),
    http: CachedHttpClient | None = None,
    with_sequences: bool = True,
) -> TargetPanel:
    """Build the receptor panel, one dataset per (receptor, endpoint)."""
    for e in endpoints:
        if e not in ENDPOINT_ASSAY_TYPE:
            raise ValueError(f"unknown endpoint {e!r}")

    http = http or CachedHttpClient(ttl=timedelta(days=30))
    rows: list[PanelRow] = []
    counts: dict[str, Any] = {"per_receptor_endpoint": {}}
    urls: list[str] = []

    for short in receptors:
        target = MELANOCORTIN_RECEPTORS[short]
        for endpoint in endpoints:
            bundle = build_dataset(target=target, endpoint=endpoint, http=http)
            urls.extend(bundle.source_urls)
            for rec in bundle.records:
                rows.append(PanelRow(
                    molecule_chembl_id=rec.molecule_chembl_id,
                    smiles=rec.smiles, receptor=short,
                    target_chembl_id=target["target_chembl_id"],
                    endpoint=endpoint, value=rec.pki,
                    value_min=rec.pki_min, value_max=rec.pki_max,
                    n_replicates=rec.n_replicates,
                    replicate_spread=rec.replicate_spread,
                    document_ids=rec.document_ids, assay_ids=rec.assay_ids,
                ))
            counts["per_receptor_endpoint"][f"{short}|{endpoint}"] = len(bundle.records)
            logger.info("%s %s: %d molecules", short, endpoint, len(bundle.records))

    counts["total_measurements"] = len(rows)
    counts["distinct_molecules"] = len({r.molecule_chembl_id for r in rows})

    return TargetPanel(
        rows=rows,
        receptors=(_receptor_sequences(http) if with_sequences
                   else dict(MELANOCORTIN_RECEPTORS)),
        endpoints=list(endpoints),
        counts=counts,
        overlap={e: _overlap_audit(rows, e) for e in endpoints},
        license_note=LICENSE_NOTE,
        built_at=utc_now().isoformat(),
        source_urls=sorted(set(urls)),
    )


def load_panel(path: Path = PANEL_PATH) -> TargetPanel:
    payload = json.loads(path.read_text())
    panel = TargetPanel(
        rows=[PanelRow(
            molecule_chembl_id=r["molecule_chembl_id"], smiles=r["smiles"],
            receptor=r["receptor"], target_chembl_id=r["target_chembl_id"],
            endpoint=r["endpoint"], value=r["value"], value_min=r["value_min"],
            value_max=r["value_max"], n_replicates=r["n_replicates"],
            replicate_spread=r["replicate_spread"],
            document_ids=tuple(r["document_ids"]),
            assay_ids=tuple(r["assay_ids"]),
        ) for r in payload["rows"]],
        receptors=payload["receptors"], endpoints=payload["endpoints"],
        counts=payload["counts"], overlap=payload["overlap"],
        license_note=payload["license_note"], built_at=payload["built_at"],
        source_urls=payload.get("source_urls", []),
    )
    if panel.version != payload["version"]:
        raise ValueError(
            f"panel at {path} records version {payload['version']} but hashes "
            f"to {panel.version}; the file has been altered"
        )
    return panel


# ------------------------------------------------------------------ splits ----

def leave_one_receptor_out(
    rows: Sequence[PanelRow], held_out: str
) -> tuple[np.ndarray, np.ndarray]:
    """Train on every receptor but one; test on the held-out receptor.

    This is the brief's "known peptide family, related target" regime. It is
    the only split that tests a target encoder at all: under any split that
    keeps all receptors in training, a model can succeed without ever having
    to generalise across targets.
    """
    if held_out not in {r.receptor for r in rows}:
        raise ValueError(f"receptor {held_out!r} is not present in these rows")
    test = np.array([i for i, r in enumerate(rows) if r.receptor == held_out])
    train = np.array([i for i, r in enumerate(rows) if r.receptor != held_out])
    return train, test


def generalisation_regimes(
    rows: Sequence[PanelRow],
    peptide_cluster: Sequence[int],
    held_out_receptor: str,
) -> dict[str, dict[str, Any]]:
    """The three regimes the brief distinguishes, as explicit index sets.

    ``peptide_cluster`` is a cluster label per row, from the existing
    Taylor-Butina clustering. The regimes differ in what the held-out rows
    share with training:

    * **known target, new peptide family** — same receptors, unseen chemistry.
    * **known peptide family, related target** — unseen receptor, chemistry
      the model has seen on other receptors.
    * **new peptide family and new target** — neither. This is the regime whose
      result must not be reported as though it were one of the others, and it
      is usually much smaller, because it needs molecules that are both on the
      held-out receptor and in clusters absent from the remaining data.
    """
    labels = np.asarray(peptide_cluster)
    if len(labels) != len(rows):
        raise ValueError(
            f"peptide_cluster has {len(labels)} labels for {len(rows)} rows"
        )
    receptor = np.array([r.receptor for r in rows])

    on_held = receptor == held_out_receptor
    clusters_elsewhere = set(labels[~on_held].tolist())

    known_target_new_family = ~on_held & ~np.isin(labels, list(clusters_elsewhere))
    related_target = on_held & np.isin(labels, list(clusters_elsewhere))
    both_new = on_held & ~np.isin(labels, list(clusters_elsewhere))

    def describe(mask: np.ndarray, train_mask: np.ndarray) -> dict[str, Any]:
        return {
            "n_test": int(mask.sum()), "n_train": int(train_mask.sum()),
            "test_idx": np.where(mask)[0].tolist(),
            "train_idx": np.where(train_mask)[0].tolist(),
        }

    return {
        "known_target_new_peptide_family": {
            **describe(known_target_new_family, ~on_held & ~known_target_new_family),
            "description": (
                "Held-out chemistry on receptors the model has seen. Tests "
                "peptide generalisation only."
            ),
        },
        "known_peptide_family_related_target": {
            **describe(related_target, ~on_held),
            "description": (
                f"{held_out_receptor} held out entirely; its molecules belong "
                "to clusters seen on other receptors. Tests transfer across "
                "receptor variants."
            ),
        },
        "new_peptide_family_and_new_target": {
            **describe(both_new, ~on_held),
            "description": (
                f"{held_out_receptor} held out, and these molecules' clusters "
                "appear nowhere in the remaining data. The hardest regime, and "
                "typically the smallest."
            ),
        },
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    panel = build_panel()
    path = panel.save()
    print(f"panel version {panel.version}")
    print(f"endpoints     {panel.endpoints}")
    for k, v in panel.counts["per_receptor_endpoint"].items():
        print(f"    {k:16s} {v}")
    for endpoint, o in panel.overlap.items():
        print(f"\noverlap, {endpoint}:")
        for k in ("n_distinct_molecules", "n_molecules_on_multiple_receptors",
                  "fraction_on_multiple_receptors", "receptors_per_molecule",
                  "median_selectivity_span_log", "max_selectivity_span_log"):
            print(f"    {k:36s} {o[k]}")
        print(f"    pairs: {o['receptor_pair_counts']}")
    for short, meta in panel.receptors.items():
        print(f"    {short}: {meta.get('uniprot_accessions')} "
              f"len={meta.get('sequence_length')}")
    print(f"\nSaved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
