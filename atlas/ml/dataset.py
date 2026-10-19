"""Reproducible dataset construction from ChEMBL bioactivity data.

The endpoint is **MC4R binding affinity (pKi)**. Melanocortin-4 receptor was
chosen because it is directly relevant to this registry -- PT-141, melanotan I
and melanotan II act on melanocortin receptors -- and because ChEMBL holds both
peptide and small-molecule ligands for it, which makes it possible to measure
how badly a model transfers between chemical modalities instead of assuming it
does.

Two deliberate restrictions keep the endpoint physically homogeneous:

* **Ki only.** ChEMBL also holds IC50 and EC50 for this target. IC50 depends on
  radioligand concentration through the Cheng-Prusoff relation, and EC50 is a
  functional rather than a binding measurement. Pooling them would add roughly
  2,800 records at the cost of mixing different quantities under one label.
* **Binding assays only** (``assay_type == "B"``), for the same reason.

Three further filters were added after inspecting what the raw query returns:

* **Wild-type receptor only.** 67 activities (2.4%) measure binding to *mutant*
  MC4R -- alanine scans such as I129A and Y268A. A mutant receptor is a
  different protein, and pooling it into a wild-type affinity label is a silent
  error. ChEMBL's structured ``assay_variant_mutation`` field is authoritative
  here and agreed with a description-text heuristic on all 67 records.
* **ChEMBL's own validity flags respected.** 32 records carry a
  ``data_validity_comment`` ("Outside typical range", "Potential transcription
  error") and are dropped rather than silently averaged in.
* **Duplicate records collapsed before aggregation.** 455 activities are
  flagged ``potential_duplicate``. Counting them as independent replicates
  inflated the apparent replication and drove the median replicate spread to
  zero. Records are now deduplicated on (document, rounded value) so that the
  replicate count reflects distinct reported measurements.

Replicate measurements of the same molecule are collapsed to their median, and
the spread between replicates is retained: it is an estimate of experimental
noise and therefore a floor on the error any model can achieve. A model whose
held-out error approaches it is as good as the data permits.
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable

from atlas.core.evidence import utc_now
from atlas.sources.base import CachedHttpClient

logger = logging.getLogger(__name__)

__all__ = [
    "ActivityRecord",
    "DatasetBundle",
    "build_dataset",
    "load_dataset",
    "DATASET_PATH",
    "MC4R",
]

API = "https://www.ebi.ac.uk/chembl/api/data"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "ml"
DATASET_PATH = DATA_DIR / "mc4r_pki.json"

MC4R = {
    "target_chembl_id": "CHEMBL259",
    "pref_name": "Melanocortin receptor 4",
    "organism": "Homo sapiens",
}

LICENSE_NOTE = (
    "Derived from ChEMBL (EMBL-EBI), released under CC BY-SA 3.0. "
    "Attribution and share-alike terms apply to redistribution of this dataset."
)

#: pKi outside this range is almost certainly a unit or transcription error.
PKI_MIN, PKI_MAX = 3.0, 12.5


@dataclass(frozen=True, slots=True)
class ActivityRecord:
    """One molecule's aggregated binding affinity against the target."""

    molecule_chembl_id: str
    smiles: str
    pki: float
    pki_min: float
    pki_max: float
    n_replicates: int
    replicate_spread: float      # max - min across replicates, in log units
    molecule_type: str | None
    document_ids: tuple[str, ...]
    assay_ids: tuple[str, ...]

    @property
    def is_replicated(self) -> bool:
        return self.n_replicates > 1

    @property
    def is_discordant(self) -> bool:
        """Replicates disagreeing by more than a log unit are not a point value.

        PT-141 is the motivating case: its records span pKi 7.61 to 9.60, a
        hundredfold range, so reporting the median alone would misrepresent
        what is known.
        """
        return self.replicate_spread > 1.0


@dataclass
class DatasetBundle:
    """A dataset plus everything needed to rebuild and audit it."""

    records: list[ActivityRecord]
    target: dict[str, str]
    endpoint: str
    filters: dict[str, Any]
    counts: dict[str, int]
    noise_estimate: dict[str, float]
    license_note: str
    built_at: str
    source_urls: list[str] = field(default_factory=list)

    @property
    def version(self) -> str:
        """Content hash over the data itself, so a changed dataset changes id."""
        body = sorted((r.molecule_chembl_id, round(r.pki, 6)) for r in self.records)
        blob = json.dumps(body, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "endpoint": self.endpoint,
            "target": self.target,
            "filters": self.filters,
            "counts": self.counts,
            "noise_estimate": self.noise_estimate,
            "license_note": self.license_note,
            "built_at": self.built_at,
            "source_urls": self.source_urls,
            "records": [asdict(r) for r in self.records],
        }

    def save(self, path: Path = DATASET_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


def _page_activities(
    http: CachedHttpClient, params: dict[str, Any]
) -> Iterable[dict[str, Any]]:
    offset = 0
    seen_urls: list[str] = []
    while True:
        payload, fetch = http.get_json(
            f"{API}/activity", {**params, "limit": 1000, "offset": offset}
        )
        seen_urls.append(fetch.url)
        acts = payload.get("activities", [])
        if not acts:
            break
        yield from acts
        offset += 1000
        if offset >= payload.get("page_meta", {}).get("total_count", 0):
            break
    _page_activities.last_urls = seen_urls  # type: ignore[attr-defined]


def build_dataset(
    *,
    target: dict[str, str] = MC4R,
    http: CachedHttpClient | None = None,
) -> DatasetBundle:
    """Fetch, filter and aggregate MC4R binding affinities from ChEMBL."""
    http = http or CachedHttpClient(ttl=timedelta(days=30))
    params = {
        "target_chembl_id": target["target_chembl_id"],
        "standard_type": "Ki",
        "standard_units": "nM",
        "standard_relation": "=",
        "assay_type": "B",
        "format": "json",
    }

    raw = list(_page_activities(http, params))
    urls = getattr(_page_activities, "last_urls", [])

    rejected: dict[str, int] = {
        "no_pchembl": 0, "no_smiles": 0, "out_of_range": 0, "unparseable": 0,
        "mutant_receptor": 0, "chembl_validity_flag": 0, "duplicate_record": 0,
    }
    by_molecule: dict[str, dict[str, Any]] = {}
    seen_measurements: set[tuple[str, str, float]] = set()

    for act in raw:
        # Mutant receptors are a different protein from the wild type.
        if act.get("assay_variant_mutation"):
            rejected["mutant_receptor"] += 1
            continue
        # ChEMBL's curators have already flagged these as suspect.
        if act.get("data_validity_comment"):
            rejected["chembl_validity_flag"] += 1
            continue
        mol_id = act.get("molecule_chembl_id")
        smiles = act.get("canonical_smiles")
        pchembl = act.get("pchembl_value")

        if not mol_id:
            rejected["unparseable"] += 1
            continue
        if pchembl in (None, ""):
            rejected["no_pchembl"] += 1
            continue
        if not smiles:
            rejected["no_smiles"] += 1
            continue
        try:
            value = float(pchembl)
        except (TypeError, ValueError):
            rejected["unparseable"] += 1
            continue
        if not (PKI_MIN <= value <= PKI_MAX):
            rejected["out_of_range"] += 1
            continue

        # Collapse duplicate records of one measurement: the same value in the
        # same document is one experiment, however many rows carry it.
        fingerprint = (mol_id, act.get("document_chembl_id") or "", round(value, 2))
        if fingerprint in seen_measurements:
            rejected["duplicate_record"] += 1
            continue
        seen_measurements.add(fingerprint)

        slot = by_molecule.setdefault(
            mol_id,
            {"smiles": smiles, "values": [], "docs": set(), "assays": set(), "type": None},
        )
        slot["values"].append(value)
        if act.get("document_chembl_id"):
            slot["docs"].add(act["document_chembl_id"])
        if act.get("assay_chembl_id"):
            slot["assays"].add(act["assay_chembl_id"])

    records: list[ActivityRecord] = []
    spreads: list[float] = []
    for mol_id, slot in by_molecule.items():
        values = slot["values"]
        spread = (max(values) - min(values)) if len(values) > 1 else 0.0
        if len(values) > 1:
            spreads.append(spread)
        records.append(
            ActivityRecord(
                molecule_chembl_id=mol_id,
                smiles=slot["smiles"],
                pki=float(statistics.median(values)),
                pki_min=float(min(values)),
                pki_max=float(max(values)),
                n_replicates=len(values),
                replicate_spread=float(spread),
                molecule_type=slot["type"],
                document_ids=tuple(sorted(slot["docs"])),
                assay_ids=tuple(sorted(slot["assays"])),
            )
        )

    records.sort(key=lambda r: r.molecule_chembl_id)

    # Experimental noise floor: how far apart repeated measurements of the same
    # molecule are. The median across all replicated molecules is 0, because
    # ChEMBL carries the same measurement under several activity records --
    # even across different documents. Only the subset whose values actually
    # differ carries information about measurement error, so that subset is
    # reported separately and is the figure worth comparing a model against.
    replicated = [r for r in records if r.is_replicated]
    differing = [s for s in spreads if s > 0]
    noise = {
        "n_replicated_molecules": len(replicated),
        "n_molecules_with_differing_values": len(differing),
        "median_spread_all_replicates_log": (
            float(statistics.median(spreads)) if spreads else 0.0
        ),
        "median_spread_among_differing_log": (
            float(statistics.median(differing)) if differing else 0.0
        ),
        "implied_mae_floor_log": (
            float(statistics.median(differing)) / 2.0 if differing else 0.0
        ),
        "n_discordant_molecules": sum(1 for r in records if r.is_discordant),
        "caveat": (
            "Computed after collapsing duplicate records, so it reflects "
            "distinct reported measurements. It still understates true "
            "inter-laboratory variation, since agreement within a curated "
            "database is not independent replication."
        ),
    }

    return DatasetBundle(
        records=records,
        target=target,
        endpoint="MC4R binding affinity (pKi, -log10 M)",
        filters={
            "standard_type": "Ki",
            "standard_units": "nM",
            "standard_relation": "=",
            "assay_type": "B (binding)",
            "pki_range": [PKI_MIN, PKI_MAX],
            "aggregation": "median across distinct replicate measurements",
            "wild_type_only": (
                "Activities with assay_variant_mutation set were excluded: a "
                "mutant receptor is a different protein."
            ),
            "validity": "Records carrying a ChEMBL data_validity_comment were excluded.",
            "deduplication": (
                "Records sharing (molecule, document, rounded pChEMBL) were "
                "collapsed to one measurement."
            ),
            "excluded": (
                "IC50 and EC50 were excluded: IC50 depends on radioligand "
                "concentration via Cheng-Prusoff, and EC50 is a functional "
                "measurement. Pooling them would mix physical quantities."
            ),
        },
        counts={
            "raw_activities": len(raw),
            "unique_molecules": len(records),
            **{f"rejected_{k}": v for k, v in rejected.items()},
        },
        noise_estimate=noise,
        license_note=LICENSE_NOTE,
        built_at=utc_now().isoformat(),
        source_urls=urls,
    )


def load_dataset(path: Path = DATASET_PATH) -> DatasetBundle:
    """Load a saved dataset, preserving its recorded version and provenance."""
    payload = json.loads(path.read_text())
    bundle = DatasetBundle(
        records=[ActivityRecord(**{**r,
                                   "document_ids": tuple(r["document_ids"]),
                                   "assay_ids": tuple(r["assay_ids"])})
                 for r in payload["records"]],
        target=payload["target"],
        endpoint=payload["endpoint"],
        filters=payload["filters"],
        counts=payload["counts"],
        noise_estimate=payload["noise_estimate"],
        license_note=payload["license_note"],
        built_at=payload["built_at"],
        source_urls=payload.get("source_urls", []),
    )
    if bundle.version != payload["version"]:
        raise ValueError(
            f"dataset at {path} has version {payload['version']} on disk but "
            f"hashes to {bundle.version}; the file has been altered"
        )
    return bundle


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    bundle = build_dataset()
    path = bundle.save()

    print(f"endpoint        {bundle.endpoint}")
    print(f"target          {bundle.target['target_chembl_id']} "
          f"({bundle.target['pref_name']})")
    print(f"dataset version {bundle.version}")
    print(f"\ncounts:")
    for k, v in bundle.counts.items():
        print(f"    {k:28} {v}")
    print(f"\nexperimental noise:")
    for k, v in bundle.noise_estimate.items():
        print(f"    {k:28} {v:.4g}" if isinstance(v, float) else f"    {k:28} {v}")
    pkis = [r.pki for r in bundle.records]
    print(f"\npKi: min {min(pkis):.2f}  median {statistics.median(pkis):.2f}  "
          f"max {max(pkis):.2f}  sd {statistics.stdev(pkis):.2f}")
    print(f"\nsaved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
