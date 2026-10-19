"""ATLAS 2.5 Stage 2: experimental assay context as an explicit covariate.

A pKi is not a property of a molecule alone. It is the result of a particular
experiment, and the 89 binding assays behind the MC4R dataset are not one
experiment repeated: 56 are cell-based, 27 use purified single protein, 4 are
cell-membrane preparations. A displacement constant measured on whole cells and
one measured on purified receptor are not interchangeable, so the brief is
right that assay context belongs in the representation rather than in the
noise term.

Measured variation across those 89 assays, which is why this module exists:

===================  ========================================================
``bao_format``       4 distinct: cell-based 56, single protein 27,
                     cell membrane 4, unspecified 2
``assay_cell_type``  7 distinct: HEK293 46, none recorded 32, CHO 4, ...
``confidence_score`` 2 distinct: 9 for 71, 8 for 18
``relationship_type`` 2 distinct: direct 71, homologous 18
``assay_type``       1 distinct: all binding, because the dataset filters to it
===================  ========================================================

**What is still absent.** pH, temperature and ionic strength are not in
ChEMBL's structured assay fields, so the microenvironment the brief asks about
cannot be encoded from this source. The residue-chemistry engine applies one
assumed pH of 7.4 uniformly and says so. Nothing here changes that.

**A caution this module cannot resolve by itself.** Assay identity correlates
with document, and document correlates with analogue series. An assay-context
feature can therefore act as a proxy for series membership, which under a
cluster split is a shortcut rather than chemistry. The benchmark ablates it for
exactly that reason, and a gain here should not be read as the model having
learned about assay physics.
"""

from __future__ import annotations

import collections
import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from atlas.core.evidence import utc_now
from atlas.ml.dataset import DATA_DIR
from atlas.sources.base import CachedHttpClient

logger = logging.getLogger(__name__)

__all__ = [
    "ASSAY_CONTEXT_PATH", "ASSAY_FIELDS", "fetch_assay_metadata",
    "assay_context_audit", "build_assay_vocabularies", "assay_encoding",
    "load_assay_metadata", "MISSING_CONDITIONS",
]

ASSAY_CONTEXT_PATH = DATA_DIR / "assay_context.json"
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"

#: The structured fields worth keeping. Each varies across this dataset's
#: assays; ``assay_type`` is recorded for provenance but does not vary, since
#: the dataset filters to binding assays.
ASSAY_FIELDS = (
    "assay_type", "bao_format", "bao_label", "assay_cell_type",
    "confidence_score", "relationship_type", "assay_organism",
)

#: Conditions the brief asks for that this source does not carry. Recorded as
#: data so a downstream consumer can see what is missing rather than assume it
#: was considered.
MISSING_CONDITIONS = (
    "ph", "temperature_c", "ionic_strength_mm", "buffer_composition",
    "incubation_time_min", "radioligand_concentration_nm",
)


def fetch_assay_metadata(
    assay_ids: Iterable[str], *, http: CachedHttpClient | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch ChEMBL assay records, 50 at a time, through the cached client."""
    ids = sorted({a for a in assay_ids if a})
    http = http or CachedHttpClient(ttl=timedelta(days=30))
    out: dict[str, dict[str, Any]] = {}
    for start in range(0, len(ids), 50):
        chunk = ids[start:start + 50]
        try:
            payload, _fetch = http.get_json(f"{CHEMBL_API}/assay", {
                "assay_chembl_id__in": ",".join(chunk),
                "limit": 50, "format": "json",
            })
        except Exception as exc:                      # pragma: no cover - network
            logger.warning("assay metadata chunk failed: %s", exc)
            continue
        for assay in payload.get("assays", []):
            key = assay.get("assay_chembl_id")
            if key:
                out[key] = {f: assay.get(f) for f in ASSAY_FIELDS}
                out[key]["description"] = (assay.get("description") or "")[:300]
    missing = [a for a in ids if a not in out]
    if missing:
        logger.warning("%d assay records unavailable", len(missing))
    return out


def load_assay_metadata(path: Path = ASSAY_CONTEXT_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def assay_context_audit(
    records: Sequence[Any], metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """How much the assay context actually varies, and how much is missing.

    Built before the encoder so that a field carrying no variation is reported
    as such rather than silently contributing a constant column.
    """
    counters: dict[str, collections.Counter] = {
        f: collections.Counter() for f in ASSAY_FIELDS
    }
    for assay in metadata.values():
        for f in ASSAY_FIELDS:
            counters[f][str(assay.get(f))] += 1

    per_molecule = [len(set(r.assay_ids)) for r in records]
    formats_per_molecule = []
    for r in records:
        formats = {
            metadata.get(a, {}).get("bao_label") for a in r.assay_ids
        } - {None}
        formats_per_molecule.append(len(formats))

    return {
        "n_assays_referenced": len({a for r in records for a in r.assay_ids}),
        "n_assays_resolved": len(metadata),
        "field_variation": {
            f: {"n_distinct": len(c), "counts": dict(c.most_common())}
            for f, c in counters.items()
        },
        "non_varying_fields": [f for f, c in counters.items() if len(c) <= 1],
        "assays_per_molecule": {
            "mean": float(np.mean(per_molecule)) if per_molecule else 0.0,
            "max": int(max(per_molecule)) if per_molecule else 0,
        },
        "molecules_spanning_several_formats": int(
            sum(1 for n in formats_per_molecule if n > 1)),
        "missing_conditions": list(MISSING_CONDITIONS),
        "caveat": (
            "Assay identity correlates with document, and document with "
            "analogue series, so this context can proxy for series membership. "
            "A gain from it is not evidence of learned assay physics and the "
            "benchmark ablates it."
        ),
    }


def build_assay_vocabularies(
    records: Sequence[Any], metadata: dict[str, dict[str, Any]], *,
    max_cell_types: int = 6,
) -> dict[str, list[str]]:
    """Vocabularies fitted on training records only.

    Fitting on everything would let the test set define the feature space,
    which is leakage even though no label is touched.
    """
    fmt: collections.Counter = collections.Counter()
    cell: collections.Counter = collections.Counter()
    for r in records:
        for a in r.assay_ids:
            meta = metadata.get(a)
            if not meta:
                continue
            fmt[str(meta.get("bao_label"))] += 1
            cell[str(meta.get("assay_cell_type"))] += 1
    return {
        "bao_label": [k for k, _ in fmt.most_common()],
        "assay_cell_type": [k for k, _ in cell.most_common(max_cell_types)],
    }


def assay_encoding(
    records: Sequence[Any], metadata: dict[str, dict[str, Any]],
    vocabularies: dict[str, list[str]],
) -> np.ndarray:
    """Assay context per molecule.

    A molecule may be measured in several assays, so format and cell type are
    encoded as the *distribution* over its measurements rather than a single
    label. An unresolved assay contributes nothing instead of a guess.
    """
    formats = vocabularies["bao_label"]
    cells = vocabularies["assay_cell_type"]
    width = len(formats) + len(cells) + 5
    out = np.zeros((len(records), width), dtype=float)

    for row, r in enumerate(records):
        metas = [metadata[a] for a in r.assay_ids if a in metadata]
        if not metas:
            out[row, -1] = 1.0            # flag: no assay context available
            continue
        for meta in metas:
            label = str(meta.get("bao_label"))
            if label in formats:
                out[row, formats.index(label)] += 1.0 / len(metas)
            cell = str(meta.get("assay_cell_type"))
            if cell in cells:
                out[row, len(formats) + cells.index(cell)] += 1.0 / len(metas)
        base = len(formats) + len(cells)
        scores = [m.get("confidence_score") for m in metas
                  if m.get("confidence_score") is not None]
        out[row, base] = (float(np.mean([float(s) for s in scores])) / 9.0
                          if scores else 0.0)
        out[row, base + 1] = float(np.mean(
            [1.0 if m.get("relationship_type") == "D" else 0.0 for m in metas]))
        out[row, base + 2] = np.log1p(len(set(r.assay_ids)))
        out[row, base + 3] = np.log1p(len(set(r.document_ids)))
    return out


def build_context(*, save: bool = True) -> dict[str, Any]:
    """Fetch and audit assay context for the MC4R dataset."""
    from atlas.ml.dataset import load_dataset

    records = load_dataset().records
    metadata = fetch_assay_metadata({a for r in records for a in r.assay_ids})
    report = {
        "audit": assay_context_audit(records, metadata),
        "metadata": metadata,
        "built_at": utc_now().isoformat(),
    }
    if save:
        ASSAY_CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
        ASSAY_CONTEXT_PATH.write_text(json.dumps(report, indent=2, default=str))
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = build_context()
    a = report["audit"]
    print(f"assays referenced {a['n_assays_referenced']}, "
          f"resolved {a['n_assays_resolved']}")
    for f, v in a["field_variation"].items():
        print(f"  {f:20s} {v['n_distinct']:2d} distinct  "
              f"{list(v['counts'].items())[:3]}")
    print(f"  non-varying: {a['non_varying_fields']}")
    print(f"  molecules spanning several formats: "
          f"{a['molecules_spanning_several_formats']}")
    print(f"  conditions absent from this source: {a['missing_conditions']}")
    print(f"\nSaved to {ASSAY_CONTEXT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
