"""Build and serve the store of source-backed pharmacokinetic parameters.

This is the artefact that unblocks human PK simulation. Until it existed, the
evidence gate refused a human pharmacokinetic simulation for all 46 registry
entries, because literature *mentioning* pharmacokinetics is not the same as
holding parameters. A compound appears here only when a label states enough to
parameterise a model without any value being invented.

Run as::

    python -m atlas.mechanistic.pk_store          # build from openFDA
    python -m atlas.mechanistic.pk_store --offline
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

from atlas.core.evidence import utc_now
from atlas.mechanistic.pk_extraction import FDALabelClient
from atlas.mechanistic.pk_validation import build_parameter_set, validate_against_label
from atlas.registry.seed import SEED_REGISTRY
from atlas.sources.base import CachedHttpClient

logger = logging.getLogger(__name__)

__all__ = ["build_store", "load_store", "parameters_for", "STORE_PATH"]

STORE_PATH = Path(__file__).resolve().parents[2] / "data" / "pk_parameters.json"

#: Registry key -> the generic name openFDA indexes the product under. Only
#: entries plausibly having an approved product are attempted; the rest have no
#: label by definition, which is itself the finding.
LABEL_NAMES: dict[str, str] = {
    "pt-141": "bremelanotide",
    "semaglutide": "semaglutide",
    "tirzepatide": "tirzepatide",
    "tesamorelin": "tesamorelin",
    "melanotan-1": "afamelanotide",
    "sermorelin": "sermorelin",
    "thymosin-alpha-1": "thymalfasin",
    "cagrilintide": "cagrilintide",
    "glutathione": "glutathione",
    "nad": "nicotinamide adenine dinucleotide",
    "vip": "aviptadil",
    "kisspeptin": "kisspeptin",
    "ara-290": "cibinetide",
}


def build_store(*, offline: bool = False) -> dict[str, Any]:
    http = CachedHttpClient(ttl=timedelta(days=30), offline=offline)
    client = FDALabelClient(http)

    entries: dict[str, Any] = {}
    attempted: list[str] = []
    for key, generic in LABEL_NAMES.items():
        attempted.append(key)
        try:
            record = client.fetch(generic)
        except Exception as exc:
            logger.warning("label fetch failed for %s: %s", generic, exc)
            continue
        if record is None:
            entries[key] = {
                "registry_key": key, "generic_name": generic,
                "status": "no_label_or_no_pk_section", "parameters": None,
            }
            continue

        ps = build_parameter_set(record)
        payload: dict[str, Any] = {
            "registry_key": key,
            "generic_name": generic,
            "brand_name": record.brand_name,
            "route": record.route,
            "dose_mg": record.dose_mg,
            "status": "usable" if ps.is_usable else "insufficient_parameters",
            "blocked_reason": ps.blocked_reason,
            "parameters": ps.parameters if ps.is_usable else None,
            "model": ps.model_name if ps.is_usable else None,
            "derived": ps.derived,
            "provenance": ps.provenance,
            "extracted": [asdict(p) | {"qualifier": p.qualifier.value}
                          for p in record.parameters],
            "warnings": record.warnings,
            "source": json.loads(record.source.model_dump_json()),
        }

        if ps.is_usable:
            report = validate_against_label(record, n_monte_carlo=1500)
            payload["validation"] = {
                "verdict": report.verdict,
                "n_targets": len(report.targets),
                "n_agreeing": report.n_agreeing,
                "all_agree": report.all_agree,
                "targets": [
                    {
                        "name": t.name, "observed": t.observed, "unit": t.unit,
                        "predicted": t.predicted,
                        "predicted_low": t.predicted_low,
                        "predicted_high": t.predicted_high,
                        "relative_error": t.relative_error,
                        "agrees": t.agrees,
                        "observed_within_interval": t.observed_within_interval,
                        "source_quote": t.source_quote,
                    }
                    for t in report.targets
                ],
                "limitations": report.limitations,
            }
        entries[key] = payload

    usable = [k for k, v in entries.items() if v.get("status") == "usable"]
    return {
        "created_at": utc_now().isoformat(),
        "n_attempted": len(attempted),
        "n_with_label": sum(1 for v in entries.values()
                            if v.get("status") != "no_label_or_no_pk_section"),
        "n_usable": len(usable),
        "usable_keys": sorted(usable),
        "note": (
            "A compound is 'usable' only when its label states clearance, "
            "volume of distribution, bioavailability as a point estimate, Tmax "
            "and a dose. Anything less is reported as insufficient rather than "
            "completed with an assumed value."
        ),
        "entries": entries,
    }


@lru_cache(maxsize=1)
def load_store(path: Path = STORE_PATH) -> dict[str, Any]:
    if not path.exists():
        return {"entries": {}, "usable_keys": [], "n_usable": 0}
    return json.loads(path.read_text())


def parameters_for(registry_key: str) -> dict[str, Any] | None:
    """Source-backed parameters for a registry entry, or ``None``."""
    entry = load_store().get("entries", {}).get(registry_key)
    if not entry or entry.get("status") != "usable":
        return None
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s",
                        stream=sys.stderr)

    store = build_store(offline=args.offline)
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(json.dumps(store, indent=2, allow_nan=False))

    print("=" * 96)
    print("PHARMACOKINETIC PARAMETER STORE".center(96))
    print("=" * 96)
    print(f"attempted {store['n_attempted']} registry entries · "
          f"{store['n_with_label']} had a label with a PK section · "
          f"{store['n_usable']} yielded a usable parameter set\n")

    for key, e in sorted(store["entries"].items()):
        status = e["status"]
        if status == "usable":
            v = e.get("validation", {})
            params = ", ".join(f"{k}={val:.4g}" for k, val in e["parameters"].items())
            print(f"  {key:18} USABLE    {params}  dose {e['dose_mg']} mg ({e['route']})")
            print(f"  {'':18}           {v.get('verdict', '')}")
            for t in v.get("targets", []):
                mark = "ok " if t["agrees"] else "DIFF"
                inside = ("inside" if t["observed_within_interval"] else "outside")
                print(f"  {'':18}           [{mark}] {t['name']:10} "
                      f"predicted {t['predicted']:.4g} vs observed {t['observed']:.4g} "
                      f"{t['unit']} ({t['relative_error']:.1%}, {inside} 90% interval)")
        elif status == "insufficient_parameters":
            print(f"  {key:18} BLOCKED   {e['blocked_reason'][:88]}")
        else:
            print(f"  {key:18} NO LABEL  openFDA has no PK section for "
                  f"{e['generic_name']!r}")
    print(f"\nstore written to {STORE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
