"""Sweep the registry through the evidence gate and report the research gaps.

Produces the table that answers "what can this platform actually say about each
compound, and what is missing?" -- entirely from measured counts and resolved
identities, with no estimated values anywhere.

Run as::

    python -m atlas.analysis.report
    python -m atlas.analysis.report --offline   # from cache only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path

from atlas.analysis.feasibility import AnalysisKind, FeasibilityEngine
from atlas.mechanistic.pk_store import parameters_for
from atlas.core.evidence import utc_now
from atlas.registry.resolve import IdentityResolver
from atlas.registry.seed import SEED_REGISTRY
from atlas.sources.base import CachedHttpClient
from atlas.sources.chembl import ChemblClient
from atlas.sources.ctgov import ClinicalTrialsClient
from atlas.sources.pubchem import PubChemClient
from atlas.sources.pubmed import PubMedClient

logger = logging.getLogger(__name__)

REPORT_PATH = Path(__file__).resolve().parents[2] / "data" / "feasibility_report.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s",
                        stream=sys.stderr)

    http = CachedHttpClient(ttl=timedelta(days=30), offline=args.offline)
    resolver = IdentityResolver(ChemblClient(http), PubChemClient(http))
    engine = FeasibilityEngine(PubMedClient(http), ClinicalTrialsClient(http))

    entries = SEED_REGISTRY
    if args.only:
        wanted = set(args.only)
        entries = tuple(e for e in SEED_REGISTRY if e.key in wanted)

    rows = []
    for i, entry in enumerate(entries, start=1):
        print(f"[{i}/{len(entries)}] {entry.key}", file=sys.stderr)
        identity = resolver.resolve(
            entry.key, entry.name, aliases=entry.aliases,
            declared_modality=entry.declared_modality,
            is_mixture=entry.is_mixture, supplier_defined=entry.supplier_defined,
        )
        pk = parameters_for(entry.key)
        assessment = engine.assess(
            entry, identity,
            pk_parameters=(pk or {}).get('parameters'),
        )
        rows.append(
            {
                "key": entry.key,
                "name": entry.name,
                "area": entry.research_area,
                "identity_status": identity.status.value,
                "modality": identity.modality.value,
                "completeness": round(identity.completeness, 3),
                "approved": identity.is_approved_drug,
                "publications": assessment.literature.total_publications,
                "human_publications": assessment.literature.human_publications,
                "human_pk_publications": assessment.literature.human_pk_publications,
                "randomized_trials": assessment.literature.randomized_trials,
                "registered_trials": assessment.literature.registered_trials,
                "trials_with_results": assessment.literature.trials_with_posted_results,
                "density": assessment.literature.evidence_density,
                "permitted": [k.value for k in assessment.permitted],
                "refused": {
                    k.value: a.reason.value for k, a in assessment.refusals.items()
                },
                "pk_parameters": (pk or {}).get("parameters"),
                "pk_validation": (pk or {}).get("validation"),
            }
        )

    payload = {
        "created_at": utc_now().isoformat(),
        "n_entries": len(rows),
        "rows": rows,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2))

    # -- printed summary ----------------------------------------------------
    print(f"\n{'=' * 118}")
    print("EVIDENCE AND FEASIBILITY REPORT".center(118))
    print(f"{'=' * 118}")
    hdr = (
        f"{'compound':34} {'identity':14} {'pubs':>6} {'human':>6} "
        f"{'humPK':>6} {'RCT':>5} {'reg':>5} {'res':>4}  {'density':12} permitted"
    )
    print(hdr)
    print("-" * 118)
    for r in sorted(rows, key=lambda x: (-x["publications"], x["key"])):
        ident = r["identity_status"].split("_")[0][:13]
        n_perm = len(r["permitted"])
        print(
            f"{r['name'][:33]:34} {ident:14} {r['publications']:>6} "
            f"{r['human_publications']:>6} {r['human_pk_publications']:>6} "
            f"{r['randomized_trials']:>5} {r['registered_trials']:>5} "
            f"{r['trials_with_results']:>4}  {r['density']:12} {n_perm}"
        )

    print(f"\n{'-' * 118}")
    zero_lit = [r["name"] for r in rows if r["publications"] == 0]
    no_human = [r["name"] for r in rows if r["human_publications"] == 0]
    no_results = [r["name"] for r in rows if r["trials_with_results"] == 0]
    print(f"entries with no indexed publications ({len(zero_lit)}): {', '.join(zero_lit)}")
    print(f"entries with no human-tagged publications ({len(no_human)}): {', '.join(no_human)}")
    print(f"entries with no trial posting results: {len(no_results)}/{len(rows)}")

    reason_counts: Counter[str] = Counter()
    for r in rows:
        reason_counts.update(r["refused"].values())
    print("\nrefusal reasons across the registry:")
    for reason, n in reason_counts.most_common():
        print(f"    {n:3d}  {reason}")

    permitted_counts: Counter[str] = Counter()
    for r in rows:
        permitted_counts.update(r["permitted"])
    print("\nanalyses permitted across the registry:")
    for kind in AnalysisKind:
        print(f"    {permitted_counts.get(kind.value, 0):3d}/{len(rows)}  {kind.value}")

    print(f"\nreport written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
