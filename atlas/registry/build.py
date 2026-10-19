"""Resolve the seed registry against live databases and write a snapshot.

The snapshot is the versioned artefact every downstream analysis cites: it
records what each database reported, when, and from which endpoint, so a stored
conclusion can be re-checked against the evidence it was derived from.

Run as::

    python -m atlas.registry.build            # resolve all, write snapshot
    python -m atlas.registry.build --offline  # rebuild from HTTP cache only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

from atlas.core.evidence import utc_now
from atlas.registry.resolve import IdentityResolver, IdentityStatus, MolecularIdentity
from atlas.registry.seed import SEED_REGISTRY, SeedEntry
from atlas.sources.base import CachedHttpClient
from atlas.sources.chembl import ChemblClient
from atlas.sources.pubchem import PubChemClient

logger = logging.getLogger(__name__)

__all__ = ["resolve_registry", "write_snapshot", "SNAPSHOT_PATH"]

SNAPSHOT_PATH = Path(__file__).resolve().parents[2] / "data" / "registry_snapshot.json"


def resolve_registry(
    entries: tuple[SeedEntry, ...] = SEED_REGISTRY,
    *,
    resolver: IdentityResolver | None = None,
) -> list[MolecularIdentity]:
    resolver = resolver or IdentityResolver()
    out: list[MolecularIdentity] = []
    for i, entry in enumerate(entries, start=1):
        logger.info("[%d/%d] resolving %s", i, len(entries), entry.key)
        try:
            identity = resolver.resolve(
                entry.key,
                entry.name,
                aliases=entry.aliases,
                declared_modality=entry.declared_modality,
                is_mixture=entry.is_mixture,
                supplier_defined=entry.supplier_defined,
            )
        except Exception as exc:  # a single bad entry must not void the build
            logger.error("resolution failed for %s: %s", entry.key, exc)
            identity = MolecularIdentity(
                registry_key=entry.key,
                query_name=entry.name,
                status=IdentityStatus.UNRESOLVED,
                modality=entry.declared_modality,
                caveats=(f"Resolution raised an error and was not completed: {exc}",),
            )
        out.append(identity)
    return out


def _snapshot_payload(identities: list[MolecularIdentity]) -> dict[str, Any]:
    body = [json.loads(i.model_dump_json()) for i in identities]
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "snapshot_id": digest[:16],
        "created_at": utc_now().isoformat(),
        "n_entries": len(body),
        "identities": body,
    }


def write_snapshot(
    identities: list[MolecularIdentity], path: Path = SNAPSHOT_PATH
) -> dict[str, Any]:
    payload = _snapshot_payload(identities)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False))
    return payload


def _summarize(identities: list[MolecularIdentity]) -> None:
    by_status: dict[str, list[str]] = {}
    for i in identities:
        by_status.setdefault(i.status.value, []).append(i.registry_key)

    print(f"\n{'=' * 72}\nREGISTRY RESOLUTION SUMMARY ({len(identities)} entries)\n{'=' * 72}")
    for status, keys in sorted(by_status.items(), key=lambda kv: -len(kv[1])):
        print(f"\n{status}  ({len(keys)})")
        for k in sorted(keys):
            print(f"    {k}")

    resolved = [i for i in identities if i.status is IdentityStatus.RESOLVED]
    print(f"\n{'-' * 72}")
    print(f"mean identity completeness (resolved only): "
          f"{sum(i.completeness for i in resolved) / max(len(resolved), 1):.0%}")
    print(f"approved per ChEMBL: "
          f"{sorted(i.registry_key for i in identities if i.is_approved_drug)}")
    print(f"blocked from structure models: "
          f"{sum(1 for i in identities if i.blocks_structure_models())}/{len(identities)}")

    flagged = [i for i in identities if i.caveats]
    print(f"\nentries carrying identity caveats: {len(flagged)}")
    for i in flagged:
        print(f"\n  {i.registry_key} [{i.status.value}]")
        for c in i.caveats:
            print(f"      - {c}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="use HTTP cache only")
    parser.add_argument("--only", nargs="*", help="restrict to these registry keys")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )

    http = CachedHttpClient(ttl=timedelta(days=30), offline=args.offline)
    resolver = IdentityResolver(ChemblClient(http), PubChemClient(http))

    entries = SEED_REGISTRY
    if args.only:
        wanted = set(args.only)
        entries = tuple(e for e in SEED_REGISTRY if e.key in wanted)
        if not entries:
            print(f"no registry entries matched {sorted(wanted)}", file=sys.stderr)
            return 2

    identities = resolve_registry(entries, resolver=resolver)
    payload = write_snapshot(identities)
    _summarize(identities)
    print(f"\nsnapshot {payload['snapshot_id']} written to {SNAPSHOT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
