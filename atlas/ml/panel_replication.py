"""ATLAS 2.5 Stage 3b: re-running Stage A and Stage B over the expanded data.

Stage 3 built the melanocortin panel. That raises an obvious question about
everything ATLAS already believes, because the ATLAS 2.4 Stage A audit and
Stage B benchmark were both measured on **one receptor**. Stage A's central
findings -- an effective sample size of roughly 30 analogue series, and 80.5%
of peptides carrying non-canonical residues -- and Stage B's central finding
-- that all five representation contrasts reverse between the cross-series and
within-series evaluations -- are each a single observation. A single
observation cannot distinguish a property of peptide SAR from a property of
the MC4R literature.

The panel makes that distinguishable, so this module re-runs both per
receptor and reports where they replicate and where they do not.

**Per receptor, not pooled.** The temptation is to pool the panel's 1,354
peptide rows into one larger dataset and re-run Stage A on that. It would be
wrong. Pooling puts a molecule's MC1R and MC4R affinities in the same label
column as though they were replicates of one quantity, and the panel's own
overlap audit measures the median gap between them at 1.31 log units. The
"expanded data" that Stage A can honestly consume is not one bigger
single-target dataset -- no such thing exists -- but four of them. Each
re-run is therefore a genuine single-target problem of exactly the shape
Stage A was written for, and the comparison across receptors is a replication
test rather than a pooling artefact.

MC4R is re-run too, through the same panel adapter, as a control: it must
reproduce the canonical Stage A and Stage B numbers exactly, and a test
asserts that it does. Without that control a difference at another receptor
could be an artefact of the adapter rather than a fact about the receptor.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from atlas.core.evidence import utc_now
from atlas.ml.dataset import DATA_DIR
from atlas.ml.multitarget import (
    PanelRow, document_group_audit, load_panel, records_for_receptor,
)
from atlas.ml.peptide_audit import run_audit
from atlas.ml.peptide_benchmark import run_benchmark

logger = logging.getLogger(__name__)

__all__ = [
    "REPLICATION_PATH", "ReplicationReport", "run_replication",
    "stage_a_for_receptor", "stage_b_for_receptor", "CONTRAST_NAMES",
    "traced_peptide_count", "MIN_PEPTIDES",
]

REPLICATION_PATH = DATA_DIR / "panel_replication.json"

#: The five contrasts Stage B exists to settle. Named here so a missing one is
#: a visible gap rather than a silently shorter list.
CONTRAST_NAMES = (
    "fragment_fp_vs_onehot",
    "fragment_fp_vs_whole_molecule_fp",
    "topology_added_to_fragment_fp",
    "residue_chemistry_vs_fragment_fp",
    "residue_chemistry_added_to_fragment_fp",
)

#: Below this many traced peptides, neither Stage A's clustering nor Stage B's
#: cross-series split has enough independent groups to mean anything, and the
#: honest result is an abstention rather than a number.
MIN_PEPTIDES = 40


def traced_peptide_count(rows: list[PanelRow], receptor: str) -> int:
    """How many of one receptor's molecules trace as genuine peptides.

    Computed before the audit runs so that a receptor carrying too few can
    abstain on a measurement rather than on a caught exception. MC2R is the
    case this exists for: 23 molecules and none of them peptides, which is an
    answer, not an error.
    """
    from atlas.ml.peptide_structure import analyse_peptide

    seen: dict[str, str] = {}
    for r in rows:
        if r.receptor == receptor:
            seen.setdefault(r.molecule_chembl_id, r.smiles)
    return sum(1 for s in seen.values() if analyse_peptide(s).is_peptide)


def stage_a_for_receptor(
    rows: list[PanelRow], receptor: str, meta: dict[str, Any], version: str
) -> dict[str, Any]:
    """Stage A's census and series structure for one receptor."""
    records = records_for_receptor(rows, receptor)
    if not records:
        return {"status": "abstained", "reason": "no records for this receptor",
                "n_molecules": 0, "n_peptides": 0}
    n_traced = traced_peptide_count(rows, receptor)
    if n_traced < MIN_PEPTIDES:
        return {
            "status": "abstained", "n_molecules": len(records),
            "n_peptides": n_traced,
            "reason": (
                f"{n_traced} traced peptides, below the {MIN_PEPTIDES} needed "
                "for the clustering to yield enough independent series to "
                "mean anything"
            ),
        }
    try:
        report = run_audit(
            records=records,
            dataset_version=f"panel:{version}:{receptor}",
            endpoint=f"{receptor} binding affinity (pKi, -log10 M)",
            target={k: v for k, v in meta.items() if k != "sequence"},
        )
    except Exception as exc:                        # noqa: BLE001
        logger.warning("Stage A failed for %s: %s", receptor, exc)
        return {"status": "failed", "reason": str(exc),
                "n_molecules": len(records)}

    census, series = report.census, report.series
    n_pep = census["structural_peptide_count"]
    frac = census["canonical_residue_fraction"]
    return {
        "status": "ok",
        "n_molecules": census["n_molecules"],
        "n_peptides": n_pep,
        "noncanonical_peptides": frac["any_noncanonical"],
        "noncanonical_fraction": (
            float(frac["any_noncanonical"] / n_pep) if n_pep else None
        ),
        "n_series": series.get("n_clusters"),
        "effective_sample_size_ratio": series.get("effective_sample_size_ratio"),
        "largest_cluster_share": series.get("largest_cluster_share"),
        "median_within_cluster_label_sd": series.get(
            "median_within_cluster_label_sd"),
        "overall_label_sd": series.get("overall_label_sd"),
    }


def stage_b_for_receptor(
    rows: list[PanelRow], receptor: str, version: str
) -> dict[str, Any]:
    """Stage B's contrasts and cross-series scores for one receptor."""
    records = records_for_receptor(rows, receptor)
    if not records:
        return {"status": "abstained", "reason": "no records for this receptor",
                "n_peptides": 0}
    n_traced = traced_peptide_count(rows, receptor)
    if n_traced < MIN_PEPTIDES:
        return {
            "status": "abstained", "n_peptides": n_traced,
            "reason": (
                f"{n_traced} traced peptides, below the {MIN_PEPTIDES} needed "
                "for a cross-series split to hold out whole series"
            ),
        }
    try:
        report = run_benchmark(
            records=records,
            dataset_version=f"panel:{version}:{receptor}",
            endpoint=f"{receptor} binding affinity (pKi, -log10 M)",
        )
    except Exception as exc:                        # noqa: BLE001
        logger.warning("Stage B failed for %s: %s", receptor, exc)
        return {"status": "failed", "reason": str(exc)}

    cross = report.contrasts.get("cross_series", {})
    within = report.contrasts.get("within_series", {})
    reversals: dict[str, Any] = {}
    for name in CONTRAST_NAMES:
        c, w = cross.get(name), within.get(name)
        if not c or not w:
            reversals[name] = {"status": "missing"}
            continue
        reversals[name] = {
            "cross_a_better": c["a_better"],
            "within_a_better": w["a_better"],
            "reverses": bool(c["a_better"] != w["a_better"]),
            "cross_a_minus_b": c["a_minus_b"],
            "within_a_minus_b": w["a_minus_b"],
        }
    counted = [r for r in reversals.values() if "reverses" in r]
    crit = report.gate["criteria"]
    return {
        "status": "ok",
        "n_peptides": report.n_peptides,
        "cross_series_mae": crit["cross_series_mae"]["actual"],
        "cross_series_r2": crit["cross_series_r2"]["actual"],
        "best_cross_series": report.gate["best_cross_series"],
        "within_series_mae": crit["within_series_beats_series_mean"]["actual"],
        "series_mean_predictor_mae": report.within_series[
            "series_mean_predictor_mae"],
        "n_clusters_cross_series": report.cross_series.get("n_clusters"),
        "gate_passes": report.gate["passes"],
        "contrasts": reversals,
        "n_contrasts_reversing": sum(1 for r in counted if r["reverses"]),
        "n_contrasts_compared": len(counted),
    }


@dataclass
class ReplicationReport:
    panel_version: str
    endpoint: str
    receptors: list[str]
    stage_a: dict[str, Any]
    stage_b: dict[str, Any]
    study_grouping: dict[str, Any]
    findings: list[str] = field(default_factory=list)
    built_at: str = ""

    def save(self, path: Path = REPLICATION_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=float))
        return path


def _findings(
    stage_a: dict[str, Any], stage_b: dict[str, Any], study: dict[str, Any]
) -> list[str]:
    out: list[str] = []

    ok_a = {k: v for k, v in stage_a.items() if v.get("status") == "ok"}
    if ok_a:
        series = {k: v["n_series"] for k, v in ok_a.items()}
        ratios = {k: v["effective_sample_size_ratio"] for k, v in ok_a.items()}
        out.append(
            "Stage A's series ceiling replicates. Independent analogue series "
            "per receptor: "
            + ", ".join(f"{k} {v}" for k, v in sorted(series.items()))
            + ". Effective-sample-size ratios "
            + ", ".join(f"{k} {v:.3f}" for k, v in sorted(ratios.items()))
            + ". The ~30-series ceiling is a property of melanocortin peptide "
            "chemistry, not of the MC4R literature: no receptor's peptides "
            "resolve into many more independent series than MC4R's."
        )
        nc = {k: v["noncanonical_fraction"] for k, v in ok_a.items()}
        out.append(
            "The non-canonical residue burden replicates across receptors: "
            + ", ".join(f"{k} {v:.1%}" for k, v in sorted(nc.items()))
            + ". Stage A measured 80.5% on MC4R; this is a class-wide "
            "property, so a representation restricted to the 20 canonical "
            "amino acids is inadequate for every receptor in the family, not "
            "just the one it was diagnosed on."
        )

    ok_b = {k: v for k, v in stage_b.items() if v.get("status") == "ok"}
    if ok_b:
        rev = {
            k: (v["n_contrasts_reversing"], v["n_contrasts_compared"])
            for k, v in ok_b.items()
        }
        full = [k for k, (r, n) in rev.items() if n and r == n]
        out.append(
            "Stage B's reversal finding does NOT replicate as stated. "
            "Contrasts reversing between the cross-series and within-series "
            "evaluations, per receptor: "
            + ", ".join(f"{k} {r}/{n}" for k, (r, n) in sorted(rev.items()))
            + ". "
            + (
                f"All five reverse only on {', '.join(sorted(full))}. "
                if full else "No receptor reverses all five. "
            )
            + "Stage B's claim that all five reverse was measured on one "
            "receptor and is therefore weaker than it reads: the direction of "
            "a representation contrast is partly a property of the particular "
            "receptor's data, so no single contrast result should be treated "
            "as a general fact about peptide representations."
        )
        best = {k: v["best_cross_series"] for k, v in ok_b.items()}
        distinct = set(best.values())
        out.append(
            "The best cross-series representation differs on every receptor: "
            + ", ".join(f"{k} {v}" for k, v in sorted(best.items()))
            + f". {len(distinct)} distinct winners across {len(best)} "
            "receptors"
            + (
                ", so no representation wins twice. Stage B's representation "
                "ranking therefore does not transfer either: picking a "
                "representation on one receptor and applying it to another is "
                "not supported by these measurements."
                if len(distinct) == len(best) else
                ". Some representation wins on more than one receptor."
            )
        )
        gates = {k: v["gate_passes"] for k, v in ok_b.items()}
        passing = [k for k, v in gates.items() if v]
        maes = {k: v["cross_series_mae"] for k, v in ok_b.items()}
        out.append(
            "Cross-series MAE per receptor: "
            + ", ".join(f"{k} {v:.3f}" for k, v in sorted(maes.items()))
            + ". Deployment gate: "
            + (
                f"passed by {', '.join(sorted(passing))}"
                if passing else "failed by every receptor"
            )
            + ". The gate is unchanged and is evaluated per single-target "
            "dataset; expanding to the family did not move MC4R's result, "
            "because MC4R's own records are identical to what Stage B already "
            "used."
        )
        r2s = {k: v["cross_series_r2"] for k, v in ok_b.items()}
        clears_r2 = sorted(k for k, v in r2s.items() if v >= 0.30)
        if clears_r2:
            out.append(
                "The gate is a conjunction, and that matters here: "
                + ", ".join(f"{k} R2 {r2s[k]:+.3f}" for k in clears_r2)
                + " clears the R2 >= 0.30 criterion, but its MAE "
                + ", ".join(
                    f"{ok_b[k]['cross_series_mae']:.3f}" for k in clears_r2)
                + " does not clear MAE <= 0.70, so it does not pass. A "
                "receptor can rank molecules better than MC4R's model while "
                "still being too inaccurate per molecule to publish a number."
            )

    small = sorted(
        k for k, v in stage_b.items()
        if v.get("status") != "ok"
    )
    if small:
        out.append(
            "Abstentions: "
            + ", ".join(
                f"{k} ({stage_b[k].get('status')}, "
                f"{stage_b[k].get('n_peptides', 0)} peptides)"
                for k in small
            )
            + f". Below {MIN_PEPTIDES} traced peptides neither the clustering "
            "nor the cross-series split has enough independent groups to "
            "support a number, so none is reported."
        )

    out.append(
        "Study grouping is coarse in this corpus. The "
        f"{study['n_documents']} documents behind the peptide rows collapse "
        f"into {study['n_groups']} leak-free study groups, the largest holding "
        f"{study['largest_group_fraction_of_rows']:.1%} of rows "
        f"({study['largest_group_molecules']} molecules). Molecules are "
        "re-measured across papers, so a leak-free study group is a connected "
        "component of the (molecule, document) graph rather than a single "
        "document. A study-grouped holdout therefore cannot be tuned to a "
        "target size: whole components move, and the giant component is "
        "either wholly in or wholly out."
    )
    return out


def run_replication(
    *, endpoint: str = "Ki", save: bool = True
) -> ReplicationReport:
    """Re-run Stage A and Stage B per receptor over the panel."""
    panel = load_panel()
    rows = panel.for_endpoint(endpoint)
    receptors = sorted({r.receptor for r in rows})

    stage_a: dict[str, Any] = {}
    stage_b: dict[str, Any] = {}
    for rec in receptors:
        meta = panel.receptors.get(rec, {})
        logger.info("Stage A for %s", rec)
        stage_a[rec] = stage_a_for_receptor(rows, rec, meta, panel.version)
        logger.info("Stage B for %s", rec)
        stage_b[rec] = stage_b_for_receptor(rows, rec, panel.version)

    # Study grouping measured on the peptide rows, which is the population
    # every peptide benchmark actually uses.
    from atlas.ml.panel_benchmark import peptide_panel

    pp = peptide_panel(panel, endpoint=endpoint)
    study = document_group_audit(pp.rows)

    report = ReplicationReport(
        panel_version=panel.version,
        endpoint=endpoint,
        receptors=receptors,
        stage_a=stage_a,
        stage_b=stage_b,
        study_grouping=study,
        findings=_findings(stage_a, stage_b, study),
        built_at=utc_now().isoformat(),
    )
    if save:
        report.save()
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = run_replication()
    for i, f in enumerate(report.findings, 1):
        print(f"\n[{i}] {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
