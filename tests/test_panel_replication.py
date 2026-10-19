"""Stage 3b: re-running Stage A and Stage B per receptor over the panel.

Two things have to hold for a per-receptor re-run to mean anything.

The audit and benchmark must be able to consume an injected dataset *without*
inheriting the saved MC4R dataset's provenance, because a report that carried
another receptor's census under MC4R's version string would be a fabricated
attribution. And the re-run must not be able to overwrite the canonical
reports that the published Stage A and Stage B findings cite.

Both are asserted here. So is the control: MC4R routed through the panel
adapter must reproduce the canonical numbers exactly, since without that a
difference at another receptor could be an artefact of the adapter.
"""

from __future__ import annotations

import json

import pytest

from atlas.ml.dataset import DATASET_PATH
from atlas.ml.multitarget import PANEL_PATH, load_panel, records_for_receptor
from atlas.ml.panel_replication import (
    CONTRAST_NAMES, MIN_PEPTIDES, REPLICATION_PATH, stage_a_for_receptor,
    stage_b_for_receptor, traced_peptide_count,
)
from atlas.ml.peptide_audit import AUDIT_PATH, run_audit
from atlas.ml.peptide_benchmark import BENCHMARK_PATH, run_benchmark


@pytest.fixture(scope="module")
def panel():
    if not PANEL_PATH.exists():
        pytest.skip(f"{PANEL_PATH.name} not built; run atlas.ml.multitarget")
    return load_panel()


@pytest.fixture(scope="module")
def report() -> dict:
    if not REPLICATION_PATH.exists():
        pytest.skip(
            f"{REPLICATION_PATH.name} not built; run atlas.ml.panel_replication")
    return json.loads(REPLICATION_PATH.read_text())


@pytest.fixture(scope="module")
def records(panel):
    return records_for_receptor(panel.for_endpoint("Ki"), "MC1R")


# ------------------------------------------------------- provenance guards ----

def test_injected_audit_records_require_their_own_provenance(records):
    with pytest.raises(ValueError, match="provenance"):
        run_audit(records=records)


def test_injected_benchmark_records_require_their_own_provenance(records):
    with pytest.raises(ValueError, match="provenance"):
        run_benchmark(records=records, endpoint="MC1R pKi")


def test_injected_audit_refuses_to_overwrite_the_canonical_report(records):
    """The Stage A findings cite peptide_audit.json by name."""
    with pytest.raises(ValueError, match="canonical"):
        run_audit(
            records=records, dataset_version="v", endpoint="MC1R pKi",
            target={"target_chembl_id": "CHEMBL3795"}, save=True,
        )


def test_injected_benchmark_refuses_to_overwrite_the_canonical_report(records):
    with pytest.raises(ValueError, match="canonical"):
        run_benchmark(
            records=records, dataset_version="v", endpoint="MC1R pKi",
            save=True,
        )


def test_injected_runs_do_not_save_by_default(records, monkeypatch):
    """An injected run writes nothing unless asked, so re-runs are read-only."""
    called: list[str] = []
    from atlas.ml import peptide_audit

    monkeypatch.setattr(
        peptide_audit.AuditReport, "save",
        lambda self, path=AUDIT_PATH: called.append(str(path)) or path,
    )
    run_audit(
        records=records[:60], dataset_version="v", endpoint="MC1R pKi",
        target={"target_chembl_id": "CHEMBL3795"},
    )
    assert called == []


# -------------------------------------------------------------- the control ----

def test_mc4r_through_the_adapter_reproduces_the_canonical_benchmark(report):
    """The adapter must not perturb the data Stage B was measured on."""
    if not BENCHMARK_PATH.exists():
        pytest.skip("peptide_benchmark.json not built")
    canonical = json.loads(BENCHMARK_PATH.read_text())
    mc4r = report["stage_b"]["MC4R"]
    assert mc4r["status"] == "ok"
    assert mc4r["n_peptides"] == canonical["n_peptides"]
    assert mc4r["cross_series_mae"] == pytest.approx(
        canonical["gate"]["criteria"]["cross_series_mae"]["actual"], abs=1e-9)
    assert mc4r["cross_series_r2"] == pytest.approx(
        canonical["gate"]["criteria"]["cross_series_r2"]["actual"], abs=1e-9)


def test_mc4r_through_the_adapter_reproduces_the_canonical_audit(report):
    if not AUDIT_PATH.exists():
        pytest.skip("peptide_audit.json not built")
    canonical = json.loads(AUDIT_PATH.read_text())
    mc4r = report["stage_a"]["MC4R"]
    assert mc4r["n_molecules"] == canonical["census"]["n_molecules"]
    assert mc4r["n_peptides"] == canonical["census"]["structural_peptide_count"]
    assert mc4r["n_series"] == canonical["series"]["n_clusters"]


# ------------------------------------------------------------- abstentions ----

def test_mc2r_contributes_no_peptides(panel):
    assert traced_peptide_count(panel.for_endpoint("Ki"), "MC2R") == 0


def test_a_receptor_without_peptides_abstains_rather_than_erroring(panel):
    """An abstention is a result here, not a failure to be caught."""
    rows = panel.for_endpoint("Ki")
    a = stage_a_for_receptor(rows, "MC2R", {}, panel.version)
    b = stage_b_for_receptor(rows, "MC2R", panel.version)
    assert a["status"] == "abstained"
    assert b["status"] == "abstained"
    assert a["n_peptides"] == 0 and b["n_peptides"] == 0
    assert str(MIN_PEPTIDES) in a["reason"]


def test_report_records_the_abstention_rather_than_dropping_it(report):
    assert "MC2R" in report["stage_a"]
    assert report["stage_a"]["MC2R"]["status"] != "ok"
    assert report["stage_b"]["MC2R"]["status"] != "ok"


# ---------------------------------------------------- endpoint separation ----

def test_replication_runs_on_one_endpoint_only(report):
    assert report["endpoint"] == "Ki"
    for side in ("stage_a", "stage_b"):
        for rec, body in report[side].items():
            if body.get("status") == "ok":
                assert "Ki" not in str(body.get("n_peptides", "")), "sanity"
    # Every receptor the report covers came from the panel's Ki rows, and the
    # report carries no second endpoint to confuse with them.
    assert isinstance(report["endpoint"], str)


# ------------------------------------------------------------- the contrasts ----

def test_every_stage_b_run_reports_all_five_contrasts(report):
    for rec, body in report["stage_b"].items():
        if body.get("status") != "ok":
            continue
        assert set(body["contrasts"]) == set(CONTRAST_NAMES)
        assert body["n_contrasts_compared"] == len(CONTRAST_NAMES)


def test_stage_b_reversal_does_not_replicate_across_receptors(report):
    """The negative result, pinned so it cannot be quietly softened.

    Stage B found all five representation contrasts reversing between the
    cross-series and within-series evaluations. That was one receptor. Here
    only MC4R reverses all five, so the finding is a property of MC4R's data
    and not a general fact about peptide representations.
    """
    ok = {k: v for k, v in report["stage_b"].items()
          if v.get("status") == "ok"}
    assert len(ok) >= 3, "need several receptors for a replication claim"
    full = [k for k, v in ok.items()
            if v["n_contrasts_reversing"] == v["n_contrasts_compared"]]
    assert full == ["MC4R"], (
        f"expected only MC4R to reverse all five contrasts, got {full}")
    assert any(
        v["n_contrasts_reversing"] < v["n_contrasts_compared"]
        for k, v in ok.items() if k != "MC4R")


def test_findings_state_the_negative_result(report):
    joined = " ".join(report["findings"])
    assert "does NOT replicate" in joined


# -------------------------------------------------------- the gate is intact ----

def test_the_deployment_gate_still_fails_on_every_receptor(report):
    """Expanding the data did not earn the right to publish estimates."""
    for rec, body in report["stage_b"].items():
        if body.get("status") == "ok":
            assert body["gate_passes"] is False, f"{rec} unexpectedly passes"


def test_stage_a_series_ceiling_replicates(report):
    """No receptor's peptides resolve into many more series than MC4R's."""
    ok = {k: v for k, v in report["stage_a"].items()
          if v.get("status") == "ok"}
    assert len(ok) >= 3
    assert all(v["n_series"] <= 40 for v in ok.values())


def test_noncanonical_burden_is_class_wide(report):
    ok = {k: v for k, v in report["stage_a"].items()
          if v.get("status") == "ok"}
    for rec, v in ok.items():
        assert v["noncanonical_fraction"] > 0.5, (
            f"{rec} at {v['noncanonical_fraction']:.1%}")


# ------------------------------------------------------------ study grouping ----

def test_study_grouping_audit_is_carried_in_the_report(report):
    s = report["study_grouping"]
    assert s["n_groups"] < s["n_documents"]
    assert 0 < s["largest_group_fraction_of_rows"] < 1
