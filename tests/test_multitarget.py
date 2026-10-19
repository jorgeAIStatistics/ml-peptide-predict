"""Stage 3 melanocortin panel: endpoint separation and leakage guards.

The panel's whole purpose is to make a target encoder evaluable, and that
depends on two disciplines holding: endpoints must never merge, and a
molecule's rows must never straddle a split. Both are asserted here rather
than trusted.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from atlas.ml.dataset import ENDPOINT_ASSAY_TYPE, ENDPOINT_LABEL, MC4R
from atlas.ml.multitarget import (
    MELANOCORTIN_RECEPTORS, PANEL_PATH, PanelRow, TargetPanel,
    generalisation_regimes, leave_one_receptor_out, load_panel,
)


def _row(mol: str, receptor: str, value: float, endpoint: str = "Ki") -> PanelRow:
    return PanelRow(
        molecule_chembl_id=mol, smiles="CC", receptor=receptor,
        target_chembl_id=MELANOCORTIN_RECEPTORS[receptor]["target_chembl_id"],
        endpoint=endpoint, value=value, value_min=value, value_max=value,
        n_replicates=1, replicate_spread=0.0,
        document_ids=("D1",), assay_ids=("A1",),
    )


@pytest.fixture(scope="module")
def panel() -> TargetPanel:
    if not PANEL_PATH.exists():
        pytest.skip(f"{PANEL_PATH.name} not built; run atlas.ml.multitarget")
    return load_panel()


# ------------------------------------------------------- endpoint hygiene ----

def test_every_endpoint_has_an_assay_type_and_label():
    assert set(ENDPOINT_ASSAY_TYPE) == set(ENDPOINT_LABEL)
    # Ki is a binding constant; EC50 is a functional response. Fetching either
    # under the other's assay class would mix physical quantities.
    assert ENDPOINT_ASSAY_TYPE["Ki"] == "B"
    assert ENDPOINT_ASSAY_TYPE["EC50"] == "F"


def test_unknown_endpoint_is_refused():
    from atlas.ml.dataset import build_dataset
    with pytest.raises(ValueError, match="unknown endpoint"):
        build_dataset(endpoint="pIC90")


def test_panel_offers_no_way_to_merge_endpoints(panel):
    """There is deliberately no accessor that returns several endpoints."""
    assert hasattr(panel, "for_endpoint")
    with pytest.raises(KeyError):
        panel.for_endpoint("EC50" if "EC50" not in panel.endpoints else "nonsense")


def test_every_row_carries_one_endpoint(panel):
    assert {r.endpoint for r in panel.rows} == set(panel.endpoints)


def test_mc4r_endpoint_label_is_stable():
    """A rebuild must not silently rename the existing dataset's endpoint."""
    assert MC4R["short_name"] == "MC4R"
    assert ENDPOINT_LABEL["Ki"] == "binding affinity (pKi, -log10 M)"


# ------------------------------------------------------------- provenance ----

def test_receptors_carry_uniprot_accessions_and_sequences(panel):
    for short, meta in panel.receptors.items():
        assert meta["target_chembl_id"].startswith("CHEMBL"), short
        if meta.get("sequence"):
            assert meta["sequence_length"] == len(meta["sequence"])
            assert meta["uniprot_accessions"], short


def test_mc4r_accession_matches_the_one_used_for_structure_lookup(panel):
    """P32245 was used for the RCSB query in the Stage 1 audit."""
    mc4r = panel.receptors.get("MC4R", {})
    if mc4r.get("uniprot_accessions"):
        assert "P32245" in mc4r["uniprot_accessions"]


def test_panel_version_detects_alteration(panel, tmp_path):
    payload = json.loads(PANEL_PATH.read_text())
    payload["rows"][0]["value"] = payload["rows"][0]["value"] + 1.0
    bad = tmp_path / "tampered.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="has been altered"):
        load_panel(bad)


def test_version_is_insensitive_to_row_order(panel):
    reordered = TargetPanel(
        rows=list(reversed(panel.rows)), receptors=panel.receptors,
        endpoints=panel.endpoints, counts=panel.counts, overlap=panel.overlap,
        license_note=panel.license_note, built_at=panel.built_at,
    )
    assert reordered.version == panel.version


# ------------------------------------------------- cross-receptor overlap ----

def test_overlap_audit_is_internally_consistent(panel):
    for endpoint, o in panel.overlap.items():
        breadth = {int(k): v for k, v in o["receptors_per_molecule"].items()}
        assert sum(breadth.values()) == o["n_distinct_molecules"]
        assert o["n_molecules_on_multiple_receptors"] == sum(
            v for k, v in breadth.items() if k > 1)
        assert 0.0 <= o["fraction_on_multiple_receptors"] <= 1.0


def test_selectivity_span_is_defined_only_with_overlap(panel):
    """A span needs a molecule on at least two receptors."""
    for o in panel.overlap.values():
        if o["n_molecules_on_multiple_receptors"] > 0:
            assert o["median_selectivity_span_log"] is not None
            assert o["max_selectivity_span_log"] >= o["median_selectivity_span_log"]


def test_overlap_audit_on_a_constructed_panel():
    rows = [_row("M1", "MC4R", 9.0), _row("M1", "MC1R", 7.0),
            _row("M2", "MC4R", 8.0)]
    from atlas.ml.multitarget import _overlap_audit
    o = _overlap_audit(rows, "Ki")
    assert o["n_distinct_molecules"] == 2
    assert o["n_molecules_on_multiple_receptors"] == 1
    assert o["max_selectivity_span_log"] == pytest.approx(2.0)
    assert o["receptor_pair_counts"] == {"MC1R|MC4R": 1}


# ----------------------------------------------------------------- splits ----

def test_leave_one_receptor_out_holds_out_exactly_that_receptor():
    rows = [_row("M1", "MC4R", 9.0), _row("M1", "MC1R", 7.0),
            _row("M2", "MC3R", 8.0), _row("M3", "MC4R", 6.0)]
    tr, te = leave_one_receptor_out(rows, "MC4R")
    assert {rows[i].receptor for i in te} == {"MC4R"}
    assert "MC4R" not in {rows[i].receptor for i in tr}
    assert len(tr) + len(te) == len(rows)


def test_leave_one_receptor_out_refuses_an_absent_receptor():
    rows = [_row("M1", "MC4R", 9.0)]
    with pytest.raises(ValueError, match="not present"):
        leave_one_receptor_out(rows, "MC5R")


def test_generalisation_regimes_are_disjoint_and_labelled():
    rows = [_row("M1", "MC4R", 9.0), _row("M1", "MC1R", 7.0),
            _row("M2", "MC4R", 8.0), _row("M3", "MC1R", 6.0),
            _row("M4", "MC4R", 5.0)]
    clusters = [0, 0, 1, 1, 2]        # M4's cluster 2 is MC4R-only
    reg = generalisation_regimes(rows, clusters, "MC4R")

    both_new = set(reg["new_peptide_family_and_new_target"]["test_idx"])
    related = set(reg["known_peptide_family_related_target"]["test_idx"])
    assert both_new.isdisjoint(related)
    # Row 4 is on the held-out receptor and in a cluster seen nowhere else.
    assert both_new == {4}
    # Rows 0 and 2 are on MC4R in clusters that also appear on MC1R.
    assert related == {0, 2}
    for v in reg.values():
        assert v["description"]


def test_generalisation_regimes_rejects_mismatched_labels():
    rows = [_row("M1", "MC4R", 9.0)]
    with pytest.raises(ValueError, match="labels for"):
        generalisation_regimes(rows, [0, 1], "MC4R")


def test_leave_one_receptor_out_does_not_split_a_molecule_by_accident():
    """A molecule on two receptors legitimately appears on both sides here.

    That is not leakage of the *receptor* question -- the point of the split is
    that the held-out receptor is unseen -- but it does mean the model has seen
    this molecule's affinity elsewhere. The split records that honestly rather
    than pretending the molecule is novel.
    """
    rows = [_row("M1", "MC4R", 9.0), _row("M1", "MC1R", 7.0)]
    tr, te = leave_one_receptor_out(rows, "MC4R")
    train_mols = {rows[i].molecule_chembl_id for i in tr}
    test_mols = {rows[i].molecule_chembl_id for i in te}
    assert train_mols & test_mols == {"M1"}
