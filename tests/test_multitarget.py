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
    document_group_audit, document_groups, generalisation_regimes,
    leave_documents_out, leave_one_receptor_out, load_panel,
    records_for_receptor,
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


# ------------------------------------------------- study/document grouping ----

def _doc_row(mol: str, receptor: str, docs: tuple[str, ...]) -> PanelRow:
    return PanelRow(
        molecule_chembl_id=mol, smiles="CC", receptor=receptor,
        target_chembl_id=MELANOCORTIN_RECEPTORS[receptor]["target_chembl_id"],
        endpoint="Ki", value=8.0, value_min=8.0, value_max=8.0,
        n_replicates=1, replicate_spread=0.0,
        document_ids=docs, assay_ids=("A1",),
    )


def test_a_molecule_shared_between_documents_fuses_their_groups():
    """The reason a study group cannot simply be one document.

    M2 is measured in both D1 and D2, so holding out D2 alone would put M2 on
    both sides of the split. The transitive closure fuses D1 and D2 instead.
    """
    rows = [
        _doc_row("M1", "MC4R", ("D1",)),
        _doc_row("M2", "MC4R", ("D1", "D2")),
        _doc_row("M3", "MC4R", ("D2",)),
        _doc_row("M4", "MC4R", ("D3",)),
    ]
    labels = document_groups(rows)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]
    assert len(set(labels.tolist())) == 2


def test_a_molecule_with_no_document_is_its_own_group():
    rows = [_doc_row("M1", "MC4R", ()), _doc_row("M2", "MC4R", ())]
    labels = document_groups(rows)
    assert len(set(labels.tolist())) == 2


def test_all_rows_of_one_molecule_share_a_study_group():
    """A molecule on several receptors is one node, so its rows cannot split."""
    rows = [
        _doc_row("M1", "MC1R", ("D1",)),
        _doc_row("M1", "MC4R", ("D2",)),
        _doc_row("M9", "MC4R", ("D9",)),
    ]
    labels = document_groups(rows)
    assert labels[0] == labels[1] != labels[2]


def test_no_molecule_and_no_document_straddles_a_study_split(panel):
    rows = panel.for_endpoint("Ki")
    train, test = leave_documents_out(rows, test_fraction=0.25)
    assert len(train) + len(test) == len(rows)
    assert not (set(train.tolist()) & set(test.tolist()))

    tr_mols = {rows[i].molecule_chembl_id for i in train}
    te_mols = {rows[i].molecule_chembl_id for i in test}
    assert not (tr_mols & te_mols), "a molecule straddles the study split"

    tr_docs = {d for i in train for d in rows[i].document_ids}
    te_docs = {d for i in test for d in rows[i].document_ids}
    assert not (tr_docs & te_docs), "a document straddles the study split"


def test_study_split_holds_out_whole_groups(panel):
    rows = panel.for_endpoint("Ki")
    labels = document_groups(rows)
    train, test = leave_documents_out(rows, test_fraction=0.25)
    assert not (set(labels[train].tolist()) & set(labels[test].tolist()))


def test_study_group_audit_is_internally_consistent(panel):
    rows = panel.for_endpoint("Ki")
    a = document_group_audit(rows)
    labels = document_groups(rows)
    assert a["n_rows"] == len(rows)
    assert a["n_groups"] == len(set(labels.tolist()))
    assert 0 < a["largest_group_rows"] <= a["n_rows"]
    assert a["largest_group_fraction_of_rows"] == pytest.approx(
        a["largest_group_rows"] / a["n_rows"])


def test_study_grouping_is_coarse_in_this_corpus(panel):
    """Recorded as a measurement, because it limits what the split can do.

    Molecules are re-measured across papers, so the leak-free closure is much
    coarser than the document count. A later stage must not ask this split for
    a precisely sized test set.
    """
    a = document_group_audit(panel.for_endpoint("Ki"))
    assert a["n_groups"] < a["n_documents"]
    assert a["largest_group_fraction_of_rows"] > 0.2


# ----------------------------------------------------- per-receptor records ----

def test_records_for_receptor_keeps_only_that_receptor(panel):
    rows = panel.for_endpoint("Ki")
    recs = records_for_receptor(rows, "MC1R")
    want = [r for r in rows if r.receptor == "MC1R"]
    assert len(recs) == len(want)
    assert {r.molecule_chembl_id for r in recs} == {
        r.molecule_chembl_id for r in want}


def test_records_for_receptor_preserves_values(panel):
    rows = panel.for_endpoint("Ki")
    by_mol = {r.molecule_chembl_id: r for r in rows if r.receptor == "MC3R"}
    for rec in records_for_receptor(rows, "MC3R"):
        src = by_mol[rec.molecule_chembl_id]
        assert rec.pki == src.value
        assert rec.pki_min == src.value_min
        assert rec.pki_max == src.value_max
        assert rec.n_replicates == src.n_replicates
        assert rec.document_ids == src.document_ids


def test_mc4r_panel_rows_reproduce_the_standalone_dataset(panel):
    """The panel's MC4R slice must equal the separately built mc4r_pki.json.

    Two independently saved artefacts built through different code paths. If
    they agree molecule for molecule and value for value, the panel pipeline
    did not perturb the data Stage A and Stage B were measured on, and any
    difference found at another receptor is a fact about that receptor rather
    than an artefact of the adapter.
    """
    from atlas.ml.dataset import DATASET_PATH, load_dataset

    if not DATASET_PATH.exists():
        pytest.skip("mc4r_pki.json not built")
    standalone = {r.molecule_chembl_id: r.pki for r in load_dataset().records}
    viapanel = {
        r.molecule_chembl_id: r.pki
        for r in records_for_receptor(panel.for_endpoint("Ki"), "MC4R")
    }
    assert viapanel == standalone


def test_per_receptor_counts_match_the_rows_they_claim(panel):
    """The recorded counts are not a free-floating claim."""
    for key, n in panel.counts["per_receptor_endpoint"].items():
        receptor, endpoint = key.split("|")
        actual = sum(
            1 for r in panel.rows
            if r.receptor == receptor and r.endpoint == endpoint)
        assert actual == n, f"{key} claims {n} but holds {actual}"
    assert panel.counts["total_measurements"] == len(panel.rows)
    assert panel.counts["distinct_molecules"] == len(
        {r.molecule_chembl_id for r in panel.rows})


@pytest.mark.parametrize("receptor", sorted(MELANOCORTIN_RECEPTORS))
def test_per_receptor_counts_match_chembl(panel, receptor):
    """Against ChEMBL itself, through the cached client.

    Skipped when the cache is cold and the network is unavailable, because the
    cache directory is not committed. ChEMBL reports raw activity rows; the
    panel holds unique molecules after filtering, so the check is that the
    panel never claims more molecules than ChEMBL has rows for that target and
    that the target is non-empty exactly when the panel says so.
    """
    from datetime import timedelta

    from atlas.ml.dataset import API, ENDPOINT_ASSAY_TYPE
    from atlas.sources.base import CachedHttpClient

    target = MELANOCORTIN_RECEPTORS[receptor]["target_chembl_id"]
    http = CachedHttpClient(ttl=timedelta(days=30))
    try:
        payload, _ = http.get_json(f"{API}/activity", {
            "target_chembl_id": target, "standard_type": "Ki",
            "standard_units": "nM", "standard_relation": "=",
            "assay_type": ENDPOINT_ASSAY_TYPE["Ki"], "format": "json",
            "limit": 1, "offset": 0,
        })
    except Exception as exc:                        # noqa: BLE001
        pytest.skip(f"ChEMBL unavailable and cache cold: {exc}")

    raw = payload.get("page_meta", {}).get("total_count")
    if raw is None:
        pytest.skip("ChEMBL returned no total_count")
    claimed = panel.counts["per_receptor_endpoint"].get(f"{receptor}|Ki", 0)
    assert claimed <= raw, (
        f"{receptor} panel claims {claimed} molecules from {raw} ChEMBL rows")
    assert (claimed > 0) == (raw > 0)
