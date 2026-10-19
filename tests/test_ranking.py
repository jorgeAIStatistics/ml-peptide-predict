"""The ranking gate and the service it licenses.

Two things matter most here. The bound must be taken over molecules rather
than pairs, because pairs sharing a molecule are not independent observations;
and no numeric affinity may appear in a ranking answer, because the affinity
gate still fails and ranking is a different claim.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from atlas.ml.ranking_gate import (
    RANKING_GATE, RANKING_PATH, bootstrap_over_groups, naive_binomial_bound,
)


@pytest.fixture(scope="module")
def report() -> dict:
    if not RANKING_PATH.exists():
        pytest.skip(f"{RANKING_PATH.name} not built; run atlas.ml.ranking_gate")
    return json.loads(RANKING_PATH.read_text())


@pytest.fixture(scope="module")
def service():
    from atlas.ml.ranking_service import get_ranking_service
    return get_ranking_service()


# ------------------------------------------------------------ statistics ----

def test_bootstrap_resamples_groups_not_trials():
    """A molecule in 50 pairs is one observation, not 50.

    Two groups, one large and wrong, one small and right. Resampling groups
    must produce a wide interval; resampling trials would not.
    """
    per_group = {"big": (0, 50), "small": (10, 10)}
    out = bootstrap_over_groups(per_group, n_boot=500, seed=0)
    assert out["n_groups"] == 2
    assert out["mean"] == pytest.approx(10 / 60)
    # With only two groups the interval must span a wide range.
    assert out["upper"] - out["lower"] > 0.3


def test_bootstrap_ignores_groups_with_no_trials():
    out = bootstrap_over_groups({"a": (5, 10), "empty": (0, 0)}, n_boot=200, seed=0)
    assert out["n_groups"] == 1


def test_bootstrap_on_no_groups_returns_nan():
    out = bootstrap_over_groups({}, n_boot=10)
    assert np.isnan(out["mean"])
    assert out["n_groups"] == 0


def test_naive_binomial_bound_is_below_the_point_estimate():
    assert naive_binomial_bound(92, 100) < 0.92
    assert 0.0 <= naive_binomial_bound(92, 100) <= 1.0
    assert np.isnan(naive_binomial_bound(0, 0))


def test_naive_bound_tightens_with_more_trials():
    """Which is exactly why it must not be applied to dependent pairs."""
    few = naive_binomial_bound(92, 100)
    many = naive_binomial_bound(9200, 10000)
    assert many > few


def test_clustered_bound_is_more_conservative_than_naive(report):
    """The measured cost of the independence assumption."""
    within = report["evaluations"]["within_series_held_out_molecules"]
    assert within["clustered"]["lower"] < within["naive_lower_bound"]


# ------------------------------------------------------- the served domain ----

def test_quoted_accuracy_is_measured_on_the_domain_served(report):
    """Cluster co-membership is looser than mutual similarity.

    The largest series holds 335 molecules within 0.4 of a centroid, so two of
    its members can be far apart. The service checks mutual similarity, so the
    gate figure must be measured under that condition.
    """
    within = report["evaluations"]["within_series_held_out_molecules"]
    assert within["n_served_pairs"] < within["n_meaningful_pairs"]
    assert within["pair_weighted_accuracy"] != within["accuracy_all_meaningful_pairs"]
    assert within["clustered_all_meaningful_pairs"]


def test_gate_criteria_use_the_served_domain_bound(report):
    within = report["evaluations"]["within_series_held_out_molecules"]
    actual = within["criteria"]["within_series_accuracy_lower_bound"]["actual"]
    assert actual == pytest.approx(within["clustered"]["lower"], rel=1e-9)


# ------------------------------------------------------------- the verdict ----

def test_within_series_passes_and_cross_series_fails(report):
    """What bounds the claim to within a series rather than leaving it open."""
    within = report["evaluations"]["within_series_held_out_molecules"]
    cross = report["evaluations"]["cross_series_held_out_series"]
    assert within["passes"] is True
    assert cross["passes"] is False
    assert report["verdict"]["within_series_passes"] is True
    assert report["verdict"]["cross_series_passes"] is False
    assert report["verdict"]["deployable_mode"] == "within_series_ranking"


def test_verdict_follows_from_the_criteria(report):
    for ev in report["evaluations"].values():
        assert ev["passes"] == all(c["met"] for c in ev["criteria"].values())


def test_accuracy_rises_with_the_size_of_the_difference(report):
    """The behaviour a real ranker shows, and a degenerate one would not."""
    bands = report["evaluations"]["within_series_held_out_molecules"]["accuracy_by_delta_band"]
    ordered = [bands[k]["accuracy"] for k in ("0-0.5", "0.5-1", "1-2", ">2") if k in bands]
    assert ordered == sorted(ordered), bands


def test_near_chance_on_negligible_differences_justifies_the_threshold(report):
    """The 0-0.5 band is near chance, which is why it is excluded."""
    bands = report["evaluations"]["within_series_held_out_molecules"]["accuracy_by_delta_band"]
    if "0-0.5" in bands:
        assert bands["0-0.5"]["accuracy"] < 0.75


def test_the_hardest_pairs_are_the_most_similar(report):
    """Activity cliffs: reported as a limitation of the claim."""
    bands = report["evaluations"]["within_series_held_out_molecules"]["accuracy_by_similarity_band"]
    if ">0.8" in bands and "0.6-0.8" in bands:
        assert bands[">0.8"]["accuracy"] < bands["0.6-0.8"]["accuracy"]
        assert any("activity cliff" in f for f in report["findings"])


def test_report_states_what_it_does_not_license(report):
    text = report["what_this_licenses"].lower()
    assert "no numeric affinity estimate" in text
    assert "knowledge-graph edge" in text
    assert "still fails" in text


# ------------------------------------------------------------- the service ----

def test_identical_structures_are_refused(service):
    r = service.rank("CC(=O)Oc1ccccc1C(O)=O", "CC(=O)Oc1ccccc1C(O)=O")
    assert r.abstained
    assert r.refusal_reason == "the_two_structures_are_the_same"


def test_dissimilar_molecules_are_refused_with_the_reason(service):
    r = service.rank(
        "CC(=O)Oc1ccccc1C(O)=O",
        "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C")
    assert r.abstained
    assert r.refusal_reason == "molecules_are_not_of_one_series"
    assert str(RANKING_GATE["pair_similarity_min"]) in r.refusal_explanation
    assert r.what_would_resolve_it


def test_unparseable_structure_is_refused(service):
    r = service.rank("not a molecule", "CC(=O)Oc1ccccc1C(O)=O")
    assert r.abstained
    assert r.refusal_reason == "unresolved_molecular_identity"


def test_a_ranking_answer_contains_no_affinity_number(service):
    """The guard that matters: the affinity gate fails, so no number escapes.

    Checked over the whole serialised payload, not only the named fields, so a
    future field cannot leak one unnoticed.
    """
    from dataclasses import asdict

    from tests.test_peptide_structure import MELANOTAN_II

    # A pair that is answerable: one molecule against itself with one atom
    # changed would be near-identical, so use two genuine same-series analogues
    # from the dataset instead.
    from atlas.ml.dataset import load_dataset

    recs = load_dataset().records
    by_id = {r.molecule_chembl_id: r for r in recs}
    # Find any two records whose structures are similar enough to pass the floor.
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    peptides = [r for r in recs if len(r.smiles) > 300][:60]
    fps = {r.molecule_chembl_id: gen.GetFingerprint(Chem.MolFromSmiles(r.smiles))
           for r in peptides}
    pair = None
    ids = list(fps)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            sim = DataStructs.TanimotoSimilarity(fps[a], fps[b])
            if RANKING_GATE["pair_similarity_min"] <= sim < 0.99:
                pair = (a, b)
                break
        if pair:
            break
    if pair is None:
        pytest.skip("no same-series pair found in the sampled records")

    r = service.rank(by_id[pair[0]].smiles, by_id[pair[1]].smiles)
    assert not r.abstained
    blob = json.dumps(asdict(r))
    for forbidden in ("predicted_pki", "pki_min", "pki_max", "ki_nm",
                      "interval_low", "interval_high"):
        assert forbidden not in blob, forbidden
    assert any("No numeric affinity is returned" in l for l in r.limitations)


def test_measurement_comparison_carries_no_model_accuracy(service):
    """Two measured molecules are compared, not ranked by a model."""
    from atlas.ml.dataset import load_dataset
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    recs = [r for r in load_dataset().records if len(r.smiles) > 300][:60]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(r.smiles)) for r in recs]
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            if RANKING_GATE["pair_similarity_min"] <= DataStructs.TanimotoSimilarity(
                    fps[i], fps[j]) < 0.99:
                r = service.rank(recs[i].smiles, recs[j].smiles)
                if r.basis == "measurement":
                    assert r.confidence is None
                    assert any("comparison of measurements" in l
                               for l in r.limitations)
                    return
    pytest.skip("no measured same-series pair found in the sampled records")


def test_gate_summary_always_states_the_affinity_gate_fails(service):
    r = service.rank("not a molecule", "CC")
    assert "still failing" in r.gate["affinity_gate"]
    assert r.gate["within_series_ranking_passes"] is True
    assert r.gate["pair_similarity_min"] == RANKING_GATE["pair_similarity_min"]
