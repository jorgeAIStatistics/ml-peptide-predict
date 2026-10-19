"""Stage B benchmark: the ceiling argument and the two-way split.

The ceiling argument is load-bearing — it is what licenses a claim about every
frozen sequence embedding without running one — so its properties are asserted
directly rather than only observed on the dataset.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from atlas.ml.peptide_benchmark import (
    BENCHMARK_PATH, GATE, cross_series_groups, oracle_ceiling,
    within_series_split, _cross_series_split,
)


# ------------------------------------------------------- ceiling argument ----

def test_unique_keys_give_a_zero_ceiling():
    """A representation that separates everything is unconstrained by collisions.

    The ceiling analysis is informative only for coarse representations, and
    this is the boundary case that says so.
    """
    y = np.array([5.0, 7.0, 9.0])
    c = oracle_ceiling(["a", "b", "c"], y)
    assert c["oracle_mae"] == 0.0
    assert c["n_colliding_molecules"] == 0


def test_identical_keys_give_the_group_spread():
    """Molecules sharing a key can only receive one number."""
    y = np.array([6.0, 8.0])
    c = oracle_ceiling(["same", "same"], y)
    assert c["n_distinct_keys"] == 1
    assert c["oracle_mae"] == pytest.approx(1.0)      # |6-7| and |8-7|
    assert c["max_group_label_range"] == pytest.approx(2.0)
    assert c["colliding_fraction"] == 1.0


def test_refining_a_representation_cannot_raise_the_ceiling():
    """A finer representation has a ceiling no worse than a coarser one.

    This monotonicity is what makes the canonical-versus-chemistry-aware
    comparison valid: naming the side chains splits groups, never merges them.
    """
    y = np.array([4.0, 6.0, 8.0, 10.0])
    coarse = oracle_ceiling(["A", "A", "B", "B"], y)
    fine = oracle_ceiling(["A1", "A2", "B", "B"], y)
    assert fine["oracle_mae"] <= coarse["oracle_mae"]
    assert fine["n_distinct_keys"] > coarse["n_distinct_keys"]


def test_ceiling_is_a_lower_bound_on_any_predictor():
    """No assignment of one value per key beats the group mean on squared error.

    Checked against random alternatives, since the claim is what the whole
    ceiling argument rests on.
    """
    rng = np.random.default_rng(0)
    y = rng.normal(7.0, 1.5, 40)
    keys = rng.integers(0, 5, 40)
    oracle_rmse = float(np.sqrt(np.mean(
        (np.array([y[keys == k].mean() for k in keys]) - y) ** 2)))

    for _ in range(200):
        guess = {k: rng.normal(7.0, 1.5) for k in set(keys.tolist())}
        pred = np.array([guess[k] for k in keys])
        assert float(np.sqrt(np.mean((pred - y) ** 2))) >= oracle_rmse

    # And the reported figure is the mean absolute error of that same optimum.
    assert oracle_ceiling(list(keys), y)["oracle_mae"] == pytest.approx(float(
        np.mean(np.abs(np.array([y[keys == k].mean() for k in keys]) - y))))


# ----------------------------------------------------------------- splits ----

def test_cross_series_split_leaves_no_cluster_on_both_sides():
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10 + [3] * 10)
    tr, te = _cross_series_split(labels, test_fraction=0.25, seed=0)
    assert set(labels[tr]).isdisjoint(set(labels[te]))
    assert len(tr) + len(te) == len(labels)


def test_within_series_split_gives_every_test_molecule_a_neighbour():
    """The defining property of the within-series evaluation."""
    labels = np.array([0] * 8 + [1] * 8 + [2] * 2)
    tr, te = within_series_split(labels, test_fraction=0.25, min_cluster=4, seed=0)
    assert len(te) > 0
    for i in te:
        same = labels[tr] == labels[i]
        assert same.sum() > 0, "held-out molecule has no same-series neighbour"


def test_within_series_split_excludes_small_clusters():
    """A singleton held out would be a cross-series test mislabelled."""
    labels = np.array([0] * 8 + [1] * 2 + [2] * 1)
    tr, te = within_series_split(labels, min_cluster=4, seed=0)
    assert set(labels[te]) == {0}
    assert 1 in set(labels[tr]) and 2 in set(labels[tr])


def test_splits_are_disjoint_and_complete():
    labels = np.array([i // 5 for i in range(40)])
    for tr, te in (_cross_series_split(labels, seed=1),
                   within_series_split(labels, seed=1)):
        assert set(tr.tolist()).isdisjoint(te.tolist())
        assert len(set(tr.tolist()) | set(te.tolist())) == len(labels)


def test_cluster_labels_cover_every_molecule():
    from atlas.ml.features import morgan_matrix
    from rdkit import Chem

    from tests.test_peptide_structure import (
        AFAMELANOTIDE, BRADYKININ, MELANOTAN_II, OXYTOCIN,
    )
    mols = [Chem.MolFromSmiles(s) for _c, s in
            (OXYTOCIN, MELANOTAN_II, AFAMELANOTIDE, BRADYKININ)]
    labels = cross_series_groups(morgan_matrix(mols))
    assert len(labels) == 4
    assert (labels >= 0).all()


# ----------------------------------------------------------- saved report ----

@pytest.fixture(scope="module")
def report() -> dict:
    if not BENCHMARK_PATH.exists():
        pytest.skip(f"{BENCHMARK_PATH.name} not built; run atlas.ml.peptide_benchmark")
    return json.loads(BENCHMARK_PATH.read_text())


def test_chemistry_aware_ceiling_beats_canonical(report):
    """The measured consequence of a twenty-letter alphabet."""
    c = report["ceilings"]
    assert (c["chemistry_aware_positional"]["oracle_mae"]
            < c["canonical_sequence_20_letter"]["oracle_mae"])
    assert (c["chemistry_aware_positional"]["n_distinct_keys"]
            > c["canonical_sequence_20_letter"]["n_distinct_keys"])


def test_positional_ceiling_beats_order_free(report):
    """Order carries information that composition discards."""
    c = report["ceilings"]
    assert (c["chemistry_aware_positional"]["oracle_mae"]
            <= c["chemistry_aware_composition_order_free"]["oracle_mae"])


def test_within_series_is_easier_than_cross_series(report):
    """If it were not, the two evaluations would be measuring one thing."""
    best = lambda d: min(v["mae"] for k, v in d["results"].items()
                         if k != "baseline_mean")
    assert best(report["within_series"]) < best(report["cross_series"])


def test_within_series_neighbours_are_closer(report):
    """The diagnostic that stops within-series success being over-read."""
    w = report["within_series"]["diagnostics"]["nearest_training_similarity"]
    x = report["cross_series"]["diagnostics"]["nearest_training_similarity"]
    assert w["median"] > x["median"]


def test_cross_series_test_contains_unseen_side_chains(report):
    """Without unseen residues, the one-hot/fragment contrast means nothing."""
    d = report["cross_series"]["diagnostics"]
    assert d["test_molecules_with_unseen_side_chain"]["n"] > 0


def test_gate_verdict_follows_from_its_criteria(report):
    gate = report["gate"]
    assert gate["passes"] == all(c["met"] for c in gate["criteria"].values())
    assert set(gate["criteria"]) == {
        "cross_series_mae", "cross_series_r2", "within_series_beats_series_mean"}


def test_gate_thresholds_are_above_the_label_noise_floor(report):
    """A gate at or below the noise floor would be unmeetable in principle."""
    assert GATE["cross_series_mae_max"] > 0.245


def test_every_result_reports_the_same_test_set_size(report):
    for ev in ("cross_series", "within_series"):
        sizes = {v["n_test"] for v in report[ev]["results"].values()}
        assert len(sizes) == 1, f"{ev} scored models on different test sets"


def test_findings_carry_measurements(report):
    assert len(report["findings"]) >= 8
    for f in report["findings"]:
        assert any(ch.isdigit() for ch in f), f
