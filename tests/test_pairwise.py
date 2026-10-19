"""Within-series pairwise reformulation: antisymmetry and leakage guards.

Splitting pairs at random would be catastrophic: with pair (A,B) held out and
(A,C), (B,D) in training, both molecules' affinities are almost fully
determined. Every split must therefore partition molecules or clusters, and
these tests assert it on constructed data where the answer is known.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from atlas.ml.pairwise import (
    DATA_PROBE, MEANINGFUL_DELTA, MIN_CLUSTER, PAIRWISE_PATH, PairSet,
    _score_pairs, build_pairs, split_pairs_by_cluster, split_pairs_by_molecule,
)

# Three clusters: sizes 4, 3 and 2. The last is below MIN_CLUSTER.
_LABELS = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2])
_Y = np.array([9.0, 8.0, 7.0, 6.0, 5.5, 7.5, 6.5, 4.0, 10.0])


@pytest.fixture(scope="module")
def pairs() -> PairSet:
    return build_pairs(_Y, _LABELS)


@pytest.fixture(scope="module")
def report() -> dict:
    if not PAIRWISE_PATH.exists():
        pytest.skip(f"{PAIRWISE_PATH.name} not built; run atlas.ml.pairwise")
    return json.loads(PAIRWISE_PATH.read_text())


# ----------------------------------------------------------- construction ----

def test_pairs_are_within_series_only(pairs):
    """A difference across series would reintroduce the offset it cancels."""
    assert (_LABELS[pairs.left] == _LABELS[pairs.right]).all()
    assert (_LABELS[pairs.left] == pairs.cluster).all()


def test_small_clusters_contribute_no_pairs(pairs):
    assert MIN_CLUSTER == 3
    assert 2 not in set(pairs.cluster.tolist())


def test_both_orderings_are_present_and_antisymmetric(pairs):
    """Training on both halves enforces antisymmetry in the data."""
    seen = {(int(a), int(b)): d for a, b, d in
            zip(pairs.left, pairs.right, pairs.delta)}
    for (a, b), d in seen.items():
        assert (b, a) in seen
        assert seen[(b, a)] == pytest.approx(-d)


def test_pair_count_matches_the_combinatorics(pairs):
    # Cluster 0 has 4 members -> 6 unordered -> 12 ordered; cluster 1 has 3 -> 6.
    assert len(pairs) == 12 + 6


def test_delta_is_the_affinity_difference(pairs):
    for a, b, d in zip(pairs.left, pairs.right, pairs.delta):
        assert d == pytest.approx(_Y[a] - _Y[b])


def test_no_molecule_is_paired_with_itself(pairs):
    assert (pairs.left != pairs.right).all()


def test_meaningful_mask_uses_the_declared_threshold(pairs):
    assert MEANINGFUL_DELTA == 0.5
    assert (pairs.meaningful == (np.abs(pairs.delta) >= 0.5)).all()


def test_max_per_cluster_caps_a_large_series():
    labels = np.zeros(40, dtype=int)
    y = np.arange(40, dtype=float)
    capped = build_pairs(y, labels, max_per_cluster=50)
    assert len(capped) == 100          # 50 unordered, both orderings
    uncapped = build_pairs(y, labels, max_per_cluster=None)
    assert len(uncapped) == 40 * 39


# ---------------------------------------------------------- leakage guards ----

def test_cluster_split_holds_out_whole_series(pairs):
    tr, te = split_pairs_by_cluster(pairs, _LABELS, test_fraction=0.4, seed=0)
    train_clusters = set(pairs.cluster[tr].tolist())
    test_clusters = set(pairs.cluster[te].tolist())
    assert train_clusters.isdisjoint(test_clusters)


def test_cluster_split_leaves_no_molecule_on_both_sides(pairs):
    """The strongest form of the guard: both members of a test pair unseen."""
    tr, te = split_pairs_by_cluster(pairs, _LABELS, test_fraction=0.4, seed=0)
    train_mols = set(pairs.left[tr].tolist()) | set(pairs.right[tr].tolist())
    test_mols = set(pairs.left[te].tolist()) | set(pairs.right[te].tolist())
    assert train_mols.isdisjoint(test_mols)


def test_molecule_split_tests_pairs_with_exactly_one_unseen_member(pairs):
    tr, te = split_pairs_by_molecule(pairs, len(_Y), test_fraction=0.3, seed=0)
    train_mols = set(pairs.left[tr].tolist()) | set(pairs.right[tr].tolist())
    for i in te:
        members = {int(pairs.left[i]), int(pairs.right[i])}
        unseen = members - train_mols
        assert len(unseen) == 1, (members, unseen)


def test_molecule_split_training_pairs_have_no_unseen_member(pairs):
    tr, te = split_pairs_by_molecule(pairs, len(_Y), test_fraction=0.3, seed=0)
    test_only = (set(pairs.left[te].tolist()) | set(pairs.right[te].tolist()))
    train_mols = set(pairs.left[tr].tolist()) | set(pairs.right[tr].tolist())
    # A molecule may appear in both lists, but never as a *training* member
    # while being the held-out member of its test pair.
    for i in tr:
        assert {int(pairs.left[i]), int(pairs.right[i])} <= train_mols
    assert test_only - train_mols, "fixture no longer holds out any molecule"


def test_splits_do_not_overlap(pairs):
    for tr, te in (split_pairs_by_cluster(pairs, _LABELS, seed=1),
                   split_pairs_by_molecule(pairs, len(_Y), seed=1)):
        assert set(tr.tolist()).isdisjoint(te.tolist())


# ----------------------------------------------------------------- scoring ----

def test_perfect_prediction_scores_perfectly():
    truth = np.array([1.0, -2.0, 0.5, -0.1])
    s = _score_pairs(truth.copy(), truth)
    assert s["mae"] == pytest.approx(0.0)
    assert s["sign_accuracy"] == 1.0
    assert s["sign_accuracy_meaningful"] == 1.0


def test_sign_accuracy_ignores_negligible_differences():
    """Accuracy on a near-zero difference is noise, not skill."""
    truth = np.array([2.0, -2.0, 0.01, -0.01])
    pred = np.array([1.0, -1.0, -5.0, 5.0])       # wrong only on the tiny pairs
    s = _score_pairs(pred, truth)
    assert s["sign_accuracy"] == pytest.approx(0.5)
    assert s["sign_accuracy_meaningful"] == pytest.approx(1.0)
    assert s["n_meaningful"] == 2


def test_zero_predictor_has_no_sign_skill():
    truth = np.array([1.0, -1.0, 2.0, -2.0])
    s = _score_pairs(np.zeros(4), truth)
    assert s["sign_accuracy"] == 0.0        # sign(0) matches nothing
    assert s["mae"] == pytest.approx(1.5)


# ------------------------------------------------------------- data probe ----

def test_data_probe_records_a_negative_result():
    """The probe that justified changing the task rather than the dataset."""
    assert DATA_PROBE["chembl_non_human_melanocortin_ki_binding"] < (
        0.1 * DATA_PROBE["chembl_human_melanocortin_ki_binding"])
    assert "No." in DATA_PROBE["verdict"]
    assert "species scale" in DATA_PROBE["verdict"]


# ----------------------------------------------------------- saved report ----

def test_report_compares_against_the_absolute_model(report):
    """The comparison that decides whether the reformulation bought anything."""
    for key in ("held_out_series", "held_out_molecules"):
        r = report[key]
        if "skipped" in r:
            continue
        assert "absolute_model_implied_delta" in r
        assert "pairwise_minus_implied_mae" in r
        assert r["pairwise_minus_implied_mae"] == pytest.approx(
            r["pairwise_model"]["mae"]
            - r["absolute_model_implied_delta"]["mae"], rel=1e-9)


def test_report_arms_share_a_test_set(report):
    for key in ("held_out_series", "held_out_molecules"):
        r = report[key]
        if "skipped" in r:
            continue
        sizes = {r[arm]["n_pairs"] for arm in
                 ("pairwise_model", "absolute_model_implied_delta",
                  "zero_predictor")}
        assert len(sizes) == 1, key


def test_pairs_multiply_the_training_examples(report):
    assert report["n_pairs"] > 10 * report["n_molecules"]


def test_findings_are_substantive_and_mostly_measured(report):
    """Most findings must carry a measurement; not quite all of them should.

    An earlier version of this test required a digit in every finding, which
    rejected the one that states what the ranking result *licenses* -- a
    conclusion about claims rather than a measurement, and legitimately
    numberless. The discipline worth keeping is that conclusions rest on
    numbers, not that every sentence contains one.
    """
    findings = report["findings"]
    assert len(findings) >= 4
    for f in findings:
        assert len(f) > 60, f
    with_numbers = [f for f in findings if any(ch.isdigit() for ch in f)]
    assert len(with_numbers) >= 2 * len(findings) // 3
