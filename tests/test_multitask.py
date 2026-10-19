"""Family pooling: the global holdout that makes the comparison honest.

A peptide measured on MC1R and MC4R contributes a row to each. Hold out its
MC4R row while leaving its MC1R row in training and the pooled model has seen
the molecule's affinity on a sister receptor, a median 1.31 log units away.
The pooled arm would then beat the single-task arm for a reason unrelated to
sample efficiency. These tests assert the holdout removes a cluster's
molecules from training for every receptor at once.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from atlas.ml.multitask import (
    MIN_TEST_ROWS, MULTITASK_PATH, global_cluster_holdout,
)


@pytest.fixture(scope="module")
def report() -> dict:
    if not MULTITASK_PATH.exists():
        pytest.skip(f"{MULTITASK_PATH.name} not built; run atlas.ml.multitask")
    return json.loads(MULTITASK_PATH.read_text())


# --------------------------------------------------------- global holdout ----

def test_holdout_removes_a_molecule_from_every_receptor_at_once():
    """The guard the whole experiment depends on.

    Four molecules in two clusters, each measured on two receptors, so eight
    rows. Whichever cluster is held out, both of its molecules' rows must be
    held out on both receptors.
    """
    molecule_index = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    mol_labels = np.array([0, 0, 1, 1])
    mask = global_cluster_holdout(molecule_index, mol_labels,
                                  test_fraction=0.5, seed=0)
    held_mols = {int(m) for m, h in zip(molecule_index, mask) if h}
    # Every row of a held molecule is held, on both receptors.
    for row, m in enumerate(molecule_index):
        assert mask[row] == (int(m) in held_mols)
    # And holding a cluster holds both its molecules.
    clusters_held = {int(mol_labels[m]) for m in held_mols}
    for m, c in enumerate(mol_labels):
        if int(c) in clusters_held:
            assert m in held_mols


def test_holdout_is_a_row_mask_of_the_right_length():
    molecule_index = np.arange(10) % 5
    mol_labels = np.array([0, 0, 1, 1, 2])
    mask = global_cluster_holdout(molecule_index, mol_labels, seed=1)
    assert mask.shape == (10,)
    assert mask.dtype == bool


def test_holdout_leaves_some_rows_on_each_side():
    molecule_index = np.repeat(np.arange(8), 2)
    mol_labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    mask = global_cluster_holdout(molecule_index, mol_labels,
                                  test_fraction=0.25, seed=0)
    assert mask.any() and not mask.all()


def test_holdout_never_splits_a_cluster():
    molecule_index = np.arange(12)
    mol_labels = np.array([0] * 4 + [1] * 4 + [2] * 4)
    mask = global_cluster_holdout(molecule_index, mol_labels,
                                  test_fraction=0.34, seed=3)
    for c in np.unique(mol_labels):
        rows = mask[mol_labels == c]
        assert rows.all() or not rows.any(), c


# ----------------------------------------------------------- saved report ----

def test_both_arms_are_scored_on_the_same_rows(report):
    """The arms must differ only in what they may train on."""
    for rec, v in report["per_receptor"].items():
        if "abstained" in v:
            continue
        sizes = {v[arm]["n_test"] for arm in
                 ("single_task", "pooled_family", "baseline_mean")}
        assert len(sizes) == 1, rec


def test_pooled_arm_trains_on_strictly_more_rows(report):
    for rec, v in report["per_receptor"].items():
        if "abstained" in v:
            continue
        assert v["pooled_family"]["n_train"] > v["single_task"]["n_train"], rec
        assert v["extra_training_rows"] == (
            v["pooled_family"]["n_train"] - v["single_task"]["n_train"])


def test_gain_is_consistent_with_its_components(report):
    for rec, v in report["per_receptor"].items():
        if "abstained" in v:
            continue
        assert v["pooled_gain_mae"] == pytest.approx(
            v["single_task"]["mae"] - v["pooled_family"]["mae"], rel=1e-9)
        assert v["pooled_helps"] == (v["pooled_gain_mae"] > 0)


def test_thin_receptors_abstain_rather_than_report_noise(report):
    """MC5R-scale data should abstain, not produce a number."""
    for rec, v in report["per_receptor"].items():
        if "abstained" in v:
            assert str(MIN_TEST_ROWS) in v["abstained"]
        else:
            assert v["single_task"]["n_test"] >= MIN_TEST_ROWS


def test_summary_partitions_the_receptors(report):
    s = report["summary"]
    usable = {k for k, v in report["per_receptor"].items() if "abstained" not in v}
    assert set(s["receptors_compared"]) == usable
    assert set(s["pooling_helps_on"]) | set(s["pooling_hurts_on"]) == usable
    assert not set(s["pooling_helps_on"]) & set(s["pooling_hurts_on"])


def test_holdout_discipline_is_recorded(report):
    note = report["summary"]["holdout"].lower()
    assert "globally" in note and "sister receptor" in note


def test_findings_carry_numbers(report):
    assert report["findings"]
    for f in report["findings"]:
        assert any(ch.isdigit() for ch in f)
