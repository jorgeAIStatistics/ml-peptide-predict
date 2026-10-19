"""Stage 3/4 panel benchmark: the grouping that keeps the comparison honest.

A molecule measured on several receptors contributes several rows. If those
rows split independently, the model is shown the molecule's affinity on one
receptor and asked for it on another — a median 1.31 log units away, with the
chemistry handed over. Every split here must therefore move whole molecules.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from atlas.ml.multitarget import PANEL_PATH, load_panel
from atlas.ml.panel_benchmark import (
    PANEL_BENCHMARK_PATH, grouped_cluster_split, molecule_cluster_labels,
    peptide_panel, receptor_onehot, receptor_sequence_features,
)


@pytest.fixture(scope="module")
def pp():
    if not PANEL_PATH.exists():
        pytest.skip(f"{PANEL_PATH.name} not built; run atlas.ml.multitarget")
    return peptide_panel(load_panel())


@pytest.fixture(scope="module")
def report() -> dict:
    if not PANEL_BENCHMARK_PATH.exists():
        pytest.skip(f"{PANEL_BENCHMARK_PATH.name} not built")
    return json.loads(PANEL_BENCHMARK_PATH.read_text())


# --------------------------------------------------------------- encoders ----

def test_unseen_receptor_gets_an_all_zero_onehot():
    """The reason one-hot cannot survive leave-one-receptor-out.

    An all-zero row is the honest encoding: this representation has nothing to
    say about a receptor it never saw.
    """
    enc = receptor_onehot(["MC4R", "MC5R"], ["MC4R", "MC1R"])
    assert enc[0].tolist() == [1.0, 0.0]
    assert enc[1].sum() == 0.0


def test_receptor_sequence_features_are_continuous_and_distinct(pp):
    panel = load_panel()
    names = sorted(panel.receptors)
    enc = receptor_sequence_features(names, panel.receptors)
    assert enc.shape == (len(names), 21)
    assert np.isfinite(enc).all()
    present = [i for i, n in enumerate(names) if panel.receptors[n].get("sequence")]
    if len(present) >= 2:
        # Different receptors must not encode identically, or the encoder
        # carries no target information at all.
        rows = {tuple(np.round(enc[i], 6)) for i in present}
        assert len(rows) == len(present)
        # And composition fractions sum to about 1 for each.
        for i in present:
            assert enc[i, :20].sum() == pytest.approx(1.0, abs=1e-6)


def test_sequence_encoding_has_no_more_capacity_than_onehot_on_seen_receptors(pp):
    """Stated as a test because it is why the cluster split cannot settle it.

    Five receptors give five distinct vectors under either encoding, so a
    sequence encoding can only matter where a receptor is unseen.
    """
    panel = load_panel()
    names = sorted(set(pp.receptor.tolist()))
    seq = receptor_sequence_features(names, panel.receptors)
    oh = receptor_onehot(names, names)
    assert len({tuple(np.round(r, 6)) for r in seq}) == len(names)
    assert len({tuple(r) for r in oh}) == len(names)


# ------------------------------------------------------- grouping guards ----

def test_cluster_labels_are_per_molecule_not_per_row(pp):
    labels = molecule_cluster_labels(pp)
    assert len(labels) == len(pp.molecules)
    assert len(labels) < len(pp)          # fewer molecules than rows
    assert (labels >= 0).all()


def test_no_molecule_straddles_the_grouped_split(pp):
    """The leakage guard this module exists for."""
    labels = molecule_cluster_labels(pp)
    tr, te = grouped_cluster_split(pp, labels)
    train_mols = {pp.rows[i].molecule_chembl_id for i in tr}
    test_mols = {pp.rows[i].molecule_chembl_id for i in te}
    assert train_mols.isdisjoint(test_mols)


def test_no_cluster_straddles_the_grouped_split(pp):
    labels = molecule_cluster_labels(pp)
    tr, te = grouped_cluster_split(pp, labels)
    train_clusters = {labels[pp.molecule_index[i]] for i in tr}
    test_clusters = {labels[pp.molecule_index[i]] for i in te}
    assert train_clusters.isdisjoint(test_clusters)


def test_grouped_split_is_complete_and_disjoint(pp):
    labels = molecule_cluster_labels(pp)
    tr, te = grouped_cluster_split(pp, labels)
    assert set(tr.tolist()).isdisjoint(te.tolist())
    assert len(set(tr.tolist()) | set(te.tolist())) == len(pp)
    assert len(te) > 0


def test_multi_receptor_rows_move_together(pp):
    """A peptide on three receptors contributes three rows to one side."""
    labels = molecule_cluster_labels(pp)
    tr, te = grouped_cluster_split(pp, labels)
    side = {}
    for i in tr:
        side.setdefault(pp.rows[i].molecule_chembl_id, set()).add("train")
    for i in te:
        side.setdefault(pp.rows[i].molecule_chembl_id, set()).add("test")
    assert all(len(v) == 1 for v in side.values())


# ----------------------------------------------------------- panel content ----

def test_only_structurally_confirmed_peptides_are_kept(pp):
    assert all(pp.analyses[m].is_peptide for m in pp.molecules)
    assert len(pp) >= len(pp.molecules)


def test_rows_align_with_labels_and_receptors(pp):
    assert len(pp.y) == len(pp) == len(pp.receptor) == len(pp.molecule_index)
    assert pp.molecule_index.max() < len(pp.molecules)


def test_panel_adds_measurements_not_merely_molecules(report):
    """The point of the family expansion, as a measurement."""
    c = report["composition"]
    assert c["n_rows"] > c["n_peptides"]
    assert c["peptides_on_multiple_receptors"] > 0


# --------------------------------------------------------- saved findings ----

def test_loro_held_out_onehot_is_verified_all_zero(report):
    for held, v in report["leave_one_receptor_out"].items():
        if "skipped" in v:
            continue
        assert v["held_out_receptor_onehot_is_all_zero"] is True, held


def test_loro_arms_share_a_test_set(report):
    for held, v in report["leave_one_receptor_out"].items():
        if "skipped" in v:
            continue
        sizes = {v[k]["n_test"] for k in
                 ("baseline_mean", "peptide_only",
                  "peptide_plus_receptor_onehot", "peptide_plus_receptor_sequence")}
        assert len(sizes) == 1, held


def test_receptor_signal_arms_share_a_test_set(report):
    sizes = {v["n_test"] for v in report["receptor_signal"]["results"].values()}
    assert len(sizes) == 1


def test_findings_do_not_claim_a_target_encoder(report):
    """A negative conclusion must survive in the report, not be smoothed away."""
    joined = " ".join(report["findings"])
    assert "not established" in joined
    assert any(ch.isdigit() for ch in joined)
