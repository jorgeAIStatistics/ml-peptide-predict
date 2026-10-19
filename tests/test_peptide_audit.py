"""Stage A audit: internal consistency and the invariants its claims rest on.

These tests do not assert particular measured values -- those belong to the
dataset and would turn a data change into a test failure. They assert that the
quantities are arithmetically possible, that the classification partitions
cleanly, and that the specific bugs found while building the audit stay fixed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from atlas.ml.peptide_audit import (
    AUDIT_PATH, composition_features, series_structure, structural_census,
)
from atlas.ml.peptide_structure import PeptideClass, analyse_peptide

from tests.test_peptide_structure import (
    AFAMELANOTIDE, ASPIRIN, MELANOTAN_II, OXYTOCIN, SOMATOSTATIN,
)


@pytest.fixture(scope="module")
def report() -> dict:
    if not AUDIT_PATH.exists():
        pytest.skip(f"{AUDIT_PATH.name} not built; run atlas.ml.peptide_audit")
    return json.loads(AUDIT_PATH.read_text())


def test_classification_partitions_the_dataset(report):
    """Every molecule gets exactly one structural class."""
    census = report["census"]
    assert sum(census["structural_classification"].values()) == census["n_molecules"]
    assert sum(census["previous_classification"].values()) == census["n_molecules"]


def test_cross_tabulation_totals_match(report):
    census = report["census"]
    assert sum(r["n"] for r in census["cross_tabulation"]) == census["n_molecules"]


def test_peptide_count_is_sum_of_peptide_classes(report):
    """The headline peptide count excludes peptidomimetics."""
    cls = report["census"]["structural_classification"]
    peptide_classes = [c.value for c in PeptideClass if c.is_peptide]
    assert report["census"]["structural_peptide_count"] == sum(
        cls.get(k, 0) for k in peptide_classes
    )
    assert "peptidomimetic" not in peptide_classes


def test_membership_disagreement_is_the_symmetric_difference(report):
    """Equal totals must not be mistaken for equal membership."""
    c = report["census"]
    assert c["membership_disagreement"] == c["n_false_positives"] + c["n_false_negatives"]


def test_document_shares_are_fractions(report):
    """Molecules cite several documents, so shares must be set-based.

    Summing per-document counts double-counts and produced a 'share' of 1.265.
    """
    for group in ("peptides", "non_peptides", "all"):
        a = report["assays"][group]
        assert 0.0 <= a["largest_document_share"] <= 1.0
        assert 0.0 <= a["top_10_document_share"] <= 1.0
        assert a["largest_document_share"] <= a["top_10_document_share"]


def test_similarity_quantiles_are_ordered_and_in_range(report):
    nn = report["shift"]["nearest_neighbour_similarity"]
    for name, q in nn.items():
        assert -1.0 <= q["min"] <= q["p10"] <= q["median"] <= q["p90"] <= q["max"] <= 1.0, name


def test_peptides_are_more_like_each_other_than_like_non_peptides(report):
    nn = report["shift"]["nearest_neighbour_similarity"]
    assert nn["peptide_to_peptide"]["median"] > nn["peptide_to_non_peptide"]["median"]


def test_series_count_does_not_exceed_molecule_count(report):
    s = report["series"]
    assert 1 <= s["n_clusters"] <= s["n_molecules"]
    assert 0.0 < s["largest_cluster_share"] <= 1.0
    assert s["largest_cluster_share"] <= s["top_5_cluster_share"] <= 1.0


def test_within_cluster_variance_is_not_greater_than_total(report):
    """Label variance inside a series cannot exceed the overall spread."""
    s = report["series"]
    assert s["median_within_cluster_label_sd"] <= s["overall_label_sd"]


def test_learning_curve_sizes_increase(report):
    curve = report["failure"]["C_dataset_size"]["learning_curve"]
    sizes = [c["n_train"] for c in curve]
    assert sizes == sorted(sizes)
    assert all(0 < c["fraction"] <= 1.0 for c in curve)


def test_transfer_comparison_uses_one_test_set(report):
    """Only the training population may vary between the transfer arms."""
    b = report["failure"]["B_domain_shift"]
    n_test = {b[k]["n_test"] for k in ("peptides_only", "non_peptides_only", "pooled")}
    assert len(n_test) == 1, "transfer arms were scored on different test sets"


def test_noise_floor_is_below_model_error(report):
    """Were it above, the model would be fitting label noise."""
    d = report["failure"]["D_irreducible_variability"]
    if d["noise_floor_mae_estimate"] is not None:
        assert d["noise_floor_mae_estimate"] < d["best_model_mae"]
        assert d["headroom_to_floor"] > 0


def test_every_finding_is_a_sentence_with_a_number(report):
    """A finding without a measurement in it is an assertion, not a finding."""
    assert report["findings"]
    for f in report["findings"]:
        assert any(ch.isdigit() for ch in f), f
        assert len(f) > 60, f


def test_caveats_are_recorded_not_dropped(report):
    assert "caveat" in report["labels"]
    assert "caveat" in report["failure"]["D_irreducible_variability"]
    assert "caveat" in report["failure"]["C_dataset_size"]["extrapolation"]


# ---------------------------------------------------------------- units ----

class _Rec:
    """Minimal stand-in for ActivityRecord."""

    def __init__(self, smiles, pki=8.0, spread=0.0, reps=1, docs=("D1",), assays=("A1",)):
        self.molecule_chembl_id = f"X{abs(hash(smiles)) % 10000}"
        self.smiles = smiles
        self.pki = pki
        self.pki_min = self.pki_max = pki
        self.replicate_spread = spread
        self.n_replicates = reps
        self.molecule_type = None
        self.document_ids = docs
        self.assay_ids = assays

    @property
    def is_replicated(self):
        return self.n_replicates > 1

    @property
    def is_discordant(self):
        return self.replicate_spread > 1.0


def test_census_separates_peptides_from_a_non_peptide():
    recs = [_Rec(s) for _cid, s in (OXYTOCIN, SOMATOSTATIN, AFAMELANOTIDE, ASPIRIN)]
    census, analyses = structural_census(recs)
    assert census["n_molecules"] == 4
    assert census["structural_peptide_count"] == 3
    assert sum(1 for a in analyses if not a.is_peptide) == 1


def test_composition_features_are_finite_and_aligned():
    analyses = [analyse_peptide(s) for _c, s in (OXYTOCIN, MELANOTAN_II, ASPIRIN)]
    x = composition_features(analyses)
    assert x.shape[0] == 3
    assert np.isfinite(x).all()
    # The non-peptide contributes an all-but-weight-zero row, not a crash.
    assert x[2, : len("ACDEFGHIKLMNPQRSTVWXY")].sum() == 0


def test_composition_distinguishes_disulfide_from_lactam():
    oxy, mt2 = (composition_features([analyse_peptide(s)])[0]
                for _c, s in (OXYTOCIN, MELANOTAN_II))
    assert not np.allclose(oxy, mt2)


def test_composition_cannot_distinguish_positional_isomers():
    """The measured weakness of the composition representation, made explicit.

    Two tripeptides with the same residues in a different order are identical
    under a bag-of-residues encoding. This is why composition scored below the
    mean baseline on a dataset of single-substitution analogue series.
    """
    ala_gly_ser = "C[C@H](N)C(=O)NCC(=O)N[C@@H](CO)C(O)=O"
    ser_gly_ala = "OC[C@H](N)C(=O)NCC(=O)N[C@@H](C)C(O)=O"
    a, b = (analyse_peptide(s) for s in (ala_gly_ser, ser_gly_ala))
    assert sorted(a.sequence) == sorted(b.sequence)
    assert a.sequence != b.sequence
    assert np.allclose(*composition_features([a, b]))


def test_series_structure_finds_one_cluster_for_near_duplicates():
    from atlas.ml.features import morgan_matrix
    from rdkit import Chem

    mols = [Chem.MolFromSmiles(s) for _c, s in (MELANOTAN_II, MELANOTAN_II)]
    fp = morgan_matrix(mols)
    s = series_structure(fp, np.array([8.0, 8.2]))
    assert s["n_clusters"] == 1
    assert s["largest_cluster"] == 2
