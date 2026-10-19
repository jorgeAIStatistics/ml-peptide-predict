"""Tests for the machine-learning layer.

These pin the properties that keep the model honest: that splits do not leak,
that undefined statistics stay undefined rather than becoming NaN, that the
applicability gates actually fire, and that the model refuses peptides on the
strength of its own measured failure on them.

They use small synthetic molecule sets where possible so they run fast and do
not depend on the network.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from atlas.ml.dataset import ActivityRecord, DatasetBundle, load_dataset
from atlas.ml.features import (
    Modality,
    applicability,
    classify_modality,
    featurize,
    morgan_matrix,
    murcko_scaffold,
)
from atlas.ml.splits import leakage_report, modality_split, random_split, scaffold_split
from atlas.ml.train import _finite_or_none, _metrics, _num

from rdkit import Chem

REGISTRY = Path(__file__).resolve().parents[1] / "data" / "ml" / "model_registry.json"
DATASET = Path(__file__).resolve().parents[1] / "data" / "ml" / "mc4r_pki.json"

# A deliberately mixed set: three small molecules, three peptide-like.
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
BENZENE = "c1ccccc1"
TRIPEPTIDE = "CC(C)C[C@H](N)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CC(=O)O)C(=O)O"
HEXAPEPTIDE = (
    "CC(C)C[C@H](N)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CO)C(=O)N[C@@H](CCCCN)"
    "C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CC(=O)O)C(=O)O"
)


def rec(mol_id: str, smiles: str, pki: float, n: int = 1, spread: float = 0.0):
    return ActivityRecord(
        molecule_chembl_id=mol_id, smiles=smiles, pki=pki, n_replicates=n,
        replicate_spread=spread, molecule_type=None,
        document_ids=("DOC1",), assay_ids=("A1",),
    )


class TestModalityClassification:
    def test_small_molecules_classified_as_such(self):
        for smi in (ASPIRIN, CAFFEINE, BENZENE):
            assert classify_modality(Chem.MolFromSmiles(smi)) is Modality.SMALL_MOLECULE_LIKE

    def test_long_peptide_classified_as_peptide(self):
        assert classify_modality(Chem.MolFromSmiles(HEXAPEPTIDE)) is Modality.PEPTIDE_LIKE

    def test_short_peptide_is_not_peptide_like_by_weight(self):
        """Both thresholds must hold; a tripeptide is below the weight cut."""
        assert classify_modality(Chem.MolFromSmiles(TRIPEPTIDE)) is Modality.SMALL_MOLECULE_LIKE

    def test_unparseable_is_its_own_category(self):
        assert classify_modality(None) is Modality.UNPARSEABLE

    def test_peptidic_modalities_are_not_small_molecules(self):
        assert Modality.PEPTIDE_LIKE != Modality.SMALL_MOLECULE_LIKE


class TestFeatures:
    def test_featurize_produces_aligned_arrays(self):
        records = [rec("A", ASPIRIN, 7.0), rec("B", CAFFEINE, 6.0)]
        fs = featurize(records)
        assert len(fs) == 2
        assert fs.descriptors.shape[0] == 2
        assert fs.fingerprints.shape == (2, 2048)
        assert fs.y.tolist() == [7.0, 6.0]

    def test_unparseable_molecules_are_dropped_not_imputed(self):
        records = [rec("A", ASPIRIN, 7.0), rec("BAD", "not-a-molecule", 5.0)]
        fs = featurize(records)
        assert len(fs) == 1 and fs.ids == ["A"]

    def test_no_nan_survives_featurisation(self):
        fs = featurize([rec(str(i), s, 7.0) for i, s in
                        enumerate([ASPIRIN, CAFFEINE, BENZENE, HEXAPEPTIDE])])
        assert not np.isnan(fs.descriptors).any()

    def test_fingerprints_are_binary(self):
        fs = featurize([rec("A", ASPIRIN, 7.0)])
        assert set(np.unique(fs.fingerprints)).issubset({0, 1})

    def test_subset_preserves_alignment(self):
        fs = featurize([rec(c, s, float(i)) for i, (c, s) in
                        enumerate([("A", ASPIRIN), ("B", CAFFEINE), ("C", BENZENE)])])
        sub = fs.subset(np.array([0, 2]))
        assert sub.ids == ["A", "C"]
        assert sub.y.tolist() == [0.0, 2.0]
        assert sub.fingerprints.shape[0] == 2

    def test_combined_features_include_both_blocks(self):
        fs = featurize([rec("A", ASPIRIN, 7.0)])
        assert fs.combined.shape[1] == fs.descriptors.shape[1] + 2048

    def test_murcko_scaffold_strips_substituents(self):
        """Aspirin and benzoic acid share a benzene scaffold."""
        a = murcko_scaffold(Chem.MolFromSmiles(ASPIRIN))
        b = murcko_scaffold(Chem.MolFromSmiles("OC(=O)c1ccccc1"))
        assert a == b

    def test_acyclic_molecule_has_empty_scaffold(self):
        assert murcko_scaffold(Chem.MolFromSmiles("CCCCO")) == ""


class TestApplicability:
    def test_identical_molecule_has_similarity_one(self):
        fp = morgan_matrix([Chem.MolFromSmiles(ASPIRIN)])
        assert applicability(fp, fp)[0] == pytest.approx(1.0)

    def test_dissimilar_molecules_score_low(self):
        a = morgan_matrix([Chem.MolFromSmiles(ASPIRIN)])
        b = morgan_matrix([Chem.MolFromSmiles(HEXAPEPTIDE)])
        assert applicability(a, b)[0] < 0.3

    def test_similarity_is_bounded(self):
        mols = [Chem.MolFromSmiles(s) for s in (ASPIRIN, CAFFEINE, BENZENE)]
        fps = morgan_matrix(mols)
        sims = applicability(fps, fps)
        assert np.all((sims >= 0.0) & (sims <= 1.0))


class TestSplits:
    def _fs(self, n: int = 40):
        smis = [ASPIRIN, CAFFEINE, BENZENE, HEXAPEPTIDE]
        return featurize([rec(f"M{i}", smis[i % len(smis)], 5.0 + i % 5)
                          for i in range(n)])

    def test_random_split_partitions_without_overlap(self):
        fs = self._fs()
        s = random_split(fs, test_fraction=0.25, seed=1)
        assert s.n_train + s.n_test == len(fs)
        assert not (set(s.train_idx.tolist()) & set(s.test_idx.tolist()))

    def test_split_rejects_leaking_indices(self):
        from atlas.ml.splits import Split
        with pytest.raises(ValueError, match="leaks"):
            Split("bad", np.array([0, 1, 2]), np.array([2, 3]), "overlapping")

    def test_scaffold_split_shares_no_scaffold(self):
        fs = self._fs()
        s = scaffold_split(fs, test_fraction=0.3, seed=0)
        lk = leakage_report(fs, s)
        assert lk["n_shared_scaffolds"] == 0

    def test_random_split_does_share_scaffolds(self):
        """The contrast is the point: random splits leak chemotypes."""
        fs = self._fs()
        s = random_split(fs, test_fraction=0.3, seed=0)
        assert leakage_report(fs, s)["n_shared_scaffolds"] > 0

    def test_modality_split_puts_all_peptides_on_one_side(self):
        fs = self._fs()
        s = modality_split(fs, holdout=Modality.PEPTIDE_LIKE)
        assert all(fs.modality[i] is Modality.PEPTIDE_LIKE for i in s.test_idx)
        assert all(fs.modality[i] is not Modality.PEPTIDE_LIKE for i in s.train_idx)

    def test_split_records_its_seed_and_description(self):
        s = random_split(self._fs(), seed=7)
        assert s.seed == 7 and s.description


class TestMetrics:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        m = _metrics(y, y.copy())
        assert m["mae"] == pytest.approx(0.0)
        assert m["r2"] == pytest.approx(1.0)

    def test_constant_prediction_has_undefined_correlation(self):
        """A constant predictor has no rank correlation: null, not NaN."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        m = _metrics(y, np.full_like(y, 2.5))
        assert m["spearman"] is None
        assert m["pearson"] is None

    def test_r2_can_be_negative(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        m = _metrics(y, np.array([4.0, 3.0, 2.0, 1.0]))
        assert m["r2"] < 0

    def test_finite_or_none_rejects_nan_and_inf(self):
        assert _finite_or_none(float("nan")) is None
        assert _finite_or_none(float("inf")) is None
        assert _finite_or_none(1.5) == 1.5

    def test_num_formats_none_as_dash(self):
        assert _num(None, 7).strip() == "—"
        assert _num(0.681, 7).strip() == "0.681"


@pytest.mark.skipif(not DATASET.exists(), reason="dataset not built")
class TestDataset:
    def test_dataset_loads_and_version_matches_content(self):
        b = load_dataset()
        assert b.version
        assert len(b.records) > 500

    def test_altered_dataset_is_rejected(self, tmp_path):
        """The version is a content hash, so tampering is detected."""
        payload = json.loads(DATASET.read_text())
        payload["records"][0]["pki"] = 99.0
        p = tmp_path / "tampered.json"
        p.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="has been altered"):
            load_dataset(p)

    def test_filters_are_recorded(self):
        b = load_dataset()
        assert b.filters["standard_type"] == "Ki"
        assert "IC50 and EC50 were excluded" in b.filters["excluded"]

    def test_license_is_recorded(self):
        assert "CC BY-SA" in load_dataset().license_note

    def test_pki_values_are_in_range(self):
        b = load_dataset()
        assert all(3.0 <= r.pki <= 12.5 for r in b.records)

    def test_molecules_are_unique(self):
        b = load_dataset()
        ids = [r.molecule_chembl_id for r in b.records]
        assert len(ids) == len(set(ids))


@pytest.mark.skipif(not REGISTRY.exists(), reason="model registry not built")
class TestTrainedModelRegistry:
    @pytest.fixture(scope="class")
    @classmethod
    def reg(cls):
        return json.loads(REGISTRY.read_text())

    def test_registry_is_strict_json(self):
        """Bare NaN is not valid JSON and breaks every browser client."""
        def reject(c):
            raise ValueError(f"bare {c} in registry")
        json.loads(REGISTRY.read_text(), parse_constant=reject)

    def test_records_seed_and_dataset_version(self, reg):
        assert reg["seed"] and reg["dataset_version"]

    def test_every_split_reports_leakage(self, reg):
        for s in reg["splits"]:
            assert "leakage" in s and "n_shared_scaffolds" in s["leakage"]

    def test_scaffold_split_has_no_shared_scaffolds(self, reg):
        s = next(s for s in reg["splits"] if s["kind"] == "scaffold")
        assert s["leakage"]["n_shared_scaffolds"] == 0

    def test_random_split_leaks_more_than_scaffold(self, reg):
        rnd = next(s for s in reg["splits"] if s["kind"] == "random")
        scf = next(s for s in reg["splits"] if s["kind"] == "scaffold")
        assert (rnd["leakage"]["fraction_test_scaffolds_seen"]
                > scf["leakage"]["fraction_test_scaffolds_seen"])

    def test_no_identical_molecules_across_any_split(self, reg):
        for s in reg["splits"]:
            assert s["leakage"]["n_identical_molecules"] == 0

    def test_scaffold_error_exceeds_random_error(self, reg):
        """The honest split must look worse; if it does not, something leaked."""
        best = {}
        for r in reg["results"]:
            if r["model"] == "baseline_mean":
                continue
            best[r["split"]] = min(best.get(r["split"], 9e9), r["mae"])
        assert best["scaffold"] > best["random"]

    def test_negative_transfer_to_peptides_is_present(self, reg):
        """Every model must do worse than the baseline on held-out peptides."""
        rows = [r for r in reg["results"] if r["split"] == "modality_peptide_holdout"]
        base = next(r for r in rows if r["model"] == "baseline_mean")
        others = [r for r in rows if r["model"] != "baseline_mean"]
        assert others
        assert all(r["mae"] >= base["mae"] for r in others), (
            "a model beating the baseline on peptides would invalidate the "
            "documented reason for refusing peptide predictions"
        )

    def test_conformal_coverage_degrades_off_distribution(self, reg):
        def cov(split, model):
            return next(r["conformal_coverage"] for r in reg["results"]
                        if r["split"] == split and r["model"] == model)
        assert cov("random", "fingerprint_knn") > cov("scaffold", "fingerprint_knn")
        assert cov("scaffold", "fingerprint_knn") > cov(
            "modality_peptide_holdout", "fingerprint_knn")

    def test_applicability_bins_are_reported(self, reg):
        for r in reg["results"]:
            assert r["applicability"]["bins"]


@pytest.mark.skipif(not DATASET.exists(), reason="dataset not built")
class TestGatedPrediction:
    @pytest.fixture(scope="class")
    @classmethod
    def predictor(cls):
        from atlas.ml.predict import get_predictor
        return get_predictor()

    def test_peptide_is_refused_with_measured_justification(self, predictor):
        r = predictor.predict(HEXAPEPTIDE)
        assert r.abstained
        assert r.abstention_reason == "out_of_model_applicability_domain"
        assert "worse than no number" in r.abstention_explanation

    def test_unparseable_input_is_refused(self, predictor):
        r = predictor.predict("not-a-molecule")
        assert r.abstained
        assert r.abstention_reason == "unresolved_molecular_identity"

    def test_remote_molecule_is_refused_as_extrapolation(self, predictor):
        r = predictor.predict(ASPIRIN)
        assert r.abstained
        assert r.max_tanimoto_to_training < 0.30

    def test_refusal_always_says_what_would_resolve_it(self, predictor):
        for smi in (HEXAPEPTIDE, ASPIRIN, "garbage"):
            assert predictor.predict(smi).what_would_resolve_it

    def test_in_domain_molecule_gets_a_bounded_prediction(self, predictor):
        from atlas.ml.dataset import load_dataset
        known = load_dataset().records[0].smiles
        r = predictor.predict(known)
        assert not r.abstained
        assert r.interval_low < r.predicted_pki < r.interval_high
        assert r.predicted_ki_nm > 0

    def test_prediction_reports_its_neighbours(self, predictor):
        known = load_dataset().records[0].smiles
        r = predictor.predict(known)
        assert r.nearest_neighbours
        assert all("tanimoto" in n for n in r.nearest_neighbours)

    def test_prediction_carries_model_and_dataset_version(self, predictor):
        known = load_dataset().records[0].smiles
        r = predictor.predict(known)
        assert r.model_version and r.dataset_version

    def test_pki_to_nanomolar_conversion(self, predictor):
        """pKi 9 is 1 nM by definition."""
        from atlas.ml.predict import PredictionResult
        assert PredictionResult(smiles="x", predicted_pki=9.0).predicted_ki_nm == pytest.approx(1.0)
        assert PredictionResult(smiles="x", predicted_pki=6.0).predicted_ki_nm == pytest.approx(1000.0)

    def test_model_card_quotes_the_scaffold_figure_not_the_random_one(self):
        from atlas.ml.predict import MODEL_CARD
        assert "scaffold" in MODEL_CARD["headline_performance"]["split"].lower()
        assert (MODEL_CARD["headline_performance"]["mae_log_units"]
                > MODEL_CARD["random_split_comparison"]["mae_log_units"])

    def test_model_card_is_not_claimed_externally_validated(self):
        from atlas.ml.predict import MODEL_CARD
        assert MODEL_CARD["quality_tier"] == "internally_validated"
        assert "not externally validated" in MODEL_CARD["quality_tier_meaning"]
