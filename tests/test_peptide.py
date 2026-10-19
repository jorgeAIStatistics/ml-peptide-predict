"""Tests for peptide featurisation, cluster splitting and measurement lookup.

The central behaviour pinned here is the precedence rule: a molecule with a
measured affinity must return the measurement, not a prediction. PT-141,
melanotan I and melanotan II all have measured MC4R values in ChEMBL, and
quietly replacing them with model output would substitute inference for
evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from atlas.ml.peptide import (
    HELM_VALIDATION,
    MACROCYCLE_MIN_RING,
    RESIDUE_SMARTS,
    compute_peptide_features,
    parse_helm_residues,
    peptide_feature_names,
    peptide_matrix,
    validate_against_helm,
)
from atlas.ml.splits import cluster_split, leakage_report

DATASET = Path(__file__).resolve().parents[1] / "data" / "ml" / "mc4r_pki.json"
PEPTIDE_REGISTRY = (
    Path(__file__).resolve().parents[1] / "data" / "ml" / "peptide_model_registry.json"
)
SNAPSHOT = Path(__file__).resolve().parents[1] / "data" / "registry_snapshot.json"

ARG_PEPTIDE = "NC(=N)NCCC[C@H](N)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O"   # Arg-Trp
TYR_HIS = "N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](Cc1cnc[nH]1)C(=O)O"          # Tyr-His
CYCLIC = "C1CCNC(=O)CCCNC(=O)CCCNC(=O)CC1"                                 # macrocycle
ACETYL_AMIDE = "CC(=O)N[C@@H](Cc1ccccc1)C(=O)N"                            # Ac-Phe-NH2


def names_index():
    return {n: i for i, n in enumerate(peptide_feature_names())}


class TestResidueExtraction:
    def test_arginine_guanidine_detected(self):
        idx = names_index()
        f = compute_peptide_features(Chem.MolFromSmiles(ARG_PEPTIDE))
        assert f[idx["res_arg_guanidine"]] == 1

    def test_tryptophan_indole_detected(self):
        idx = names_index()
        f = compute_peptide_features(Chem.MolFromSmiles(ARG_PEPTIDE))
        assert f[idx["res_trp_indole"]] == 1

    def test_tyrosine_and_histidine_detected(self):
        idx = names_index()
        f = compute_peptide_features(Chem.MolFromSmiles(TYR_HIS))
        assert f[idx["res_tyr_phenol"]] == 1
        assert f[idx["res_his_imidazole"]] == 1

    def test_absent_residues_score_zero(self):
        idx = names_index()
        f = compute_peptide_features(Chem.MolFromSmiles(TYR_HIS))
        assert f[idx["res_cys_thiol"]] == 0
        assert f[idx["res_disulfide"]] == 0

    def test_every_residue_pattern_compiles(self):
        for name, smarts in RESIDUE_SMARTS.items():
            assert Chem.MolFromSmarts(smarts) is not None, name

    def test_feature_vector_length_matches_names(self):
        f = compute_peptide_features(Chem.MolFromSmiles(ARG_PEPTIDE))
        assert len(f) == len(peptide_feature_names())

    def test_matrix_stacks_rows(self):
        mols = [Chem.MolFromSmiles(s) for s in (ARG_PEPTIDE, TYR_HIS, CYCLIC)]
        assert peptide_matrix(mols).shape == (3, len(peptide_feature_names()))

    def test_no_nan_in_features(self):
        for smi in (ARG_PEPTIDE, TYR_HIS, CYCLIC, ACETYL_AMIDE):
            assert not np.isnan(compute_peptide_features(Chem.MolFromSmiles(smi))).any()


class TestTopologyFeatures:
    def test_macrocycle_detected(self):
        idx = names_index()
        f = compute_peptide_features(Chem.MolFromSmiles(CYCLIC))
        assert f[idx["is_macrocyclic"]] == 1.0
        assert f[idx["max_ring_size"]] >= MACROCYCLE_MIN_RING

    def test_linear_peptide_is_not_macrocyclic(self):
        idx = names_index()
        f = compute_peptide_features(Chem.MolFromSmiles(ARG_PEPTIDE))
        assert f[idx["is_macrocyclic"]] == 0.0

    def test_terminal_modifications_detected(self):
        idx = names_index()
        f = compute_peptide_features(Chem.MolFromSmiles(ACETYL_AMIDE))
        assert f[idx["has_n_terminal_acetyl"]] == 1.0
        assert f[idx["has_c_terminal_amide"]] == 1.0

    def test_free_acid_counted_on_unmodified_terminus(self):
        idx = names_index()
        f = compute_peptide_features(Chem.MolFromSmiles(TYR_HIS))
        assert f[idx["n_free_acids"]] >= 1

    def test_residue_count_is_positive(self):
        idx = names_index()
        for smi in (ARG_PEPTIDE, TYR_HIS, CYCLIC):
            assert compute_peptide_features(Chem.MolFromSmiles(smi))[idx["est_residue_count"]] >= 1

    def test_per_residue_normalisation_is_finite(self):
        idx = names_index()
        f = compute_peptide_features(Chem.MolFromSmiles(ARG_PEPTIDE))
        assert np.isfinite(f[idx["mw_per_residue"]])
        assert f[idx["mw_per_residue"]] > 0


class TestHelmParsing:
    def test_simple_sequence(self):
        assert parse_helm_residues("PEPTIDE1{H.F.R.W}$$$$") == ["H", "F", "R", "W"]

    def test_terminal_caps_are_dropped(self):
        assert parse_helm_residues("PEPTIDE1{[ac].H.R.[am]}$$$$") == ["H", "R"]

    def test_d_amino_acids_map_to_their_side_chain(self):
        """[dF] is D-phenylalanine: a different centre, the same side chain."""
        assert parse_helm_residues("PEPTIDE1{[dF].[dW]}$$$$") == ["F", "W"]

    def test_non_canonical_residues_are_preserved(self):
        assert "Nle" in parse_helm_residues("PEPTIDE1{Y.V.[Nle].G}$$$$")

    def test_multiple_blocks_are_concatenated(self):
        out = parse_helm_residues("PEPTIDE1{H.R}|PEPTIDE2{[ac].W}$x$$$")
        assert out == ["H", "R", "W"]

    def test_empty_notation_is_empty(self):
        assert parse_helm_residues("") == []


class TestHelmValidation:
    def test_recorded_accuracy_is_high_and_honest(self):
        assert 0.90 <= HELM_VALIDATION["accuracy"] <= 1.0
        assert HELM_VALIDATION["n_exact_agreements"] <= HELM_VALIDATION["n_residue_type_comparisons"]

    def test_validation_runs_on_synthetic_ground_truth(self):
        pairs = [("M1", ARG_PEPTIDE), ("M2", TYR_HIS)]
        helm = {"M1": "PEPTIDE1{R.W}$$$$", "M2": "PEPTIDE1{Y.H}$$$$"}
        out = validate_against_helm(pairs, helm)
        assert out["n_molecules"] == 2
        assert out["accuracy"] == 1.0

    def test_validation_detects_disagreement(self):
        """A wrong sequence must lower the score, not be silently accepted."""
        pairs = [("M1", ARG_PEPTIDE)]
        out = validate_against_helm(pairs, {"M1": "PEPTIDE1{C.C.C}$$$$"})
        assert out["accuracy"] < 1.0

    def test_molecules_without_helm_are_skipped(self):
        out = validate_against_helm([("M1", ARG_PEPTIDE)], {})
        assert out["n_molecules"] == 0


class TestClusterSplit:
    def _fps(self, n=60):
        from atlas.ml.features import morgan_matrix
        smis = [ARG_PEPTIDE, TYR_HIS, CYCLIC, ACETYL_AMIDE]
        mols = [Chem.MolFromSmiles(smis[i % len(smis)]) for i in range(n)]
        return morgan_matrix(mols)

    def test_split_partitions_without_overlap(self):
        fps = self._fps()
        s = cluster_split(fps, test_fraction=0.25, seed=0)
        assert s.n_train + s.n_test == len(fps)
        assert not (set(s.train_idx.tolist()) & set(s.test_idx.tolist()))

    def test_split_records_cluster_count(self):
        s = cluster_split(self._fps(), test_fraction=0.25, seed=0)
        assert "clusters" in s.description

    def test_tighter_cutoff_makes_more_clusters(self):
        fps = self._fps()
        loose = cluster_split(fps, distance_cutoff=0.8, seed=0)
        tight = cluster_split(fps, distance_cutoff=0.1, seed=0)
        assert tight.description != loose.description


@pytest.mark.skipif(not DATASET.exists(), reason="dataset not built")
class TestMeasurementPrecedence:
    @pytest.fixture(scope="class")
    @classmethod
    def svc(cls):
        from atlas.ml.peptide_service import get_affinity_service
        return get_affinity_service()

    @pytest.fixture(scope="class")
    @classmethod
    def registry_smiles(cls):
        snap = json.loads(SNAPSHOT.read_text())
        out = {}
        for i in snap["identities"]:
            sm = (i.get("smiles") or {}).get("value")
            if sm:
                out[i["registry_key"]] = sm
        return out

    @pytest.mark.parametrize("key", ["pt-141", "melanotan-1", "melanotan-2"])
    def test_melanocortin_peptides_return_measurements(self, svc, registry_smiles, key):
        a = svc.lookup(registry_smiles[key])
        assert a.is_measured, f"{key} should resolve to a measured value"
        assert a.evidence_class.startswith("A_")
        assert a.match_kind == "exact_structure"
        assert a.molecule_chembl_id

    def test_measurements_are_in_a_plausible_affinity_range(self, svc, registry_smiles):
        for key in ("pt-141", "melanotan-1", "melanotan-2"):
            a = svc.lookup(registry_smiles[key])
            assert 4.0 <= a.pki <= 12.0
            assert a.ki_nm > 0

    def test_measurement_cites_its_documents(self, svc, registry_smiles):
        a = svc.lookup(registry_smiles["melanotan-1"])
        assert a.document_ids
        assert a.n_replicates >= 1

    def test_unmeasured_peptide_abstains_with_numbers(self, svc):
        long_pep = (
            "CC(C)C[C@H](N)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CO)C(=O)"
            "N[C@@H](CCCCN)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@@H](CC(=O)O)C(=O)O"
        )
        a = svc.lookup(long_pep)
        assert a.abstained
        assert "mean-predictor baseline" in a.abstention_explanation
        assert a.what_would_resolve_it

    def test_unparseable_input_abstains(self, svc):
        a = svc.lookup("not-a-molecule")
        assert a.abstained
        assert a.abstention_reason == "unresolved_molecular_identity"

    def test_findings_record_that_the_model_is_not_deployed(self):
        from atlas.ml.peptide_service import PEPTIDE_MODEL_FINDINGS
        assert PEPTIDE_MODEL_FINDINGS["deployed"] is False
        assert PEPTIDE_MODEL_FINDINGS["cluster_split"]["r2"] < 0.2

    def test_findings_record_the_negative_feature_result(self):
        from atlas.ml.peptide_service import PEPTIDE_MODEL_FINDINGS
        note = PEPTIDE_MODEL_FINDINGS["peptide_features_did_not_help"]["note"]
        assert "fingerprints beat them" in note


@pytest.mark.skipif(not PEPTIDE_REGISTRY.exists(), reason="peptide registry not built")
class TestPeptideRegistry:
    @pytest.fixture(scope="class")
    @classmethod
    def reg(cls):
        return json.loads(PEPTIDE_REGISTRY.read_text())

    def test_registry_is_strict_json(self):
        def reject(c):
            raise ValueError(f"bare {c}")
        json.loads(PEPTIDE_REGISTRY.read_text(), parse_constant=reject)

    def test_scaffold_split_looks_clean_but_leaks_by_similarity(self, reg):
        """The finding that justifies using a cluster split for peptides."""
        scaf = next(s for s in reg["splits"] if s["kind"] == "scaffold")
        assert scaf["leakage"]["n_shared_scaffolds"] == 0
        assert scaf["fraction_test_with_close_neighbour"] > 0.9

    def test_cluster_split_is_genuinely_harder(self, reg):
        scaf = next(s for s in reg["splits"] if s["kind"] == "scaffold")
        clus = next(s for s in reg["splits"] if s["kind"] == "cluster")
        assert clus["median_test_similarity"] < scaf["median_test_similarity"]
        assert (clus["fraction_test_with_close_neighbour"]
                < scaf["fraction_test_with_close_neighbour"])

    def test_peptide_model_beats_the_small_molecule_model(self, reg):
        rows = [r for r in reg["results"] if r["split"] == "cluster"]
        ref = next(r for r in rows if r["model"] == "small_molecule_model_applied")
        best = min((r for r in rows if r["model"] not in
                    ("baseline_mean", "small_molecule_model_applied")),
                   key=lambda x: x["mae"])
        assert best["mae"] < ref["mae"]

    def test_peptide_features_do_not_beat_fingerprints(self, reg):
        """A negative result the documentation depends on."""
        rows = [r for r in reg["results"] if r["split"] == "cluster"
                and r["model"] not in ("baseline_mean", "small_molecule_model_applied")]
        best_pep = min((r for r in rows if r["features"] == "peptide"),
                       key=lambda x: x["mae"])
        best_fp = min((r for r in rows if r["features"] == "fingerprints"),
                      key=lambda x: x["mae"])
        assert best_fp["mae"] <= best_pep["mae"]

    def test_cluster_split_error_exceeds_optimistic_splits(self, reg):
        best = {}
        for r in reg["results"]:
            if r["model"] in ("baseline_mean", "small_molecule_model_applied"):
                continue
            best[r["split"]] = min(best.get(r["split"], 9e9), r["mae"])
        assert best["cluster"] > best["scaffold"] > 0
        assert best["cluster"] > best["random"]


@pytest.mark.skipif(not DATASET.exists(), reason="dataset not built")
class TestAPIMeasurementFlow:
    @pytest.fixture(scope="class")
    @classmethod
    def client(cls):
        from atlas.api.app import create_app
        app = create_app()
        app.config.update(TESTING=True)
        return app.test_client()

    @pytest.mark.parametrize("key", ["pt-141", "melanotan-1", "melanotan-2"])
    def test_registry_peptides_return_measurements(self, client, key):
        r = client.post("/api/predict", json={"compound_key": key}).get_json()
        assert r["answer_kind"] == "measurement"
        assert r["evidence_class"].startswith("A_")
        assert r["measurement"]["pki"] > 0
        assert r["measurement"]["chembl_url"].startswith("https://")

    def test_measurement_reports_study_count(self, client):
        r = client.post("/api/predict", json={"compound_key": "melanotan-1"}).get_json()
        assert r["measurement"]["n_documents"] >= 1

    def test_remote_small_molecule_is_refused_on_target_relevance(self, client):
        """A fingerprint can be computed for anything; relevance must be shown."""
        smi = "CC(C)(C)NC(=O)[C@@H]1CCCN1C(=O)[C@H](Cc1ccccc1)NC(=O)c1ccccc1"
        r = client.post("/api/predict", json={"smiles": smi}).get_json()
        assert r["answer_kind"] == "abstention"
        assert r["target_relevance"].startswith("C_")
        assert r["may_create_graph_edge"] is False

    def test_close_analogue_of_a_documented_active_is_predicted(self, client):
        """Structural evidence alone can reach class B when no name is given."""
        from atlas.ml.dataset import load_dataset
        from rdkit import Chem
        rec = next(r for r in load_dataset().records if r.pki >= 8.0)
        flat = Chem.MolToSmiles(Chem.MolFromSmiles(rec.smiles), isomericSmiles=False)
        r = client.post("/api/predict", json={"smiles": flat}).get_json()
        assert r["answer_kind"] == "prediction"
        assert r["target_relevance"].startswith("B_")
        assert r["may_create_graph_edge"] is False
        p = r["prediction"]
        assert p["interval_low"] < p["predicted_pki"] < p["interval_high"]

    def test_unmeasured_peptide_abstains(self, client):
        r = client.post("/api/predict", json={"compound_key": "semaglutide"}).get_json()
        assert r["answer_kind"] == "abstention"
        assert r["target_relevance"][0] in "CD"
        assert r["may_create_graph_edge"] is False

    def test_mixture_abstains_on_identity(self, client):
        r = client.post("/api/predict", json={"compound_key": "glow"}).get_json()
        assert r["answer_kind"] == "abstention"
        assert r["abstention"]["reason"] == "unresolved_molecular_identity"

    def test_model_card_exposes_peptide_findings(self, client):
        r = client.get("/api/model-card").get_json()
        assert r["peptide_findings"]["deployed"] is False
        assert r["peptide_report"]["available"] is True
