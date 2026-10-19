"""Tests for uncertainty calibration and target-relevance gating.

The two behaviours pinned here are the ones the milestone turned on: intervals
must mean what they claim, and a number must not be published for a compound
with no established relationship to the target.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from atlas.ml.calibration import (
    MIN_BAND_CALIBRATION,
    SIMILARITY_BANDS,
    CalibrationReport,
    ConformalCalibrator,
    band_of,
    evaluate_calibration,
    four_way_scaffold_split,
)
from atlas.ml.features import Modality
from atlas.ml.target_relevance import (
    FAMILY_SIMILARITY_FLOOR,
    MIN_TARGET_FRACTION,
    MIN_TARGET_PAPERS,
    TargetRelevance,
    TargetRelevanceAssessor,
    literature_names,
)

CAL_REPORT = (
    Path(__file__).resolve().parents[1] / "data" / "ml" / "calibration_report.json"
)
PEPV2_REPORT = (
    Path(__file__).resolve().parents[1] / "data" / "ml" / "peptide_v2_report.json"
)


class StubPubMed:
    """Returns fixed counts so literature logic is testable offline."""

    def __init__(self, counts: dict[str, tuple[int, int]]):
        self.counts = counts
        self.calls: list[str] = []

    def search(self, query, retmax=1, mindate=None):
        self.calls.append(query)
        for name, (total, target) in self.counts.items():
            if f'"{name}"' in query:
                # The target-filtered query carries an AND clause.
                return [], (target if " AND " in query else total), None
        return [], 0, None


class TestConformalQuantile:
    def test_quantile_is_finite_sample_corrected(self):
        """The conformal level is ceil((n+1)(1-a))/n, above the plain quantile."""
        resid = np.arange(1, 101, dtype=float)
        cal = ConformalCalibrator(alpha=0.10).fit(resid)
        assert cal.global_quantile >= float(np.quantile(resid, 0.90))

    def test_empty_calibration_gives_nan(self):
        cal = ConformalCalibrator().fit(np.array([]))
        assert np.isnan(cal.global_quantile)

    def test_coverage_is_approximately_nominal_when_exchangeable(self):
        """Sanity check of the machinery itself on exchangeable data."""
        rng = np.random.default_rng(0)
        cal_resid = np.abs(rng.normal(0, 1, 2000))
        cal = ConformalCalibrator(alpha=0.10).fit(cal_resid)
        test_resid = np.abs(rng.normal(0, 1, 4000))
        coverage = float(np.mean(test_resid <= cal.global_quantile))
        assert 0.87 <= coverage <= 0.93

    def test_tighter_alpha_widens_the_interval(self):
        resid = np.abs(np.random.default_rng(1).normal(0, 1, 1000))
        wide = ConformalCalibrator(alpha=0.01).fit(resid).global_quantile
        narrow = ConformalCalibrator(alpha=0.20).fit(resid).global_quantile
        assert wide > narrow


class TestMondrianCalibration:
    def _fit(self, n=400):
        rng = np.random.default_rng(0)
        sims = rng.uniform(0, 1, n)
        # Error grows as similarity falls, which is the effect bands capture.
        resid = np.abs(rng.normal(0, 1, n)) * (1.5 - sims)
        return ConformalCalibrator(alpha=0.10, mondrian=True).fit(resid, sims), sims

    def test_band_quantiles_are_learned(self):
        cal, _ = self._fit()
        assert len(cal.band_quantiles) == len(SIMILARITY_BANDS)

    def test_low_similarity_band_gets_a_wider_interval(self):
        cal, _ = self._fit()
        assert cal.half_width(0.1) > cal.half_width(0.9)

    def test_sparse_band_falls_back_and_says_so(self):
        sims = np.concatenate([np.full(200, 0.9), np.full(3, 0.1)])
        resid = np.abs(np.random.default_rng(0).normal(0, 1, len(sims)))
        cal = ConformalCalibrator(alpha=0.10, mondrian=True).fit(resid, sims)
        assert "0.0–0.4" in cal.bands_using_fallback
        assert cal.half_width(0.1) == cal.global_quantile

    def test_mondrian_requires_similarities(self):
        with pytest.raises(ValueError, match="requires similarities"):
            ConformalCalibrator(mondrian=True).fit(np.array([1.0, 2.0]))

    def test_band_assignment_covers_the_range(self):
        for s in (0.0, 0.39, 0.4, 0.69, 0.7, 1.0):
            assert band_of(s) in {b[2] for b in SIMILARITY_BANDS}


class TestFourWaySplit:
    def _fs(self, n=200):
        from atlas.ml.dataset import ActivityRecord
        from atlas.ml.features import featurize
        smis = ["c1ccccc1C(=O)O", "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
                "CC(=O)Oc1ccccc1C(=O)O", "c1ccc2ccccc2c1", "CCCCO"]
        recs = [ActivityRecord(f"M{i}", smis[i % len(smis)], 5.0 + (i % 5),
                               5.0, 5.0, 1, 0.0, None, ("D",), ("A",))
                for i in range(n)]
        return featurize(recs)

    def test_roles_are_strictly_disjoint(self):
        split = four_way_scaffold_split(self._fs(), seed=0)
        total = sum(split.sizes.values())
        assert total == 200  # every molecule has exactly one role

    def test_overlapping_roles_are_rejected(self):
        from atlas.ml.calibration import FourWaySplit
        with pytest.raises(ValueError, match="strictly separate"):
            FourWaySplit(np.array([0, 1]), np.array([1, 2]), np.array([3]),
                         np.array([4]), "bad", 0)

    def test_fractions_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1"):
            four_way_scaffold_split(self._fs(), fractions=(0.5, 0.5, 0.5, 0.5))

    def test_no_scaffold_spans_two_roles(self):
        fs = self._fs()
        split = four_way_scaffold_split(fs, seed=0)
        roles = {
            "train": {fs.scaffolds[i] for i in split.train_idx},
            "select": {fs.scaffolds[i] for i in split.select_idx},
            "calib": {fs.scaffolds[i] for i in split.calib_idx},
            "test": {fs.scaffolds[i] for i in split.test_idx},
        }
        names = list(roles)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                assert not (roles[a] & roles[b]), f"{a}/{b} share a scaffold"


class TestCalibrationReport:
    def _report(self, coverage_points):
        y = np.zeros(len(coverage_points))
        pred = np.array(coverage_points, dtype=float)
        cal = ConformalCalibrator(alpha=0.10).fit(np.array([1.0] * 100))
        return evaluate_calibration(
            strategy="t", y_true=y, y_pred=pred, calibrator=cal,
            similarities=np.full(len(y), 0.8),
            modalities=[Modality.SMALL_MOLECULE_LIKE] * len(y),
            calibration_set_size=100,
        )

    def test_coverage_gap_sign(self):
        r = self._report([0.0] * 90 + [5.0] * 10)
        assert r.empirical_coverage == pytest.approx(0.90, abs=0.01)
        assert abs(r.coverage_gap) < 0.02

    def test_undercoverage_is_not_called_adequate(self):
        r = self._report([0.0] * 50 + [5.0] * 50)
        assert not r.is_adequately_calibrated
        assert r.coverage_gap < 0

    def test_overcoverage_is_also_flagged(self):
        """Intervals far wider than needed are miscalibrated too."""
        r = self._report([0.0] * 100)
        assert r.empirical_coverage == 1.0
        assert not r.is_adequately_calibrated

    def test_limitations_always_mention_exchangeability(self):
        r = self._report([0.0] * 90 + [5.0] * 10)
        assert any("exchangeab" in l for l in r.limitations)

    def test_subgroups_are_reported(self):
        r = self._report([0.0] * 100)
        assert set(r.coverage_by_similarity) == {b[2] for b in SIMILARITY_BANDS}
        assert "peptide_like" in r.coverage_by_modality


@pytest.mark.skipif(not CAL_REPORT.exists(), reason="calibration report not built")
class TestMeasuredCalibration:
    @pytest.fixture(scope="class")
    @classmethod
    def report(cls):
        return json.loads(CAL_REPORT.read_text())

    def test_roles_are_recorded_and_disjoint_by_construction(self, report):
        sizes = report["split"]["sizes"]
        assert set(sizes) == {"train", "select", "calibrate", "test"}
        assert all(v > 0 for v in sizes.values())

    def test_model_was_selected_without_the_test_split(self, report):
        assert "never the test" in report["model_selection"]["selected_on"]

    def test_disjoint_calibration_beats_random_calibration(self, report):
        by = {r["strategy"]: r for r in report["reports"]}
        random = next(v for k, v in by.items() if k.startswith("random_calibration"))
        disjoint = by["disjoint_calibration"]
        assert abs(disjoint["empirical_coverage"] - 0.90) < abs(
            random["empirical_coverage"] - 0.90)

    def test_disjoint_calibration_reaches_nominal(self, report):
        disjoint = next(r for r in report["reports"]
                        if r["strategy"] == "disjoint_calibration")
        assert abs(disjoint["empirical_coverage"] - 0.90) <= 0.03

    def test_previous_approach_undercovers(self, report):
        random = next(r for r in report["reports"]
                      if r["strategy"].startswith("random_calibration"))
        assert random["empirical_coverage"] < 0.87

    def test_correct_coverage_costs_interval_width(self, report):
        by = {r["strategy"]: r for r in report["reports"]}
        random = next(v for k, v in by.items() if k.startswith("random_calibration"))
        assert by["disjoint_calibration"]["median_interval_width"] > random[
            "median_interval_width"]

    def test_subgroup_coverage_is_reported_for_every_strategy(self, report):
        for r in report["reports"]:
            assert r["coverage_by_similarity"]
            assert r["coverage_by_modality"]


class TestLiteratureNames:
    def test_ambiguous_short_alias_dropped(self):
        """MT-2 retrieves metallothionein-2 and must not widen the query."""
        assert "MT-2" not in literature_names(["melanotan II", "MT-2"])

    def test_primary_name_always_kept(self):
        assert literature_names(["KPV"]) == ["KPV"]

    def test_informative_aliases_kept(self):
        out = literature_names(["PT-141", "bremelanotide", "Vyleesi"])
        assert "bremelanotide" in out and "Vyleesi" in out

    def test_empty_input(self):
        assert literature_names([]) == []


class TestTargetRelevance:
    def _assessor(self, counts=None):
        return TargetRelevanceAssessor(pubmed=StubPubMed(counts or {}))

    def test_measurement_is_class_a(self):
        ev = self._assessor().assess(measured_pki=7.5, measurement_source="CHEMBL1")
        assert ev.relevance is TargetRelevance.A_DOCUMENTED_ENGAGEMENT
        assert ev.may_publish_affinity and ev.may_create_graph_edge

    def test_out_of_domain_is_class_d(self):
        ev = self._assessor().assess(in_applicability_domain=False,
                                     max_similarity_to_actives=0.1)
        assert ev.relevance is TargetRelevance.D_UNSUPPORTED
        assert not ev.may_publish_affinity

    def test_distant_analogue_in_domain_is_class_c(self):
        """Pinealon: inside the domain, 0.375 from any documented active."""
        ev = self._assessor({"Pinealon": (18, 0)}).assess(
            names=["Pinealon"], in_applicability_domain=True,
            max_similarity_to_actives=0.375, nearest_active_id="CHEMBL1")
        assert ev.relevance is TargetRelevance.C_EXPLORATORY_SCREENING_HYPOTHESIS
        assert not ev.may_publish_affinity
        assert not ev.may_create_graph_edge

    def test_class_c_explains_itself(self):
        ev = self._assessor({"X": (18, 0)}).assess(
            names=["X"], in_applicability_domain=True,
            max_similarity_to_actives=0.3)
        assert "no affinity is reported" in ev.rationale
        assert any("knowledge graph" in c for c in ev.caveats)

    def test_close_analogue_with_supporting_literature_is_class_b(self):
        ev = self._assessor({"Y": (100, 30)}).assess(
            names=["Y"], in_applicability_domain=True,
            max_similarity_to_actives=0.85, nearest_active_id="CHEMBL2",
            nearest_active_pki=9.0)
        assert ev.relevance is TargetRelevance.B_VALIDATED_DOMAIN_PREDICTION
        assert ev.may_publish_affinity
        assert not ev.may_create_graph_edge

    def test_literature_can_veto_a_close_analogue(self):
        """Structural resemblance uncorroborated by literature drops to C."""
        ev = self._assessor({"Z": (200, 0)}).assess(
            names=["Z"], in_applicability_domain=True,
            max_similarity_to_actives=0.85, nearest_active_id="CHEMBL3")
        assert ev.relevance is TargetRelevance.C_EXPLORATORY_SCREENING_HYPOTHESIS
        assert "not corroborated" in ev.rationale

    def test_absent_literature_does_not_block_a_close_analogue(self):
        """A bare SMILES has no name to search; structure alone can suffice."""
        ev = self._assessor().assess(
            names=[], in_applicability_domain=True,
            max_similarity_to_actives=0.9, nearest_active_id="CHEMBL4")
        assert ev.relevance is TargetRelevance.B_VALIDATED_DOMAIN_PREDICTION

    def test_only_a_measurement_may_make_a_graph_edge(self):
        for rel in TargetRelevance:
            assert rel.may_create_graph_edge == (
                rel is TargetRelevance.A_DOCUMENTED_ENGAGEMENT)

    def test_thresholds_require_both_count_and_fraction(self):
        # 4 papers: below the count floor even at a high fraction.
        ev = self._assessor({"W": (6, 4)}).assess(
            names=["W"], in_applicability_domain=True,
            max_similarity_to_actives=0.9)
        assert ev.relevance is TargetRelevance.C_EXPLORATORY_SCREENING_HYPOTHESIS
        assert MIN_TARGET_PAPERS == 5 and MIN_TARGET_FRACTION == 0.05

    def test_family_floor_is_documented(self):
        assert 0.5 < FAMILY_SIMILARITY_FLOOR < 0.9


@pytest.mark.skipif(not PEPV2_REPORT.exists(), reason="peptide v2 report not built")
class TestEndpointSeparation:
    @pytest.fixture(scope="class")
    @classmethod
    def report(cls):
        return json.loads(PEPV2_REPORT.read_text())

    def test_endpoints_are_defined_as_distinct_quantities(self, report):
        defs = report["endpoint_definitions"]
        assert defs["ki_binding"]["quantity"] != defs["ec50_functional"]["quantity"]

    def test_endpoint_difference_justifies_separation(self, report):
        rel = report["endpoint_relation"]
        # Correlated, but with enough spread that pooling would add real noise.
        assert 0.4 < rel["pearson"] < 0.9
        assert rel["sd_of_difference"] > 0.5

    def test_peptide_descriptors_do_not_beat_fingerprints(self, report):
        """Recorded negative result; the documentation depends on it."""
        for ep in ("ki_binding", "ec50_functional"):
            rows = [r for r in report["results"]
                    if r["endpoint"] == ep and r["model"] != "baseline_mean"]
            pep = min(r["mae"] for r in rows
                      if r["representation"] == "peptide_descriptors")
            fp = min(r["mae"] for r in rows
                     if r["representation"] == "fingerprints")
            assert fp <= pep, f"{ep}: peptide descriptors now beat fingerprints"

    def test_multitask_helps_ki_binding(self, report):
        rows = [r for r in report["results"] if r["endpoint"] == "ki_binding"]
        multi = min(r["mae"] for r in rows
                    if r["representation"] == "multitask_fingerprints")
        single = min(r["mae"] for r in rows
                     if r["representation"] == "fingerprints"
                     and r["model"] != "baseline_mean")
        assert multi < single

    def test_every_result_reports_test_similarity(self, report):
        assert all(r["median_test_similarity"] is not None for r in report["results"])
