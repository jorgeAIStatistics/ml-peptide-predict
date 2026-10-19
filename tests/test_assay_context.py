"""Assay context: variation, vocabulary hygiene, and what is missing.

A pKi is the result of an experiment, not a property of a molecule alone, so
assay context is a covariate. These tests assert that the encoder reports its
own limits: fields that do not vary, conditions the source does not carry, and
the series-proxy risk the benchmark had to ablate.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from atlas.ml.assay_context import (
    ASSAY_CONTEXT_PATH, ASSAY_FIELDS, MISSING_CONDITIONS, assay_context_audit,
    assay_encoding, build_assay_vocabularies, load_assay_metadata,
)


class _Rec:
    def __init__(self, mol, assays, docs=("D1",)):
        self.molecule_chembl_id = mol
        self.assay_ids = tuple(assays)
        self.document_ids = tuple(docs)


_META = {
    "A1": {"bao_label": "cell-based format", "assay_cell_type": "HEK293",
           "confidence_score": 9, "relationship_type": "D", "assay_type": "B"},
    "A2": {"bao_label": "single protein format", "assay_cell_type": None,
           "confidence_score": 8, "relationship_type": "H", "assay_type": "B"},
    "A3": {"bao_label": "cell membrane format", "assay_cell_type": "CHO",
           "confidence_score": 9, "relationship_type": "D", "assay_type": "B"},
}


@pytest.fixture(scope="module")
def saved() -> dict:
    if not ASSAY_CONTEXT_PATH.exists():
        pytest.skip(f"{ASSAY_CONTEXT_PATH.name} not built; run atlas.ml.assay_context")
    return load_assay_metadata()


# ----------------------------------------------------------------- audit ----

def test_audit_names_fields_that_do_not_vary(saved):
    """assay_type is constant because the dataset filters to binding assays.

    A constant column is reported, not silently encoded.
    """
    audit = saved["audit"]
    assert "assay_type" in audit["non_varying_fields"]
    assert audit["field_variation"]["assay_type"]["n_distinct"] == 1


def test_audit_finds_real_variation_in_assay_format(saved):
    v = saved["audit"]["field_variation"]["bao_label"]
    assert v["n_distinct"] >= 3, v
    assert "cell-based format" in v["counts"]


def test_molecules_span_several_formats(saved):
    """The fact that makes assay context a covariate and not only a series tag.

    The same molecule measured on whole cells and on purified protein gives
    within-molecule variation that series membership cannot explain.
    """
    assert saved["audit"]["molecules_spanning_several_formats"] > 0


def test_absent_conditions_are_recorded_as_data(saved):
    """pH, temperature and ionic strength are not in this source."""
    missing = saved["audit"]["missing_conditions"]
    for key in ("ph", "temperature_c", "ionic_strength_mm"):
        assert key in missing
    assert set(MISSING_CONDITIONS) <= set(missing)


def test_audit_records_the_series_proxy_caveat(saved):
    caveat = saved["audit"]["caveat"].lower()
    assert "series" in caveat and "ablates" in caveat


def test_every_referenced_assay_was_resolved(saved):
    a = saved["audit"]
    assert a["n_assays_resolved"] == a["n_assays_referenced"]


def test_audit_on_constructed_records():
    recs = [_Rec("M1", ["A1", "A2"]), _Rec("M2", ["A1"])]
    audit = assay_context_audit(recs, _META)
    assert audit["n_assays_referenced"] == 2
    assert audit["assays_per_molecule"]["max"] == 2
    assert audit["molecules_spanning_several_formats"] == 1


# ------------------------------------------------------------ vocabulary ----

def test_vocabulary_is_fitted_on_training_records_only():
    """Letting the test set define the feature space is leakage even with no
    label touched."""
    train = [_Rec("M1", ["A1"])]
    vocab = build_assay_vocabularies(train, _META)
    assert vocab["bao_label"] == ["cell-based format"]
    assert "single protein format" not in vocab["bao_label"]


def test_unknown_assay_contributes_nothing_rather_than_a_guess():
    recs = [_Rec("M1", ["NOT-IN-METADATA"])]
    vocab = build_assay_vocabularies([_Rec("M0", ["A1"])], _META)
    enc = assay_encoding(recs, _META, vocab)
    # Only the "no context available" flag is set.
    assert enc[0, -1] == 1.0
    assert enc[0, :-1].sum() == 0.0


# -------------------------------------------------------------- encoding ----

def test_encoding_is_a_distribution_over_a_molecule_s_assays():
    """A molecule in two assay formats is half in each, not labelled with one."""
    vocab = build_assay_vocabularies(
        [_Rec("M0", ["A1", "A2", "A3"])], _META)
    enc = assay_encoding([_Rec("M1", ["A1", "A2"])], _META, vocab)
    fmt = enc[0, :len(vocab["bao_label"])]
    assert fmt.sum() == pytest.approx(1.0)
    assert sorted(fmt[fmt > 0].tolist()) == pytest.approx([0.5, 0.5])


def test_encoding_width_and_finiteness():
    vocab = build_assay_vocabularies([_Rec("M0", ["A1", "A2", "A3"])], _META)
    enc = assay_encoding(
        [_Rec("M1", ["A1"]), _Rec("M2", ["A2", "A3"])], _META, vocab)
    assert enc.shape == (2, len(vocab["bao_label"]) + len(vocab["assay_cell_type"]) + 5)
    assert np.isfinite(enc).all()


def test_confidence_is_scaled_and_relationship_is_a_fraction():
    vocab = build_assay_vocabularies([_Rec("M0", ["A1", "A2", "A3"])], _META)
    base = len(vocab["bao_label"]) + len(vocab["assay_cell_type"])
    direct = assay_encoding([_Rec("M1", ["A1"])], _META, vocab)
    homologous = assay_encoding([_Rec("M2", ["A2"])], _META, vocab)
    assert direct[0, base] == pytest.approx(1.0)          # confidence 9 / 9
    assert homologous[0, base] == pytest.approx(8 / 9)
    assert direct[0, base + 1] == 1.0                     # relationship D
    assert homologous[0, base + 1] == 0.0                 # relationship H


def test_all_declared_fields_are_captured(saved):
    for assay in list(saved["metadata"].values())[:5]:
        for field in ASSAY_FIELDS:
            assert field in assay


# ------------------------------------------------- measured benchmark result ----

def test_assay_context_is_recorded_as_harmful_cross_series():
    """The measured outcome, pinned so it is not quietly restated as a gain.

    Alone, assay context scores far worse than the mean cross-series, because
    assay identity points at training series when whole series are held out.
    """
    from atlas.ml.peptide_benchmark import BENCHMARK_PATH

    if not BENCHMARK_PATH.exists():
        pytest.skip("benchmark not built")
    report = json.loads(BENCHMARK_PATH.read_text())
    cross = report["cross_series"]["results"]
    within = report["within_series"]["results"]
    if "assay_context__forest" not in cross:
        pytest.skip("assay context arm not present")
    assert cross["assay_context__forest"]["r2"] < 0
    assert within["assay_context__forest"]["r2"] > cross["assay_context__forest"]["r2"]
    contrast = report["contrasts"]["cross_series"]["assay_context_added_to_fragment_fp"]
    assert contrast["a_better"] is False
