"""ATLAS 2.6: the properties the language-model arm's conclusion rests on.

A null result is only worth reporting if the arm it came from was wired
correctly. These tests assert the three things that could make the null an
artefact instead of a measurement: that the embedding is a per-sequence
constant (so it cannot leak across the split), that it is aligned to the
molecules the benchmark scores, and that a sequence the cache does not cover
raises rather than silently becoming zeros.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from atlas.ml.peptide_benchmark import BENCHMARK_PATH, _extension_roster
from atlas.ml.peptide_representations import (
    build_representations, canonical_sequence,
)
from atlas.ml.peptide_structure import analyse_peptide
from atlas.ml.plm_embeddings import (
    PLM_DIM, PLM_EMBEDDING_PATH, PLM_MODEL, PlmEmbeddings, distinct_sequences,
    load_embeddings, plm_matrix,
)

from tests.test_peptide_structure import (
    AFAMELANOTIDE, BRADYKININ, CYCLOSPORINE, MELANOTAN_II, OXYTOCIN,
    SOMATOSTATIN,
)


@pytest.fixture(scope="module")
def peptides():
    return [analyse_peptide(s) for _c, s in (
        OXYTOCIN, SOMATOSTATIN, AFAMELANOTIDE, MELANOTAN_II, BRADYKININ,
        CYCLOSPORINE,
    )]


def _fake(peptides, dim: int = 8, *, drop: str | None = None) -> PlmEmbeddings:
    """A cache with a deterministic vector per sequence, no torch required."""
    seqs = [s for s in distinct_sequences(peptides) if s != drop]
    rng = np.random.default_rng(0)
    return PlmEmbeddings(
        model="test/fake", dim=dim,
        vectors={s: rng.normal(size=dim) for s in seqs},
        versions={"torch": "none"}, n_sequences=len(seqs),
    )


# ------------------------------------------------------- what gets embedded ----

def test_distinct_sequences_dedupes_and_sorts(peptides):
    seqs = distinct_sequences(peptides)
    assert seqs == sorted(set(seqs))
    assert len(seqs) <= len(peptides)
    assert set(seqs) == {canonical_sequence(a) for a in peptides}


def test_distinct_sequences_collapses_what_the_alphabet_cannot_name():
    """Two different side chains, one canonical string, one embedding.

    This is the ceiling restated as an executable fact: a frozen sequence
    model is handed the same input for both molecules, so no downstream model
    can separate them however good the embedding is.
    """
    chloro = analyse_peptide("CC(N)C(=O)NC(Cc1ccc(Cl)cc1)C(=O)NCC(O)=O")
    naphthyl = analyse_peptide("CC(N)C(=O)NC(Cc1ccc2ccccc2c1)C(=O)NCC(O)=O")
    assert canonical_sequence(chloro) == canonical_sequence(naphthyl)
    assert len(distinct_sequences([chloro, naphthyl])) == 1

    emb = _fake([chloro, naphthyl])
    mat = plm_matrix([chloro, naphthyl], emb)
    assert np.array_equal(mat[0], mat[1])


# -------------------------------------------------------------- alignment ----

def test_plm_matrix_is_aligned_and_shaped(peptides):
    emb = _fake(peptides, dim=8)
    mat = plm_matrix(peptides, emb)
    assert mat.shape == (len(peptides), 8)
    for row, a in enumerate(peptides):
        assert np.array_equal(mat[row], emb[canonical_sequence(a)])


def test_plm_matrix_is_a_per_molecule_function_of_its_own_sequence(peptides):
    """The property that makes computing all 630 vectors at once not leakage.

    Reordering the molecules permutes the rows and changes nothing else. No
    row depends on which other molecules were present, so there is no
    train/test boundary for the embedding step to cross.
    """
    emb = _fake(peptides, dim=8)
    order = [3, 0, 5, 1, 4, 2]
    full = plm_matrix(peptides, emb)
    shuffled = plm_matrix([peptides[i] for i in order], emb)
    assert np.array_equal(shuffled, full[order])
    assert np.array_equal(plm_matrix(peptides[:2], emb), full[:2])


# ------------------------------------------------- refusal, not zero-filling ----

def test_an_uncached_sequence_raises_rather_than_scoring_badly(peptides):
    """Zero-filling would have been reported as a measurement."""
    missing = canonical_sequence(peptides[0])
    emb = _fake(peptides, drop=missing)
    with pytest.raises(KeyError, match="absent from the embedding cache"):
        plm_matrix(peptides, emb)


def test_non_strict_zero_fills_and_is_opt_in(peptides):
    missing = canonical_sequence(peptides[0])
    emb = _fake(peptides, dim=8, drop=missing)
    mat = plm_matrix(peptides, emb, strict=False)
    assert not mat[0].any()
    assert mat[1:].any()


# ------------------------------------------------------------- the cache ----

def test_cache_round_trips_with_its_provenance(tmp_path, peptides):
    from atlas.ml.plm_embeddings import build_embeddings  # noqa: F401

    seqs = distinct_sequences(peptides)
    rng = np.random.default_rng(1)
    mat = rng.normal(size=(len(seqs), 4)).astype(np.float32)
    path = tmp_path / "e.npz"
    np.savez_compressed(
        path, sequences=np.array(seqs), vectors=mat,
        model=np.array("test/fake"), pooling=np.array("mean_over_residue_tokens"),
        version_torch=np.array("9.9.9"),
    )
    emb = load_embeddings(path)
    assert emb.model == "test/fake"
    assert emb.dim == 4
    assert emb.pooling == "mean_over_residue_tokens"
    assert emb.versions == {"torch": "9.9.9"}
    assert set(emb.vectors) == set(seqs)
    assert np.allclose(emb[seqs[0]], mat[0])


def test_a_truncated_cache_is_refused(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path, sequences=np.array(["AAA", "BBB"]),
        vectors=np.zeros((1, 4), dtype=np.float32),
        model=np.array("test/fake"), pooling=np.array("mean"),
    )
    with pytest.raises(ValueError, match="sequences but"):
        load_embeddings(path)


def test_committed_cache_is_the_model_it_claims():
    if not PLM_EMBEDDING_PATH.exists():
        pytest.skip("embedding cache not built; run atlas.ml.plm_embeddings")
    emb = load_embeddings()
    assert emb.model == PLM_MODEL
    assert emb.dim == PLM_DIM
    assert emb.pooling == "mean_over_residue_tokens"
    assert {"torch", "transformers"} <= set(emb.versions)
    mat = np.vstack(list(emb.vectors.values()))
    assert np.isfinite(mat).all()
    # A constant embedding would make the arm a disguised intercept.
    assert mat.std(axis=0).mean() > 0


# -------------------------------------------------- how the benchmark sees it ----

def test_extra_block_becomes_its_own_arm_and_a_concatenation(peptides):
    emb = _fake(peptides, dim=8)
    mat = plm_matrix(peptides, emb)
    reps = build_representations(peptides, extra={"esm2_embedding": mat})
    assert "esm2_embedding" in reps.names
    assert "positional_fragment_fp+esm2_embedding" in reps.names
    joined = reps["positional_fragment_fp+esm2_embedding"]
    assert joined.shape[1] == reps["positional_fragment_fp"].shape[1] + 8
    assert np.array_equal(joined[:, -8:], mat)


def test_extension_roster_counts_only_arms_that_ran():
    with_plm = _extension_roster({"results": {"esm2_embedding__forest": {}}})
    without = _extension_roster({"results": {}})
    assert with_plm.startswith("six times")
    assert "protein language model" in with_plm
    assert without.startswith("five times")
    assert "protein language model" not in without


@pytest.fixture(scope="module")
def report() -> dict:
    if not BENCHMARK_PATH.exists():
        pytest.skip(f"{BENCHMARK_PATH.name} not built; run atlas.ml.peptide_benchmark")
    return json.loads(BENCHMARK_PATH.read_text())


def test_all_three_arms_were_scored(report):
    """The honest experiment is three arms, and all three must be present."""
    for ev in ("cross_series", "within_series"):
        results = report[ev]["results"]
        for arm in ("esm2_embedding", "positional_fragment_fp+esm2_embedding",
                    "positional_fragment_fp"):
            assert f"{arm}__forest" in results, f"{ev}/{arm}"
            assert f"{arm}__ridge" in results, f"{ev}/{arm}"


def test_plm_contrasts_are_read_off_one_model(report):
    for ev in ("cross_series", "within_series"):
        for key in ("esm2_embedding_vs_fragment_fp",
                    "esm2_embedding_added_to_fragment_fp"):
            c = report["contrasts"][ev][key]
            assert c["model"] == "forest"
            assert c["a_better"] == (c["best_mae_a"] < c["best_mae_b"])


def test_the_null_is_not_explained_by_the_ceiling(report):
    """The argument the findings doc makes, asserted against the numbers.

    If the canonical-sequence oracle were above the best achieved MAE, a
    sequence-only arm failing would prove nothing. It is well below, so the
    headroom was real and the arm declined it.
    """
    seq = report["ceilings"]["canonical_sequence_20_letter"]
    best = min(v["mae"] for k, v in report["cross_series"]["results"].items()
               if k != "baseline_mean")
    assert seq["oracle_mae"] < best
