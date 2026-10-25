"""The noise band under every cross-series number this project has reported.

Every benchmark comparison here was drawn with seed 0. These tests pin the
statistic that says whether a reported difference was larger than the split it
was measured on.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from atlas.ml.split_noise import (
    CONTRASTS, EXTRA_ARMS, ON_TARGET_MAX_TEST_FRACTION, REFERENCE_ARM, SEEDS,
    SPLIT_NOISE_PATH, paired_summary,
)


@pytest.fixture(scope="module")
def report() -> dict:
    if not SPLIT_NOISE_PATH.exists():
        pytest.skip(f"{SPLIT_NOISE_PATH.name} not built; run atlas.ml.split_noise")
    return json.loads(SPLIT_NOISE_PATH.read_text())


# ------------------------------------------------------------- statistics ----

def test_a_consistent_difference_is_demonstrated():
    s = paired_summary([0.30, 0.28, 0.32, 0.29, 0.31])
    assert s["demonstrated"]
    assert s["mean_difference"] == pytest.approx(0.30, abs=0.01)
    assert s["sign_agreement"] == 1.0


def test_a_difference_smaller_than_its_own_spread_is_not_demonstrated():
    """The case the whole module exists for: a real-looking mean, swamped."""
    s = paired_summary([0.02, -0.15, 0.19, -0.08, 0.11])
    assert not s["demonstrated"]
    assert abs(s["mean_difference"]) < s["sd_of_difference"]


def test_sign_agreement_exposes_a_mean_driven_by_one_outlier():
    """Low agreement is the diagnostic, and it means what it says.

    Three splits agree on a small positive difference and one disagrees
    hugely; the mean follows the outlier, so only a quarter of splits agree
    with it. That is the shape of residue_chemistry in the saved report (33%),
    and it is the reason a mean alone is not a result.
    """
    s = paired_summary([0.1, 0.1, 0.1, -0.9])
    assert s["mean_difference"] < 0, "one large outlier flips the mean"
    assert s["sign_agreement"] == pytest.approx(0.25)
    assert not s["demonstrated"]


def test_direction_names_which_way_the_extension_went():
    assert paired_summary([0.2] * 5)["direction"] == "extension is worse"
    assert paired_summary([-0.2] * 5)["direction"] == "extension is better"


def test_non_finite_differences_are_dropped_not_propagated():
    """A degenerate arm yields nan; one nan must not erase the whole contrast."""
    s = paired_summary([0.2, float("nan"), 0.22, 0.18])
    assert s["n_seeds"] == 3
    assert np.isfinite(s["mean_difference"])


def test_too_few_seeds_refuses_to_summarise():
    s = paired_summary([0.2])
    assert not s["demonstrated"]
    assert "too few" in s["note"]


def test_a_single_seed_cannot_demonstrate_anything():
    """Which is the point: every prior comparison had exactly one."""
    assert not paired_summary([5.0])["demonstrated"]


# ----------------------------------------------------------- saved report ----

def test_the_sweep_covers_every_reported_extension(report):
    """If an extension is benchmarked but never noise-checked, a reader cannot
    tell whether its number was real."""
    assert set(report["paired_contrasts"]) <= set(CONTRASTS)
    missing = set(CONTRASTS) - set(report["paired_contrasts"])
    assert not missing - {"esm2_embedding", "assay_context"}, missing


def test_every_contrast_is_measured_on_several_splits(report):
    for name, c in report["paired_contrasts"].items():
        assert c["n_seeds"] >= 2, name
        assert c["reference"] == REFERENCE_ARM


def test_the_reference_arm_has_a_spread_worth_reporting(report):
    """If the split made no difference the module would have nothing to say."""
    ref = report["per_arm"][REFERENCE_ARM]
    assert ref["n_seeds"] == len(SEEDS)
    assert ref["sd"] > 0.0
    assert ref["max"] > ref["min"]


def test_paired_spread_is_smaller_than_unpaired_spread(report):
    """Pairing is only worth doing if it removes the shared split difficulty.

    The paired difference's spread must be smaller than the spread of the
    reference arm itself; otherwise the two arms are not tracking together
    across splits and pairing buys nothing.
    """
    ref_sd = report["per_arm"][REFERENCE_ARM]["sd"]
    for name, c in report["paired_contrasts"].items():
        assert c["sd_of_difference"] < ref_sd * 2, (name, c["sd_of_difference"], ref_sd)


def test_no_extension_improves_the_mean_across_splits(report):
    """The six null results, re-read across twelve splits rather than one.

    This is the assertion that would have to be deleted if an extension ever
    started helping, which is exactly when someone should be made to look.
    """
    helped = {n: c["mean_difference"] for n, c in report["paired_contrasts"].items()
              if c["mean_difference"] < 0}
    assert not helped, f"an extension now improves the mean: {helped}"


def test_findings_distinguish_demonstrated_from_within_noise(report):
    joined = " ".join(report["findings"]).lower()
    assert "noise" in joined
    assert any(ch.isdigit() for ch in joined)
    assert len(report["findings"]) >= 3


def test_report_records_which_arms_were_fitted(report):
    for arm in (REFERENCE_ARM, *EXTRA_ARMS):
        assert arm in report["per_arm"], arm


# ----------------------------------------- the split-target stratification ----

def test_the_splitter_overshoots_its_target_on_some_seeds(report):
    """The structural instability the sweep exposed.

    The splitter assigns whole clusters toward a 25% test target, and the
    largest cluster holds 53% of the set. If this ever stops being true the
    stratification below is pointless and should be removed.
    """
    fracs = [v["test_fraction"] for v in report["split_sizes"].values()]
    assert max(fracs) > 0.5, "no seed overshoots; stratification is moot"
    assert min(fracs) <= 0.30


def test_on_target_seeds_are_the_ones_that_met_the_target(report):
    on_target = set(report["on_target_seeds"])
    for seed, v in report["split_sizes"].items():
        expected = v["test_fraction"] <= ON_TARGET_MAX_TEST_FRACTION
        assert (int(seed) in on_target) == expected, seed
    assert on_target, "no seed met its target"


def test_both_strata_are_reported(report):
    """Reporting only one would hide that they disagree."""
    assert report["paired_contrasts"]
    assert report["paired_contrasts_on_target"]
    assert set(report["paired_contrasts_on_target"]) <= set(report["paired_contrasts"])


def test_degenerate_splits_compress_the_contrasts(report):
    """Why the on-target subset is primary rather than merely a subgroup.

    A seed holding out 71% of the set trains on ~200 molecules, where every
    arm collapses toward the baseline together. Pooling those in shrinks the
    measured differences; it does not add evidence.
    """
    all_c, on_c = report["paired_contrasts"], report["paired_contrasts_on_target"]
    shrunk = [k for k in on_c
              if abs(on_c[k]["mean_difference"]) > abs(all_c[k]["mean_difference"])]
    assert len(shrunk) >= len(on_c) // 2, (
        "pooling degenerate splits no longer compresses the contrasts; "
        "re-examine whether the stratification is still justified")


def test_seed_zero_is_recorded_beside_the_summary(report):
    """Every published comparison used seed 0; it must stay auditable."""
    for name, c in report["paired_contrasts"].items():
        assert "seed_0_difference" in c, name


def test_seed_zero_overstated_at_least_one_effect(report):
    """The finding that prompted this module.

    If no published single-seed number was inflated relative to the paired
    effect, the sweep found nothing and this test should fail loudly rather
    than pass quietly.
    """
    on_c = report["paired_contrasts_on_target"]
    inflated = [
        name for name, c in on_c.items()
        if np.isfinite(c["seed_0_difference"])
        and abs(c["seed_0_difference"]) > 1.5 * abs(c["mean_difference"])
    ]
    assert inflated, "no single-seed difference was inflated; sweep found nothing"


def test_no_extension_helps_on_the_on_target_subset(report):
    helped = {n: c["mean_difference"]
              for n, c in report["paired_contrasts_on_target"].items()
              if c["mean_difference"] < 0}
    assert not helped, f"an extension now improves the mean: {helped}"
