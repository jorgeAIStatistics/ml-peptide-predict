"""Uncertainty calibration with strict separation of data roles.

The previous model reported 90% nominal intervals that covered 79.4% of
held-out molecules. The diagnosis is structural rather than a tuning problem:
the calibration set was drawn at random from the training pool, so it was
in-distribution, while the test set was scaffold-shifted. Split conformal
guarantees coverage when calibration and test are exchangeable. They were not,
so the quantile was estimated against easier data than it was applied to.

Three strategies are implemented and measured against each other:

``random_calibration``
    The previous behaviour. Calibration drawn from the training pool.

``disjoint_calibration``
    Calibration is its own scaffold-disjoint block, so it carries the same kind
    of chemotype shift as the test set.

``mondrian``
    Disjoint calibration, with a separate quantile per similarity band, giving
    group-conditional rather than merely marginal coverage.

Data roles are kept strictly apart: **train** fits the model, **select**
chooses between models, **calibrate** sets the interval width, and **test** is
touched once. No set serves two purposes.

Nominal coverage is not guaranteed under distribution shift and is not claimed.
What is reported is the coverage actually measured, overall and by subgroup.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from atlas.ml.features import FeatureSet, Modality, applicability

logger = logging.getLogger(__name__)

__all__ = [
    "FourWaySplit",
    "four_way_scaffold_split",
    "ConformalCalibrator",
    "CalibrationReport",
    "evaluate_calibration",
    "SIMILARITY_BANDS",
]

#: Similarity bands used for group-conditional calibration and reporting.
SIMILARITY_BANDS: tuple[tuple[float, float, str], ...] = (
    (0.0, 0.4, "0.0–0.4"),
    (0.4, 0.55, "0.4–0.55"),
    (0.55, 0.7, "0.55–0.7"),
    (0.7, 1.01, "0.7–1.0"),
)

#: A band needs at least this many calibration points for its own quantile;
#: below it the global quantile is used and the fallback is recorded.
MIN_BAND_CALIBRATION = 25


def band_of(similarity: float) -> str:
    for lo, hi, name in SIMILARITY_BANDS:
        if lo <= similarity < hi:
            return name
    return SIMILARITY_BANDS[-1][2]


@dataclass(frozen=True)
class FourWaySplit:
    """Disjoint train / select / calibrate / test indices."""

    train_idx: np.ndarray
    select_idx: np.ndarray
    calib_idx: np.ndarray
    test_idx: np.ndarray
    description: str
    seed: int

    def __post_init__(self) -> None:
        sets = {
            "train": set(self.train_idx.tolist()),
            "select": set(self.select_idx.tolist()),
            "calib": set(self.calib_idx.tolist()),
            "test": set(self.test_idx.tolist()),
        }
        names = list(sets)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                overlap = sets[a] & sets[b]
                if overlap:
                    raise ValueError(
                        f"{a} and {b} share {len(overlap)} indices; data roles "
                        "must be strictly separate"
                    )

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train_idx), "select": len(self.select_idx),
            "calibrate": len(self.calib_idx), "test": len(self.test_idx),
        }


def four_way_scaffold_split(
    fs: FeatureSet,
    *,
    fractions: tuple[float, float, float, float] = (0.50, 0.15, 0.15, 0.20),
    seed: int = 0,
) -> FourWaySplit:
    """Partition scaffold groups into four disjoint roles.

    Whole scaffold groups move together, so no chemotype appears in two roles.
    This is what makes the calibration set carry the same shift as the test set
    instead of being easier than it.
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")

    groups: dict[str, list[int]] = defaultdict(list)
    for i, scaf in enumerate(fs.scaffolds):
        groups[scaf].append(i)

    rng = np.random.default_rng(seed)
    order = list(groups.values())
    rng.shuffle(order)

    n = len(fs)
    targets = [f * n for f in fractions]
    buckets: list[list[int]] = [[], [], [], []]

    # Assign each group to whichever role is furthest below its target, which
    # keeps the four sets near their intended sizes without splitting a group.
    for group in sorted(order, key=len, reverse=True):
        deficits = [t - len(b) for t, b in zip(targets, buckets)]
        buckets[int(np.argmax(deficits))].extend(group)

    return FourWaySplit(
        train_idx=np.sort(np.array(buckets[0], dtype=int)),
        select_idx=np.sort(np.array(buckets[1], dtype=int)),
        calib_idx=np.sort(np.array(buckets[2], dtype=int)),
        test_idx=np.sort(np.array(buckets[3], dtype=int)),
        description=(
            f"{len(groups)} Bemis-Murcko scaffold groups partitioned into four "
            "disjoint roles; no scaffold appears in more than one role."
        ),
        seed=seed,
    )


class ConformalCalibrator:
    """Split-conformal intervals, optionally conditioned on similarity band."""

    def __init__(self, alpha: float = 0.10, mondrian: bool = False) -> None:
        self.alpha = alpha
        self.mondrian = mondrian
        self.global_quantile: float = float("nan")
        self.band_quantiles: dict[str, float] = {}
        self.band_counts: dict[str, int] = {}
        self.bands_using_fallback: list[str] = []

    @staticmethod
    def _conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
        """Finite-sample conformal quantile of absolute residuals."""
        n = len(residuals)
        if n == 0:
            return float("nan")
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        return float(np.quantile(residuals, level))

    def fit(
        self,
        residuals: np.ndarray,
        similarities: np.ndarray | None = None,
    ) -> "ConformalCalibrator":
        residuals = np.abs(np.asarray(residuals, dtype=float))
        self.global_quantile = self._conformal_quantile(residuals, self.alpha)

        if self.mondrian:
            if similarities is None:
                raise ValueError("mondrian calibration requires similarities")
            bands = np.array([band_of(s) for s in similarities])
            for _, _, name in SIMILARITY_BANDS:
                sel = bands == name
                self.band_counts[name] = int(sel.sum())
                if sel.sum() >= MIN_BAND_CALIBRATION:
                    self.band_quantiles[name] = self._conformal_quantile(
                        residuals[sel], self.alpha
                    )
                else:
                    # Too few points to estimate a band quantile honestly.
                    self.band_quantiles[name] = self.global_quantile
                    self.bands_using_fallback.append(name)
        return self

    def half_width(self, similarity: float | None = None) -> float:
        if self.mondrian and similarity is not None:
            return self.band_quantiles.get(band_of(similarity), self.global_quantile)
        return self.global_quantile

    def intervals(
        self, predictions: np.ndarray, similarities: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        predictions = np.asarray(predictions, dtype=float)
        if self.mondrian and similarities is not None:
            half = np.array([self.half_width(s) for s in similarities])
        else:
            half = np.full(len(predictions), self.global_quantile)
        return predictions - half, predictions + half, half


@dataclass
class CalibrationReport:
    """Everything needed to judge whether intervals mean what they claim."""

    strategy: str
    nominal_coverage: float
    empirical_coverage: float
    n_test: int
    mae: float
    rmse: float
    median_interval_width: float
    mean_interval_width: float
    coverage_by_similarity: dict[str, dict[str, Any]] = field(default_factory=dict)
    coverage_by_modality: dict[str, dict[str, Any]] = field(default_factory=dict)
    error_by_similarity: dict[str, dict[str, Any]] = field(default_factory=dict)
    calibration_set_size: int = 0
    bands_using_fallback: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    @property
    def coverage_gap(self) -> float:
        """Signed shortfall against nominal. Negative means undercoverage."""
        return self.empirical_coverage - self.nominal_coverage

    @property
    def is_adequately_calibrated(self) -> bool:
        """Within 3 percentage points of nominal, and never claimed otherwise.

        Deliberately strict and deliberately two-sided: intervals far wider
        than needed are also miscalibrated, just in a less dangerous direction.
        """
        return abs(self.coverage_gap) <= 0.03

    def summary(self) -> str:
        verdict = "adequate" if self.is_adequately_calibrated else "NOT adequate"
        return (
            f"{self.strategy}: nominal {self.nominal_coverage:.2f}, empirical "
            f"{self.empirical_coverage:.3f} ({self.coverage_gap:+.3f}) — {verdict}; "
            f"median width {self.median_interval_width:.3f} log units"
        )


def _group_stats(
    mask: np.ndarray, covered: np.ndarray, abs_err: np.ndarray, width: np.ndarray
) -> dict[str, Any]:
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "coverage": None, "mae": None, "median_width": None}
    return {
        "n": n,
        "coverage": float(covered[mask].mean()),
        "mae": float(abs_err[mask].mean()),
        "median_width": float(np.median(width[mask])),
    }


def evaluate_calibration(
    *,
    strategy: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    calibrator: ConformalCalibrator,
    similarities: np.ndarray,
    modalities: Sequence[Modality],
    calibration_set_size: int,
    limitations: Sequence[str] = (),
) -> CalibrationReport:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    lo, hi, half = calibrator.intervals(y_pred, similarities)
    width = hi - lo
    covered = (y_true >= lo) & (y_true <= hi)
    abs_err = np.abs(y_pred - y_true)

    cov_sim: dict[str, dict[str, Any]] = {}
    err_sim: dict[str, dict[str, Any]] = {}
    for lo_b, hi_b, name in SIMILARITY_BANDS:
        mask = (similarities >= lo_b) & (similarities < hi_b)
        cov_sim[name] = _group_stats(mask, covered, abs_err, width)
        err_sim[name] = {
            "n": int(mask.sum()),
            "mae": float(abs_err[mask].mean()) if mask.any() else None,
            "rmse": (
                float(np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2)))
                if mask.any() else None
            ),
        }

    cov_mod: dict[str, dict[str, Any]] = {}
    for modality in (Modality.PEPTIDE_LIKE, Modality.SMALL_MOLECULE_LIKE):
        mask = np.array([m is modality for m in modalities])
        cov_mod[modality.value] = _group_stats(mask, covered, abs_err, width)

    notes = list(limitations)
    notes.append(
        "Conformal coverage is guaranteed only under exchangeability of the "
        "calibration and test sets. Under chemotype or modality shift that "
        "assumption fails, and nominal coverage is not guaranteed."
    )
    if calibrator.bands_using_fallback:
        notes.append(
            "Similarity bands "
            + ", ".join(calibrator.bands_using_fallback)
            + f" had fewer than {MIN_BAND_CALIBRATION} calibration points and "
            "fall back to the global quantile, so their conditional coverage is "
            "not separately guaranteed."
        )

    return CalibrationReport(
        strategy=strategy,
        nominal_coverage=1.0 - calibrator.alpha,
        empirical_coverage=float(covered.mean()),
        n_test=len(y_true),
        mae=float(abs_err.mean()),
        rmse=float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        median_interval_width=float(np.median(width)),
        mean_interval_width=float(width.mean()),
        coverage_by_similarity=cov_sim,
        coverage_by_modality=cov_mod,
        error_by_similarity=err_sim,
        calibration_set_size=calibration_set_size,
        bands_using_fallback=list(calibrator.bands_using_fallback),
        limitations=notes,
    )
