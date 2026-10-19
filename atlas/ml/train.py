"""Train, evaluate and register models for MC4R binding affinity.

Everything reported here is measured on held-out data. No target performance is
asserted anywhere in this module, and a model that fails is recorded as failing.

The experiment is deliberately comparative along two axes:

* **Model family** -- from a mean-predictor baseline upward. A complex model
  earns its place only by beating the simple ones on the same split.
* **Split** -- random, scaffold, and modality holdout. The gap between random
  and scaffold scores is a direct measure of how much a random split flatters
  the model; the modality holdout answers whether a model fitted on small
  molecules may be used on peptides.

Uncertainty uses split conformal prediction, which gives marginal coverage
under exchangeability of calibration and test data. Under a scaffold or
modality split that assumption is deliberately violated, so measured coverage
is expected to fall short of nominal -- and that shortfall is reported rather
than hidden, because it is the honest signal that the model is extrapolating.
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from atlas.core.evidence import utc_now
from atlas.ml.dataset import DatasetBundle, load_dataset
from atlas.ml.features import FeatureSet, Modality, applicability, featurize
from atlas.ml.splits import (
    Split,
    leakage_report,
    modality_split,
    random_split,
    scaffold_split,
)

logger = logging.getLogger(__name__)

__all__ = ["ModelSpec", "EvalResult", "MODELS", "evaluate", "run_experiment", "REGISTRY_PATH"]

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "data" / "ml" / "model_registry.json"

SEED = 20261018
CONFORMAL_ALPHA = 0.10  # nominal 90% prediction intervals


@dataclass(frozen=True)
class ModelSpec:
    """A model family, the features it consumes, and how to build it."""

    name: str
    features: str  # "descriptors" | "fingerprints" | "combined"
    build: Callable[[], Any]
    description: str
    supports_ensemble_variance: bool = False


def _features_for(fs: FeatureSet, kind: str) -> np.ndarray:
    if kind == "descriptors":
        return fs.descriptors
    if kind == "fingerprints":
        # Boolean, because the Jaccard (Tanimoto) metric is defined on sets.
        # Tree models split identically on bool and on 0/1 floats.
        return fs.fingerprints.astype(bool)
    if kind == "combined":
        return fs.combined
    raise ValueError(f"unknown feature kind {kind!r}")


MODELS: list[ModelSpec] = [
    ModelSpec(
        "baseline_mean", "descriptors",
        lambda: DummyRegressor(strategy="mean"),
        "Predicts the training mean for every molecule. Any useful model must "
        "beat this.",
    ),
    ModelSpec(
        "descriptors_ridge", "descriptors",
        lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ]),
        "Linear ridge regression on 15 interpretable physicochemical "
        "descriptors, with alpha chosen by internal cross-validation.",
    ),
    ModelSpec(
        "descriptors_rf", "descriptors",
        lambda: RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, max_features=0.5,
            n_jobs=-1, random_state=SEED,
        ),
        "Random forest on physicochemical descriptors.",
        supports_ensemble_variance=True,
    ),
    ModelSpec(
        "fingerprint_knn", "fingerprints",
        lambda: KNeighborsRegressor(n_neighbors=5, metric="jaccard", weights="distance"),
        "Nearest-neighbour regression on Morgan fingerprints with Jaccard "
        "(Tanimoto) distance -- the classical similarity baseline for QSAR.",
    ),
    ModelSpec(
        "fingerprint_rf", "fingerprints",
        lambda: RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            n_jobs=-1, random_state=SEED,
        ),
        "Random forest on 2048-bit Morgan fingerprints.",
        supports_ensemble_variance=True,
    ),
    ModelSpec(
        "combined_gb", "combined",
        lambda: HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            random_state=SEED,
        ),
        "Gradient-boosted trees on descriptors concatenated with fingerprints.",
    ),
    ModelSpec(
        "combined_rf", "combined",
        lambda: RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            n_jobs=-1, random_state=SEED,
        ),
        "Random forest on descriptors concatenated with fingerprints.",
        supports_ensemble_variance=True,
    ),
]


def _num(v: float | None, width: int = 7, places: int = 3) -> str:
    """Format a possibly-undefined statistic for a fixed-width table."""
    return f"{'—':>{width}}" if v is None else f"{v:>{width}.{places}f}"


@dataclass
class EvalResult:
    """Held-out performance of one model on one split."""

    model: str
    split: str
    n_train: int
    n_test: int
    mae: float
    rmse: float
    r2: float | None
    spearman: float | None
    pearson: float | None
    fit_seconds: float
    conformal_coverage: float | None = None
    conformal_nominal: float = 1.0 - CONFORMAL_ALPHA
    conformal_half_width: float | None = None
    applicability: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def beats_baseline_by(self) -> float | None:
        return None  # filled in by the comparison table

    def row(self) -> str:
        return (
            f"{self.model:20} {self.split:26} {self.n_train:>5} {self.n_test:>5} "
            f"{self.mae:>7.3f} {self.rmse:>7.3f} {_num(self.r2, 7)} "
            f"{_num(self.spearman, 8)} {_num(self.conformal_coverage, 8)}"
        )


def _finite_or_none(v: Any) -> float | None:
    """NaN means a statistic is undefined, which is not a number.

    A constant predictor has no rank correlation at all: scipy returns NaN, and
    serialising that as a bare NaN produces invalid JSON. Undefined statistics
    are therefore reported as null, consistent with how the rest of the platform
    distinguishes "not defined" from a value.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    resid = y_pred - y_true
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    # R-squared against the *test* mean. On a shifted test population this can
    # be negative, which correctly says the model is worse than predicting that
    # population's own mean.
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    # A constant predictor makes correlation undefined; scipy warns and returns
    # NaN. That case is handled explicitly below, so the warning is silenced
    # rather than printed on every baseline evaluation.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", stats.ConstantInputWarning)
        sp = stats.spearmanr(y_true, y_pred).statistic if len(y_true) > 2 else None
        pe = stats.pearsonr(y_true, y_pred).statistic if len(y_true) > 2 else None
    return {
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "r2": _finite_or_none(r2),
        "spearman": _finite_or_none(sp),
        "pearson": _finite_or_none(pe),
    }


def evaluate(
    spec: ModelSpec,
    fs: FeatureSet,
    split: Split,
    *,
    conformal: bool = True,
    seed: int = SEED,
) -> EvalResult:
    """Fit on the training indices and score on the held-out indices."""
    X = _features_for(fs, spec.features)
    y = fs.y
    tr, te = split.train_idx, split.test_idx
    notes: list[str] = []

    # Split conformal needs a calibration set carved out of training data, so
    # the model never sees it during fitting.
    cal_idx = np.array([], dtype=int)
    fit_idx = tr
    if conformal and len(tr) >= 100:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(tr)
        n_cal = max(50, int(0.15 * len(tr)))
        cal_idx, fit_idx = shuffled[:n_cal], shuffled[n_cal:]

    model = spec.build()
    t0 = time.perf_counter()
    model.fit(X[fit_idx], y[fit_idx])
    fit_seconds = time.perf_counter() - t0

    pred = model.predict(X[te])
    m = _metrics(y[te], pred)

    coverage = half_width = None
    if len(cal_idx):
        cal_resid = np.abs(model.predict(X[cal_idx]) - y[cal_idx])
        # Finite-sample conformal quantile.
        n = len(cal_resid)
        level = min(1.0, np.ceil((n + 1) * (1 - CONFORMAL_ALPHA)) / n)
        half_width = float(np.quantile(cal_resid, level))
        coverage = float(np.mean(np.abs(pred - y[te]) <= half_width))
        if coverage < (1 - CONFORMAL_ALPHA) - 0.05:
            notes.append(
                f"conformal coverage {coverage:.3f} falls short of the nominal "
                f"{1 - CONFORMAL_ALPHA:.2f}; calibration and test are not "
                "exchangeable under this split, which is the expected signal "
                "of extrapolation"
            )

    # Applicability domain: error as a function of similarity to training data.
    sim = applicability(fs.fingerprints[fit_idx], fs.fingerprints[te])
    abs_err = np.abs(pred - y[te])
    bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.01)]
    ad: dict[str, Any] = {
        "median_max_tanimoto": float(np.median(sim)),
        "fraction_below_0.3": float(np.mean(sim < 0.3)),
        "bins": [],
    }
    for lo, hi in bins:
        sel = (sim >= lo) & (sim < hi)
        ad["bins"].append({
            "range": f"{lo:.1f}-{hi if hi <= 1 else 1.0:.1f}",
            "n": int(sel.sum()),
            "mae": float(np.mean(abs_err[sel])) if sel.any() else None,
        })

    return EvalResult(
        model=spec.name,
        split=split.kind,
        n_train=len(fit_idx),
        n_test=len(te),
        fit_seconds=fit_seconds,
        conformal_coverage=coverage,
        conformal_half_width=half_width,
        applicability=ad,
        notes=notes,
        **m,
    )


def build_splits(fs: FeatureSet, seed: int = SEED) -> list[Split]:
    return [
        random_split(fs, seed=seed),
        scaffold_split(fs, seed=seed),
        modality_split(fs, holdout=Modality.PEPTIDE_LIKE),
        modality_split(fs, holdout=Modality.SMALL_MOLECULE_LIKE),
    ]


def run_experiment(
    bundle: DatasetBundle | None = None, *, seed: int = SEED
) -> dict[str, Any]:
    """Run every model against every split and return a registry payload."""
    bundle = bundle or load_dataset()
    fs = featurize(bundle.records)
    splits = build_splits(fs, seed=seed)

    results: list[EvalResult] = []
    for split in splits:
        for spec in MODELS:
            logger.info("fitting %s on %s split", spec.name, split.kind)
            results.append(evaluate(spec, fs, split, seed=seed))

    n_pep = sum(1 for m in fs.modality if m is Modality.PEPTIDE_LIKE)
    return {
        "created_at": utc_now().isoformat(),
        "dataset_version": bundle.version,
        "dataset_endpoint": bundle.endpoint,
        "dataset_counts": bundle.counts,
        "noise_estimate": bundle.noise_estimate,
        "seed": seed,
        "n_molecules": len(fs),
        "n_peptide_like": n_pep,
        "n_small_molecule_like": len(fs) - n_pep,
        "conformal_alpha": CONFORMAL_ALPHA,
        "models": [
            {"name": s.name, "features": s.features, "description": s.description}
            for s in MODELS
        ],
        "splits": [
            {
                "kind": s.kind,
                "description": s.description,
                "n_train": s.n_train,
                "n_test": s.n_test,
                "leakage": leakage_report(fs, s),
            }
            for s in splits
        ],
        "results": [asdict(r) for r in results],
    }


def _print_report(payload: dict[str, Any]) -> None:
    print("=" * 104)
    print("MC4R BINDING AFFINITY (pKi) — HELD-OUT EVALUATION".center(104))
    print("=" * 104)
    print(f"dataset {payload['dataset_version']}  "
          f"{payload['n_molecules']} molecules "
          f"({payload['n_peptide_like']} peptide-like, "
          f"{payload['n_small_molecule_like']} small-molecule-like)  seed {payload['seed']}")

    noise = payload["noise_estimate"]
    print(f"experimental noise floor: median disagreement "
          f"{noise['median_spread_among_differing_log']:.3f} log units across "
          f"{noise['n_molecules_with_differing_values']} molecules "
          f"-> MAE floor ~{noise['implied_mae_floor_log']:.3f}")

    print("\nsplit leakage:")
    for s in payload["splits"]:
        lk = s["leakage"]
        print(f"    {s['kind']:28} train {s['n_train']:>5} test {s['n_test']:>5}  "
              f"shared scaffolds {lk['n_shared_scaffolds']:>4}  "
              f"test scaffolds already seen {lk['fraction_test_scaffolds_seen']:.1%}  "
              f"identical molecules {lk['n_identical_molecules']}")

    by_split: dict[str, list[dict]] = {}
    for r in payload["results"]:
        by_split.setdefault(r["split"], []).append(r)

    for split, rows in by_split.items():
        print(f"\n{'-' * 104}\n{split}\n{'-' * 104}")
        print(f"{'model':20} {'split':26} {'ntr':>5} {'nte':>5} {'MAE':>7} "
              f"{'RMSE':>7} {'R2':>7} {'spearman':>8} {'cover':>8}")
        base = next((r for r in rows if r["model"] == "baseline_mean"), None)
        for r in sorted(rows, key=lambda x: x["mae"]):
            flag = ""
            if base and r["model"] != "baseline_mean":
                flag = " *" if r["mae"] < base["mae"] else "  (worse than baseline)"
            print(f"{r['model']:20} {r['split']:26} {r['n_train']:>5} {r['n_test']:>5} "
                  f"{r['mae']:>7.3f} {r['rmse']:>7.3f} {_num(r['r2'], 7)} "
                  f"{_num(r['spearman'], 8)} {_num(r['conformal_coverage'], 8)}{flag}")

    print(f"\n{'-' * 104}\napplicability domain (best model per split)\n{'-' * 104}")
    for split, rows in by_split.items():
        best = min(rows, key=lambda x: x["mae"])
        ad = best["applicability"]
        bins = "  ".join(
            f"{b['range']}: n={b['n']} MAE={b['mae']:.3f}" if b["mae"] is not None
            else f"{b['range']}: n=0"
            for b in ad["bins"]
        )
        print(f"{split:28} {best['model']:20} median similarity "
              f"{ad['median_max_tanimoto']:.3f}")
        print(f"{'':28} {bins}")

    notes = [(r["model"], r["split"], n) for r in payload["results"] for n in r["notes"]]
    if notes:
        print(f"\n{'-' * 104}\ncalibration notes\n{'-' * 104}")
        seen = set()
        for model, split, note in notes:
            key = (split, note[:40])
            if key in seen:
                continue
            seen.add(key)
            print(f"  [{split}] {model}: {note}")


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    payload = run_experiment()
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False))
    _print_report(payload)
    print(f"\nregistry written to {REGISTRY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
