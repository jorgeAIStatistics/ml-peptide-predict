"""Compare calibration strategies under strict separation of data roles.

Answers one question with measurement: does calibrating on a scaffold-disjoint
block fix the undercoverage seen when calibrating on in-distribution data?

The model is fitted on **train**, chosen on **select**, its interval width set
on **calibrate**, and scored once on **test**. All four are scaffold-disjoint.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from atlas.core.evidence import utc_now
from atlas.ml.calibration import (
    ConformalCalibrator,
    evaluate_calibration,
    four_way_scaffold_split,
)
from atlas.ml.dataset import load_dataset
from atlas.ml.features import applicability, featurize
from atlas.ml.train import SEED, _metrics

logger = logging.getLogger(__name__)

REPORT_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "ml" / "calibration_report.json"
)

ALPHA = 0.10

CANDIDATES = {
    "fingerprint_knn": lambda: KNeighborsRegressor(
        n_neighbors=5, metric="jaccard", weights="distance"),
    "fingerprint_rf": lambda: RandomForestRegressor(
        n_estimators=500, min_samples_leaf=2, max_features="sqrt",
        n_jobs=-1, random_state=SEED),
    "fingerprint_ridge": lambda: Pipeline([
        ("scale", StandardScaler(with_mean=False)),
        ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
    ]),
}


def run(seed: int = SEED) -> dict[str, Any]:
    bundle = load_dataset()
    fs = featurize(bundle.records)
    split = four_way_scaffold_split(fs, seed=seed)

    X = fs.fingerprints.astype(bool)
    y = fs.y

    # --- model selection happens on `select`, never on `test`.
    selection: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    for name, build in CANDIDATES.items():
        model = build()
        model.fit(X[split.train_idx], y[split.train_idx])
        fitted[name] = model
        m = _metrics(y[split.select_idx], model.predict(X[split.select_idx]))
        selection.append({"model": name, **m})
        logger.info("select %s MAE %.3f", name, m["mae"])

    chosen = min(selection, key=lambda r: r["mae"])["model"]
    model = fitted[chosen]

    # --- similarity of each point to the training set it was fitted on
    sim_calib = applicability(fs.fingerprints[split.train_idx],
                              fs.fingerprints[split.calib_idx])
    sim_test = applicability(fs.fingerprints[split.train_idx],
                             fs.fingerprints[split.test_idx])

    pred_test = model.predict(X[split.test_idx])
    y_test = y[split.test_idx]
    modal_test = [fs.modality[i] for i in split.test_idx]

    reports: list[Any] = []

    # 1. The previous behaviour, reproduced faithfully: hold calibration
    #    points out of fitting -- so they are not memorised -- but draw them at
    #    random from the training pool, where they share scaffolds with the
    #    fitted data. A separate model is fitted for this arm, because reusing
    #    the model above would calibrate on points it had already seen and
    #    measure memorisation rather than in-distribution error.
    rng = np.random.default_rng(seed)
    pool = rng.permutation(split.train_idx)
    n_cal = len(split.calib_idx)
    in_dist_cal, in_dist_fit = pool[:n_cal], pool[n_cal:]
    model_random = CANDIDATES[chosen]()
    model_random.fit(X[in_dist_fit], y[in_dist_fit])
    resid_in = np.abs(model_random.predict(X[in_dist_cal]) - y[in_dist_cal])
    cal_random = ConformalCalibrator(alpha=ALPHA).fit(resid_in)
    pred_test_random = model_random.predict(X[split.test_idx])
    sim_in_cal = applicability(fs.fingerprints[in_dist_fit],
                               fs.fingerprints[in_dist_cal])
    reports.append(evaluate_calibration(
        strategy="random_calibration (previous behaviour)",
        y_true=y_test, y_pred=pred_test_random, calibrator=cal_random,
        similarities=sim_test, modalities=modal_test,
        calibration_set_size=len(in_dist_cal),
        limitations=[
            "Calibration points were held out of fitting but drawn from the "
            "training pool, so they share scaffolds with the fitted data. "
            "Residuals there are smaller than on unseen chemotypes, and the "
            f"quantile is set against median similarity "
            f"{float(np.median(sim_in_cal)):.3f} while being applied at "
            f"{float(np.median(sim_test)):.3f}.",
        ],
    ))

    # 2. The fix: calibrate on a scaffold-disjoint block carrying the same kind
    #    of shift as the test set.
    resid_disjoint = np.abs(model.predict(X[split.calib_idx]) - y[split.calib_idx])
    cal_disjoint = ConformalCalibrator(alpha=ALPHA).fit(resid_disjoint)
    reports.append(evaluate_calibration(
        strategy="disjoint_calibration",
        y_true=y_test, y_pred=pred_test, calibrator=cal_disjoint,
        similarities=sim_test, modalities=modal_test,
        calibration_set_size=len(split.calib_idx),
    ))

    # 3. Group-conditional: one quantile per similarity band.
    cal_mondrian = ConformalCalibrator(alpha=ALPHA, mondrian=True).fit(
        resid_disjoint, sim_calib)
    reports.append(evaluate_calibration(
        strategy="mondrian (per-similarity-band)",
        y_true=y_test, y_pred=pred_test, calibrator=cal_mondrian,
        similarities=sim_test, modalities=modal_test,
        calibration_set_size=len(split.calib_idx),
    ))

    return {
        "created_at": utc_now().isoformat(),
        "dataset_version": bundle.version,
        "seed": seed,
        "alpha": ALPHA,
        "split": {
            "description": split.description,
            "sizes": split.sizes,
            "median_similarity_calibration_to_train": float(np.median(sim_calib)),
            "median_similarity_test_to_train": float(np.median(sim_test)),
        },
        "model_selection": {
            "candidates": selection,
            "chosen": chosen,
            "selected_on": "select split (never the test split)",
        },
        "band_quantiles": cal_mondrian.band_quantiles,
        "band_calibration_counts": cal_mondrian.band_counts,
        "reports": [r.__dict__ for r in reports],
    }


def _print(payload: dict[str, Any]) -> None:
    print("=" * 100)
    print("UNCERTAINTY CALIBRATION — STRICT ROLE SEPARATION".center(100))
    print("=" * 100)
    sp = payload["split"]
    print(f"dataset {payload['dataset_version']} · seed {payload['seed']} · "
          f"nominal coverage {1 - payload['alpha']:.2f}")
    print(f"roles: " + ", ".join(f"{k} {v}" for k, v in sp["sizes"].items()))
    print(f"{sp['description']}")
    print(f"median similarity to train — calibration {sp['median_similarity_calibration_to_train']:.3f}, "
          f"test {sp['median_similarity_test_to_train']:.3f}")

    ms = payload["model_selection"]
    print(f"\nmodel selection on the select split (test never consulted):")
    for c in sorted(ms["candidates"], key=lambda r: r["mae"]):
        r2 = "—" if c["r2"] is None else f"{c['r2']:.3f}"
        print(f"    {c['model']:20} MAE {c['mae']:.3f}  R2 {r2}")
    print(f"    chosen: {ms['chosen']}")

    print(f"\n{'-' * 100}")
    print(f"{'strategy':40} {'nominal':>8} {'empirical':>10} {'gap':>8} "
          f"{'width':>8} {'MAE':>7} {'RMSE':>7}")
    print("-" * 100)
    for r in payload["reports"]:
        verdict = "ok" if abs(r["empirical_coverage"] - r["nominal_coverage"]) <= 0.03 else "MISCALIBRATED"
        print(f"{r['strategy'][:39]:40} {r['nominal_coverage']:>8.2f} "
              f"{r['empirical_coverage']:>10.3f} "
              f"{r['empirical_coverage'] - r['nominal_coverage']:>+8.3f} "
              f"{r['median_interval_width']:>8.3f} {r['mae']:>7.3f} "
              f"{r['rmse']:>7.3f}   {verdict}")

    for r in payload["reports"]:
        print(f"\n{'-' * 100}\n{r['strategy']}\n{'-' * 100}")
        print("  coverage by similarity to training set:")
        for band, st in r["coverage_by_similarity"].items():
            if not st["n"]:
                print(f"      {band:10} n=0")
                continue
            print(f"      {band:10} n={st['n']:>4}  coverage {st['coverage']:.3f}  "
                  f"MAE {st['mae']:.3f}  median width {st['median_width']:.3f}")
        print("  coverage by modality:")
        for mod, st in r["coverage_by_modality"].items():
            if not st["n"]:
                print(f"      {mod:24} n=0")
                continue
            print(f"      {mod:24} n={st['n']:>4}  coverage {st['coverage']:.3f}  "
                  f"MAE {st['mae']:.3f}")

    print(f"\n{'-' * 100}\nlimitations\n{'-' * 100}")
    seen = set()
    for r in payload["reports"]:
        for lim in r["limitations"]:
            if lim in seen:
                continue
            seen.add(lim)
            print(f"  - {lim}")


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    payload = run()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False))
    _print(payload)
    print(f"\nreport written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
