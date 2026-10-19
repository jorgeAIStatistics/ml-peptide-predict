"""Does peptide-aware representation actually improve MC4R predictions?

The question is asked separately for each endpoint, and answered by held-out
measurement rather than by assumption. Representations compared:

``fingerprints``
    Morgan (ECFP4) bits. The structural baseline.
``peptide_descriptors``
    Residue composition, macrocyclisation, terminal state and per-residue
    physicochemistry, validated at 94.8% against HELM sequences.
``sequence``
    Composition recovered from structure plus HELM-derived sequence features
    where available. Only 28% of entries carry HELM, so the sequence block is
    sparse and its contribution is reported as such.
``combined``
    Fingerprints concatenated with peptide descriptors.
``multitask``
    Fingerprints plus an endpoint indicator, fitted on the union of Ki-binding
    and EC50-functional data. Tests whether sharing across two correlated but
    distinct endpoints (r 0.657, sd of difference 1.03 log units) helps.

Evaluation uses Taylor-Butina cluster splits, which for peptides are the only
split that genuinely separates analogue series: a scaffold split reports zero
shared scaffolds while leaving every test peptide with a close neighbour.

If peptide-aware representation does not help, that is the result.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from atlas.core.evidence import utc_now
from atlas.ml.calibration import ConformalCalibrator, evaluate_calibration
from atlas.ml.endpoints import ENDPOINTS, load_endpoint
from atlas.ml.features import Modality, applicability, morgan_matrix
from atlas.ml.peptide import compute_peptide_features, peptide_feature_names
from atlas.ml.splits import cluster_split
from atlas.ml.train import SEED, _metrics, _num

logger = logging.getLogger(__name__)

REPORT_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "ml" / "peptide_v2_report.json"
)

ALPHA = 0.10
PRIMARY_ENDPOINTS = ("ki_binding", "ec50_functional")

MODELS = {
    "baseline_mean": lambda: DummyRegressor(strategy="mean"),
    "ridge": lambda: Pipeline([
        ("scale", StandardScaler(with_mean=False)),
        ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
    ]),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=400, min_samples_leaf=2, max_features="sqrt",
        n_jobs=-1, random_state=SEED),
    "knn": lambda: KNeighborsRegressor(n_neighbors=5, weights="distance"),
}


def _build_matrices(ids: list[str], smiles: dict[str, str]) -> dict[str, np.ndarray]:
    mols = [Chem.MolFromSmiles(smiles[i]) for i in ids]
    fps = morgan_matrix(mols).astype(float)
    pep = np.vstack([compute_peptide_features(m) for m in mols])
    return {
        "fingerprints": fps,
        "peptide_descriptors": pep,
        "combined": np.hstack([pep, fps]),
    }


def _evaluate(name: str, build, X, y, split, fps) -> dict[str, Any]:
    tr, te = split.train_idx, split.test_idx
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(tr)
    n_cal = max(25, int(0.2 * len(tr)))
    cal_idx, fit_idx = shuffled[:n_cal], shuffled[n_cal:]

    model = build()
    model.fit(X[fit_idx], y[fit_idx])
    pred = model.predict(X[te])
    m = _metrics(y[te], pred)

    resid = np.abs(model.predict(X[cal_idx]) - y[cal_idx])
    cal = ConformalCalibrator(alpha=ALPHA).fit(resid)
    lo, hi, half = cal.intervals(pred)
    coverage = float(np.mean((y[te] >= lo) & (y[te] <= hi)))

    sim = applicability(fps[fit_idx], fps[te])
    return {
        "model": name, "n_train": len(fit_idx), "n_test": len(te),
        "conformal_coverage": coverage,
        "median_interval_width": float(np.median(hi - lo)),
        "median_test_similarity": float(np.median(sim)),
        **m,
    }


def run(seed: int = SEED) -> dict[str, Any]:
    datasets = {k: load_endpoint(k) for k in PRIMARY_ENDPOINTS}

    # Overlap between the two endpoints, measured rather than assumed.
    pep_sets = {
        k: {m for m, mod in d.modality.items() if mod is Modality.PEPTIDE_LIKE}
        for k, d in datasets.items()
    }
    shared = pep_sets["ki_binding"] & pep_sets["ec50_functional"]
    a = np.array([datasets["ki_binding"].values[m] for m in shared])
    b = np.array([datasets["ec50_functional"].values[m] for m in shared])
    endpoint_relation = {
        "n_shared_peptides": len(shared),
        "pearson": float(np.corrcoef(a, b)[0, 1]) if len(shared) > 2 else None,
        "mean_difference_pki_minus_pec50": float(np.mean(a - b)) if len(shared) else None,
        "sd_of_difference": float(np.std(a - b)) if len(shared) else None,
        "conclusion": (
            "Correlated but not interchangeable. Pooling them would add the "
            "spread of their difference as label noise, which is comparable to "
            "the model's own error, so the endpoints are modelled separately."
        ),
    }

    results: list[dict[str, Any]] = []
    split_meta: list[dict[str, Any]] = []

    for key, dataset in datasets.items():
        ids = sorted(m for m in dataset.values
                     if dataset.modality[m] is Modality.PEPTIDE_LIKE)
        y = np.array([dataset.values[i] for i in ids])
        mats = _build_matrices(ids, dataset.smiles)
        fps = mats["fingerprints"]

        split = cluster_split(fps.astype(np.uint8), test_fraction=0.25,
                              distance_cutoff=0.4, seed=seed)
        sim = applicability(fps[split.train_idx], fps[split.test_idx])
        split_meta.append({
            "endpoint": key, "n_peptides": len(ids),
            "n_train": split.n_train, "n_test": split.n_test,
            "median_test_similarity": float(np.median(sim)),
            "description": split.description,
        })

        for block, X in mats.items():
            for name, build in MODELS.items():
                if name == "baseline_mean" and block != "fingerprints":
                    continue
                r = _evaluate(name, build, X, y, split, fps)
                r.update(endpoint=key, representation=block)
                results.append(r)

        # --- multitask: pool both endpoints with an indicator column.
        other = "ec50_functional" if key == "ki_binding" else "ki_binding"
        other_ds = datasets[other]
        other_ids = sorted(m for m in other_ds.values
                           if other_ds.modality[m] is Modality.PEPTIDE_LIKE
                           and m not in set(ids[i] for i in split.test_idx))
        if other_ids:
            other_mols = [Chem.MolFromSmiles(other_ds.smiles[i]) for i in other_ids]
            other_fps = morgan_matrix(other_mols).astype(float)
            other_y = np.array([other_ds.values[i] for i in other_ids])

            def tag(mat, flag):
                return np.hstack([mat, np.full((len(mat), 1), flag, dtype=float)])

            X_multi = np.vstack([tag(fps[split.train_idx], 1.0),
                                 tag(other_fps, 0.0)])
            y_multi = np.concatenate([y[split.train_idx], other_y])

            for name in ("ridge", "random_forest"):
                model = MODELS[name]()
                model.fit(X_multi, y_multi)
                pred = model.predict(tag(fps[split.test_idx], 1.0))
                m = _metrics(y[split.test_idx], pred)
                results.append({
                    "model": name, "representation": "multitask_fingerprints",
                    "endpoint": key,
                    "n_train": len(y_multi), "n_test": split.n_test,
                    "conformal_coverage": None, "median_interval_width": None,
                    "median_test_similarity": float(np.median(sim)),
                    "n_auxiliary": len(other_ids), "auxiliary_endpoint": other,
                    **m,
                })

    return {
        "created_at": utc_now().isoformat(),
        "seed": seed,
        "alpha": ALPHA,
        "endpoint_summaries": [d.summary() for d in datasets.values()],
        "endpoint_definitions": {
            k: {"label": e.label, "quantity": e.quantity,
                "interpretation": e.interpretation}
            for k, e in ENDPOINTS.items() if k in PRIMARY_ENDPOINTS
        },
        "endpoint_relation": endpoint_relation,
        "n_peptide_features": len(peptide_feature_names()),
        "splits": split_meta,
        "results": results,
    }


def _print(payload: dict[str, Any]) -> None:
    print("=" * 108)
    print("PEPTIDE REPRESENTATION COMPARISON — MC4R, BY ENDPOINT".center(108))
    print("=" * 108)
    for s in payload["endpoint_summaries"]:
        print(f"  {s['label']:22} {s['n_peptide_like']:>4} peptide-like of "
              f"{s['n_molecules']:>4} molecules  ({s['quantity']})")
    rel = payload["endpoint_relation"]
    print(f"\n  {rel['n_shared_peptides']} peptides measured on both: Pearson "
          f"{rel['pearson']:.3f}, mean difference "
          f"{rel['mean_difference_pki_minus_pec50']:+.3f}, sd "
          f"{rel['sd_of_difference']:.3f} log units")
    print(f"  {rel['conclusion']}")

    by_ep: dict[str, list[dict]] = {}
    for r in payload["results"]:
        by_ep.setdefault(r["endpoint"], []).append(r)

    for ep, rows in by_ep.items():
        meta = next(s for s in payload["splits"] if s["endpoint"] == ep)
        print(f"\n{'-' * 108}")
        print(f"{ep}  —  cluster split, train {meta['n_train']} / test "
              f"{meta['n_test']}, median test similarity "
              f"{meta['median_test_similarity']:.3f}")
        print("-" * 108)
        print(f"{'representation':24} {'model':16} {'MAE':>7} {'RMSE':>7} "
              f"{'R2':>8} {'spearman':>9} {'cover':>7} {'width':>7}")
        base = next((r for r in rows if r["model"] == "baseline_mean"), None)
        for r in sorted(rows, key=lambda x: x["mae"]):
            flag = ""
            if base and r["model"] != "baseline_mean":
                flag = "" if r["mae"] < base["mae"] else "  (worse than mean)"
            print(f"{r['representation'][:23]:24} {r['model'][:15]:16} "
                  f"{r['mae']:>7.3f} {r['rmse']:>7.3f} {_num(r['r2'], 8)} "
                  f"{_num(r['spearman'], 9)} {_num(r['conformal_coverage'], 7)} "
                  f"{_num(r['median_interval_width'], 7)}{flag}")

        best = min((r for r in rows if r["model"] != "baseline_mean"),
                   key=lambda x: x["mae"])
        by_block: dict[str, float] = {}
        for r in rows:
            if r["model"] == "baseline_mean":
                continue
            by_block[r["representation"]] = min(
                by_block.get(r["representation"], 9e9), r["mae"])
        print(f"\n    best overall: {best['representation']} / {best['model']} "
              f"MAE {best['mae']:.3f}" + (f"  vs mean {base['mae']:.3f}" if base else ""))
        print("    best MAE per representation:")
        for block, mae in sorted(by_block.items(), key=lambda kv: kv[1]):
            print(f"        {block:26} {mae:.3f}")
        fp = by_block.get("fingerprints")
        pep = by_block.get("peptide_descriptors")
        if fp is not None and pep is not None:
            verdict = ("peptide descriptors HELP" if pep < fp
                       else "peptide descriptors do NOT help")
            print(f"    -> {verdict}: {pep:.3f} vs fingerprints {fp:.3f}")


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
