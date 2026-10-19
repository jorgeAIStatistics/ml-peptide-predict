"""Train and evaluate a peptide-ligand affinity model for MC4R.

The small-molecule model refuses peptides, for a measured reason: fitted on
small molecules it scored worse than a mean predictor on held-out peptides.
That left the registry's own melanocortin peptides -- PT-141, melanotan I and
II -- unserved. This module asks whether a peptide-specific model does better,
and reports the answer either way.

The comparison that decides deployment is against two references, not one:

* the **mean predictor**, which any useful model must beat; and
* the **small-molecule model applied to these peptides**, which is the status
  quo the new model has to improve on.

Splits run from optimistic to honest. For peptides the scaffold split is itself
misleading -- it reports zero shared scaffolds while leaving every test molecule
with a near neighbour in training -- so the cluster split is the figure quoted.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from atlas.core.evidence import utc_now
from atlas.ml.dataset import load_dataset
from atlas.ml.features import FeatureSet, Modality, applicability, featurize
from atlas.ml.peptide import (
    HELM_VALIDATION,
    peptide_feature_names,
    peptide_matrix,
)
from atlas.ml.splits import Split, cluster_split, leakage_report, random_split, scaffold_split
from atlas.ml.train import CONFORMAL_ALPHA, SEED, _metrics, _num

logger = logging.getLogger(__name__)

__all__ = ["run_peptide_experiment", "PEPTIDE_REGISTRY_PATH"]

PEPTIDE_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "ml" / "peptide_model_registry.json"
)

#: Feature blocks compared. "peptide" is the residue-composition and topology
#: set; the point of the experiment is whether it adds anything over structure
#: fingerprints alone.
FEATURE_BLOCKS = ("peptide", "fingerprints", "combined")


def _build_models() -> dict[str, Any]:
    return {
        "baseline_mean": lambda: DummyRegressor(strategy="mean"),
        "ridge": lambda: Pipeline([
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ]),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            n_jobs=-1, random_state=SEED,
        ),
        "knn": lambda: KNeighborsRegressor(n_neighbors=5, weights="distance"),
    }


def _peptide_subset(fs: FeatureSet) -> tuple[FeatureSet, np.ndarray]:
    idx = np.where(fs.mask(Modality.PEPTIDE_LIKE))[0]
    sub = fs.subset(idx)
    return sub, peptide_matrix(sub.mols)


def _matrix(block: str, pep_feats: np.ndarray, fs: FeatureSet) -> np.ndarray:
    if block == "peptide":
        return pep_feats
    if block == "fingerprints":
        return fs.fingerprints.astype(float)
    if block == "combined":
        return np.hstack([pep_feats, fs.fingerprints.astype(float)])
    raise ValueError(block)


def _evaluate(
    name: str, build, X: np.ndarray, y: np.ndarray, split: Split, block: str,
    fingerprints: np.ndarray,
) -> dict[str, Any]:
    tr, te = split.train_idx, split.test_idx

    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(tr)
    n_cal = max(30, int(0.15 * len(tr)))
    cal_idx, fit_idx = shuffled[:n_cal], shuffled[n_cal:]

    model = build()
    model.fit(X[fit_idx], y[fit_idx])
    pred = model.predict(X[te])
    m = _metrics(y[te], pred)

    cal_resid = np.abs(model.predict(X[cal_idx]) - y[cal_idx])
    n = len(cal_resid)
    level = min(1.0, np.ceil((n + 1) * (1 - CONFORMAL_ALPHA)) / n)
    half = float(np.quantile(cal_resid, level))
    coverage = float(np.mean(np.abs(pred - y[te]) <= half))

    sim = applicability(fingerprints[fit_idx], fingerprints[te])
    return {
        "model": name,
        "features": block,
        "split": split.kind,
        "n_train": len(fit_idx),
        "n_test": len(te),
        "conformal_coverage": coverage,
        "conformal_half_width": half,
        "median_test_similarity": float(np.median(sim)),
        **m,
    }


def _small_molecule_reference(
    fs_all: FeatureSet, pep_fs: FeatureSet, split: Split
) -> dict[str, Any]:
    """The status quo: the deployed small-molecule model scoring these peptides.

    Fitted on small-molecule-like ligands only, then asked for the held-out
    peptides of this split. This is what the new model has to beat to be worth
    deploying.
    """
    small_idx = np.where(fs_all.mask(Modality.SMALL_MOLECULE_LIKE))[0]
    small = fs_all.subset(small_idx)
    model = KNeighborsRegressor(n_neighbors=5, metric="jaccard", weights="distance")
    model.fit(small.fingerprints.astype(bool), small.y)
    pred = model.predict(pep_fs.fingerprints[split.test_idx].astype(bool))
    m = _metrics(pep_fs.y[split.test_idx], pred)
    return {
        "model": "small_molecule_model_applied",
        "features": "fingerprints",
        "split": split.kind,
        "n_train": len(small_idx),
        "n_test": len(split.test_idx),
        "conformal_coverage": None,
        "conformal_half_width": None,
        "median_test_similarity": None,
        **m,
    }


def run_peptide_experiment(*, seed: int = SEED) -> dict[str, Any]:
    bundle = load_dataset()
    fs_all = featurize(bundle.records)
    pep_fs, pep_feats = _peptide_subset(fs_all)

    splits = [
        random_split(pep_fs, test_fraction=0.25, seed=seed),
        scaffold_split(pep_fs, test_fraction=0.25, seed=seed),
        cluster_split(pep_fs.fingerprints, test_fraction=0.25,
                      distance_cutoff=0.4, seed=seed),
    ]

    models = _build_models()
    results: list[dict[str, Any]] = []

    for split in splits:
        for block in FEATURE_BLOCKS:
            X = _matrix(block, pep_feats, pep_fs)
            for name, build in models.items():
                # The mean predictor ignores features; evaluate it once.
                if name == "baseline_mean" and block != "peptide":
                    continue
                logger.info("%s / %s / %s", split.kind, block, name)
                results.append(
                    _evaluate(name, build, X, pep_fs.y, split, block,
                              pep_fs.fingerprints)
                )
        results.append(_small_molecule_reference(fs_all, pep_fs, split))

    split_meta = []
    for s in splits:
        lk = leakage_report(pep_fs, s)
        sim = applicability(pep_fs.fingerprints[s.train_idx],
                            pep_fs.fingerprints[s.test_idx])
        split_meta.append({
            "kind": s.kind,
            "description": s.description,
            "n_train": s.n_train,
            "n_test": s.n_test,
            "leakage": lk,
            "median_test_similarity": float(np.median(sim)),
            "fraction_test_with_close_neighbour": float(np.mean(sim > 0.7)),
        })

    return {
        "created_at": utc_now().isoformat(),
        "dataset_version": bundle.version,
        "endpoint": bundle.endpoint,
        "population": "peptide-like ligands only",
        "n_peptides": len(pep_fs),
        "seed": seed,
        "conformal_alpha": CONFORMAL_ALPHA,
        "feature_blocks": list(FEATURE_BLOCKS),
        "n_peptide_features": pep_feats.shape[1],
        "peptide_feature_names": peptide_feature_names(),
        "helm_validation": HELM_VALIDATION,
        "splits": split_meta,
        "results": results,
    }


def _print_report(payload: dict[str, Any]) -> None:
    print("=" * 104)
    print("MC4R PEPTIDE-LIGAND AFFINITY (pKi) — HELD-OUT EVALUATION".center(104))
    print("=" * 104)
    print(f"{payload['n_peptides']} peptide-like ligands · dataset "
          f"{payload['dataset_version']} · {payload['n_peptide_features']} "
          f"peptide features · seed {payload['seed']}")
    hv = payload["helm_validation"]
    print(f"residue extractor validated against HELM: "
          f"{hv['n_exact_agreements']}/{hv['n_residue_type_comparisons']} "
          f"exact ({hv['accuracy']:.1%}) over {hv['n_molecules']} molecules")

    print("\nsplit characteristics:")
    for s in payload["splits"]:
        print(f"    {s['kind']:10} train {s['n_train']:>4} test {s['n_test']:>4}  "
              f"shared scaffolds {s['leakage']['n_shared_scaffolds']:>3}  "
              f"median test-to-train similarity {s['median_test_similarity']:.3f}  "
              f"{s['fraction_test_with_close_neighbour']:.0%} have a >0.7 neighbour")

    by_split: dict[str, list[dict]] = {}
    for r in payload["results"]:
        by_split.setdefault(r["split"], []).append(r)

    for split, rows in by_split.items():
        print(f"\n{'-' * 104}\n{split}\n{'-' * 104}")
        print(f"{'model':16} {'features':14} {'ntr':>5} {'nte':>5} {'MAE':>8} "
              f"{'RMSE':>8} {'R2':>8} {'spearman':>9} {'cover':>8}")
        base = next((r for r in rows if r["model"] == "baseline_mean"), None)
        ref = next((r for r in rows if r["model"] == "small_molecule_model_applied"), None)
        for r in sorted(rows, key=lambda x: x["mae"]):
            flag = ""
            if base and r["model"] not in ("baseline_mean",):
                flag = " *" if r["mae"] < base["mae"] else "  (worse than mean)"
            print(f"{r['model']:16} {r['features']:14} {r['n_train']:>5} "
                  f"{r['n_test']:>5} {r['mae']:>8.3f} {r['rmse']:>8.3f} "
                  f"{_num(r['r2'], 8)} {_num(r['spearman'], 9)} "
                  f"{_num(r['conformal_coverage'], 8)}{flag}")
        if base and ref:
            best = min((r for r in rows if r["model"] not in
                        ("baseline_mean", "small_molecule_model_applied")),
                       key=lambda x: x["mae"])
            print(f"\n    best peptide model : {best['model']} on {best['features']} "
                  f"— MAE {best['mae']:.3f}")
            print(f"    mean predictor     : MAE {base['mae']:.3f}  "
                  f"({'beaten' if best['mae'] < base['mae'] else 'NOT beaten'})")
            print(f"    small-molecule model applied to these peptides: "
                  f"MAE {ref['mae']:.3f}  "
                  f"({'improved on' if best['mae'] < ref['mae'] else 'NOT improved on'})")


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    payload = run_peptide_experiment()
    PEPTIDE_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    PEPTIDE_REGISTRY_PATH.write_text(json.dumps(payload, indent=2, allow_nan=False))
    _print_report(payload)
    print(f"\nregistry written to {PEPTIDE_REGISTRY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
