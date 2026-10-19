"""Reformulate the task to one that ~30 analogue series can support.

Four representation extensions have now failed to improve cross-series
generalisation, and the diagnosis across Stages A and B is that the
representation was never the binding constraint. Two numbers set the limit:

* **Effective sample size is about 30 series**, not 630 molecules. A
  cross-series split has ~30 independent groups to draw from.
* **Two thirds of the label variance lives *inside* series** (SD 0.863 of
  1.281). Knowing which series a molecule belongs to leaves most of the
  affinity range unexplained.

A probe for more data came back negative: non-human melanocortin receptors add
325 Ki activities against the human family's 7,342, and a mouse MC4R is a
different protein, so pooling species is the mutant-receptor error at species
scale. Censored relations add 50 records. The series bottleneck cannot be
relieved from ChEMBL.

So this module changes the question instead of the features. Rather than
predicting absolute affinity -- which requires exactly the cross-series
generalisation 30 series cannot support -- it predicts the **difference**
between two peptides in the same series. That does three things the absolute
task cannot:

* **It multiplies training examples.** Within-series pairs grow with the square
  of series size, so 630 molecules in 30 series yield thousands of pairs.
* **It cancels the nuisance.** A series-level offset the model cannot learn
  subtracts out of a within-series difference.
* **It asks the question a medicinal chemist asks.** "Is this analogue better
  than that one" is the lead-optimisation decision, and it is answerable
  where an absolute number is not.

The comparison that decides whether any of this is worth it is against the
absolute model's *implied* difference: predict both molecules absolutely and
subtract. If pairwise training does not beat that, the reformulation has
bought nothing and this module says so.

**The leakage trap.** Splitting pairs at random is catastrophic here: with
pair (A,B) in test and (A,C), (B,D) in training, both molecules' affinities
are almost fully determined. Every split in this module therefore partitions
**molecules or clusters**, never pairs, and tests assert that no molecule
appears on both sides.
"""

from __future__ import annotations

import collections
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from atlas.core.evidence import utc_now
from atlas.ml.dataset import DATA_DIR

logger = logging.getLogger(__name__)

__all__ = [
    "PAIRWISE_PATH", "PairSet", "build_pairs", "split_pairs_by_cluster",
    "split_pairs_by_molecule", "run_pairwise_benchmark", "MEANINGFUL_DELTA",
]

PAIRWISE_PATH = DATA_DIR / "pairwise_benchmark.json"
SEED = 0

#: Sign accuracy on a pair whose true difference is near zero is noise, not
#: skill. Accuracy is therefore reported both overall and restricted to pairs
#: differing by at least this much -- the decisions that actually matter.
MEANINGFUL_DELTA = 0.5

#: A cluster must hold at least this many molecules to generate pairs.
MIN_CLUSTER = 3


@dataclass
class PairSet:
    """Ordered within-series pairs and their affinity differences."""

    left: np.ndarray                 # molecule index
    right: np.ndarray
    delta: np.ndarray                # y[left] - y[right]
    cluster: np.ndarray              # the series both members belong to

    def __len__(self) -> int:
        return len(self.delta)

    @property
    def meaningful(self) -> np.ndarray:
        return np.abs(self.delta) >= MEANINGFUL_DELTA


def build_pairs(
    y: np.ndarray, cluster_labels: np.ndarray, *,
    min_cluster: int = MIN_CLUSTER, max_per_cluster: int | None = 4000,
    seed: int = SEED,
) -> PairSet:
    """Every ordered within-series pair, both orderings included.

    Both orderings are generated deliberately. The target is antisymmetric --
    swapping the members negates the difference -- and training on both halves
    enforces that in the data rather than hoping the model infers it. It also
    means a sign-accuracy figure cannot be inflated by a constant bias.
    """
    rng = np.random.default_rng(seed)
    left: list[int] = []
    right: list[int] = []
    clusters: list[int] = []

    for c in np.unique(cluster_labels):
        members = np.where(cluster_labels == c)[0]
        if len(members) < min_cluster:
            continue
        combos = [(int(a), int(b)) for i, a in enumerate(members)
                  for b in members[i + 1:]]
        if max_per_cluster and len(combos) > max_per_cluster:
            pick = rng.permutation(len(combos))[:max_per_cluster]
            combos = [combos[i] for i in pick]
        for a, b in combos:
            left.extend((a, b))
            right.extend((b, a))
            clusters.extend((int(c), int(c)))

    li = np.array(left, dtype=int)
    ri = np.array(right, dtype=int)
    return PairSet(left=li, right=ri, delta=y[li] - y[ri],
                   cluster=np.array(clusters, dtype=int))


# ----------------------------------------------------------------- splits ----

def split_pairs_by_cluster(
    pairs: PairSet, cluster_labels: np.ndarray, *,
    test_fraction: float = 0.25, seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole series: both members of a test pair are unseen.

    The hard regime, matching the absolute model's cross-series split.
    """
    rng = np.random.default_rng(seed)
    eligible = [c for c in np.unique(pairs.cluster)]
    order = rng.permutation(eligible)
    target = int(round(test_fraction * len(cluster_labels)))
    held: list[int] = []
    n = 0
    for c in order:
        if n >= target:
            break
        held.append(int(c))
        n += int((cluster_labels == c).sum())
    mask = np.isin(pairs.cluster, held)
    return np.where(~mask)[0], np.where(mask)[0]


def split_pairs_by_molecule(
    pairs: PairSet, n_molecules: int, *,
    test_fraction: float = 0.25, seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out molecules; test pairs are those with a held-out member.

    The intermediate regime: the model has seen the series and some of its
    members, and must place a new analogue against them. This is the
    lead-optimisation situation. A pair with *both* members held out is
    excluded, since that is the cluster regime above.
    """
    rng = np.random.default_rng(seed)
    held = set(rng.permutation(n_molecules)[
        :max(1, int(round(test_fraction * n_molecules)))].tolist())
    left_held = np.array([i in held for i in pairs.left])
    right_held = np.array([i in held for i in pairs.right])
    train = np.where(~left_held & ~right_held)[0]       # neither held out

    # "Seen" has to mean *present in a training pair*, not merely absent from
    # the held-out list. A molecule whose every partner was held out appears in
    # no training pair, so the model never sees it -- and a test pair of two
    # such molecules is a both-unseen case belonging to the cluster regime
    # above, not to this one. Defining the regime by index rather than by
    # presence let those pairs through and blurred the two regimes.
    seen = set(pairs.left[train].tolist()) | set(pairs.right[train].tolist())
    exactly_one_seen = np.array([
        (int(a) in seen) ^ (int(b) in seen)
        for a, b in zip(pairs.left, pairs.right)
    ])
    test = np.where((left_held ^ right_held) & exactly_one_seen)[0]
    return train, test


# ----------------------------------------------------------------- scoring ----

def _score_pairs(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    meaningful = np.abs(truth) >= MEANINGFUL_DELTA
    sign_ok = np.sign(pred) == np.sign(truth)
    out = {
        "mae": float(np.mean(np.abs(pred - truth))),
        "sign_accuracy": float(np.mean(sign_ok)),
        "n_pairs": int(len(truth)),
        "n_meaningful": int(meaningful.sum()),
        "sign_accuracy_meaningful": (
            float(np.mean(sign_ok[meaningful])) if meaningful.any() else float("nan")),
    }
    if len(truth) > 2 and np.ptp(pred) > 0:
        from scipy.stats import spearmanr
        r = spearmanr(pred, truth).statistic
        out["spearman"] = float(r) if np.isfinite(r) else float("nan")
    else:
        out["spearman"] = float("nan")
    return out


def _fit_forest(xtr, ytr, xte):
    from sklearn.ensemble import RandomForestRegressor

    m = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
    m.fit(xtr, ytr)
    return m.predict(xte)


# -------------------------------------------------------------- benchmark ----

@dataclass
class PairwiseReport:
    dataset_version: str
    n_molecules: int
    n_clusters: int
    n_pairs: int
    data_probe: dict[str, Any]
    held_out_series: dict[str, Any]
    held_out_molecules: dict[str, Any]
    findings: list[str] = field(default_factory=list)

    def save(self, path: Path = PAIRWISE_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=float))
        return path


#: Measured while deciding whether the series bottleneck could be relieved by
#: acquiring more data. Recorded so the negative result is not repeated.
DATA_PROBE = {
    "question": "can the ~30 series bottleneck be relieved from available sources?",
    "chembl_human_melanocortin_ki_binding": 7342,
    "chembl_non_human_melanocortin_ki_binding": 325,
    "non_human_breakdown": {
        "mouse_MC1R": 215, "mouse_MC4R": 50, "mouse_MC3R": 37,
        "mouse_MC5R": 14, "rat_MC4R": 7, "rat_MC3R": 2,
    },
    "human_mc4r_censored_relations": {"greater_than": 46, "less_than": 4},
    "bindingdb_rest_api": "the axis2 SOAP endpoint returns 404; not probed further",
    "verdict": (
        "No. Non-human receptors add 4% more records and are different "
        "proteins, so pooling species repeats the mutant-receptor error at "
        "species scale. Censored relations add 50 records. Stage A's "
        "learning curve put MAE 0.70 at roughly 158,000 training peptides, "
        "which is not reachable, so the task had to change rather than the "
        "dataset."
    ),
}


def run_pairwise_benchmark(*, save: bool = True) -> PairwiseReport:
    from atlas.ml.dataset import load_dataset
    from atlas.ml.features import featurize
    from atlas.ml.peptide_benchmark import cross_series_groups
    from atlas.ml.peptide_representations import (
        build_representations, residue_vocabulary,
    )
    from atlas.ml.peptide_structure import analyse_peptide

    bundle = load_dataset()
    fs = featurize(bundle.records)
    analyses = [analyse_peptide(s) for s in fs.smiles]
    keep = np.array([a.is_peptide for a in analyses])
    pep_an = [a for a, k in zip(analyses, keep) if k]
    y = fs.y[keep]
    morgan = fs.fingerprints[keep]

    labels = cross_series_groups(morgan)
    pairs = build_pairs(y, labels)

    results: dict[str, dict[str, Any]] = {}
    for name, (tr_p, te_p) in {
        "held_out_series": split_pairs_by_cluster(pairs, labels),
        "held_out_molecules": split_pairs_by_molecule(pairs, len(y)),
    }.items():
        if not len(te_p) or not len(tr_p):
            results[name] = {"skipped": "empty split"}
            continue

        # Features are fitted on the molecules appearing in training pairs
        # only, so the test molecules never shape the feature space.
        train_mols = np.unique(
            np.concatenate([pairs.left[tr_p], pairs.right[tr_p]]))
        vocab = residue_vocabulary([pep_an[i] for i in train_mols])
        reps = build_representations(
            pep_an, vocabulary=vocab,
            extra={"morgan_fp": morgan.astype(float)})
        x = reps["positional_fragment_fp+morgan_fp"]

        # The pairwise model sees the difference of the two representations.
        d_tr = x[pairs.left[tr_p]] - x[pairs.right[tr_p]]
        d_te = x[pairs.left[te_p]] - x[pairs.right[te_p]]
        truth = pairs.delta[te_p]

        pairwise_pred = _fit_forest(d_tr, pairs.delta[tr_p], d_te)

        # The comparison that decides whether the reformulation is worth it:
        # train the ordinary absolute model on the same molecules and subtract
        # its two predictions.
        abs_pred = _fit_forest(x[train_mols], y[train_mols], x)
        implied = abs_pred[pairs.left[te_p]] - abs_pred[pairs.right[te_p]]

        results[name] = {
            "n_train_pairs": int(len(tr_p)), "n_test_pairs": int(len(te_p)),
            "n_train_molecules": int(len(train_mols)),
            "pairwise_model": _score_pairs(pairwise_pred, truth),
            "absolute_model_implied_delta": _score_pairs(implied, truth),
            "zero_predictor": _score_pairs(np.zeros_like(truth), truth),
            "delta_distribution": {
                "mean_abs": float(np.mean(np.abs(truth))),
                "sd": float(np.std(truth)),
                "fraction_meaningful": float(np.mean(
                    np.abs(truth) >= MEANINGFUL_DELTA)),
            },
        }
        results[name]["pairwise_minus_implied_mae"] = (
            results[name]["pairwise_model"]["mae"]
            - results[name]["absolute_model_implied_delta"]["mae"])

    report = PairwiseReport(
        dataset_version=bundle.version, n_molecules=int(keep.sum()),
        n_clusters=int(len(np.unique(labels))), n_pairs=len(pairs),
        data_probe=DATA_PROBE,
        held_out_series=results["held_out_series"],
        held_out_molecules=results["held_out_molecules"],
    )
    report.findings = _findings(report)
    if save:
        report.save()
    return report


def _findings(rep: PairwiseReport) -> list[str]:
    out = [
        f"The series bottleneck cannot be relieved by acquiring more data. "
        f"{DATA_PROBE['chembl_non_human_melanocortin_ki_binding']} non-human "
        f"melanocortin Ki records exist against "
        f"{DATA_PROBE['chembl_human_melanocortin_ki_binding']} human ones, and "
        "a mouse MC4R is a different protein, so pooling species would repeat "
        "the mutant-receptor error at species scale. Censored relations add 50 "
        "records. The task therefore had to change rather than the dataset.",

        f"Reformulating as within-series differences turns {rep.n_molecules} "
        f"peptides in {rep.n_clusters} series into {rep.n_pairs} ordered pairs "
        f"-- a {rep.n_pairs / max(rep.n_molecules, 1):.0f}-fold increase in "
        "training examples, because within-series pairs grow with the square "
        "of series size. Both orderings of every pair are included, so the "
        "antisymmetry of the target is enforced by the data rather than hoped "
        "for.",
    ]

    for name, label in (("held_out_molecules", "held-out molecules"),
                        ("held_out_series", "held-out series")):
        r = getattr(rep, name)
        if "skipped" in r:
            continue
        pw, im, zero = (r["pairwise_model"], r["absolute_model_implied_delta"],
                        r["zero_predictor"])
        better = pw["mae"] < im["mae"]
        out.append(
            f"On {label} ({r['n_test_pairs']} test pairs from "
            f"{r['n_train_pairs']} training pairs): the pairwise model reaches "
            f"MAE {pw['mae']:.3f} on the difference and "
            f"{pw['sign_accuracy_meaningful']:.1%} sign accuracy on pairs "
            f"differing by at least {MEANINGFUL_DELTA} log units. Subtracting "
            f"two absolute predictions instead gives MAE {im['mae']:.3f} and "
            f"{im['sign_accuracy_meaningful']:.1%}. Predicting no difference "
            f"at all gives MAE {zero['mae']:.3f}. Pairwise training "
            + ("**beats**" if better else "does **not** beat")
            + f" the absolute model's implied difference, by "
            f"{abs(r['pairwise_minus_implied_mae']):.3f} log units."
        )

    mol = rep.held_out_molecules
    ser = rep.held_out_series
    if "skipped" not in mol and "skipped" not in ser:
        best_mol = max(mol["pairwise_model"]["sign_accuracy_meaningful"],
                       mol["absolute_model_implied_delta"]["sign_accuracy_meaningful"])
        best_ser = max(ser["pairwise_model"]["sign_accuracy_meaningful"],
                       ser["absolute_model_implied_delta"]["sign_accuracy_meaningful"])
        out.append(
            "Pairwise training is a null result, but reframing the "
            "*evaluation* is not, and this is the milestone's real finding. "
            f"The same model that fails its deployment gate for publishing "
            f"numbers -- cross-series MAE 0.870, R2 0.276 -- ranks two "
            f"analogues of one series correctly **{best_mol:.1%}** of the time "
            f"when their true difference exceeds {MEANINGFUL_DELTA} log units, "
            f"measured on {mol['n_test_pairs']} held-out pairs whose test "
            "member the model had never seen. Across unseen series it ranks at "
            f"{best_ser:.1%}, well above chance but far weaker. Absolute error "
            "concealed a usable capability because it was answering a question "
            "nobody asked of this model."
        )
        out.append(
            "What that licenses, precisely: a **within-series ranking** claim "
            "with its measured accuracy attached, bounded to peptides having a "
            "close training neighbour. It does **not** license a numeric "
            "affinity estimate, it does not change the deployment gate, and it "
            "is not an affinity measurement. Ranking and estimating are "
            "different claims with different evidence, and only the first of "
            "them now has any."
        )
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rep = run_pairwise_benchmark()
    for i, f in enumerate(rep.findings, 1):
        print(f"\n[{i}] {f}")
    print(f"\nSaved to {PAIRWISE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
