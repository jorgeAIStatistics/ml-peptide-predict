"""A deployment gate for ranking, which is not the gate for estimating.

The affinity gate asks whether a predicted pKi may be published as a number.
It fails: cross-series MAE 0.870 against 0.70 required, R2 0.276 against 0.30.
That verdict stands and nothing here touches it.

Ranking is a different claim. "Of these two analogues, which binds tighter"
needs no absolute number to be useful, and the pairwise evaluation found the
same model answers it correctly 92% of the time within a series. A different
claim needs its own criteria and its own evidence, so this module defines them
and measures against them.

**The statistical point that shapes everything here.** The 92% figure was
computed over 5,880 held-out pairs, but those pairs are not independent: they
are drawn from roughly 150 molecules, each appearing in many pairs. A binomial
interval on 5,880 trials would be about +/-0.7% and badly overconfident. The
bound that matters is taken over *molecules* -- accuracy per held-out molecule,
then a bootstrap over molecules -- and this module reports the naive bound
alongside it so the difference is visible rather than assumed away.

**What the gate does not do.** Passing it licenses an ordering with a measured
accuracy attached, inside a stated applicability domain. It never licenses a
numeric affinity estimate, never makes a measurement, and never creates a
knowledge-graph edge. Ranking and estimating are different claims; only one of
them has evidence.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from atlas.core.evidence import utc_now
from atlas.ml.dataset import DATA_DIR
from atlas.ml.pairwise import MEANINGFUL_DELTA

logger = logging.getLogger(__name__)

__all__ = [
    "RANKING_GATE", "RANKING_PATH", "RankingEvaluation", "evaluate_ranking",
    "bootstrap_over_groups", "naive_binomial_bound",
]

RANKING_PATH = DATA_DIR / "ranking_gate.json"
SEED = 0
N_BOOTSTRAP = 2000

#: Criteria a ranking capability must clear before the console may offer it.
#: Chosen, not derived -- like the affinity gate's thresholds -- and justified:
#:
#: * 0.80 on the *lower* bound means the model is right four times for each
#:   error on calls that matter, against a coin's 0.50. A point estimate would
#:   not do: the question is what can be relied on, not what was observed once.
#: * Magnitude discrimination guards against a degenerate "ranker" that orders
#:   pairs correctly by series position while knowing nothing about size. If
#:   predicted magnitude does not track true magnitude, the ordering is a
#:   by-product rather than a capability.
#: * The pair-similarity floor is the applicability domain, and unlike the
#:   affinity model's it is checkable at prediction time from the two queries
#:   alone, with no label needed.
RANKING_GATE: dict[str, float] = {
    "within_series_accuracy_lower_bound_min": 0.80,
    "magnitude_discrimination_spearman_min": 0.30,
    "pair_similarity_min": 0.60,
    "meaningful_delta_log_units": MEANINGFUL_DELTA,
}


def naive_binomial_bound(successes: int, trials: int, *, alpha: float = 0.05) -> float:
    """Wilson lower bound treating every pair as independent.

    Reported only as the figure *not* to use, so the cost of the independence
    assumption is visible.
    """
    if trials == 0:
        return float("nan")
    from scipy.stats import norm

    z = float(norm.ppf(1 - alpha / 2))
    phat = successes / trials
    denom = 1 + z**2 / trials
    centre = phat + z**2 / (2 * trials)
    margin = z * np.sqrt(phat * (1 - phat) / trials + z**2 / (4 * trials**2))
    return float((centre - margin) / denom)


def bootstrap_over_groups(
    per_group: dict[Any, tuple[int, int]], *, alpha: float = 0.05,
    n_boot: int = N_BOOTSTRAP, seed: int = SEED,
) -> dict[str, float]:
    """Bootstrap a success rate by resampling *groups*, not trials.

    ``per_group`` maps a group key to (successes, trials). Resampling whole
    groups respects the dependence between pairs that share a molecule, which
    is what makes the resulting bound honest.
    """
    keys = [k for k, (_s, t) in per_group.items() if t > 0]
    if not keys:
        return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan"),
                "n_groups": 0}
    rng = np.random.default_rng(seed)
    successes = np.array([per_group[k][0] for k in keys], dtype=float)
    trials = np.array([per_group[k][1] for k in keys], dtype=float)

    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        # Pooled rate within the resample, so a molecule with many pairs
        # carries its weight without one molecule dominating the interval.
        means[b] = successes[pick].sum() / trials[pick].sum()
    return {
        "mean": float(successes.sum() / trials.sum()),
        "lower": float(np.percentile(means, 100 * alpha / 2)),
        "upper": float(np.percentile(means, 100 * (1 - alpha / 2))),
        "n_groups": len(keys),
        "median_trials_per_group": float(np.median(trials)),
    }


@dataclass
class RankingEvaluation:
    regime: str
    n_pairs: int
    n_meaningful_pairs: int
    n_served_pairs: int
    n_groups: int
    pair_weighted_accuracy: float            # on the served domain
    accuracy_all_meaningful_pairs: float = float("nan")
    clustered: dict[str, float] = field(default_factory=dict)
    clustered_all_meaningful_pairs: dict[str, float] = field(default_factory=dict)
    naive_lower_bound: float = float("nan")
    magnitude_spearman: float = float("nan")
    accuracy_by_delta_band: dict[str, Any] = field(default_factory=dict)
    accuracy_by_similarity_band: dict[str, Any] = field(default_factory=dict)
    criteria: dict[str, Any] = field(default_factory=dict)
    passes: bool = False


def _bands(values: np.ndarray, edges: tuple[float, ...]) -> list[tuple[str, np.ndarray]]:
    out = []
    for lo, hi in zip(edges, edges[1:] + (float("inf"),)):
        label = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}"
        out.append((label, (values >= lo) & (values < hi)))
    return out


def _evaluate_regime(
    regime: str, predicted: np.ndarray, truth: np.ndarray,
    groups: np.ndarray, pair_similarity: np.ndarray,
) -> RankingEvaluation:
    from scipy.stats import spearmanr

    meaningful = np.abs(truth) >= MEANINGFUL_DELTA
    correct = np.sign(predicted) == np.sign(truth)

    # The gate must be measured on the domain the service actually serves.
    # Cluster co-membership is not the same condition as mutual similarity:
    # the largest series holds 335 molecules within 0.4 distance of a centroid,
    # so two of its members can be far apart from each other. The deployed
    # mode checks mutual similarity, because that is what it can check from
    # two queries, so the quoted accuracy is computed under the same condition
    # rather than over all within-series pairs.
    served = meaningful & (pair_similarity >= RANKING_GATE["pair_similarity_min"])

    def grouped(sel: np.ndarray) -> dict[Any, tuple[int, int]]:
        out: dict[Any, tuple[int, int]] = {}
        for g in np.unique(groups[sel]):
            s_ = sel & (groups == g)
            out[int(g)] = (int(correct[s_].sum()), int(s_.sum()))
        return out

    clustered = bootstrap_over_groups(grouped(served))
    all_pairs_clustered = bootstrap_over_groups(grouped(meaningful))
    naive = naive_binomial_bound(
        int(correct[served].sum()), int(served.sum()))

    rho = (spearmanr(np.abs(predicted[served]), np.abs(truth[served])).statistic
           if served.sum() > 2 else float("nan"))

    by_delta = {}
    for label, sel in _bands(np.abs(truth), (0.0, 0.5, 1.0, 2.0)):
        if sel.any():
            by_delta[label] = {
                "n": int(sel.sum()),
                "accuracy": float(correct[sel].mean()),
            }
    # Per-band figures get their own clustered bound, because the service
    # quotes the band a pair actually falls in rather than the pooled figure.
    # A band accuracy without an interval would be quoted as if it were as
    # well established as the headline, and the >0.8 band is both the smallest
    # and the worst.
    by_sim = {}
    for label, sel in _bands(pair_similarity, (0.0, 0.4, 0.6, 0.8)):
        sel = sel & meaningful
        if sel.any():
            band = bootstrap_over_groups(grouped(sel))
            by_sim[label] = {
                "n": int(sel.sum()),
                "accuracy": float(correct[sel].mean()),
                "lower": band["lower"],
                "upper": band["upper"],
                "n_groups": band["n_groups"],
                "served": bool(
                    float(label.lstrip(">").split("-")[0])
                    >= RANKING_GATE["pair_similarity_min"]),
            }

    criteria = {
        "within_series_accuracy_lower_bound": {
            "actual": clustered["lower"],
            "required": f">= {RANKING_GATE['within_series_accuracy_lower_bound_min']}",
            "met": bool(clustered["lower"]
                        >= RANKING_GATE["within_series_accuracy_lower_bound_min"]),
        },
        "magnitude_discrimination_spearman": {
            "actual": float(rho) if np.isfinite(rho) else float("nan"),
            "required": f">= {RANKING_GATE['magnitude_discrimination_spearman_min']}",
            "met": bool(np.isfinite(rho)
                        and rho >= RANKING_GATE["magnitude_discrimination_spearman_min"]),
        },
    }
    return RankingEvaluation(
        regime=regime, n_pairs=int(len(truth)),
        n_meaningful_pairs=int(meaningful.sum()),
        n_served_pairs=int(served.sum()), n_groups=clustered["n_groups"],
        pair_weighted_accuracy=(
            float(correct[served].mean()) if served.any() else float("nan")),
        accuracy_all_meaningful_pairs=float(correct[meaningful].mean()),
        clustered=clustered, clustered_all_meaningful_pairs=all_pairs_clustered,
        naive_lower_bound=naive,
        magnitude_spearman=float(rho) if np.isfinite(rho) else float("nan"),
        accuracy_by_delta_band=by_delta, accuracy_by_similarity_band=by_sim,
        criteria=criteria,
        passes=all(c["met"] for c in criteria.values()),
    )


def evaluate_ranking(*, save: bool = True) -> dict[str, Any]:
    """Measure ranking reliability in both regimes and apply the gate."""
    from rdkit import Chem

    from atlas.ml.dataset import load_dataset
    from atlas.ml.features import featurize, morgan_matrix
    from atlas.ml.pairwise import (
        build_pairs, split_pairs_by_cluster, split_pairs_by_molecule,
    )
    from atlas.ml.peptide_benchmark import cross_series_groups
    from atlas.ml.peptide_representations import (
        build_representations, residue_vocabulary,
    )
    from atlas.ml.peptide_structure import analyse_peptide
    from sklearn.ensemble import RandomForestRegressor

    bundle = load_dataset()
    fs = featurize(bundle.records)
    analyses = [analyse_peptide(s) for s in fs.smiles]
    keep = np.array([a.is_peptide for a in analyses])
    pep_an = [a for a, k in zip(analyses, keep) if k]
    y = fs.y[keep]
    morgan = fs.fingerprints[keep].astype(np.float32)

    labels = cross_series_groups(morgan)
    pairs = build_pairs(y, labels)

    # Tanimoto between the two members of each pair: the applicability measure
    # the deployed mode can check from the queries alone.
    inter = morgan @ morgan.T
    sums = morgan.sum(axis=1, keepdims=True)
    tanimoto = inter / np.maximum(sums + sums.T - inter, 1e-9)
    pair_sim = np.array([
        float(tanimoto[a, b]) for a, b in zip(pairs.left, pairs.right)])

    regimes = {
        "within_series_held_out_molecules": split_pairs_by_molecule(pairs, len(y)),
        "cross_series_held_out_series": split_pairs_by_cluster(pairs, labels),
    }

    evaluations: dict[str, Any] = {}
    for regime, (tr_p, te_p) in regimes.items():
        if not len(te_p) or not len(tr_p):
            continue
        train_mols = np.unique(
            np.concatenate([pairs.left[tr_p], pairs.right[tr_p]]))
        vocab = residue_vocabulary([pep_an[i] for i in train_mols])
        mols = [Chem.MolFromSmiles(a.smiles) for a in pep_an]
        reps = build_representations(
            pep_an, vocabulary=vocab,
            extra={"morgan_fp": morgan_matrix(mols).astype(float)})
        x = reps["positional_fragment_fp+morgan_fp"]

        # Rank by subtracting two absolute predictions. The pairwise-trained
        # model was measurably worse at this (90.9% against 92.0%), so the
        # deployable ranker is the ordinary affinity model -- the same one that
        # fails the gate for publishing its own numbers.
        model = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
        model.fit(x[train_mols], y[train_mols])
        pred = model.predict(x)
        predicted = pred[pairs.left[te_p]] - pred[pairs.right[te_p]]

        # Groups are the held-out molecule of each pair, which is the unit the
        # dependence runs through.
        seen = set(train_mols.tolist())
        groups = np.array([
            a if int(a) not in seen else b
            for a, b in zip(pairs.left[te_p], pairs.right[te_p])
        ])
        evaluations[regime] = asdict(_evaluate_regime(
            regime, predicted, pairs.delta[te_p], groups, pair_sim[te_p]))

    within = evaluations.get("within_series_held_out_molecules", {})
    cross = evaluations.get("cross_series_held_out_series", {})
    report = {
        "dataset_version": bundle.version,
        "gate": RANKING_GATE,
        "ranker": "absolute affinity model, differences of two predictions",
        "evaluations": evaluations,
        "verdict": {
            "within_series_passes": bool(within.get("passes")),
            "cross_series_passes": bool(cross.get("passes")),
            "deployable_mode": (
                "within_series_ranking" if within.get("passes") else None),
        },
        "what_this_licenses": (
            "An ordering of two peptides of the same series, with the measured "
            "accuracy of the similarity band the pair falls in attached, and "
            "the applicability condition checked from the two queries. It "
            "licenses no numeric affinity estimate, makes no measurement, and "
            "creates no knowledge-graph edge. The affinity deployment gate is "
            "unchanged and still fails."
        ),
        "cliff_band": {
            "threshold": 0.8,
            "note": (
                "Pairs above 0.8 mutual similarity are the hardest: a "
                "near-identical pair with a real affinity gap is an activity "
                "cliff. Their band accuracy is lower than the pooled figure "
                "and is quoted in its place, with a warning, rather than "
                "letting the pooled number flatter the answer."
            ),
        },
        "built_at": utc_now().isoformat(),
        "findings": _findings(within, cross),
    }
    if save:
        RANKING_PATH.parent.mkdir(parents=True, exist_ok=True)
        RANKING_PATH.write_text(json.dumps(report, indent=2, default=float))
    return report


def _findings(within: dict[str, Any], cross: dict[str, Any]) -> list[str]:
    out: list[str] = []
    if within:
        c = within["clustered"]
        out.append(
            f"Within series, on the domain the service actually serves -- "
            f"pairs of mutual similarity at least "
            f"{RANKING_GATE['pair_similarity_min']} differing by at least "
            f"{MEANINGFUL_DELTA} log units -- the ranker is correct on "
            f"{within['pair_weighted_accuracy']:.1%} of "
            f"{within['n_served_pairs']} held-out pairs. "
            f"Bootstrapping over the {c['n_groups']} held-out molecules those "
            f"pairs come from, rather than over the pairs, gives a 95% "
            f"interval of [{c['lower']:.1%}, {c['upper']:.1%}]. A binomial "
            f"bound treating every pair as independent would have claimed "
            f"{within['naive_lower_bound']:.1%} -- "
            f"{abs(within['naive_lower_bound'] - c['lower']) * 100:.1f} points "
            "too confident, because a molecule appearing in "
            f"{c['median_trials_per_group']:.0f} pairs is one observation, not "
            f"{c['median_trials_per_group']:.0f}."
        )
        allm = within.get("clustered_all_meaningful_pairs", {})
        if allm:
            out.append(
                f"Restricting to the served domain matters: over *all* "
                f"{within['n_meaningful_pairs']} within-series meaningful "
                f"pairs the figure is "
                f"{within['accuracy_all_meaningful_pairs']:.1%} with a bound of "
                f"{allm['lower']:.1%}, because cluster co-membership is a looser "
                "condition than mutual similarity -- the largest series holds "
                "335 molecules within 0.4 distance of a centroid, so two of "
                "its members can be far apart from each other. The quoted "
                "figure is the one measured under the condition the service "
                "checks."
            )
        out.append(
            f"The ranker also discriminates magnitude: predicted and true "
            f"differences correlate at Spearman "
            f"{within['magnitude_spearman']:+.3f}, so the ordering is not a "
            "by-product of knowing a series' position. Accuracy rises with "
            "the size of the true difference — "
            + ", ".join(
                f"{band} log units: {v['accuracy']:.1%} of {v['n']}"
                for band, v in within["accuracy_by_delta_band"].items())
            + " — which is the behaviour a usable ranker should show."
        )
        bands = within.get("accuracy_by_similarity_band", {})
        if ">0.8" in bands and "0.6-0.8" in bands:
            hi, mid = bands[">0.8"], bands["0.6-0.8"]
            if hi["accuracy"] < mid["accuracy"]:
                out.append(
                    f"The hardest calls are between the *most* similar "
                    f"molecules: accuracy falls to {hi['accuracy']:.1%} on "
                    f"{hi['n']} pairs above 0.8 similarity, against "
                    f"{mid['accuracy']:.1%} on {mid['n']} pairs between 0.6 and "
                    "0.8. A near-identical pair with a real affinity gap is an "
                    "activity cliff, and the ranker is worst exactly where a "
                    "chemist most needs it. That is a limitation of the claim, "
                    "not a reason to withhold it, and it is reported with the "
                    "answer."
                )
    if cross:
        cc = cross["clustered"]
        out.append(
            f"Across unseen series the same ranker reaches "
            f"{cross['pair_weighted_accuracy']:.1%} with a clustered interval "
            f"of [{cc['lower']:.1%}, {cc['upper']:.1%}] over {cc['n_groups']} "
            f"groups, and magnitude discrimination of "
            f"{cross['magnitude_spearman']:+.3f}. It "
            + ("passes" if cross.get("passes") else "fails")
            + " the same criteria, which is what bounds the claim to within a "
            "series rather than leaving it open."
        )
    if within:
        out.append(
            f"Gate verdict: within-series ranking "
            + ("**passes**" if within.get("passes") else "**fails**")
            + "; "
            + "; ".join(
                f"{k} = {v['actual']:.3f} against {v['required']} "
                f"({'met' if v['met'] else 'not met'})"
                for k, v in within["criteria"].items())
            + ". Passing licenses an ordering with its accuracy attached, not "
            "a numeric affinity estimate. The affinity gate is untouched and "
            "still fails."
        )
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = evaluate_ranking()
    for i, f in enumerate(report["findings"], 1):
        print(f"\n[{i}] {f}")
    print(f"\nSaved to {RANKING_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
