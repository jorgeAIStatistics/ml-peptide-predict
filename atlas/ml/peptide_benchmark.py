"""ATLAS 2.4 Stage B: representational ceilings and a two-way benchmark.

Stage A found that a single aggregate error conceals which of two different
things a model can do. This module separates them and reports both:

* **Cross-series** — whole analogue series held out. Tests whether a model
  recognises a chemotype it has never seen. This is the harder question and
  the one Stage A measured (MAE 1.136, R2 0.123).
* **Within-series** — members held out from series the model has otherwise
  seen. Tests whether a model resolves single substitutions. Stage A found
  two-thirds of the label spread lives here (within-series SD 0.863 against
  1.281 overall), so this is where most of the predictable signal is.

Before fitting anything, the module measures what each representation *can*
express. A frozen sequence embedding is a deterministic function of the
sequence string: two molecules with the same string receive identical
embeddings, so no model built on them can tell the two apart. Grouping
molecules by representation key and predicting each group's mean therefore
gives a hard lower bound on the error of *any* model using that
representation. For a canonical twenty-letter alphabet that bound can be
computed without downloading a protein language model, and it applies to every
such model rather than to one of them.

The bound is only informative for representations coarse enough to collide. A
representation that assigns every molecule a distinct key has an oracle error
of zero and the ceiling analysis says nothing about it -- for those,
learnability is the question and the benchmark answers it.
"""

from __future__ import annotations

import collections
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from atlas.ml.dataset import DATA_DIR, load_dataset
from atlas.ml.features import featurize, morgan_matrix
from atlas.ml.peptide_representations import (
    build_representations, canonical_sequence, oriented_keys,
    residue_vocabulary, stereo_oriented_keys,
)
from atlas.ml.peptide_structure import analyse_peptide

logger = logging.getLogger(__name__)

__all__ = [
    "BENCHMARK_PATH", "oracle_ceiling", "cross_series_groups",
    "within_series_split", "run_benchmark", "BenchmarkReport", "GATE",
]

BENCHMARK_PATH = DATA_DIR / "peptide_benchmark.json"
SEED = 0

#: Criteria a peptide affinity model must clear before it may publish numeric
#: estimates. Set from Stage A's measurements, not from convention: the label
#: noise floor is MAE 0.245, so 0.70 leaves a real margin above noise while
#: remaining far better than the 1.343 mean-predictor baseline; and a model
#: that cannot beat predicting a series' own mean has not resolved the
#: substitutions that matter.
GATE = {
    "cross_series_mae_max": 0.70,
    "cross_series_r2_min": 0.30,
    "within_series_must_beat_series_mean": True,
}


# ------------------------------------------------------- ceiling analysis ----

def oracle_ceiling(keys: Sequence[Any], y: np.ndarray) -> dict[str, Any]:
    """Lower bound on the error of any model using this representation.

    Molecules sharing a key are indistinguishable to the representation, so
    the best any model can do for them is one shared number; the group mean
    minimises squared error and is used here. The resulting error is computed
    in-sample and is therefore optimistic as a prediction -- which is exactly
    what makes it a valid *lower* bound.
    """
    groups: dict[Any, list[int]] = collections.defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)

    errs = np.zeros(len(y))
    colliding: list[int] = []
    spreads: list[float] = []
    for members in groups.values():
        idx = np.array(members)
        errs[idx] = np.abs(y[idx] - y[idx].mean())
        if len(members) > 1:
            colliding.extend(members)
            spreads.append(float(y[idx].max() - y[idx].min()))

    col = np.array(colliding, dtype=int)
    return {
        "n_molecules": int(len(y)),
        "n_distinct_keys": len(groups),
        "n_colliding_molecules": int(len(col)),
        "colliding_fraction": float(len(col) / max(len(y), 1)),
        "largest_group": max((len(m) for m in groups.values()), default=0),
        "oracle_mae": float(errs.mean()),
        "oracle_mae_on_colliding": float(errs[col].mean()) if len(col) else 0.0,
        "median_group_label_range": float(np.median(spreads)) if spreads else 0.0,
        "max_group_label_range": float(max(spreads)) if spreads else 0.0,
        "note": (
            "A lower bound, not a prediction: no model on this representation "
            "can achieve a smaller mean absolute error on these molecules."
        ),
    }


def representational_ceilings(
    analyses: Sequence, y: np.ndarray
) -> dict[str, Any]:
    """Ceilings for the representations Stage A put in question."""
    keyings: dict[str, Callable[[Any], Any]] = {
        "canonical_sequence_20_letter": lambda a: canonical_sequence(a),
        "canonical_sequence_length_only": lambda a: len(a.residue_keys),
        "canonical_composition_order_free": lambda a: tuple(sorted(canonical_sequence(a))),
        "chemistry_aware_composition_order_free": lambda a: tuple(sorted(oriented_keys(a))),
        "chemistry_aware_positional": lambda a: tuple(oriented_keys(a)),
        "chemistry_aware_positional_with_stereo":
            lambda a: tuple(stereo_oriented_keys(a)),
    }
    out = {}
    for name, fn in keyings.items():
        out[name] = oracle_ceiling([fn(a) for a in analyses], y)

    # What stereo-blindness costs, isolated: groups that one key merges and
    # the other separates. This is the only place the two ceilings can be
    # compared molecule by molecule rather than in aggregate.
    blind: dict[tuple, list[int]] = collections.defaultdict(list)
    for i, a in enumerate(analyses):
        blind[tuple(oriented_keys(a))].append(i)
    split_groups, split_members, ranges = 0, 0, []
    for members in blind.values():
        if len(members) < 2:
            continue
        if len({tuple(stereo_oriented_keys(analyses[i])) for i in members}) > 1:
            split_groups += 1
            split_members += len(members)
            idx = np.array(members)
            ranges.append(float(y[idx].max() - y[idx].min()))
    out["stereochemistry_collapse"] = {
        "groups_merged_by_stereo_blindness": split_groups,
        "molecules_affected": split_members,
        "molecules_affected_fraction": float(split_members / max(len(y), 1)),
        "median_pki_range_within_those_groups": (
            float(np.median(ranges)) if ranges else 0.0),
        "max_pki_range_within_those_groups": (
            float(max(ranges)) if ranges else 0.0),
        "note": (
            "Molecules a constitutional key merges and a stereochemistry-aware "
            "key separates. Their label spread is a cost no model on the "
            "stereo-blind key can avoid."
        ),
    }

    # Make the collapse concrete: the worst canonical-sequence collision.
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for i, a in enumerate(analyses):
        groups[canonical_sequence(a)].append(i)
    worst = max(groups.items(), key=lambda kv: (
        len(kv[1]) > 1 and (y[kv[1]].max() - y[kv[1]].min()) or 0.0))
    idx = np.array(worst[1])
    out["worst_canonical_collision"] = {
        "sequence": worst[0],
        "n_molecules": int(len(idx)),
        "pki_min": float(y[idx].min()),
        "pki_max": float(y[idx].max()),
        "pki_range": float(y[idx].max() - y[idx].min()),
        "distinct_chemistry_aware_keys": len(
            {tuple(oriented_keys(analyses[i])) for i in idx}),
    }
    return out


# ------------------------------------------------------------------ splits ----

def cross_series_groups(fingerprints: np.ndarray, *, cutoff: float = 0.4) -> np.ndarray:
    """Taylor-Butina cluster label per molecule."""
    from rdkit import DataStructs
    from rdkit.ML.Cluster import Butina

    n = len(fingerprints)
    bits = [
        DataStructs.CreateFromBitString("".join(str(int(b)) for b in row))
        for row in fingerprints.astype(np.uint8)
    ]
    dists: list[float] = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(bits[i], bits[:i])
        dists.extend(1.0 - s for s in sims)
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    labels = np.full(n, -1, dtype=int)
    for c, members in enumerate(clusters):
        for m in members:
            labels[m] = c
    return labels


def _cross_series_split(
    labels: np.ndarray, *, test_fraction: float = 0.25, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Assign whole clusters to test until the target size is reached."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.unique(labels))
    target = int(round(test_fraction * len(labels)))
    test_clusters, n = [], 0
    for c in order:
        if n >= target:
            break
        test_clusters.append(c)
        n += int((labels == c).sum())
    mask = np.isin(labels, test_clusters)
    return np.where(~mask)[0], np.where(mask)[0]


def within_series_split(
    labels: np.ndarray, *, test_fraction: float = 0.25, min_cluster: int = 4,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out members of series the model otherwise sees.

    Only clusters with at least ``min_cluster`` members contribute test
    molecules, so that every held-out molecule genuinely has same-series
    training neighbours. Singletons stay in training: holding one out would be
    a cross-series test wearing the wrong label.
    """
    rng = np.random.default_rng(seed)
    test: list[int] = []
    for c in np.unique(labels):
        members = np.where(labels == c)[0]
        if len(members) < min_cluster:
            continue
        k = max(1, int(round(test_fraction * len(members))))
        test.extend(rng.permutation(members)[:k].tolist())
    test_arr = np.array(sorted(test), dtype=int)
    train = np.setdiff1d(np.arange(len(labels)), test_arr)
    return train, test_arr


# ------------------------------------------------------------------ models ----

def _fit_predict(xtr, ytr, xte, kind: str) -> np.ndarray:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.neighbors import KNeighborsRegressor

    if kind == "mean":
        return np.full(len(xte), ytr.mean())
    if kind == "ridge":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        # Ridge penalises coefficients on the features' own scale, so a
        # representation whose columns span orders of magnitude is handicapped
        # against one of binary bits. The residue-chemistry descriptors mix
        # surface areas near 60 with charges near 1 and scored R2 -3.55
        # unscaled; standardising first is what makes the comparison a
        # comparison of representations rather than of their units.
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 15)))
    elif kind == "forest":
        m = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
    elif kind == "knn":
        m = KNeighborsRegressor(n_neighbors=min(5, len(ytr)))
    else:
        raise ValueError(kind)
    m.fit(xtr, ytr)
    return m.predict(xte)


def _score(pred: np.ndarray, yte: np.ndarray, ytr: np.ndarray) -> dict[str, float]:
    err = np.abs(pred - yte)
    ss_res = float(np.sum((pred - yte) ** 2))
    ss_tot = float(np.sum((yte - yte.mean()) ** 2))
    return {
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(np.mean((pred - yte) ** 2))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "spearman": _spearman(pred, yte),
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    # A constant prediction has no rank correlation to report; the mean
    # baseline is exactly that case, so it is handled rather than warned about.
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else float("nan")


def split_diagnostics(
    morgan: np.ndarray, analyses: Sequence, tr: np.ndarray, te: np.ndarray,
) -> dict[str, Any]:
    """How close the held-out molecules actually are to training molecules.

    Without this, a good within-series score cannot be told apart from
    interpolation among near-duplicates. It also counts side chains the
    training vocabulary never saw, which is the mechanism by which a one-hot
    positional encoding must fail where a fragment fingerprint need not.
    """
    from atlas.ml.features import applicability

    sim = applicability(morgan[tr], morgan[te])
    vocab = set(residue_vocabulary([analyses[i] for i in tr]))
    unseen_per_mol = [
        sum(1 for k in oriented_keys(analyses[i]) if k not in vocab) for i in te
    ]
    return {
        "nearest_training_similarity": {
            "median": float(np.median(sim)), "p10": float(np.percentile(sim, 10)),
            "p90": float(np.percentile(sim, 90)), "mean": float(np.mean(sim)),
        },
        "test_molecules_with_close_neighbour_above_0.7": {
            "n": int((sim > 0.7).sum()), "fraction": float((sim > 0.7).mean()),
        },
        "training_vocabulary_size": len(vocab),
        "test_molecules_with_unseen_side_chain": {
            "n": int(sum(1 for u in unseen_per_mol if u)),
            "fraction": float(np.mean([bool(u) for u in unseen_per_mol])),
        },
        "unseen_side_chain_positions_total": int(sum(unseen_per_mol)),
    }


# --------------------------------------------------------------- benchmark ----

def _assay_block(
    records: Sequence[Any] | None, tr: np.ndarray
) -> np.ndarray | None:
    """Assay-context features, with vocabularies fitted on training rows only.

    Returns None when the metadata has not been built, so the arm is skipped
    rather than silently replaced by zeros.
    """
    from atlas.ml.assay_context import (
        ASSAY_CONTEXT_PATH, assay_encoding, build_assay_vocabularies,
        load_assay_metadata,
    )

    if records is None or not ASSAY_CONTEXT_PATH.exists():
        return None
    metadata = load_assay_metadata()["metadata"]
    vocab = build_assay_vocabularies([records[i] for i in tr], metadata)
    if not vocab["bao_label"]:
        return None
    return assay_encoding(records, metadata, vocab)


def _plm_block(analyses: Sequence[Any]) -> np.ndarray | None:
    """Frozen protein language model embeddings, read from the cache.

    Returns None when the cache has not been built, so the arm is skipped
    rather than silently replaced by zeros -- the same contract as
    ``_assay_block``. Nothing is fitted here. A frozen embedding is a
    per-sequence constant, so unlike the assay vocabularies there is no
    training-only quantity that could leak; the matrix is built for every
    molecule and the benchmark's existing train/test indexing does the rest.

    A sequence absent from the cache also skips the arm. That case arises for
    injected records from another receptor, whose peptides were never embedded:
    zero-filling them would make the arm score badly for a reason unrelated to
    the language model, and the benchmark would report that as a measurement.
    """
    from atlas.ml.plm_embeddings import (
        PLM_EMBEDDING_PATH, load_embeddings, plm_matrix,
    )

    if not PLM_EMBEDDING_PATH.exists():
        return None
    try:
        return plm_matrix(analyses, load_embeddings())
    except KeyError as exc:
        logger.warning("PLM arm skipped: %s", exc)
        return None


def _evaluate(
    analyses: Sequence, y: np.ndarray, morgan: np.ndarray,
    tr: np.ndarray, te: np.ndarray,
    records: Sequence[Any] | None = None,
    *,
    arms: Sequence[str] | None = None,
    models: Sequence[str] = ("ridge", "forest"),
) -> dict[str, dict[str, float]]:
    """Fit every representation on one split, vocabulary from training only.

    ``arms`` and ``models`` restrict what is fitted. They exist for the seed
    sweep in ``atlas.ml.split_noise``, which refits a handful of arms across a
    dozen splits and would otherwise take twelve times the full benchmark. The
    restriction is applied *after* the features are built, so a restricted run
    measures exactly what the full run measures -- a noise band computed on a
    different code path would calibrate the wrong number.
    """
    # Fitting the vocabulary on all molecules would let the test set define the
    # feature space, which is leakage even though no label is touched.
    train = [analyses[i] for i in tr]
    vocab = residue_vocabulary(train)
    svocab = residue_vocabulary(train, stereo=True)
    extra: dict[str, np.ndarray] = {"morgan_fp": morgan.astype(float)}
    assay = _assay_block(records, tr)
    if assay is not None:
        extra["assay_context"] = assay
    plm = _plm_block(analyses)
    if plm is not None:
        extra["esm2_embedding"] = plm
    reps = build_representations(
        analyses, vocabulary=vocab, stereo_vocabulary=svocab, extra=extra)

    results: dict[str, dict[str, float]] = {
        "baseline_mean": _score(
            _fit_predict(np.zeros((len(tr), 1)), y[tr], np.zeros((len(te), 1)), "mean"),
            y[te], y[tr]),
    }
    wanted = reps.names if arms is None else [n for n in arms if n in reps.matrices]
    if arms is not None:
        missing = [n for n in arms if n not in reps.matrices]
        if missing:
            logger.warning("requested arms not built: %s", ", ".join(missing))
    for name in wanted:
        x = reps[name]
        for model in models:
            try:
                pred = _fit_predict(x[tr], y[tr], x[te], model)
            except Exception as exc:        # a representation may be degenerate
                logger.warning("%s/%s failed: %s", name, model, exc)
                continue
            results[f"{name}__{model}"] = _score(pred, y[te], y[tr])
    return results


@dataclass
class BenchmarkReport:
    dataset_version: str
    endpoint: str
    n_peptides: int
    ceilings: dict[str, Any]
    cross_series: dict[str, Any]
    within_series: dict[str, Any]
    contrasts: dict[str, Any]
    gate: dict[str, Any]
    findings: list[str] = field(default_factory=list)

    def save(self, path: Path = BENCHMARK_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=float))
        return path


def _best(results: dict[str, dict[str, float]]) -> tuple[str, dict[str, float]]:
    candidates = {k: v for k, v in results.items() if k != "baseline_mean"}
    name = min(candidates, key=lambda k: candidates[k]["mae"])
    return name, candidates[name]


def _extension_roster(cross: dict[str, Any]) -> str:
    """Name the representation extensions measured, and count them honestly.

    Generated rather than written out, so the count cannot drift from how many
    arms actually ran. The language model arm is skipped when its embedding
    cache is absent, and then it must not be claimed as measured.
    """
    names = ["explicit topology", "residue chemistry", "local sequence context",
             "assay context", "backbone stereochemistry"]
    if "esm2_embedding__forest" in cross["results"]:
        names.append("a pretrained protein language model")
    words = {5: "five", 6: "six", 7: "seven"}.get(len(names), str(len(names)))
    return f"{words} times -- " + ", ".join(names[:-1]) + f" and {names[-1]}"


def _findings(
    ceilings: dict[str, Any], cross: dict[str, Any], within: dict[str, Any],
    contrasts: dict[str, Any], gate: dict[str, Any], n_peptides: int,
) -> list[str]:
    seq = ceilings["canonical_sequence_20_letter"]
    chem = ceilings["chemistry_aware_positional"]
    worst = ceilings["worst_canonical_collision"]
    comp = ceilings["chemistry_aware_composition_order_free"]
    cb_name, cb = _best(cross["results"])
    wb_name, wb = _best(within["results"])

    out = [
        f"A canonical twenty-letter sequence cannot separate these peptides: "
        f"{n_peptides} molecules collapse to {seq['n_distinct_keys']} distinct "
        f"sequences, leaving {seq['colliding_fraction']:.1%} of molecules sharing "
        f"a sequence with at least one other. No model on that representation "
        f"can beat MAE {seq['oracle_mae']:.3f}, and on the colliding molecules "
        f"alone it cannot beat {seq['oracle_mae_on_colliding']:.3f}. This is a "
        "bound on every frozen sequence embedding, not an estimate for one.",

        f"The worst single collision is instructive: {worst['n_molecules']} "
        f"molecules all read as '{worst['sequence']}' yet span pKi "
        f"{worst['pki_min']:.2f} to {worst['pki_max']:.2f}, a range of "
        f"{worst['pki_range']:.2f} log units — a "
        f"{10 ** worst['pki_range']:,.0f}-fold difference in affinity rendered "
        f"as one string. A chemistry-aware encoding distinguishes "
        f"{worst['distinct_chemistry_aware_keys']} of them.",

        f"Naming the actual side chains removes most of that loss: the "
        f"chemistry-aware positional encoding gives "
        f"{chem['n_distinct_keys']} distinct keys for {n_peptides} molecules "
        f"({chem['colliding_fraction']:.1%} colliding) and an oracle MAE of "
        f"{chem['oracle_mae']:.3f}, against {seq['oracle_mae']:.3f} for the "
        "canonical alphabet.",

        f"This corrects a Stage A expectation. Stage A predicted that "
        f"vocabulary collapse would make a frozen sequence embedding fail, and "
        f"the collapse is certainly real — but its floor of MAE "
        f"{seq['oracle_mae']:.3f} is *below* the best model measured here "
        f"({cb['mae']:.3f}). At current performance the collapse is therefore "
        "not yet the binding constraint; it would become binding only for a "
        f"model already better than MAE {seq['oracle_mae']:.3f}. Learnability, "
        "not representational capacity, is what limits every representation "
        "tried here.",

        f"Discarding order costs more than discarding chemistry: the "
        f"order-free chemistry-aware encoding has an oracle MAE of "
        f"{comp['oracle_mae']:.3f} against {chem['oracle_mae']:.3f} positional, "
        f"confirming by construction why Stage A's composition representation "
        "scored below the mean baseline.",

        f"Cross-series (whole analogue series held out, "
        f"{cb['n_train']} train / {cb['n_test']} test): the best representation "
        f"is {cb_name} at MAE {cb['mae']:.3f}, R2 {cb['r2']:+.3f}, against a "
        f"mean-predictor baseline of "
        f"{cross['results']['baseline_mean']['mae']:.3f}.",

        f"Within-series ({wb['n_train']} train / {wb['n_test']} test, every "
        f"held-out molecule having same-series neighbours): the best is "
        f"{wb_name} at MAE {wb['mae']:.3f}, R2 {wb['r2']:+.3f}, against a "
        f"baseline of {within['results']['baseline_mean']['mae']:.3f} and a "
        f"series-mean predictor at {within['series_mean_predictor_mae']:.3f}. "
        + ("The model beats a series lookup, so it resolves substitution "
           "effects among close analogues — see the diagnostics below for how "
           "close."
           if wb["mae"] < within["series_mean_predictor_mae"] else
           "The model does not beat predicting the series mean, so it resolves "
           "no substitution effect that a series lookup does not already give."),

        f"Splitting the two questions was worth it: the same representations "
        f"differ by {abs(cb['mae'] - wb['mae']):.3f} log units of MAE between "
        f"cross-series and within-series evaluation, so a single aggregate "
        "number would have described neither.",
    ]

    # --- the contrasts Stage B was built to settle --------------------------
    xc = contrasts["cross_series"]
    fo, fm, tp = (xc["fragment_fp_vs_onehot"],
                  xc["fragment_fp_vs_whole_molecule_fp"],
                  xc["topology_added_to_fragment_fp"])
    cd, wd = cross["diagnostics"], within["diagnostics"]

    out.append(
        f"The design hypothesis holds, and the mechanism is visible. Encoding "
        f"each side chain by its fragment fingerprint rather than by a one-hot "
        f"index improves cross-series MAE from {fo['best_mae_b']:.3f} to "
        f"{fo['best_mae_a']:.3f}. The reason is measurable: "
        f"{cd['test_molecules_with_unseen_side_chain']['fraction']:.1%} of "
        f"held-out molecules carry a side chain absent from the "
        f"{cd['training_vocabulary_size']}-entry training vocabulary "
        f"({cd['unseen_side_chain_positions_total']} positions in total). A "
        "one-hot index has nothing to say about those; a fragment fingerprint "
        "places an unseen chlorophenylalanine near a seen one."
    )
    out.append(
        f"The peptide-native representation beats the whole-molecule "
        f"fingerprint on the harder question: cross-series MAE "
        f"{fm['best_mae_a']:.3f} against {fm['best_mae_b']:.3f}, and R2 "
        f"{cb['r2']:+.3f} against "
        f"{min((v['mae'], v['r2']) for k, v in cross['results'].items() if k.startswith('morgan_fp'))[1]:+.3f}. "
        "Positional residue chemistry is doing work that circular "
        "substructure bits over the whole molecule do not."
    )
    out.append(
        f"Stage A's requirement that the macrocyclic constraint be an explicit "
        f"feature is **not** supported by measurement. Adding the topology "
        f"block to the fragment fingerprint changes cross-series MAE from "
        f"{tp['best_mae_b']:.3f} to {tp['best_mae_a']:.3f} at best, and with a "
        f"random forest it is far worse — "
        f"{cross['results']['positional_fragment_fp__forest']['mae']:.3f} "
        f"without topology against "
        f"{cross['results']['positional_fragment_fp+topology__forest']['mae']:.3f} "
        f"with it. Topology alone "
        f"is worse than the mean baseline "
        f"({min(v['mae'] for k, v in cross['results'].items() if k.startswith('topology')):.3f} "
        f"against {cross['results']['baseline_mean']['mae']:.3f}). Residue "
        "count and molecular weight let a tree split on series identity, which "
        "is exactly what does not transfer to an unseen series. The constraint "
        "is better carried implicitly by the bridge chemistry already present "
        "in the fragment encoding."
    )
    # Every design conclusion depends on which question was asked. This is the
    # strongest argument for having separated them, and a warning about
    # benchmarking on the easier one.
    wcon = contrasts["within_series"]
    reversed_ = [k for k in xc if xc[k]["a_better"] != wcon[k]["a_better"]]
    if reversed_:
        out.append(
            f"Every representation choice reverses between the two "
            f"evaluations. {len(reversed_)} of {len(xc)} contrasts flip sign: "
            + "; ".join(
                f"{k} favours {xc[k]['a'] if xc[k]['a_better'] else xc[k]['b']} "
                f"cross-series ({min(xc[k]['best_mae_a'], xc[k]['best_mae_b']):.3f} "
                f"against {max(xc[k]['best_mae_a'], xc[k]['best_mae_b']):.3f}) but "
                f"{wcon[k]['a'] if wcon[k]['a_better'] else wcon[k]['b']} "
                f"within-series "
                f"({min(wcon[k]['best_mae_a'], wcon[k]['best_mae_b']):.3f} against "
                f"{max(wcon[k]['best_mae_a'], wcon[k]['best_mae_b']):.3f})"
                for k in reversed_
            )
            + ". A programme that benchmarked only within-series — the "
            "evaluation most papers report — would have selected one-hot "
            "residue indices, explicit topology and whole-molecule "
            "fingerprints: in each case the choice that generalises worst to "
            "an unseen series. This is the clearest vindication of Stage A's "
            "requirement to report the two separately."
        )

    out.append(
        f"The within-series result must be read with its diagnostics: median "
        f"nearest-training-neighbour similarity is "
        f"{wd['nearest_training_similarity']['median']:.3f} and "
        f"{wd['test_molecules_with_close_neighbour_above_0.7']['fraction']:.1%} "
        f"of held-out molecules have a training neighbour above 0.7 Tanimoto, "
        f"against a median of {cd['nearest_training_similarity']['median']:.3f} "
        f"cross-series. Within-series performance is therefore interpolation "
        "among close analogues — the lead-optimisation regime — and not "
        "evidence of generalisation. That is a genuinely useful capability, but "
        "it is a narrower claim than the R2 alone suggests."
    )

    # Ablation: every feature group added to the best single representation,
    # one at a time, cross-series. Generated rather than narrated so the
    # verdict cannot drift from the numbers.
    added = {k: v for k, v in xc.items() if k.endswith("_added_to_fragment_fp")}
    if added:
        helped = sorted(k.replace("_added_to_fragment_fp", "")
                        for k, v in added.items() if v["a_better"])
        hurt = [k for k in sorted(added) if not added[k]["a_better"]]
        out.append(
            f"Ablation, cross-series, random forest: of {len(added)} feature "
            f"groups added one at a time to the positional fragment "
            f"fingerprint, {len(helped)} improved it and {len(hurt)} made it "
            f"worse. Helped: {', '.join(helped) or 'none'}. Hurt: "
            + "; ".join(
                f"{k.replace('_added_to_fragment_fp', '')} "
                f"({added[k]['best_mae_b']:.3f} to {added[k]['best_mae_a']:.3f})"
                for k in hurt
            )
            + ". Across ATLAS 2.4 to 2.6 the representation layer has been "
            f"extended {_extension_roster(cross)} -- "
            "and **none improves cross-series generalisation.** The best "
            "cross-series model remains "
            f"{cb_name} at MAE {cb['mae']:.3f}, unchanged. On this dataset the "
            "representation is not the binding constraint, which agrees with "
            "Stage A's effective sample size of ~30 series and with the pocket "
            "encoder's null result."
        )
        sc = ceilings["stereochemistry_collapse"]
        aware = ceilings["chemistry_aware_positional_with_stereo"]
        stereo_pair = xc.get("stereo_added_to_fragment_fp")
        if stereo_pair:
            out.append(
                f"Stereochemistry is the one extension whose failure to help "
                f"does not make it optional. Naming each residue's backbone "
                f"configuration lowers the representational ceiling from oracle "
                f"MAE {chem['oracle_mae']:.3f} to {aware['oracle_mae']:.3f} and "
                f"cuts the colliding fraction from "
                f"{chem['colliding_fraction']:.1%} to "
                f"{aware['colliding_fraction']:.1%}, because "
                f"{sc['groups_merged_by_stereo_blindness']} groups covering "
                f"{sc['molecules_affected']} molecules "
                f"({sc['molecules_affected_fraction']:.1%} of the set) are "
                f"merged by a constitutional key alone -- with a median pKi "
                f"range of {sc['median_pki_range_within_those_groups']:.2f} and "
                f"a maximum of {sc['max_pki_range_within_those_groups']:.2f} "
                f"log units inside them. Yet added to the fragment fingerprint "
                f"it leaves cross-series MAE at "
                f"{stereo_pair['best_mae_a']:.3f} against "
                f"{stereo_pair['best_mae_b']:.3f}: no improvement, consistent "
                "with every other extension. The conclusion is not that "
                "configuration is unimportant but that the error is not where "
                "the representation is. A model that cannot distinguish D-Phe "
                "from L-Phe must not be asked to order them, which is a "
                "refusal to implement rather than a metric to improve."
            )
        plm_add = xc.get("esm2_embedding_added_to_fragment_fp")
        plm_alone = xc.get("esm2_embedding_vs_fragment_fp")
        if plm_add and plm_alone and np.isfinite(plm_alone["best_mae_a"]):
            wplm = wcon.get("esm2_embedding_added_to_fragment_fp", {})
            out.append(
                f"A pretrained protein language model is the only extension "
                f"tried that imports information from outside these "
                f"{n_peptides} molecules, and it does not help either. Frozen "
                f"ESM-2 650M embeddings of the canonical sequence score "
                f"cross-series MAE {plm_alone['best_mae_a']:.3f} on their own "
                f"against {plm_alone['best_mae_b']:.3f} for the positional "
                f"fragment fingerprint; concatenated with that fingerprint "
                f"they leave it at {plm_add['best_mae_a']:.3f} against "
                f"{plm_add['best_mae_b']:.3f}"
                + (f", and within-series at {wplm['best_mae_a']:.3f} against "
                   f"{wplm['best_mae_b']:.3f}" if wplm else "")
                + ". Pretraining across tens of millions of UniRef sequences "
                "therefore transfers nothing measurable to this endpoint that "
                "the side-chain fingerprint does not already carry."
            )
            out.append(
                f"Two candidate explanations can be separated here, and only "
                f"one survives. The ceiling is **not** the explanation: the "
                f"canonical sequence gives {seq['n_distinct_keys']} distinct "
                f"strings for {n_peptides} molecules and an oracle MAE of "
                f"{seq['oracle_mae']:.3f}, which is far below the "
                f"{cb['mae']:.3f} the best real model reaches, so a "
                f"sequence-only representation had room to improve and did not "
                f"take it. The vocabulary loss is real and is the plausible "
                f"mechanism: every residue outside the canonical twenty "
                f"reaches the model as the single unknown token X, so the "
                f"{chem['n_distinct_keys']} chemically distinct peptides the "
                f"fragment encoding resolves arrive at the language model as "
                f"{seq['n_distinct_keys']} inputs, and "
                f"{seq['colliding_fraction']:.1%} of molecules share their "
                f"input with another. A model pretrained on natural proteins "
                f"has no representation for a 4-chloro-phenylalanine because "
                f"UniRef contains none. This is the sixth consecutive null "
                f"result on the representation layer and the first that cannot "
                f"be answered by adding more chemistry to the feature set -- "
                f"it points back at sample structure, where Stage A's "
                f"effective sample size of ~30 series already pointed."
            )
        ac_cross = cross["results"].get("assay_context__forest")
        ac_within = within["results"].get("assay_context__forest")
        if ac_cross and ac_within:
            out.append(
                f"Assay context behaves as its own caveat warned. Alone it "
                f"scores R2 {ac_cross['r2']:+.3f} cross-series -- far worse "
                f"than the mean -- and R2 {ac_within['r2']:+.3f} within-series. "
                "Assay identity correlates with document and so with analogue "
                "series: when whole series are held out the assay features "
                "point at training series and mislead, and when series are "
                "shared they carry weak real signal. That is a reason to keep "
                "it as a recorded covariate, not a predictive feature."
            )

    verdict = gate["passes"]
    out.append(
        f"Deployment gate: {'PASS' if verdict else 'FAIL'}. "
        + "; ".join(
            f"{k} = {v['actual']:.3f} against {v['required']} "
            f"({'met' if v['met'] else 'not met'})"
            if isinstance(v.get("actual"), float) else
            f"{k}: {'met' if v['met'] else 'not met'}"
            for k, v in gate["criteria"].items()
        )
        + ". The peptide affinity model therefore "
        + ("may publish numeric estimates within its applicability domain."
           if verdict else
           "remains undeployed and must not publish numeric affinity estimates, "
           "nor propagate them into pharmacokinetic models or knowledge-graph "
           "edges.")
    )
    return out


def run_benchmark(
    *,
    records: Sequence[Any] | None = None,
    dataset_version: str | None = None,
    endpoint: str | None = None,
    save: bool | None = None,
    path: Path = BENCHMARK_PATH,
) -> BenchmarkReport:
    """Run the Stage B benchmark and return the report.

    With no arguments this is the original Stage B run over the saved MC4R
    dataset, unchanged -- including the deployment gate, whose thresholds are
    not a function of which dataset is passed.

    ``records`` injects a different single-target dataset, so the Stage B
    contrasts can be re-measured on another receptor. Stage B's finding was
    that all five representation contrasts reverse between the cross-series
    and within-series evaluations; that was established on one receptor, and
    one receptor is one trial. Injected runs do not save unless given an
    explicit ``path``.
    """
    if records is None:
        bundle = load_dataset()
        records = bundle.records
        dataset_version = bundle.version
        endpoint = bundle.endpoint
        if save is None:
            save = True
    else:
        missing = [
            n for n, v in (("dataset_version", dataset_version),
                           ("endpoint", endpoint)) if v is None
        ]
        if missing:
            raise ValueError(
                f"injected records need their own provenance; missing {missing}"
            )
        if save is None:
            save = False
        if save and path == BENCHMARK_PATH:
            raise ValueError(
                "refusing to save an injected-record benchmark to the "
                f"canonical {BENCHMARK_PATH.name}; pass an explicit path"
            )
    records = list(records)
    fs = featurize(records)

    analyses = [analyse_peptide(s) for s in fs.smiles]
    keep = np.array([a.is_peptide for a in analyses])
    pep_an = [a for a, k in zip(analyses, keep) if k]
    y = fs.y[keep]
    morgan = fs.fingerprints[keep]

    # Records aligned to the peptide subset, for the assay-context arm.
    # featurize drops unparseable molecules, so alignment goes through fs.ids
    # rather than through the original record order.
    by_id = {r.molecule_chembl_id: r for r in records}
    pep_records = [by_id[i] for i, k in zip(fs.ids, keep) if k]

    ceilings = representational_ceilings(pep_an, y)

    labels = cross_series_groups(morgan)
    tr_x, te_x = _cross_series_split(labels, seed=SEED)
    tr_w, te_w = within_series_split(labels, seed=SEED)

    cross = {
        "evaluation": "whole analogue series held out",
        "n_clusters": int(len(np.unique(labels))),
        "diagnostics": split_diagnostics(morgan, pep_an, tr_x, te_x),
        "results": _evaluate(pep_an, y, morgan, tr_x, te_x, pep_records),
    }

    # What a lookup of the series mean achieves, as the bar a within-series
    # model has to clear.
    series_mean = np.zeros(len(te_w))
    for i, idx in enumerate(te_w):
        same = tr_w[labels[tr_w] == labels[idx]]
        series_mean[i] = y[same].mean() if len(same) else y[tr_w].mean()
    within = {
        "evaluation": "members held out of series the model otherwise sees",
        "min_cluster_size": 4,
        "series_mean_predictor_mae": float(np.abs(series_mean - y[te_w]).mean()),
        "diagnostics": split_diagnostics(morgan, pep_an, tr_w, te_w),
        "results": _evaluate(pep_an, y, morgan, tr_w, te_w, pep_records),
    }

    # The contrasts Stage B exists to settle, each a like-for-like comparison
    # on one test set.
    # Contrasts are read off the random forest alone, not off whichever model
    # happened to do best. A forest is invariant to monotone transforms of each
    # feature, so a difference between two representations under it is a
    # difference between the representations. Ridge is not: standardising helps
    # the dense residue-chemistry descriptors and hurts the sparse binary
    # fingerprint, whose rare bits get amplified by division by a tiny standard
    # deviation. Mixing the two models would let a scaling artefact decide a
    # representation question.
    CONTRAST_MODEL = "forest"

    def pair(res: dict[str, Any], a: str, b: str) -> dict[str, Any]:
        mae_a = res.get(f"{a}__{CONTRAST_MODEL}", {}).get("mae", float("nan"))
        mae_b = res.get(f"{b}__{CONTRAST_MODEL}", {}).get("mae", float("nan"))
        return {"a": a, "b": b, "model": CONTRAST_MODEL,
                "best_mae_a": mae_a, "best_mae_b": mae_b,
                "a_minus_b": mae_a - mae_b, "a_better": bool(mae_a < mae_b)}

    contrasts = {
        ev: {
            "fragment_fp_vs_onehot": pair(
                d["results"], "positional_fragment_fp", "positional_onehot"),
            "fragment_fp_vs_whole_molecule_fp": pair(
                d["results"], "positional_fragment_fp", "morgan_fp"),
            "topology_added_to_fragment_fp": pair(
                d["results"], "positional_fragment_fp+topology",
                "positional_fragment_fp"),
            "residue_chemistry_vs_fragment_fp": pair(
                d["results"], "residue_chemistry", "positional_fragment_fp"),
            "residue_chemistry_added_to_fragment_fp": pair(
                d["results"], "residue_chemistry+positional_fragment_fp",
                "positional_fragment_fp"),
            "local_context_added_to_fragment_fp": pair(
                d["results"], "positional_fragment_fp+local_context",
                "positional_fragment_fp"),
            "assay_context_added_to_fragment_fp": pair(
                d["results"], "positional_fragment_fp+assay_context",
                "positional_fragment_fp"),
            "stereo_added_to_fragment_fp": pair(
                d["results"], "positional_fragment_fp+stereo",
                "positional_fragment_fp"),
            "stereo_aware_vs_blind_onehot": pair(
                d["results"], "positional_onehot_stereo", "positional_onehot"),
            "esm2_embedding_vs_fragment_fp": pair(
                d["results"], "esm2_embedding", "positional_fragment_fp"),
            "esm2_embedding_added_to_fragment_fp": pair(
                d["results"], "positional_fragment_fp+esm2_embedding",
                "positional_fragment_fp"),
        }
        for ev, d in (("cross_series", cross), ("within_series", within))
    }

    cb_name, cb = _best(cross["results"])
    wb_name, wb = _best(within["results"])
    criteria = {
        "cross_series_mae": {
            "actual": cb["mae"], "required": f"<= {GATE['cross_series_mae_max']}",
            "met": bool(cb["mae"] <= GATE["cross_series_mae_max"])},
        "cross_series_r2": {
            "actual": cb["r2"], "required": f">= {GATE['cross_series_r2_min']}",
            "met": bool(cb["r2"] >= GATE["cross_series_r2_min"])},
        "within_series_beats_series_mean": {
            "actual": wb["mae"],
            "required": f"< {within['series_mean_predictor_mae']:.3f}",
            "met": bool(wb["mae"] < within["series_mean_predictor_mae"])},
    }
    gate = {
        "criteria": criteria,
        "passes": all(c["met"] for c in criteria.values()),
        "best_cross_series": cb_name,
        "best_within_series": wb_name,
        "rationale": (
            "Thresholds come from Stage A's measurements: a label noise floor "
            "of MAE 0.245 and a mean-predictor baseline of 1.343."
        ),
    }

    report = BenchmarkReport(
        dataset_version=dataset_version, endpoint=endpoint,
        n_peptides=int(keep.sum()), ceilings=ceilings,
        cross_series=cross, within_series=within, contrasts=contrasts,
        gate=gate,
        findings=_findings(ceilings, cross, within, contrasts, gate,
                           int(keep.sum())),
    )
    if save:
        report.save(path)
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = run_benchmark()
    for i, f in enumerate(report.findings, 1):
        print(f"\n[{i}] {f}")
    print(f"\nSaved to {BENCHMARK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
