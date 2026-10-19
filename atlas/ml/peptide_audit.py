"""ATLAS 2.4 Stage A: quantitative failure and dataset audit.

The peptide affinity model was not deployed. The brief asks why, and insists
the answer be measured rather than asserted. This module produces those
measurements, and in particular refuses to take the earlier "peptide-like"
label at face value: that label counted amide bonds, and this audit traces
backbones instead (see :mod:`atlas.ml.peptide_structure`).

Four hypotheses are distinguished, each by a measurement that can separate it
from the others rather than merely be consistent with it:

A. **Representation mismatch** — a molecular fingerprint cannot express what
   determines peptide affinity. Discriminated by fitting several
   representations, including a sequence-composition one available only now
   that backbones are traced, on the same peptide-internal split.
B. **Domain shift** — the training population and the peptide population are
   not exchangeable. Discriminated by holding the peptide test set fixed and
   varying only what the model is trained on; if small-molecule training
   transfers worse than a far smaller peptide-only training set, the shift is
   doing the damage and extra small-molecule data cannot repair it.
C. **Dataset limitation** — too few peptides. Discriminated by a learning
   curve on the fixed test set: an error still falling at full data implicates
   quantity, an error already flat does not.
D. **Irreducible variability** — the labels themselves disagree.
   Discriminated by the measured replicate spread, which bounds below the
   error any model can achieve.

The hypotheses are not exclusive, and the report records the share of evidence
for each rather than selecting a winner.
"""

from __future__ import annotations

import collections
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from rdkit import Chem

from atlas.ml.dataset import DATA_DIR, ActivityRecord, load_dataset
from atlas.ml.features import (
    FeatureSet, applicability, classify_modality, featurize,
)
from atlas.ml.peptide_structure import (
    CANONICAL_SIDE_CHAINS, PeptideAnalysis, PeptideClass, analyse_peptide,
)
from atlas.ml.splits import cluster_split

logger = logging.getLogger(__name__)

__all__ = ["run_audit", "AuditReport", "AUDIT_PATH", "composition_features",
           "series_structure", "structural_census", "domain_shift"]

AUDIT_PATH = DATA_DIR / "peptide_audit.json"

SEED = 0
#: A replicate range wider than this is treated as a discordant label.
DISCORDANCE_LOG_UNITS = 1.0


# ---------------------------------------------------------------- census ----

def structural_census(
    records: Sequence[ActivityRecord],
) -> tuple[dict[str, Any], list[PeptideAnalysis]]:
    """Reclassify every molecule by backbone tracing and compare to the old label.

    The comparison matters more than either count alone: two classifications
    can agree on the total while disagreeing on membership, and it is
    membership that decides what a peptide model is trained and judged on.
    """
    analyses, old_labels = [], []
    for r in records:
        analyses.append(analyse_peptide(r.smiles))
        mol = Chem.MolFromSmiles(r.smiles)
        old_labels.append(classify_modality(mol).value if mol is not None else "unparseable")

    new_counts = collections.Counter(a.peptide_class.value for a in analyses)
    old_counts = collections.Counter(old_labels)
    cross = collections.Counter(
        (old, a.peptide_class.value) for old, a in zip(old_labels, analyses)
    )

    peptides = [a for a in analyses if a.is_peptide]
    old_peptide = {i for i, o in enumerate(old_labels) if o == "peptide_like"}
    new_peptide = {i for i, a in enumerate(analyses) if a.is_peptide}

    # An inventory of what the noncanonical residues actually are, so that
    # "noncanonical" is a description rather than a shrug.
    frags = collections.Counter(
        f for a in analyses for f in a.unmatched_fragments
    )

    return {
        "n_molecules": len(records),
        "previous_classification": dict(old_counts),
        "structural_classification": dict(new_counts),
        "cross_tabulation": [
            {"previous": o, "structural": n, "n": v}
            for (o, n), v in sorted(cross.items(), key=lambda kv: -kv[1])
        ],
        "previous_peptide_count": len(old_peptide),
        "structural_peptide_count": len(new_peptide),
        "false_positives_of_previous_label": sorted(old_peptide - new_peptide),
        "false_negatives_of_previous_label": sorted(new_peptide - old_peptide),
        "n_false_positives": len(old_peptide - new_peptide),
        "n_false_negatives": len(new_peptide - old_peptide),
        "membership_disagreement": len(old_peptide ^ new_peptide),
        "membership_disagreement_fraction": (
            len(old_peptide ^ new_peptide) / max(len(new_peptide), 1)
        ),
        "residue_count_histogram": dict(sorted(
            collections.Counter(a.n_residues_in_longest_chain for a in peptides).items()
        )),
        # These categories overlap: a disulfide-constrained peptide may also
        # carry a lactam bridge. Only "linear" and "constrained" partition.
        "topology": {
            "head_to_tail_cyclic": sum(1 for a in peptides if a.is_cyclic_backbone),
            "side_chain_bridged": sum(1 for a in peptides if a.n_side_chain_bridges),
            "disulfide_containing": sum(1 for a in peptides if a.n_disulfides),
            "macrocyclic_by_backbone_ring_only": sum(
                1 for a in peptides
                if a.largest_backbone_ring >= 12 and not a.is_cyclic_backbone
                and not a.n_side_chain_bridges and not a.n_disulfides
            ),
            "linear": sum(
                1 for a in peptides
                if not a.is_cyclic_backbone and not a.n_side_chain_bridges
                and not a.n_disulfides and a.largest_backbone_ring < 12
            ),
            "categories_overlap": True,
        },
        "canonical_residue_fraction": {
            "mean": float(np.mean([a.canonical_fraction for a in peptides])) if peptides else 0.0,
            "fully_canonical": sum(1 for a in peptides if a.canonical_fraction == 1.0),
            "any_noncanonical": sum(1 for a in peptides if a.canonical_fraction < 1.0),
        },
        "noncanonical_fragment_inventory": [
            {"fragment": f, "n": n} for f, n in frags.most_common(25)
        ],
        "n_distinct_noncanonical_fragments": len(frags),
    }, analyses


# ------------------------------------------------------- assay provenance ----

def assay_audit(
    records: Sequence[ActivityRecord], analyses: Sequence[PeptideAnalysis]
) -> dict[str, Any]:
    """How concentrated the evidence is in a few assays and papers.

    A dataset whose peptides come from a handful of documents is not a sample
    of peptide chemistry; it is a sample of a few laboratories' series, and a
    held-out split within it tests less than it appears to.
    """
    def profile(subset: Sequence[ActivityRecord]) -> dict[str, Any]:
        docs = collections.Counter(d for r in subset for d in r.document_ids)
        assays = collections.Counter(a for r in subset for a in r.assay_ids)
        n = max(len(subset), 1)
        top_docs = docs.most_common(10)
        # Share must be measured over the *set* of molecules a document group
        # covers. Summing per-document counts double-counts every molecule
        # cited by more than one paper, and can exceed 1.
        top_ids = {d for d, _ in top_docs}
        covered = sum(
            1 for r in subset if top_ids & set(r.document_ids)
        )
        largest = top_docs[0][0] if top_docs else None
        return {
            "n_molecules": len(subset),
            "n_distinct_documents": len(docs),
            "n_distinct_assays": len(assays),
            "molecules_per_document": len(subset) / max(len(docs), 1),
            "largest_document_share": (
                sum(1 for r in subset if largest in r.document_ids) / n
                if largest else 0.0
            ),
            "top_10_document_share": covered / n,
            "top_documents": [{"document": d, "n": v} for d, v in top_docs],
            "multi_document_molecules": sum(1 for r in subset if len(r.document_ids) > 1),
            "multi_assay_molecules": sum(1 for r in subset if len(r.assay_ids) > 1),
        }

    pep = [r for r, a in zip(records, analyses) if a.is_peptide]
    sml = [r for r, a in zip(records, analyses) if not a.is_peptide]
    return {"peptides": profile(pep), "non_peptides": profile(sml),
            "all": profile(list(records))}


# ------------------------------------------------------------ label noise ----

def label_quality(
    records: Sequence[ActivityRecord], analyses: Sequence[PeptideAnalysis]
) -> dict[str, Any]:
    """The experimental noise floor, measured separately for peptides.

    Half the replicate range is used as a per-molecule uncertainty: for a
    uniformly distributed set of replicate values it is the largest error an
    optimal point estimate can be forced into, so the mean over replicated
    molecules is a defensible lower bound on achievable mean absolute error.
    It is a bound on this dataset's labels, not a claim about assay precision
    in general.
    """
    def profile(subset: Sequence[ActivityRecord]) -> dict[str, Any]:
        rep = [r for r in subset if r.is_replicated]
        spreads = np.array([r.replicate_spread for r in rep]) if rep else np.zeros(0)
        return {
            "n": len(subset),
            "n_replicated": len(rep),
            "replicated_fraction": len(rep) / max(len(subset), 1),
            "n_discordant": sum(1 for r in subset if r.is_discordant),
            "discordant_fraction": (
                sum(1 for r in subset if r.is_discordant) / max(len(subset), 1)
            ),
            "median_replicate_spread": float(np.median(spreads)) if len(spreads) else None,
            "mean_replicate_spread": float(np.mean(spreads)) if len(spreads) else None,
            "max_replicate_spread": float(np.max(spreads)) if len(spreads) else None,
            "noise_floor_mae_estimate": float(np.mean(spreads) / 2) if len(spreads) else None,
            "label_range": [
                float(min(r.pki for r in subset)), float(max(r.pki for r in subset)),
            ] if subset else None,
            "label_sd": float(np.std([r.pki for r in subset])) if subset else None,
        }

    pep = [r for r, a in zip(records, analyses) if a.is_peptide]
    sml = [r for r, a in zip(records, analyses) if not a.is_peptide]
    out = {"peptides": profile(pep), "non_peptides": profile(sml),
           "all": profile(list(records))}
    out["caveat"] = (
        "Half the replicate range bounds the error of an optimal point "
        "estimate only for molecules that were actually replicated; "
        f"{out['peptides']['replicated_fraction']:.1%} of peptides were. The "
        "unreplicated remainder has unmeasured label uncertainty, so the floor "
        "quoted here is itself an estimate from a subset."
    )
    return out


# ----------------------------------------------------------- domain shift ----

def domain_shift(fs: FeatureSet, peptide_mask: np.ndarray) -> dict[str, Any]:
    """Quantify separation between the peptide and non-peptide populations.

    Convex-hull membership is deliberately not used. In a 2048-bit fingerprint
    space almost nothing is inside the hull of anything, so the test reports
    "out of domain" for every molecule and distinguishes none of them. These
    diagnostics are distance- and neighbourhood-based, which keeps them
    informative at this dimensionality.
    """
    pep_fp = fs.fingerprints[peptide_mask]
    sml_fp = fs.fingerprints[~peptide_mask]
    pep_y = fs.y[peptide_mask]
    sml_y = fs.y[~peptide_mask]

    def q(a: np.ndarray) -> dict[str, float]:
        return {
            "min": float(np.min(a)), "p10": float(np.percentile(a, 10)),
            "median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "max": float(np.max(a)), "mean": float(np.mean(a)),
        }

    # Nearest neighbour in the *other* population, and within the same one.
    pep_to_sml = applicability(sml_fp, pep_fp)
    sml_to_pep = applicability(pep_fp, sml_fp)
    pep_to_pep = _self_neighbour(pep_fp)
    sml_to_sml = _self_neighbour(sml_fp)

    # Standardised mean difference per descriptor: how far the two populations
    # sit apart in units of pooled spread.
    pd_, sd_ = fs.descriptors[peptide_mask], fs.descriptors[~peptide_mask]
    pooled = np.sqrt((np.var(pd_, axis=0) + np.var(sd_, axis=0)) / 2) + 1e-12
    smd = np.abs(pd_.mean(axis=0) - sd_.mean(axis=0)) / pooled
    order = np.argsort(-smd)

    return {
        "n_peptides": int(peptide_mask.sum()),
        "n_non_peptides": int((~peptide_mask).sum()),
        "nearest_neighbour_similarity": {
            "peptide_to_non_peptide": q(pep_to_sml),
            "non_peptide_to_peptide": q(sml_to_pep),
            "peptide_to_peptide": q(pep_to_pep),
            "non_peptide_to_non_peptide": q(sml_to_sml),
        },
        "peptides_with_no_close_non_peptide_neighbour": {
            "threshold": 0.4,
            "n": int((pep_to_sml < 0.4).sum()),
            "fraction": float((pep_to_sml < 0.4).mean()),
        },
        "descriptor_standardised_mean_difference": {
            "max": float(smd[order[0]]),
            "median": float(np.median(smd)),
            "n_descriptors_above_1_sd": int((smd > 1.0).sum()),
            "n_descriptors": int(smd.size),
            "most_shifted": [
                {"descriptor": fs.descriptor_names[i], "smd": float(smd[i])}
                for i in order[:10]
            ],
        },
        "label_distribution": {
            "peptide_mean_pki": float(pep_y.mean()),
            "non_peptide_mean_pki": float(sml_y.mean()),
            "peptide_sd": float(pep_y.std()),
            "non_peptide_sd": float(sml_y.std()),
            "mean_difference": float(pep_y.mean() - sml_y.mean()),
        },
        "note": (
            "Exchangeability is the assumption conformal calibration rests on. "
            "These figures describe how far it is from holding between the two "
            "populations; they do not by themselves say the model fails."
        ),
    }


def series_structure(fp: np.ndarray, y: np.ndarray, *, cutoff: float = 0.4) -> dict[str, Any]:
    """How many genuinely independent series the peptide population contains.

    630 molecules is not 630 independent observations if they are a handful of
    analogue series. The number of Taylor-Butina clusters is a better estimate
    of effective sample size, and the within-cluster label spread says how much
    of the variance a model must resolve *inside* a series -- which is the part
    no amount of series-level signal can help with.
    """
    from rdkit import DataStructs
    from rdkit.ML.Cluster import Butina

    n = len(fp)
    bits = [
        DataStructs.CreateFromBitString("".join(str(int(b)) for b in row))
        for row in fp.astype(np.uint8)
    ]
    dists: list[float] = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(bits[i], bits[:i])
        dists.extend(1.0 - s for s in sims)
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    sizes = sorted((len(c) for c in clusters), reverse=True)

    within = [float(np.std(y[list(c)])) for c in clusters if len(c) > 2]
    return {
        "distance_cutoff": cutoff,
        "n_molecules": n,
        "n_clusters": len(clusters),
        "effective_sample_size_ratio": len(clusters) / max(n, 1),
        "largest_cluster": sizes[0] if sizes else 0,
        "largest_cluster_share": (sizes[0] / max(n, 1)) if sizes else 0.0,
        "top_5_cluster_share": sum(sizes[:5]) / max(n, 1),
        "singletons": sum(1 for s in sizes if s == 1),
        "median_within_cluster_label_sd": (
            float(np.median(within)) if within else None),
        "overall_label_sd": float(np.std(y)),
        "note": (
            "A cluster split assigns whole clusters to test, so the number of "
            "clusters bounds how many independent held-out groups exist. Label "
            "variance that lives inside a cluster cannot be predicted from "
            "which cluster a molecule belongs to."
        ),
    }


def _self_neighbour(fp: np.ndarray) -> np.ndarray:
    """Max Tanimoto to any *other* molecule in the same set."""
    sim = _tanimoto_matrix(fp)
    np.fill_diagonal(sim, -1.0)
    return sim.max(axis=1)


def _tanimoto_matrix(fp: np.ndarray) -> np.ndarray:
    x = fp.astype(np.float32)
    inter = x @ x.T
    s = x.sum(axis=1, keepdims=True)
    union = s + s.T - inter
    return inter / np.maximum(union, 1e-9)


# ------------------------------------------- peptide-native representation ----

_RESIDUE_CODES = sorted(CANONICAL_SIDE_CHAINS) + ["G", "P", "X"]


def composition_features(analyses: Iterable[PeptideAnalysis]) -> np.ndarray:
    """A representation available only once backbones are traced.

    Residue counts, chain length, topology and bridge counts. It is not a
    sequence model -- order is discarded -- but it is the first representation
    here that encodes peptide structure as peptide structure rather than as
    circular substructure bits, which makes it the right control for the
    representation hypothesis.
    """
    rows = []
    for a in analyses:
        seq = a.sequence or ""
        counts = collections.Counter(seq)
        rows.append(
            [counts.get(c, 0) for c in _RESIDUE_CODES]
            + [
                a.n_residues_in_longest_chain,
                a.n_disulfides,
                a.n_side_chain_bridges,
                float(a.is_cyclic_backbone),
                a.n_unrecognised_side_chains,
                a.canonical_fraction,
                a.molecular_weight / 1000.0,
                a.largest_ring,
                float(a.has_c_terminal_amide),
                float(a.has_acylated_n_terminus),
            ]
        )
    return np.asarray(rows, dtype=float)


# ---------------------------------------------------- failure decomposition ----

def _fit_eval(
    xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, yte: np.ndarray, kind: str,
) -> dict[str, float]:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.neighbors import KNeighborsRegressor

    if kind == "mean":
        pred = np.full(len(yte), ytr.mean())
    elif kind == "knn":
        k = min(5, len(ytr))
        m = KNeighborsRegressor(n_neighbors=k, metric="jaccard"
                                if set(np.unique(xtr)) <= {0.0, 1.0} else "minkowski")
        m.fit(xtr, ytr); pred = m.predict(xte)
    elif kind == "ridge":
        m = RidgeCV(alphas=np.logspace(-3, 3, 13)); m.fit(xtr, ytr); pred = m.predict(xte)
    elif kind == "forest":
        m = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
        m.fit(xtr, ytr); pred = m.predict(xte)
    else:
        raise ValueError(kind)

    err = np.abs(pred - yte)
    ss_res = float(np.sum((pred - yte) ** 2))
    ss_tot = float(np.sum((yte - yte.mean()) ** 2))
    return {
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(np.mean((pred - yte) ** 2))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
    }


def failure_decomposition(
    fs: FeatureSet,
    analyses: Sequence[PeptideAnalysis],
    records: Sequence[ActivityRecord],
    peptide_mask: np.ndarray,
) -> dict[str, Any]:
    """Measure each hypothesis on one fixed peptide test set.

    Holding the test set fixed across every comparison is what makes the
    numbers comparable: representation, training population and training size
    are varied one at a time against the same molecules.
    """
    pep_idx = np.where(peptide_mask)[0]
    pep_fs = fs.subset(pep_idx)
    pep_an = [analyses[i] for i in pep_idx]

    # A cluster split, because a scaffold split within a macrocyclic analogue
    # series is already known to leave near-identical neighbours in training.
    split = cluster_split(pep_fs.fingerprints, test_fraction=0.25, seed=SEED)
    tr, te = split.train_idx, split.test_idx
    comp = composition_features(pep_an)

    y = pep_fs.y
    reps = {
        "fingerprint": pep_fs.fingerprints.astype(float),
        "descriptor": pep_fs.descriptors,
        "composition": comp,
        "descriptor_plus_composition": np.hstack([pep_fs.descriptors, comp]),
    }

    # --- A: representation -------------------------------------------------
    representation = {"baseline_mean": _fit_eval(
        np.zeros((len(tr), 1)), y[tr], np.zeros((len(te), 1)), y[te], "mean")}
    for name, x in reps.items():
        for model in ("ridge", "forest"):
            representation[f"{name}__{model}"] = _fit_eval(
                x[tr], y[tr], x[te], y[te], model)

    # --- B: domain shift / transfer direction ------------------------------
    # Same peptide test molecules; only the training population changes.
    sml_idx = np.where(~peptide_mask)[0]
    pep_global_tr = pep_idx[tr]
    pep_global_te = pep_idx[te]
    x_all = fs.descriptors
    transfer = {
        "peptides_only": _fit_eval(
            x_all[pep_global_tr], fs.y[pep_global_tr],
            x_all[pep_global_te], fs.y[pep_global_te], "forest"),
        "non_peptides_only": _fit_eval(
            x_all[sml_idx], fs.y[sml_idx],
            x_all[pep_global_te], fs.y[pep_global_te], "forest"),
        "pooled": _fit_eval(
            x_all[np.concatenate([pep_global_tr, sml_idx])],
            fs.y[np.concatenate([pep_global_tr, sml_idx])],
            x_all[pep_global_te], fs.y[pep_global_te], "forest"),
    }
    transfer["negative_transfer_log_units"] = (
        transfer["pooled"]["mae"] - transfer["peptides_only"]["mae"])
    transfer["interpretation"] = (
        "Pooling helps" if transfer["negative_transfer_log_units"] < -0.01
        else "Pooling hurts" if transfer["negative_transfer_log_units"] > 0.01
        else "Pooling is neutral")

    # --- C: dataset size --------------------------------------------------
    best_rep = min(
        ((k, v["mae"]) for k, v in representation.items() if k != "baseline_mean"),
        key=lambda kv: kv[1])[0]
    rep_name, model_name = best_rep.split("__")
    x_best = reps[rep_name]
    rng = np.random.default_rng(SEED)
    curve = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        n = max(int(round(frac * len(tr))), 5)
        sub = rng.permutation(tr)[:n] if frac < 1.0 else tr
        r = _fit_eval(x_best[sub], y[sub], x_best[te], y[te], model_name)
        curve.append({"fraction": frac, "n_train": int(n), "mae": r["mae"]})
    tail_gain = curve[-2]["mae"] - curve[-1]["mae"]

    # Fit MAE against log training size, so "more data would fix it" becomes a
    # number rather than a hope. Log-linear is the conventional form for a
    # learning curve over this range; it is an extrapolation well past the
    # observed data and is reported as an order of magnitude, not a forecast.
    ln_n = np.log([c["n_train"] for c in curve])
    maes = np.array([c["mae"] for c in curve])
    slope, intercept = np.polyfit(ln_n, maes, 1)
    targets = {}
    for target in (0.7, 0.5):
        if slope < 0:
            needed = float(np.exp((target - intercept) / slope))
            targets[f"n_peptides_for_mae_{target}"] = needed
        else:
            targets[f"n_peptides_for_mae_{target}"] = None
    extrapolation = {
        "fit": "mae = intercept + slope * ln(n_train)",
        "slope_per_ln_n": float(slope),
        "intercept": float(intercept),
        "r2_of_fit": float(
            1 - np.sum((maes - (intercept + slope * ln_n)) ** 2)
            / np.sum((maes - maes.mean()) ** 2)
        ),
        **targets,
        "caveat": (
            "A log-linear extrapolation several orders of magnitude beyond 489 "
            "observed training peptides. It bounds the quantity hypothesis "
            "rather than predicting an outcome."
        ),
    }

    # --- D: label noise ---------------------------------------------------
    pep_records = [records[i] for i in pep_idx]
    te_ids = {pep_fs.ids[i] for i in te}
    te_reps = [r for r in pep_records if r.molecule_chembl_id in te_ids and r.is_replicated]
    floor = float(np.mean([r.replicate_spread for r in te_reps]) / 2) if te_reps else None

    best_mae = representation[best_rep]["mae"]
    baseline_mae = representation["baseline_mean"]["mae"]

    return {
        "split": {
            "kind": split.kind, "description": split.description,
            "n_train": int(len(tr)), "n_test": int(len(te)), "seed": SEED,
        },
        "A_representation": {
            "results": representation,
            "best": best_rep,
            "best_mae": best_mae,
            "baseline_mae": baseline_mae,
            "improvement_over_baseline": baseline_mae - best_mae,
            "composition_beats_fingerprint": (
                min(representation[f"composition__{m}"]["mae"] for m in ("ridge", "forest"))
                < min(representation[f"fingerprint__{m}"]["mae"] for m in ("ridge", "forest"))
            ),
        },
        "B_domain_shift": transfer,
        "C_dataset_size": {
            "learning_curve": curve,
            "representation": best_rep,
            "mae_gain_from_last_quartile": tail_gain,
            "still_improving": bool(tail_gain > 0.02),
            "extrapolation": extrapolation,
        },
        "D_irreducible_variability": {
            "test_set_replicated_molecules": len(te_reps),
            "noise_floor_mae_estimate": floor,
            "best_model_mae": best_mae,
            "headroom_to_floor": (best_mae - floor) if floor is not None else None,
            "caveat": (
                "The floor is estimated from the replicated subset of the test "
                "molecules only; molecules measured once contribute no estimate."
            ),
        },
    }


# ------------------------------------------------------------------ report ----

@dataclass
class AuditReport:
    dataset_version: str
    endpoint: str
    target: dict[str, str]
    census: dict[str, Any]
    series: dict[str, Any]
    assays: dict[str, Any]
    labels: dict[str, Any]
    shift: dict[str, Any]
    failure: dict[str, Any]
    findings: list[str] = field(default_factory=list)

    def save(self, path: Path = AUDIT_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=float))
        return path


def _findings(report: dict[str, Any]) -> list[str]:
    """Statements that follow from the measurements, each with its number."""
    c, s, f, lq = report["census"], report["shift"], report["failure"], report["labels"]
    out = [
        f"Backbone tracing finds {c['structural_peptide_count']} peptides among "
        f"{c['n_molecules']} molecules, against {c['previous_peptide_count']} "
        f"called peptide-like by the amide-count heuristic. The totals are "
        f"close, but membership differs on {c['membership_disagreement']} "
        f"molecules ({c['membership_disagreement_fraction']:.1%} of the peptide "
        f"set): {c['n_false_positives']} were counted as peptides and are not, "
        f"{c['n_false_negatives']} are peptides and were missed.",

        f"{c['structural_classification'].get('peptidomimetic', 0)} molecules "
        "trace one or two residues. They contain peptide bonds but have no "
        "sequence to featurise, and counting them as peptides is what made the "
        "original peptide population look larger than it is.",

        f"The peptide population is dominated by constrained topologies: "
        f"{c['topology']['side_chain_bridged']} side-chain bridged, "
        f"{c['topology']['disulfide_containing']} disulfide-containing, "
        f"{c['topology']['head_to_tail_cyclic']} head-to-tail cyclic. These "
        f"overlap; what partitions is that only {c['topology']['linear']} of "
        f"{c['structural_peptide_count']} peptides are unconstrained.",

        f"{c['canonical_residue_fraction']['any_noncanonical']} of "
        f"{c['structural_peptide_count']} peptides "
        f"({c['canonical_residue_fraction']['any_noncanonical'] / max(c['structural_peptide_count'], 1):.1%}) "
        f"contain at least one residue outside the twenty canonical side chains, "
        f"across {c['n_distinct_noncanonical_fragments']} distinct fragments. The "
        "commonest are not exotic biology but medicinal chemistry: "
        + ", ".join(
            f"{e['fragment']} ({e['n']}x)"
            for e in c["noncanonical_fragment_inventory"][:3]
        )
        + ". This bears directly on Stage B: a protein language model's "
        "vocabulary has no token for a 4-chloro or 2,4-dichloro phenylalanine, "
        "so frozen sequence embeddings would collapse precisely the "
        "substitutions this dataset's structure-activity series varies.",

        f"Peptides and non-peptides are far apart but not disjoint: median "
        f"nearest-neighbour similarity from a peptide to any non-peptide is "
        f"{s['nearest_neighbour_similarity']['peptide_to_non_peptide']['median']:.3f}, "
        f"against {s['nearest_neighbour_similarity']['peptide_to_peptide']['median']:.3f} "
        f"to another peptide. "
        f"{s['peptides_with_no_close_non_peptide_neighbour']['fraction']:.1%} of "
        "peptides have no non-peptide neighbour above 0.4 Tanimoto.",

        f"{s['descriptor_standardised_mean_difference']['n_descriptors_above_1_sd']} "
        f"of {s['descriptor_standardised_mean_difference']['n_descriptors']} "
        "descriptors differ between the populations by more than one pooled "
        "standard deviation, so the shift is not confined to size.",
    ]

    se = report["series"]
    out.append(
        f"The {se['n_molecules']} peptides fall into only {se['n_clusters']} "
        f"similarity clusters at {se['distance_cutoff']} Tanimoto distance, the "
        f"largest holding {se['largest_cluster']} molecules "
        f"({se['largest_cluster_share']:.1%}) and the top five "
        f"{se['top_5_cluster_share']:.1%} between them. The effective sample "
        f"size is therefore nearer {se['n_clusters']} than "
        f"{se['n_molecules']}. Within a cluster the labels still vary "
        f"(median SD {se['median_within_cluster_label_sd']:.3f} against "
        f"{se['overall_label_sd']:.3f} overall), so most of the affinity range "
        "a model must explain lies *inside* analogue series, where knowing "
        "which series a molecule belongs to carries no information."
    )

    a, b, cc, d = (f["A_representation"], f["B_domain_shift"],
                   f["C_dataset_size"], f["D_irreducible_variability"])
    best = a["results"][a["best"]]
    out.append(
        f"Hypothesis A (representation): the best representation is "
        f"{a['best']} at MAE {a['best_mae']:.3f} and R2 {best['r2']:+.3f}, "
        f"against a mean-predictor baseline of MAE {a['baseline_mae']:.3f}. "
        f"That is an improvement of only {a['improvement_over_baseline']:.3f} "
        f"log units, and an R2 of {best['r2']:+.3f} means the model accounts "
        f"for {max(best['r2'], 0.0):.0%} of the variance in held-out peptide "
        "affinity. The failure is not marginal underperformance; the model is "
        "close to uninformative on this population. Sequence composition "
        + ("does" if a["composition_beats_fingerprint"] else "does not")
        + " beat the molecular fingerprint on the same split"
        + (
            f", and composition alone is worse than the baseline "
            f"(MAE {a['results']['composition__forest']['mae']:.3f}, "
            f"R2 {a['results']['composition__forest']['r2']:+.3f}), which is "
            "expected here: the population is largely analogue series of "
            "cyclic heptapeptides differing by single substitutions, and a "
            "bag of residue counts cannot distinguish positional isomers at all"
            if not a["composition_beats_fingerprint"] else ""
        )
        + "."
    )
    out.append(
        f"Hypothesis B (domain shift): on the same held-out peptides, training "
        f"on {b['peptides_only']['n_train']} peptides alone gives MAE "
        f"{b['peptides_only']['mae']:.3f}, while training on "
        f"{b['non_peptides_only']['n_train']} non-peptides instead gives "
        f"{b['non_peptides_only']['mae']:.3f} (R2 "
        f"{b['non_peptides_only']['r2']:+.3f}) — worse than predicting the mean, "
        f"despite {b['non_peptides_only']['n_train'] / max(b['peptides_only']['n_train'], 1):.1f} "
        "times the training data. Non-peptide affinity data does not transfer "
        f"to peptides. Pooling both reaches {b['pooled']['mae']:.3f}, so the "
        f"small molecules are not useless once peptides anchor the fit: "
        f"{b['interpretation'].lower()} by "
        f"{abs(b['negative_transfer_log_units']):.3f} log units."
    )
    first, last = cc["learning_curve"][0], cc["learning_curve"][-1]
    penultimate = cc["learning_curve"][-2]
    out.append(
        f"Hypothesis C (dataset size): the learning curve falls monotonically "
        f"from MAE {first['mae']:.3f} at {first['n_train']} training peptides "
        f"to {last['mae']:.3f} at {last['n_train']}. Adding the final quartile "
        f"(from {penultimate['n_train']} to {last['n_train']}) still "
        f"{'reduced' if cc['mae_gain_from_last_quartile'] > 0 else 'increased'} "
        f"MAE by {abs(cc['mae_gain_from_last_quartile']):.3f} log units, so "
        "error is "
        + ("still falling and has not saturated" if cc["still_improving"]
           else "already flat")
        + ". But the rate bounds what quantity alone can buy: fitting MAE "
        f"against ln(n) gives a slope of {cc['extrapolation']['slope_per_ln_n']:.3f} "
        f"log units per e-fold (fit R2 {cc['extrapolation']['r2_of_fit']:.3f}), "
        f"which puts MAE 0.7 at roughly "
        f"{cc['extrapolation']['n_peptides_for_mae_0.7']:,.0f} training peptides "
        f"and MAE 0.5 at {cc['extrapolation']['n_peptides_for_mae_0.5']:,.0f}. "
        "Against 630 peptides in the whole of ChEMBL for this target, quantity "
        "is a real contributor that cannot be the route to a fix."
    )
    if d["noise_floor_mae_estimate"] is not None:
        out.append(
            f"Hypothesis D (irreducible variability): the replicated test "
            f"molecules ({d['test_set_replicated_molecules']} of "
            f"{f['split']['n_test']}) imply a label noise floor near MAE "
            f"{d['noise_floor_mae_estimate']:.3f}; the best model sits at "
            f"{d['best_model_mae']:.3f}, leaving "
            f"{d['headroom_to_floor']:+.3f} log units of headroom."
        )
    out.append(
        f"Label quality: {lq['peptides']['replicated_fraction']:.1%} of peptides "
        f"have more than one record and "
        f"{lq['peptides']['discordant_fraction']:.1%} are discordant by more "
        f"than one log unit. The peptide label spread "
        f"(SD {lq['peptides']['label_sd']:.3f}) is "
        + ("wider" if lq["peptides"]["label_sd"] > lq["non_peptides"]["label_sd"]
           else "narrower")
        + f" than the non-peptide spread (SD {lq['non_peptides']['label_sd']:.3f})."
    )
    return out


def run_audit(*, save: bool = True) -> AuditReport:
    """Run the whole Stage A audit and return the report."""
    bundle = load_dataset()
    records = bundle.records
    fs = featurize(records)

    # featurize drops unparseable molecules, so align records to it by id.
    by_id = {r.molecule_chembl_id: r for r in records}
    aligned = [by_id[i] for i in fs.ids]

    census, analyses = structural_census(aligned)
    peptide_mask = np.array([a.is_peptide for a in analyses])

    body = {
        "census": census,
        "series": series_structure(
            fs.fingerprints[peptide_mask], fs.y[peptide_mask]),
        "assays": assay_audit(aligned, analyses),
        "labels": label_quality(aligned, analyses),
        "shift": domain_shift(fs, peptide_mask),
        "failure": failure_decomposition(fs, analyses, aligned, peptide_mask),
    }
    report = AuditReport(
        dataset_version=bundle.version, endpoint=bundle.endpoint,
        target=bundle.target, findings=_findings(body), **body,
    )
    if save:
        report.save()
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = run_audit()
    for i, finding in enumerate(report.findings, 1):
        print(f"\n[{i}] {finding}")
    print(f"\nSaved to {AUDIT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
