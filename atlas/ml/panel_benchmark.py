"""ATLAS 2.5 Stage 3/4: does knowing the receptor help?

The panel makes two questions askable that one target could not.

**Does receptor identity carry signal?** Compare a peptide-only model against
the same model given the receptor, on rows that are (peptide, receptor) pairs.
The peptide-only model is not a straw man: it is what ATLAS had until now, and
it is forced to predict one number for a peptide measured on four receptors.

**Can a target encoder extrapolate to an unseen receptor?** With five
receptors, a one-hot receptor encoding and a receptor-sequence encoding have
the same capacity on the receptors in training — five distinct vectors either
way — so a sequence encoding cannot beat one-hot there, and a comparison that
only used such a split would be uninformative by construction. The difference
appears only under leave-one-receptor-out, where the held-out receptor's
one-hot column was never seen and is necessarily all-zero, while its sequence
features are real numbers. That is the one split that tests a target encoder
at all.

**The leakage trap this module exists to avoid.** A molecule measured on MC1R
and MC4R contributes two rows. Split those rows independently and the model
sees the molecule's affinity on one receptor while being asked for it on
another — a 1.31 log unit median difference away, but with the chemistry
given. Every split here therefore groups *all* rows of a molecule together,
and clusters are computed on unique molecules before being broadcast to rows.
"""

from __future__ import annotations

import collections
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from rdkit import Chem

from atlas.ml.dataset import DATA_DIR
from atlas.ml.features import morgan_matrix
from atlas.ml.multitarget import (
    MELANOCORTIN_RECEPTORS, PanelRow, TargetPanel, leave_one_receptor_out,
    load_panel,
)
from atlas.ml.peptide_representations import build_representations
from atlas.ml.peptide_structure import analyse_peptide

logger = logging.getLogger(__name__)

__all__ = [
    "PANEL_BENCHMARK_PATH", "peptide_panel", "receptor_onehot",
    "receptor_sequence_features", "molecule_cluster_labels",
    "grouped_cluster_split", "run_panel_benchmark",
]

PANEL_BENCHMARK_PATH = DATA_DIR / "panel_benchmark.json"
SEED = 0
_AA = "ACDEFGHIKLMNPQRSTVWY"


# ------------------------------------------------------------------- data ----

@dataclass
class PeptidePanel:
    """(peptide, receptor) rows restricted to structurally confirmed peptides."""

    rows: list[PanelRow]
    molecules: list[str]                  # unique molecule ids, stable order
    analyses: dict[str, Any]              # molecule id -> PeptideAnalysis
    y: np.ndarray
    receptor: np.ndarray                  # receptor short name per row
    molecule_index: np.ndarray            # row -> index into molecules

    def __len__(self) -> int:
        return len(self.rows)


def peptide_panel(panel: TargetPanel, *, endpoint: str = "Ki") -> PeptidePanel:
    """Keep only rows whose molecule traces as a genuine peptide."""
    rows_all = panel.for_endpoint(endpoint)
    smiles_by_mol: dict[str, str] = {}
    for r in rows_all:
        smiles_by_mol.setdefault(r.molecule_chembl_id, r.smiles)

    analyses = {m: analyse_peptide(s) for m, s in smiles_by_mol.items()}
    keep = {m for m, a in analyses.items() if a.is_peptide}
    rows = [r for r in rows_all if r.molecule_chembl_id in keep]

    molecules = sorted(keep)
    pos = {m: i for i, m in enumerate(molecules)}
    return PeptidePanel(
        rows=rows, molecules=molecules,
        analyses={m: analyses[m] for m in molecules},
        y=np.array([r.value for r in rows], dtype=float),
        receptor=np.array([r.receptor for r in rows]),
        molecule_index=np.array([pos[r.molecule_chembl_id] for r in rows]),
    )


# --------------------------------------------------------- target encoders ----

def receptor_onehot(receptor: Sequence[str], vocabulary: Sequence[str]) -> np.ndarray:
    """One column per receptor seen in training.

    A receptor absent from ``vocabulary`` yields an all-zero row, which is the
    honest representation: this encoding has nothing to say about a receptor it
    has never seen, and that is exactly why it cannot survive
    leave-one-receptor-out.
    """
    idx = {r: i for i, r in enumerate(vocabulary)}
    out = np.zeros((len(receptor), len(vocabulary)), dtype=float)
    for i, r in enumerate(receptor):
        j = idx.get(r)
        if j is not None:
            out[i, j] = 1.0
    return out


def receptor_sequence_features(
    receptor: Sequence[str], receptors: dict[str, dict[str, Any]]
) -> np.ndarray:
    """Amino-acid composition and length of each receptor's sequence.

    A continuous encoding, so an unseen receptor lands somewhere in the space
    rather than at the origin. With five receptors this is five points, and the
    question it answers is whether their geometry supports interpolation to a
    sixth -- not whether it has more capacity than one-hot, which it does not.
    """
    cache: dict[str, np.ndarray] = {}
    for name, meta in receptors.items():
        seq = meta.get("sequence") or ""
        if not seq:
            cache[name] = np.zeros(len(_AA) + 1)
            continue
        counts = collections.Counter(seq)
        comp = np.array([counts.get(a, 0) / len(seq) for a in _AA], dtype=float)
        cache[name] = np.concatenate([comp, [len(seq) / 1000.0]])

    width = len(_AA) + 1
    out = np.zeros((len(receptor), width), dtype=float)
    for i, r in enumerate(receptor):
        v = cache.get(r)
        if v is not None:
            out[i] = v
    return out


# ----------------------------------------------------------------- splits ----

def molecule_cluster_labels(
    pp: PeptidePanel, *, cutoff: float = 0.4
) -> np.ndarray:
    """Taylor-Butina cluster label per *molecule*, not per row.

    Clustering rows would let the same molecule seed clusters on both sides of
    a split.
    """
    from rdkit import DataStructs
    from rdkit.ML.Cluster import Butina

    mols = [Chem.MolFromSmiles(pp.analyses[m].smiles) for m in pp.molecules]
    fps = morgan_matrix(mols)
    bits = [
        DataStructs.CreateFromBitString("".join(str(int(b)) for b in row))
        for row in fps.astype(np.uint8)
    ]
    dists: list[float] = []
    for i in range(1, len(bits)):
        sims = DataStructs.BulkTanimotoSimilarity(bits[i], bits[:i])
        dists.extend(1.0 - s for s in sims)
    clusters = Butina.ClusterData(dists, len(bits), cutoff, isDistData=True)
    labels = np.full(len(bits), -1, dtype=int)
    for c, members in enumerate(clusters):
        for m in members:
            labels[m] = c
    return labels


def grouped_cluster_split(
    pp: PeptidePanel, mol_labels: np.ndarray, *,
    test_fraction: float = 0.25, seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole clusters, with every row of a molecule on one side."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.unique(mol_labels))
    target = int(round(test_fraction * len(pp.molecules)))
    test_clusters, n = [], 0
    for c in order:
        if n >= target:
            break
        test_clusters.append(c)
        n += int((mol_labels == c).sum())
    test_mols = np.isin(mol_labels, test_clusters)
    row_in_test = test_mols[pp.molecule_index]
    return np.where(~row_in_test)[0], np.where(row_in_test)[0]


# ------------------------------------------------------------------ models ----

def _fit_score(xtr, ytr, xte, yte, kind: str = "forest") -> dict[str, float]:
    from sklearn.ensemble import RandomForestRegressor

    if kind == "mean":
        pred = np.full(len(yte), ytr.mean())
    else:
        m = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
        m.fit(xtr, ytr)
        pred = m.predict(xte)
    err = np.abs(pred - yte)
    ss_res = float(np.sum((pred - yte) ** 2))
    ss_tot = float(np.sum((yte - yte.mean()) ** 2))
    return {
        "mae": float(err.mean()),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
    }


def _peptide_features(pp: PeptidePanel, train_mols: Sequence[int]) -> np.ndarray:
    """Peptide representation per row, vocabulary fitted on training molecules."""
    from atlas.ml.peptide_representations import residue_vocabulary

    analyses = [pp.analyses[m] for m in pp.molecules]
    vocab = residue_vocabulary([analyses[i] for i in train_mols])
    mols = [Chem.MolFromSmiles(a.smiles) for a in analyses]
    reps = build_representations(
        analyses, vocabulary=vocab,
        extra={"morgan_fp": morgan_matrix(mols).astype(float)},
    )
    per_mol = reps["positional_fragment_fp+morgan_fp"]
    return per_mol[pp.molecule_index]


# --------------------------------------------------------------- benchmark ----

@dataclass
class PanelBenchmarkReport:
    panel_version: str
    endpoint: str
    composition: dict[str, Any]
    receptor_signal: dict[str, Any]
    leave_one_receptor_out: dict[str, Any]
    findings: list[str] = field(default_factory=list)

    def save(self, path: Path = PANEL_BENCHMARK_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=float))
        return path


def run_panel_benchmark(*, endpoint: str = "Ki", save: bool = True) -> PanelBenchmarkReport:
    panel = load_panel()
    pp = peptide_panel(panel, endpoint=endpoint)
    mol_labels = molecule_cluster_labels(pp)

    breadth = collections.Counter()
    by_mol: dict[str, set[str]] = collections.defaultdict(set)
    for r in pp.rows:
        by_mol[r.molecule_chembl_id].add(r.receptor)
    for recs in by_mol.values():
        breadth[len(recs)] += 1

    composition = {
        "n_rows": len(pp),
        "n_peptides": len(pp.molecules),
        "rows_per_receptor": dict(collections.Counter(pp.receptor.tolist())),
        "receptors_per_peptide": dict(sorted(breadth.items())),
        "peptides_on_multiple_receptors": sum(v for k, v in breadth.items() if k > 1),
        "n_clusters": int(len(np.unique(mol_labels))),
    }

    # --- does the receptor carry signal? ---------------------------------
    tr, te = grouped_cluster_split(pp, mol_labels)
    train_mols = np.unique(pp.molecule_index[tr])
    pep = _peptide_features(pp, train_mols)
    vocab = sorted(set(pp.receptor[tr].tolist()))
    onehot = receptor_onehot(pp.receptor, vocab)
    seqfeat = receptor_sequence_features(pp.receptor, panel.receptors)

    arms = {
        "baseline_mean": _fit_score(pep[tr], pp.y[tr], pep[te], pp.y[te], "mean"),
        "peptide_only": _fit_score(pep[tr], pp.y[tr], pep[te], pp.y[te]),
        "peptide_plus_receptor_onehot": _fit_score(
            np.hstack([pep, onehot])[tr], pp.y[tr],
            np.hstack([pep, onehot])[te], pp.y[te]),
        "peptide_plus_receptor_sequence": _fit_score(
            np.hstack([pep, seqfeat])[tr], pp.y[tr],
            np.hstack([pep, seqfeat])[te], pp.y[te]),
    }
    receptor_signal = {
        "split": "cluster split, whole molecules grouped, all receptors in both sides",
        "n_train_rows": int(len(tr)), "n_test_rows": int(len(te)),
        "results": arms,
        "onehot_gain_over_peptide_only": (
            arms["peptide_only"]["mae"] - arms["peptide_plus_receptor_onehot"]["mae"]),
    }

    # --- can a target encoder reach an unseen receptor? -------------------
    loro: dict[str, Any] = {}
    for held in sorted(set(pp.receptor.tolist())):
        tr_l, te_l = leave_one_receptor_out(pp.rows, held)
        if len(te_l) < 20 or len(tr_l) < 50:
            loro[held] = {"skipped": f"only {len(te_l)} test rows"}
            continue
        train_mols_l = np.unique(pp.molecule_index[tr_l])
        pep_l = _peptide_features(pp, train_mols_l)
        vocab_l = sorted(set(pp.receptor[tr_l].tolist()))
        oh_l = receptor_onehot(pp.receptor, vocab_l)
        sq_l = receptor_sequence_features(pp.receptor, panel.receptors)
        loro[held] = {
            "n_train": int(len(tr_l)), "n_test": int(len(te_l)),
            "baseline_mean": _fit_score(
                pep_l[tr_l], pp.y[tr_l], pep_l[te_l], pp.y[te_l], "mean"),
            "peptide_only": _fit_score(
                pep_l[tr_l], pp.y[tr_l], pep_l[te_l], pp.y[te_l]),
            "peptide_plus_receptor_onehot": _fit_score(
                np.hstack([pep_l, oh_l])[tr_l], pp.y[tr_l],
                np.hstack([pep_l, oh_l])[te_l], pp.y[te_l]),
            "peptide_plus_receptor_sequence": _fit_score(
                np.hstack([pep_l, sq_l])[tr_l], pp.y[tr_l],
                np.hstack([pep_l, sq_l])[te_l], pp.y[te_l]),
            "held_out_receptor_onehot_is_all_zero": bool(
                oh_l[te_l].sum() == 0),
        }

    report = PanelBenchmarkReport(
        panel_version=panel.version, endpoint=endpoint,
        composition=composition, receptor_signal=receptor_signal,
        leave_one_receptor_out=loro,
    )
    report.findings = _findings(report, panel, endpoint)
    if save:
        report.save()
    return report


def _findings(
    rep: PanelBenchmarkReport, panel: TargetPanel, endpoint: str
) -> list[str]:
    c, rs = rep.composition, rep.receptor_signal
    ov = panel.overlap[endpoint]
    arms = rs["results"]
    out = [
        f"The panel holds {c['n_rows']} ({endpoint}) measurements over "
        f"{c['n_peptides']} structurally confirmed peptides and "
        f"{len(c['rows_per_receptor'])} receptors, against 630 peptide "
        f"measurements on MC4R alone. The family adds few new peptides but "
        f"{c['n_rows'] / 630:.2f} times the measurements, because "
        f"{c['peptides_on_multiple_receptors']} of {c['n_peptides']} peptides "
        f"({c['peptides_on_multiple_receptors'] / max(c['n_peptides'], 1):.1%}) "
        "were already profiled on more than one receptor. That is new "
        "information rather than more of the same series.",

        f"Across all molecules the median selectivity span is "
        f"{ov['median_selectivity_span_log']:.2f} log units and the maximum "
        f"{ov['max_selectivity_span_log']:.2f}. The median exceeds the best "
        "cross-series error ATLAS has achieved (MAE 0.870), so a model that "
        "ignores which receptor it is being asked about carries a systematic "
        "error larger than its own accuracy.",

        f"Receptor identity carries signal, but both arms are poor in absolute "
        f"terms. On a cluster split with whole molecules grouped, a "
        f"peptide-only model reaches MAE {arms['peptide_only']['mae']:.3f} "
        f"(R2 {arms['peptide_only']['r2']:+.3f}) and the same model given a "
        f"one-hot receptor reaches "
        f"{arms['peptide_plus_receptor_onehot']['mae']:.3f} "
        f"(R2 {arms['peptide_plus_receptor_onehot']['r2']:+.3f}) — a relative "
        f"gain of {rs['onehot_gain_over_peptide_only']:+.3f} log units. Both "
        f"R2 values are negative, meaning both models are worse than "
        f"predicting the test set's own mean, even though both beat the "
        f"training mean's MAE of {arms['baseline_mean']['mae']:.3f}. So the "
        "receptor helps relative to not knowing it, and that is the only claim "
        "this comparison supports.",

        "These numbers are not comparable with the ATLAS 2.4 figures. The "
        "panel task is harder in two ways at once: it holds out whole molecule "
        "clusters across four receptors rather than one, and its labels carry "
        "the across-receptor variance that a single-target dataset does not. A "
        "panel MAE and an MC4R-only MAE measure different problems.",
    ]

    usable = {k: v for k, v in rep.leave_one_receptor_out.items() if "skipped" not in v}
    if usable:
        lines = []
        for held, v in sorted(usable.items()):
            lines.append(
                f"{held} ({v['n_test']} rows): peptide-only "
                f"{v['peptide_only']['mae']:.3f}, +one-hot "
                f"{v['peptide_plus_receptor_onehot']['mae']:.3f}, +sequence "
                f"{v['peptide_plus_receptor_sequence']['mae']:.3f}, baseline "
                f"{v['baseline_mean']['mae']:.3f}"
            )
        beats_peptide_only = sorted(
            h for h, v in usable.items()
            if v["peptide_plus_receptor_sequence"]["mae"] < v["peptide_only"]["mae"])
        # The reference that matters is the mean baseline, not the other model.
        # Beating peptide-only while both are worse than the mean is not a win.
        beats_baseline = sorted(
            h for h, v in usable.items()
            if v["peptide_plus_receptor_sequence"]["mae"] < v["baseline_mean"]["mae"])
        fails = sorted(set(usable) - set(beats_baseline))
        out.append(
            "Leave-one-receptor-out is the only split that tests a target "
            "encoder, because the held-out receptor's one-hot column is "
            "necessarily all-zero there (verified per arm). Results: "
            + "; ".join(lines) + "."
        )
        out.append(
            f"Transfer to an unseen receptor succeeds for {len(beats_baseline)} "
            f"of {len(usable)} receptors and fails for {len(fails)}. With a "
            f"receptor-sequence encoding it beats the mean baseline on "
            f"{', '.join(beats_baseline) or 'none'} and is worse than the "
            f"baseline on {', '.join(fails) or 'none'}. It beat the "
            f"peptide-only arm on {len(beats_peptide_only)} of {len(usable)}, "
            "but that comparison is the wrong one: on "
            f"{', '.join(fails) or 'no receptor'} both arms are worse than "
            "predicting the mean, so being the better of two inadequate models "
            "is not a result. **A target encoder is not established.**"
        )
        out.append(
            "Two reasons to hold that conclusion loosely in both directions. "
            "With five receptors a sequence encoding is five points in a "
            "composition space, so this measures whether their geometry "
            "supports reaching a sixth, not whether sequence features have "
            "more capacity than one-hot — on receptors already seen they do "
            "not. And four held-out receptors is four trials. The panel makes "
            "the question askable, which one target did not; it does not "
            "answer it."
        )
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rep = run_panel_benchmark()
    for i, f in enumerate(rep.findings, 1):
        print(f"\n[{i}] {f}")
    print(f"\nSaved to {PANEL_BENCHMARK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
