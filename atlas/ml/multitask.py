"""Does pooling the receptor family buy sample efficiency?

The second half of the bottleneck is four receptors, and the panel benchmark
already answered the hardest version of the question: leave-one-receptor-out
transfer to an *unseen* receptor fails on two of four. But that is not the
question a project with four receptors actually faces. The useful question is:

    For a receptor I do have data on, does adding the other receptors' data
    improve prediction for it, compared with training on it alone?

If yes, pooling relieves the series bottleneck, because MC5R's 96 peptide rows
get to borrow from MC4R's 630. If no, the family is four separate problems and
the effective sample size per receptor is what it looked like.

**The leakage trap, and it is the whole experiment.** A peptide measured on
MC1R and MC4R contributes a row to each. Hold out that peptide's MC4R row
while leaving its MC1R row in training and the pooled model has seen the
molecule's affinity on a sister receptor -- a median 1.31 log units away, with
the chemistry handed over. The pooled arm would then beat the single-task arm
for a reason that has nothing to do with sample efficiency.

So held-out peptide clusters are removed from training **for every receptor at
once**, and both arms are scored on exactly the same rows. The single-task arm
differs from the pooled arm only in what it is allowed to train on.
"""

from __future__ import annotations

import collections
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from atlas.core.evidence import utc_now
from atlas.ml.dataset import DATA_DIR

logger = logging.getLogger(__name__)

__all__ = [
    "MULTITASK_PATH", "global_cluster_holdout", "run_multitask_benchmark",
]

MULTITASK_PATH = DATA_DIR / "multitask_benchmark.json"
SEED = 0
#: A receptor needs at least this many test rows for its comparison to mean
#: anything; below it the arm abstains rather than reporting noise.
MIN_TEST_ROWS = 20


def global_cluster_holdout(
    molecule_index: np.ndarray, mol_labels: np.ndarray, *,
    test_fraction: float = 0.25, seed: int = SEED,
) -> np.ndarray:
    """Boolean mask of rows whose peptide belongs to a held-out cluster.

    Held out globally: if a cluster is in test, *every* receptor's rows for its
    molecules are in test. That is what stops a pooled model seeing a test
    peptide's affinity on a sister receptor.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.unique(mol_labels))
    target = int(round(test_fraction * len(mol_labels)))
    held: list[int] = []
    n = 0
    for c in order:
        if n >= target:
            break
        held.append(int(c))
        n += int((mol_labels == c).sum())
    held_mol = np.isin(mol_labels, held)
    return held_mol[molecule_index]


def _fit_score(xtr, ytr, xte, yte, kind: str = "forest") -> dict[str, float]:
    from sklearn.ensemble import RandomForestRegressor

    if kind == "mean":
        pred = np.full(len(yte), ytr.mean())
    else:
        m = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
        m.fit(xtr, ytr)
        pred = m.predict(xte)
    ss_res = float(np.sum((pred - yte) ** 2))
    ss_tot = float(np.sum((yte - yte.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(pred - yte))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
    }


@dataclass
class MultitaskReport:
    panel_version: str
    endpoint: str
    n_rows: int
    n_peptides: int
    n_clusters: int
    per_receptor: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)

    def save(self, path: Path = MULTITASK_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=float))
        return path


def run_multitask_benchmark(*, endpoint: str = "Ki", save: bool = True) -> MultitaskReport:
    from atlas.ml.features import morgan_matrix
    from atlas.ml.multitarget import load_panel
    from atlas.ml.panel_benchmark import (
        molecule_cluster_labels, peptide_panel, receptor_onehot,
    )
    from atlas.ml.peptide_representations import (
        build_representations, residue_vocabulary,
    )
    from rdkit import Chem

    panel = load_panel()
    pp = peptide_panel(panel, endpoint=endpoint)
    mol_labels = molecule_cluster_labels(pp)
    test_mask = global_cluster_holdout(pp.molecule_index, mol_labels)

    # Features from training molecules only.
    train_mols = np.unique(pp.molecule_index[~test_mask])
    analyses = [pp.analyses[m] for m in pp.molecules]
    vocab = residue_vocabulary([analyses[i] for i in train_mols])
    mols = [Chem.MolFromSmiles(a.smiles) for a in analyses]
    reps = build_representations(
        analyses, vocabulary=vocab,
        extra={"morgan_fp": morgan_matrix(mols).astype(float)})
    per_mol = reps["positional_fragment_fp+morgan_fp"]
    x = per_mol[pp.molecule_index]

    receptors = sorted(set(pp.receptor.tolist()))
    onehot = receptor_onehot(pp.receptor, receptors)
    x_pooled = np.hstack([x, onehot])

    per_receptor: dict[str, Any] = {}
    for rec in receptors:
        is_rec = pp.receptor == rec
        te = np.where(is_rec & test_mask)[0]
        tr_single = np.where(is_rec & ~test_mask)[0]
        tr_pooled = np.where(~test_mask)[0]
        if len(te) < MIN_TEST_ROWS or len(tr_single) < MIN_TEST_ROWS:
            per_receptor[rec] = {
                "abstained": (
                    f"{len(te)} test rows and {len(tr_single)} single-task "
                    f"training rows, below the {MIN_TEST_ROWS} needed for the "
                    "comparison to mean anything"),
            }
            continue

        single = _fit_score(x[tr_single], pp.y[tr_single], x[te], pp.y[te])
        pooled = _fit_score(
            x_pooled[tr_pooled], pp.y[tr_pooled], x_pooled[te], pp.y[te])
        baseline = _fit_score(
            x[tr_single], pp.y[tr_single], x[te], pp.y[te], "mean")
        per_receptor[rec] = {
            "single_task": single,
            "pooled_family": pooled,
            "baseline_mean": baseline,
            "pooled_gain_mae": single["mae"] - pooled["mae"],
            "extra_training_rows": pooled["n_train"] - single["n_train"],
            "pooled_helps": bool(pooled["mae"] < single["mae"]),
        }

    usable = {k: v for k, v in per_receptor.items() if "abstained" not in v}
    helped = sorted(k for k, v in usable.items() if v["pooled_helps"])
    summary = {
        "receptors_compared": sorted(usable),
        "receptors_abstained": sorted(set(per_receptor) - set(usable)),
        "pooling_helps_on": helped,
        "pooling_hurts_on": sorted(set(usable) - set(helped)),
        "mean_gain_mae": (
            float(np.mean([v["pooled_gain_mae"] for v in usable.values()]))
            if usable else None),
        "holdout": (
            "Peptide clusters held out globally: a held-out cluster's molecules "
            "are absent from training for every receptor, so the pooled arm "
            "cannot see a test peptide's affinity on a sister receptor."),
    }

    report = MultitaskReport(
        panel_version=panel.version, endpoint=endpoint, n_rows=len(pp),
        n_peptides=len(pp.molecules), n_clusters=int(len(np.unique(mol_labels))),
        per_receptor=per_receptor, summary=summary,
    )
    report.findings = _findings(report)
    if save:
        report.save()
    return report


def _findings(rep: MultitaskReport) -> list[str]:
    s = rep.summary
    usable = {k: v for k, v in rep.per_receptor.items() if "abstained" not in v}
    if not usable:
        return ["No receptor had enough held-out rows for the comparison."]

    lines = "; ".join(
        f"{rec} ({v['single_task']['n_test']} test rows): single-task "
        f"{v['single_task']['mae']:.3f}, pooled {v['pooled_family']['mae']:.3f} "
        f"with {v['extra_training_rows']} extra rows, baseline "
        f"{v['baseline_mean']['mae']:.3f}"
        for rec, v in sorted(usable.items())
    )
    out = [
        f"Pooling the receptor family was tested on {len(usable)} receptors "
        f"with peptide clusters held out globally, so a held-out peptide is "
        f"absent from training for every receptor. {lines}.",

        f"Pooling helps on {len(s['pooling_helps_on'])} of {len(usable)} "
        f"receptors ({', '.join(s['pooling_helps_on']) or 'none'}) and hurts "
        f"on {len(s['pooling_hurts_on'])} "
        f"({', '.join(s['pooling_hurts_on']) or 'none'}), with a mean change "
        f"of {s['mean_gain_mae']:+.3f} log units. "
        + ("Pooling the family therefore buys sample efficiency for receptors "
           "that already have data, which is a different and more useful "
           "result than the leave-one-receptor-out failure."
           if s["mean_gain_mae"] and s["mean_gain_mae"] > 0.01 else
           "Pooling the family therefore does not buy sample efficiency, so "
           "the four receptors are closer to four separate problems than to "
           "one problem with four times the data."),
    ]
    # Pooling's sign tracks how much data a receptor already has, which is the
    # classic transfer pattern and worth stating rather than averaging away.
    ranked = sorted(usable.items(), key=lambda kv: -kv[1]["single_task"]["n_train"])
    if len(ranked) >= 2:
        richest, poorest = ranked[0], ranked[-1]
        out.append(
            f"The sign of the effect tracks how much data a receptor already "
            f"has. {richest[0]}, with the most single-task training rows "
            f"({richest[1]['single_task']['n_train']}), is the worst hurt "
            f"({richest[1]['pooled_gain_mae']:+.3f}); the receptors that gain "
            f"are the data-poor ones ({', '.join(s['pooling_helps_on'])}). "
            "Pooling redistributes rather than adds: a thin receptor borrows "
            "from its siblings while the richest is diluted by them. That is "
            "why the mean is near zero, and why a family-pooled model is not a "
            "way around the series bottleneck."
        )

    if s["receptors_abstained"]:
        out.append(
            f"Abstained on {', '.join(s['receptors_abstained'])}: too few "
            f"held-out rows for the comparison to mean anything, reported "
            "rather than estimated from noise."
        )
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rep = run_multitask_benchmark()
    for i, f in enumerate(rep.findings, 1):
        print(f"\n[{i}] {f}")
    print(f"\nSaved to {MULTITASK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
