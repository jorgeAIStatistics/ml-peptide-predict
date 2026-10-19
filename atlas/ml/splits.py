"""Dataset splits, ordered by how honestly they estimate generalisation.

A random split of a medicinal-chemistry dataset is optimistic: analogue series
share a scaffold, so near-identical molecules land on both sides and the model
is scored largely on compounds it has effectively seen. A scaffold split keeps
whole Bemis-Murcko series together and answers the question that matters --
how well does this transfer to a chemotype it was not fitted on?

The modality split goes further: it trains on one chemical population and tests
on the other, which is the direct test of whether a model fitted on small
molecules may be deployed on peptides.

All three are provided so the difference between them can be *reported*. That
gap is itself the finding.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from atlas.ml.features import FeatureSet, Modality

__all__ = ["Split", "random_split", "scaffold_split", "modality_split", "SPLIT_KINDS"]

SPLIT_KINDS = ("random", "scaffold", "modality_peptide_holdout",
               "modality_smallmol_holdout")


@dataclass(frozen=True)
class Split:
    """Train/test index arrays plus a description of how they were made."""

    kind: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    description: str
    seed: int | None = None

    def __post_init__(self) -> None:
        overlap = set(self.train_idx.tolist()) & set(self.test_idx.tolist())
        if overlap:
            raise ValueError(
                f"{self.kind} split leaks {len(overlap)} indices between "
                "train and test"
            )

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_test(self) -> int:
        return len(self.test_idx)


def random_split(fs: FeatureSet, *, test_fraction: float = 0.2, seed: int = 0) -> Split:
    """Uniformly random split. Optimistic; included as a reference point."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(fs))
    n_test = int(round(len(fs) * test_fraction))
    return Split(
        kind="random",
        train_idx=np.sort(idx[n_test:]),
        test_idx=np.sort(idx[:n_test]),
        description=(
            "Uniformly random assignment. Analogue series are split across "
            "train and test, so this overestimates generalisation to new "
            "chemotypes."
        ),
        seed=seed,
    )


def scaffold_split(fs: FeatureSet, *, test_fraction: float = 0.2, seed: int = 0) -> Split:
    """Bemis-Murcko scaffold split: whole series go to one side.

    Scaffold groups are assigned largest-first to the training set until it is
    full, which is the standard deterministic construction; the seed only
    shuffles groups of equal size.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, scaf in enumerate(fs.scaffolds):
        # Molecules with no ring system share the empty scaffold; keeping them
        # as one group is conservative, since it prevents acyclic analogues
        # from being split across the boundary.
        groups[scaf].append(i)

    rng = np.random.default_rng(seed)
    ordered = sorted(groups.values(), key=lambda g: (-len(g), rng.random()))

    n_test_target = int(round(len(fs) * test_fraction))
    train: list[int] = []
    test: list[int] = []
    for group in ordered:
        if len(test) < n_test_target:
            test.extend(group)
        else:
            train.extend(group)

    # The largest groups go to train first in the conventional formulation;
    # here the first groups fill the test set, so swap if that inverted it.
    if len(train) < len(test):
        train, test = test, train

    return Split(
        kind="scaffold",
        train_idx=np.sort(np.array(train, dtype=int)),
        test_idx=np.sort(np.array(test, dtype=int)),
        description=(
            f"Bemis-Murcko scaffold split over {len(groups)} scaffold groups. "
            "No scaffold appears on both sides, so the test set measures "
            "transfer to unseen chemotypes."
        ),
        seed=seed,
    )


def modality_split(fs: FeatureSet, *, holdout: Modality) -> Split:
    """Train on one chemical modality, test on the other.

    This is the deployment question stated plainly: a model fitted on small
    molecules is being asked to score peptides, or the reverse.
    """
    test_mask = fs.mask(holdout)
    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]
    other = (
        Modality.SMALL_MOLECULE_LIKE
        if holdout is Modality.PEPTIDE_LIKE
        else Modality.PEPTIDE_LIKE
    )
    kind = (
        "modality_peptide_holdout"
        if holdout is Modality.PEPTIDE_LIKE
        else "modality_smallmol_holdout"
    )
    return Split(
        kind=kind,
        train_idx=train_idx,
        test_idx=test_idx,
        description=(
            f"Trained on {other.value} molecules only and tested on "
            f"{holdout.value} molecules. Measures transfer across chemical "
            "modality rather than within it."
        ),
    )


def leakage_report(fs: FeatureSet, split: Split) -> dict[str, float | int]:
    """Quantify how much the test set resembles the training set.

    Shared scaffolds and identical molecules are the two concrete leakage
    routes; both are counted rather than assumed absent.
    """
    train_scaf = {fs.scaffolds[i] for i in split.train_idx}
    test_scaf = {fs.scaffolds[i] for i in split.test_idx}
    train_smiles = {fs.smiles[i] for i in split.train_idx}
    test_smiles = {fs.smiles[i] for i in split.test_idx}
    shared = train_scaf & test_scaf
    return {
        "n_train_scaffolds": len(train_scaf),
        "n_test_scaffolds": len(test_scaf),
        "n_shared_scaffolds": len(shared),
        "fraction_test_scaffolds_seen": (
            len(shared) / len(test_scaf) if test_scaf else 0.0
        ),
        "n_identical_molecules": len(train_smiles & test_smiles),
    }
