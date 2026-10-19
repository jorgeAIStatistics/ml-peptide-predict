"""Molecular featurisation, modality classification and applicability domain.

Featurisation is deliberately modality-aware. The MC4R dataset spans molecular
weights from 190 to 5166 Da: a drug-like small molecule and a cyclic peptide are
different chemical populations, and descriptor sets designed for the former do
not necessarily describe the latter. Each molecule is therefore labelled with
its modality so that transfer between populations can be *measured* rather than
assumed, which is the point of the experiment in :mod:`atlas.ml.train`.

The applicability domain is Tanimoto similarity to the training set. A test
molecule whose nearest training neighbour is remote is outside the domain the
model learned, and a prediction for it should be treated as an extrapolation
regardless of how confident the model appears.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)

__all__ = [
    "Modality",
    "classify_modality",
    "descriptor_names",
    "compute_descriptors",
    "morgan_matrix",
    "featurize",
    "murcko_scaffold",
    "applicability",
    "FeatureSet",
]

#: A molecule counts as peptide-like with at least this many amide bonds and
#: this molecular weight. Chosen from the dataset's own bimodal structure, not
#: from an external convention; both thresholds must hold.
PEPTIDE_MIN_AMIDES = 4
PEPTIDE_MIN_MW = 600.0

_AMIDE = Chem.MolFromSmarts("[NX3][CX3](=[OX1])")

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


class Modality(str, enum.Enum):
    PEPTIDE_LIKE = "peptide_like"
    SMALL_MOLECULE_LIKE = "small_molecule_like"
    UNPARSEABLE = "unparseable"


def classify_modality(mol: Chem.Mol | None) -> Modality:
    if mol is None:
        return Modality.UNPARSEABLE
    amides = len(mol.GetSubstructMatches(_AMIDE))
    mw = Descriptors.MolWt(mol)
    if amides >= PEPTIDE_MIN_AMIDES and mw >= PEPTIDE_MIN_MW:
        return Modality.PEPTIDE_LIKE
    return Modality.SMALL_MOLECULE_LIKE


#: Interpretable physicochemical descriptors. Kept small and named so that a
#: feature attribution can be read by a chemist.
_DESCRIPTORS: list[tuple[str, callable]] = [
    ("mol_weight", Descriptors.MolWt),
    ("clogp", Descriptors.MolLogP),
    ("tpsa", Descriptors.TPSA),
    ("h_donors", rdMolDescriptors.CalcNumHBD),
    ("h_acceptors", rdMolDescriptors.CalcNumHBA),
    ("rotatable_bonds", rdMolDescriptors.CalcNumRotatableBonds),
    ("rings", rdMolDescriptors.CalcNumRings),
    ("aromatic_rings", rdMolDescriptors.CalcNumAromaticRings),
    ("fraction_csp3", rdMolDescriptors.CalcFractionCSP3),
    ("heavy_atoms", lambda m: float(m.GetNumHeavyAtoms())),
    ("formal_charge", lambda m: float(Chem.GetFormalCharge(m))),
    ("amide_bonds", lambda m: float(len(m.GetSubstructMatches(_AMIDE)))),
    ("stereocentres", lambda m: float(len(Chem.FindMolChiralCenters(
        m, includeUnassigned=True, useLegacyImplementation=False)))),
    ("molar_refractivity", Descriptors.MolMR),
    ("qed", Descriptors.qed),
]


def descriptor_names() -> list[str]:
    return [n for n, _ in _DESCRIPTORS]


def compute_descriptors(mol: Chem.Mol) -> np.ndarray:
    out = np.empty(len(_DESCRIPTORS), dtype=float)
    for i, (name, fn) in enumerate(_DESCRIPTORS):
        try:
            out[i] = float(fn(mol))
        except Exception:  # a single failing descriptor must not void the row
            logger.debug("descriptor %s failed", name)
            out[i] = np.nan
    return out


def morgan_matrix(mols: list[Chem.Mol]) -> np.ndarray:
    """ECFP4-style binary fingerprints, 2048 bits."""
    arr = np.zeros((len(mols), 2048), dtype=np.uint8)
    for i, m in enumerate(mols):
        fp = _MORGAN.GetFingerprint(m)
        onbits = np.frombuffer(
            bytes(DataStructs.BitVectToBinaryText(fp)), dtype=np.uint8
        )
        arr[i] = np.unpackbits(onbits)[:2048]
    return arr


def murcko_scaffold(mol: Chem.Mol) -> str:
    """Bemis-Murcko scaffold SMILES; '' when the molecule has no ring system."""
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf) if scaf is not None else ""
    except Exception:
        return ""


@dataclass
class FeatureSet:
    """Featurised dataset with everything a split or audit needs."""

    ids: list[str]
    smiles: list[str]
    y: np.ndarray
    descriptors: np.ndarray
    fingerprints: np.ndarray
    modality: list[Modality]
    scaffolds: list[str]
    mols: list[Chem.Mol]
    descriptor_names: list[str]

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def combined(self) -> np.ndarray:
        """Descriptors concatenated with fingerprints."""
        return np.hstack([self.descriptors, self.fingerprints.astype(float)])

    def mask(self, modality: Modality) -> np.ndarray:
        return np.array([m is modality for m in self.modality])

    def subset(self, idx: np.ndarray) -> "FeatureSet":
        idx = np.asarray(idx)
        return FeatureSet(
            ids=[self.ids[i] for i in idx],
            smiles=[self.smiles[i] for i in idx],
            y=self.y[idx],
            descriptors=self.descriptors[idx],
            fingerprints=self.fingerprints[idx],
            modality=[self.modality[i] for i in idx],
            scaffolds=[self.scaffolds[i] for i in idx],
            mols=[self.mols[i] for i in idx],
            descriptor_names=self.descriptor_names,
        )


def featurize(records) -> FeatureSet:
    """Featurise activity records, dropping any molecule RDKit cannot parse."""
    ids, smiles, ys, mols = [], [], [], []
    for r in records:
        mol = Chem.MolFromSmiles(r.smiles)
        if mol is None:
            logger.warning("unparseable SMILES for %s; dropped", r.molecule_chembl_id)
            continue
        ids.append(r.molecule_chembl_id)
        smiles.append(r.smiles)
        ys.append(r.pki)
        mols.append(mol)

    desc = np.vstack([compute_descriptors(m) for m in mols])
    # Impute the rare failed descriptor with the column median; record nothing
    # silently -- the count is logged.
    n_nan = int(np.isnan(desc).sum())
    if n_nan:
        logger.warning("imputing %d missing descriptor values with column medians", n_nan)
        medians = np.nanmedian(desc, axis=0)
        inds = np.where(np.isnan(desc))
        desc[inds] = np.take(medians, inds[1])

    return FeatureSet(
        ids=ids,
        smiles=smiles,
        y=np.asarray(ys, dtype=float),
        descriptors=desc,
        fingerprints=morgan_matrix(mols),
        modality=[classify_modality(m) for m in mols],
        scaffolds=[murcko_scaffold(m) for m in mols],
        mols=mols,
        descriptor_names=descriptor_names(),
    )


def applicability(
    train_fps: np.ndarray, test_fps: np.ndarray
) -> np.ndarray:
    """Max Tanimoto similarity of each test molecule to the training set.

    Low values mean the model is extrapolating: the molecule does not resemble
    anything it was fitted on.
    """
    train = train_fps.astype(np.float32)
    test = test_fps.astype(np.float32)
    inter = test @ train.T
    t_sum = test.sum(axis=1, keepdims=True)
    r_sum = train.sum(axis=1, keepdims=True).T
    union = t_sum + r_sum - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        tan = np.where(union > 0, inter / union, 0.0)
    return tan.max(axis=1)
