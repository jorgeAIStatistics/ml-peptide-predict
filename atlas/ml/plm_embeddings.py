"""ATLAS 2.6: protein language model embeddings, and what they are worth.

Five representation extensions have now failed to improve cross-series
generalisation on this dataset -- explicit topology, residue chemistry, local
sequence context, assay context and backbone stereochemistry. Every one of
them rearranged information already present in the same 630 molecules. A
pretrained protein language model is the only lever available that imports
information from *outside* the dataset: ESM-2 was trained on UniRef50, tens of
millions of sequences, none of them these peptides.

That is the argument for trying it. The argument for expecting little is
measured and specific:

* **The model is handed X's.** 80.5% of these peptides carry residues outside
  the canonical twenty, across 160 distinct side chains, and
  :func:`atlas.ml.peptide_representations.canonical_sequence` renders every one
  as ``X`` because no twenty-letter alphabet can name them. ESM-2's vocabulary
  contains ``X`` as a single unknown token, so a 4-chloro-phenylalanine, a
  2-naphthylalanine and an Aib are the same input symbol to it.
* **The ceiling is known in advance.** A frozen embedding is a deterministic
  function of the sequence string, so two molecules with the same string get
  identical vectors and no model built on them can separate the two. Stage B
  measured that bound: oracle MAE 0.382 for the canonical twenty-letter
  sequence. The best real model is at 0.870, so capacity is *not* the
  constraint -- there is ample headroom, and a null result here cannot be
  explained by the ceiling.

So the experiment is informative either way, which is why it is run rather
than argued about.

**What is frozen and what is fitted.** The language model's weights are never
updated and it never sees a label. Each sequence is embedded independently of
every other, with no batch statistics and no cross-row normalisation, so
computing all 630 vectors in one pass is not leakage: the result for one
molecule does not depend on which other molecules were present. The only
fitted objects are the downstream ridge and forest, which
:mod:`atlas.ml.peptide_benchmark` fits on training rows alone exactly as it
does for every other representation.

**Why the vectors are cached on disk.** The embedding of a sequence is a
constant, the dataset has 630 molecules collapsing to far fewer distinct
sequences, and a 2.5 GB model download is not a test dependency. The cache is
committed, so the benchmark and the suite reproduce the arm without torch
installed; rebuilding it needs torch and transformers and is a deliberate act
(``python3 -m atlas.ml.plm_embeddings``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from atlas.ml.dataset import DATA_DIR
from atlas.ml.peptide_representations import canonical_sequence

logger = logging.getLogger(__name__)

__all__ = [
    "PLM_EMBEDDING_PATH", "PLM_MODEL", "PLM_DIM", "PlmEmbeddings",
    "distinct_sequences", "embed_sequences", "load_embeddings",
    "plm_matrix", "build_embeddings",
]

PLM_EMBEDDING_PATH = DATA_DIR / "plm_embeddings.npz"

#: ESM-2 650M, the size published frozen-embedding work uses for property
#: prediction. Recorded in the cache so a stored vector can never be silently
#: attributed to a different model.
PLM_MODEL = "facebook/esm2_t33_650M_UR50D"
PLM_DIM = 1280

#: A peptide whose residues could not be traced has no sequence to embed. It
#: receives a zero vector rather than being dropped, so the feature matrix
#: stays aligned with every other representation, and the count of such rows
#: is reported rather than absorbed.
_EMPTY = ""


@dataclass
class PlmEmbeddings:
    """Frozen per-sequence vectors, with the provenance to interpret them."""

    model: str
    dim: int
    vectors: dict[str, np.ndarray]
    pooling: str = "mean_over_residue_tokens"
    versions: dict[str, str] = field(default_factory=dict)
    n_sequences: int = 0

    def __contains__(self, seq: str) -> bool:
        return seq in self.vectors

    def __getitem__(self, seq: str) -> np.ndarray:
        return self.vectors[seq]


def distinct_sequences(analyses: Sequence[Any]) -> list[str]:
    """The distinct canonical sequences to embed, in a stable order.

    Deduplicating here is not an optimisation detail but the same fact the
    ceiling analysis reports: 630 molecules do not have 630 sequences. Sorting
    makes the cache byte-reproducible across runs.
    """
    return sorted({canonical_sequence(a) for a in analyses})


def embed_sequences(
    sequences: Sequence[str], *, model_name: str = PLM_MODEL,
    batch_size: int = 16,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Mean-pooled ESM-2 representations, one vector per sequence string.

    Pooling is the mean over *residue* token positions. The flanking ``<cls>``
    and ``<eos>`` tokens are excluded: ESM-2 is trained with a masked-language
    objective and no sentence-level head, so its first token carries no pooled
    summary the way a fine-tuned BERT's does, and averaging it in would mix a
    position-marker into the residue average.

    torch and transformers are imported here rather than at module scope so
    that reading the cache -- which is what the benchmark and the test suite
    do -- needs neither installed.
    """
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, dtype=torch.float32)
    model.eval()

    real = [s for s in sequences if s != _EMPTY]
    out: dict[str, np.ndarray] = {}
    if _EMPTY in sequences:
        out[_EMPTY] = np.zeros(model.config.hidden_size, dtype=np.float32)

    with torch.inference_mode():
        for start in range(0, len(real), batch_size):
            chunk = real[start:start + batch_size]
            enc = tok(list(chunk), return_tensors="pt", padding=True)
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].clone()
            lengths = mask.sum(dim=1)
            mask[:, 0] = 0                                   # <cls>
            for row, n in enumerate(lengths.tolist()):
                mask[row, n - 1] = 0                         # <eos>
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            denom = weights.sum(dim=1).clamp(min=1.0)
            pooled = (hidden * weights).sum(dim=1) / denom
            for seq, vec in zip(chunk, pooled):
                out[seq] = vec.to(torch.float32).numpy()
            logger.info("embedded %d/%d sequences", min(start + batch_size,
                                                        len(real)), len(real))

    versions = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
    }
    return out, versions


def load_embeddings(path: Path = PLM_EMBEDDING_PATH) -> PlmEmbeddings:
    """Read the cached vectors. Raises if the cache is absent."""
    with np.load(path, allow_pickle=False) as z:
        seqs = [str(s) for s in z["sequences"]]
        mat = np.asarray(z["vectors"], dtype=float)
        model = str(z["model"].item() if z["model"].shape == () else z["model"][0])
        pooling = str(z["pooling"].item() if z["pooling"].shape == ()
                      else z["pooling"][0])
        version_keys = [k for k in z.files if k.startswith("version_")]
        versions = {k[len("version_"):]: str(z[k].item()) for k in version_keys}
    if len(seqs) != len(mat):
        raise ValueError(
            f"{path.name} holds {len(seqs)} sequences but {len(mat)} vectors")
    return PlmEmbeddings(
        model=model, dim=int(mat.shape[1]),
        vectors={s: mat[i] for i, s in enumerate(seqs)},
        pooling=pooling, versions=versions, n_sequences=len(seqs),
    )


def plm_matrix(
    analyses: Sequence[Any], embeddings: PlmEmbeddings, *, strict: bool = True,
) -> np.ndarray:
    """One embedding row per molecule, aligned to ``analyses``.

    ``strict`` is the default because the alternative is worse than an error:
    substituting a zero row for a sequence missing from the cache would make
    the arm score badly for a reason that has nothing to do with the language
    model, and the benchmark would report that as a measurement.
    """
    missing = sorted(
        {canonical_sequence(a) for a in analyses} - set(embeddings.vectors))
    if missing and strict:
        raise KeyError(
            f"{len(missing)} sequence(s) absent from the embedding cache "
            f"(first: {missing[0]!r}); rebuild with "
            "`python3 -m atlas.ml.plm_embeddings`"
        )
    out = np.zeros((len(analyses), embeddings.dim), dtype=float)
    for row, a in enumerate(analyses):
        vec = embeddings.vectors.get(canonical_sequence(a))
        if vec is not None:
            out[row] = vec
    return out


def build_embeddings(
    *, save: bool = True, path: Path = PLM_EMBEDDING_PATH,
    model_name: str = PLM_MODEL,
) -> dict[str, Any]:
    """Embed every distinct canonical sequence in the saved dataset, once."""
    from atlas.ml.dataset import load_dataset
    from atlas.ml.features import featurize
    from atlas.ml.peptide_structure import analyse_peptide

    bundle = load_dataset()
    fs = featurize(bundle.records)
    analyses = [analyse_peptide(s) for s in fs.smiles]
    pep = [a for a in analyses if a.is_peptide]
    seqs = distinct_sequences(pep)
    logger.info("%d peptides -> %d distinct canonical sequences", len(pep),
                len(seqs))

    vectors, versions = embed_sequences(seqs, model_name=model_name)
    mat = np.vstack([vectors[s] for s in seqs]).astype(np.float32)

    report = {
        "model": model_name,
        "dim": int(mat.shape[1]),
        "n_peptides": len(pep),
        "n_distinct_sequences": len(seqs),
        "sequences_per_peptide": len(seqs) / max(len(pep), 1),
        "pooling": "mean_over_residue_tokens",
        "versions": versions,
        "dataset_version": bundle.version,
        "path": str(path),
    }
    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, sequences=np.array(seqs), vectors=mat,
            model=np.array(model_name), pooling=np.array(report["pooling"]),
            **{f"version_{k}": np.array(v) for k, v in versions.items()},
        )
        logger.info("saved %s", path)
    return report


def main() -> int:
    import json

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(json.dumps(build_embeddings(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
