"""Embedding service.

Primary: local sentence-transformers model (all-MiniLM-L6-v2). Runs offline,
needs no API key, downloads ~90MB on first use and caches under ~/.cache/.

Fallback (only if sentence-transformers isn't importable for some reason): a
deterministic hash-based bag-of-words embedding. Lower quality, but keeps the
pipeline running rather than hard-failing.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Literal

import numpy as np

log = logging.getLogger(__name__)

InputType = Literal["document", "query"]

_LOCAL_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_FALLBACK_DIM = 256

_local_model = None
_local_load_attempted = False


def _get_local_model():
    """Lazily load the local embedding model. Returns None if it can't load."""
    global _local_model, _local_load_attempted
    if _local_load_attempted:
        return _local_model
    _local_load_attempted = True
    try:
        from sentence_transformers import SentenceTransformer

        log.info("loading local embedding model %s (first run downloads ~90MB)", _LOCAL_MODEL_NAME)
        _local_model = SentenceTransformer(_LOCAL_MODEL_NAME)
        log.info("local embedding model ready")
    except Exception as exc:
        log.warning("could not load local embedding model: %s — using hash fallback", exc)
        _local_model = None
    return _local_model


def _hash_embed(text: str, dim: int = _FALLBACK_DIM) -> list[float]:
    """Deterministic hashed bag-of-words embedding. Only used if the local model fails to load."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", text.lower())
    for tok in tokens:
        h = hashlib.blake2b(tok.encode(), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = 1.0 if h[4] & 1 else -1.0
        vec[idx] += sign
    n = float(np.linalg.norm(vec))
    if n > 0:
        vec /= n
    else:
        vec[0] = 1.0
    return vec.tolist()


def embed(texts: list[str], input_type: InputType = "document") -> list[list[float]]:
    """Embed a list of texts. `input_type` is accepted for API compatibility but ignored —
    the local model treats documents and queries identically."""
    if not texts:
        return []
    model = _get_local_model()
    if model is None:
        return [_hash_embed(t) for t in texts]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_one(text: str, input_type: InputType = "document") -> list[float]:
    return embed([text], input_type=input_type)[0]


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def is_local_model_available() -> bool:
    return _get_local_model() is not None
