"""
retriever.py — FAISS vector search backed by S3-persisted index.

On first call the index is downloaded from S3 and cached in /tmp so
subsequent invocations within the same Lambda container skip the download.
"""

import os
import json
import pickle
import tempfile
import logging
from pathlib import Path

import boto3
import faiss
import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)

# ── module-level cache (survives Lambda warm starts) ────────────────────────
_faiss_index: faiss.Index | None = None
_chunks: list[str] = []

EMBED_MODEL = "text-embedding-3-small"
TOP_K = int(os.environ.get("TOP_K", "5"))
S3_BUCKET = os.environ.get("S3_BUCKET", "")
INDEX_KEY = os.environ.get("INDEX_KEY", "faiss_index/index.faiss")
CHUNKS_KEY = os.environ.get("CHUNKS_KEY", "faiss_index/chunks.pkl")
TMP_INDEX = Path(tempfile.gettempdir()) / "index.faiss"
TMP_CHUNKS = Path(tempfile.gettempdir()) / "chunks.pkl"

_openai = OpenAI()
_s3 = boto3.client("s3")


def _download_index() -> None:
    """Download FAISS index and chunk list from S3 into /tmp."""
    logger.info("Downloading FAISS index from s3://%s/%s", S3_BUCKET, INDEX_KEY)
    _s3.download_file(S3_BUCKET, INDEX_KEY, str(TMP_INDEX))
    _s3.download_file(S3_BUCKET, CHUNKS_KEY, str(TMP_CHUNKS))


def _load_index() -> None:
    global _faiss_index, _chunks
    if not TMP_INDEX.exists():
        _download_index()
    _faiss_index = faiss.read_index(str(TMP_INDEX))
    with open(TMP_CHUNKS, "rb") as f:
        _chunks = pickle.load(f)
    logger.info("Loaded FAISS index with %d vectors", _faiss_index.ntotal)


def _embed(text: str) -> np.ndarray:
    response = _openai.embeddings.create(input=text, model=EMBED_MODEL)
    vec = np.array(response.data[0].embedding, dtype="float32")
    return vec.reshape(1, -1)


def search(query: str, top_k: int = TOP_K) -> list[str]:
    """Return the top-k most relevant text chunks for *query*."""
    if _faiss_index is None:
        _load_index()

    query_vec = _embed(query)
    distances, indices = _faiss_index.search(query_vec, top_k)

    results: list[str] = []
    for idx in indices[0]:
        if 0 <= idx < len(_chunks):
            results.append(_chunks[idx])
    return results
