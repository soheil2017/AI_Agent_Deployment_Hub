"""
indexer.py — Build and upload a FAISS vector index from documents stored in S3.

Usage (run locally or as a one-off Lambda/ECS task):
    python src/indexer.py

Environment variables:
    OPENAI_API_KEY  — OpenAI API key
    S3_BUCKET       — S3 bucket name
    DOCS_PREFIX     — S3 prefix where raw .txt documents live  (default: "docs/")
    INDEX_KEY       — S3 key for the FAISS index file          (default: "faiss_index/index.faiss")
    CHUNKS_KEY      — S3 key for the pickled chunk list        (default: "faiss_index/chunks.pkl")
    CHUNK_SIZE      — characters per chunk                     (default: 512)
    CHUNK_OVERLAP   — character overlap between chunks         (default: 64)
"""

import os
import pickle
import logging
import tempfile
from pathlib import Path

import boto3
import faiss
import numpy as np
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── configuration ────────────────────────────────────────────────────────────
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536  # dimension for text-embedding-3-small
BATCH_SIZE = 100  # embeddings per API call

S3_BUCKET = os.environ["S3_BUCKET"]
DOCS_PREFIX = os.environ.get("DOCS_PREFIX", "docs/")
INDEX_KEY = os.environ.get("INDEX_KEY", "faiss_index/index.faiss")
CHUNKS_KEY = os.environ.get("CHUNKS_KEY", "faiss_index/chunks.pkl")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "64"))

_openai = OpenAI()
_s3 = boto3.client("s3")


# ── helpers ──────────────────────────────────────────────────────────────────

def list_docs(bucket: str, prefix: str) -> list[str]:
    """Return all S3 keys under *prefix* that end with .txt."""
    keys: list[str] = []
    paginator = _s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".txt"):
                keys.append(obj["Key"])
    logger.info("Found %d document(s) in s3://%s/%s", len(keys), bucket, prefix)
    return keys


def load_text(bucket: str, key: str) -> str:
    response = _s3.get_object(Bucket=bucket, Key=key)
    return response["Body"].read().decode("utf-8")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split *text* into overlapping fixed-size character chunks."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start += size - overlap
    return [c for c in chunks if c]


def embed_chunks(chunks: list[str]) -> np.ndarray:
    """Embed all chunks in batches; returns (N, EMBED_DIM) float32 array."""
    all_vecs: list[list[float]] = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        logger.info("Embedding batch %d-%d / %d", i, i + len(batch), len(chunks))
        response = _openai.embeddings.create(input=batch, model=EMBED_MODEL)
        all_vecs.extend([item.embedding for item in response.data])
    return np.array(all_vecs, dtype="float32")


def build_index(vectors: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    logger.info("Built FAISS index with %d vectors", index.ntotal)
    return index


def upload(local_path: str, bucket: str, key: str) -> None:
    logger.info("Uploading %s → s3://%s/%s", local_path, bucket, key)
    _s3.upload_file(local_path, bucket, key)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Collect and chunk all documents
    keys = list_docs(S3_BUCKET, DOCS_PREFIX)
    if not keys:
        raise RuntimeError(
            f"No .txt files found at s3://{S3_BUCKET}/{DOCS_PREFIX}. "
            "Upload documents before running the indexer."
        )

    all_chunks: list[str] = []
    for key in keys:
        text = load_text(S3_BUCKET, key)
        chunks = chunk_text(text)
        logger.info("  %s → %d chunk(s)", key, len(chunks))
        all_chunks.extend(chunks)

    logger.info("Total chunks: %d", len(all_chunks))

    # 2. Embed chunks
    vectors = embed_chunks(all_chunks)

    # 3. Build FAISS index
    index = build_index(vectors)

    # 4. Save locally then upload to S3
    with tempfile.TemporaryDirectory() as tmpdir:
        idx_path = str(Path(tmpdir) / "index.faiss")
        pkl_path = str(Path(tmpdir) / "chunks.pkl")

        faiss.write_index(index, idx_path)
        with open(pkl_path, "wb") as f:
            pickle.dump(all_chunks, f)

        upload(idx_path, S3_BUCKET, INDEX_KEY)
        upload(pkl_path, S3_BUCKET, CHUNKS_KEY)

    logger.info("Indexing complete.")


if __name__ == "__main__":
    main()
