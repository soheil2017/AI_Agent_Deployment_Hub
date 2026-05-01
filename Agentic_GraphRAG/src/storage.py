"""
storage.py — Two database clients used across the project.

  Neo4jClient   — graph database (entities + relationships)
  ChromaClient  — vector database (semantic document search)

Observability (traces, spans, scores) is handled entirely by Langfuse —
no separate logging database is needed.

Both clients are module-level singletons so Railway container warm starts
reuse connections.
"""

import logging
import os

import chromadb
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

# ── Environment variables ─────────────────────────────────────────────────────

# Neo4j — provided by Railway as a service URL
NEO4J_URI      = os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.environ.get("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD",  "password")

# ChromaDB — provided by Railway as a service URL
CHROMA_HOST       = os.environ.get("CHROMA_HOST",        "localhost")
CHROMA_PORT       = int(os.environ.get("CHROMA_PORT",    "8000"))
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION",  "conduit_docs")


# ── Neo4j ─────────────────────────────────────────────────────────────────────

class Neo4jClient:
    """Thin wrapper around the Neo4j driver. Runs Cypher queries."""

    def __init__(self):
        self._driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        logger.info("Neo4jClient: connected to %s", NEO4J_URI)

    def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        """Run a read query and return all records as plain dicts."""
        params = params or {}
        with self._driver.session() as session:
            result = session.run(cypher, params)
            records = [dict(record) for record in result]
        logger.info("Neo4jClient.query: returned %d records", len(records))
        return records

    def write(self, cypher: str, params: dict | None = None) -> None:
        """Run a write query (CREATE / MERGE)."""
        params = params or {}
        with self._driver.session() as session:
            session.run(cypher, params)

    def close(self):
        self._driver.close()


# ── ChromaDB ──────────────────────────────────────────────────────────────────

class ChromaClient:
    """
    Wrapper around ChromaDB for semantic document search.

    Documents stored here:
      - FHIR spec excerpts
      - Payer prior-auth policies
      - CARIN Blue Button / TEFCA guidelines
      - Clinical notes
    """

    def __init__(self):
        # HTTP mode when CHROMA_USE_HTTP=true (Railway remote ChromaDB service)
        # Persistent local mode otherwise (local dev)
        if os.environ.get("CHROMA_USE_HTTP", "false").lower() == "true":
            self._client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        else:
            self._client = chromadb.PersistentClient(path="/tmp/chroma")

        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaClient: collection=%s docs=%d",
            CHROMA_COLLECTION, self._collection.count(),
        )

    def add(self, doc_id: str, text: str, metadata: dict | None = None) -> None:
        """Add a single document (ChromaDB computes the embedding automatically)."""
        self._collection.upsert(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata or {}],
        )

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Return the top_k most semantically similar documents.
        Each result: {id, text, metadata, distance}
        """
        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, max(self._collection.count(), 1)),
        )
        docs = []
        for i, doc_id in enumerate(results["ids"][0]):
            docs.append({
                "id":       doc_id,
                "text":     results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        logger.info("ChromaClient.search: returned %d results", len(docs))
        return docs

    @property
    def count(self) -> int:
        return self._collection.count()


# ── Module-level singletons (reused across container requests) ────────────────

_neo4j:  Neo4jClient  | None = None
_chroma: ChromaClient | None = None


def get_neo4j() -> Neo4jClient:
    global _neo4j
    if _neo4j is None:
        _neo4j = Neo4jClient()
    return _neo4j


def get_chroma() -> ChromaClient:
    global _chroma
    if _chroma is None:
        _chroma = ChromaClient()
    return _chroma
