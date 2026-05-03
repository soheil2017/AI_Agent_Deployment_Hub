#!/bin/sh
# startup.sh — Railway container entrypoint
#
# Starts the API server immediately (so health check passes),
# then runs data ingestion in the background once Neo4j is ready.

# Background job: wait for Neo4j, then ingest
(
    echo "[ingestion] Waiting for Neo4j to be ready..."
    for i in $(seq 1 20); do
        python -c "
from neo4j import GraphDatabase
import os
uri  = os.environ.get('NEO4J_URI', 'NOT SET')
user = os.environ.get('NEO4J_USER', 'neo4j')
pwd  = os.environ.get('NEO4J_PASSWORD', 'NOT SET')
print(f'Connecting to {uri} as {user}')
try:
    d = GraphDatabase.driver(uri, auth=(user, pwd))
    d.verify_connectivity()
    d.close()
    exit(0)
except Exception as e:
    print(f'Error: {e}')
    exit(1)
" && break
        echo "[ingestion] Attempt $i failed, retrying in 10s..."
        sleep 10
    done
    echo "[ingestion] Running ingestion..."
    python ingestion.py
    echo "[ingestion] Done."
) &

# Start API server immediately so the health check passes
echo "DEPLOY_VERSION: v7"
echo "Starting API server..."
uvicorn main:api --host 0.0.0.0 --port ${PORT:-8000}
