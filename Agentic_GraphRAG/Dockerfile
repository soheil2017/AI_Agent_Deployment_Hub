# Dockerfile — for Railway deployment (FastAPI app)
#
# Railway uses this to build and run the Graph RAG agent container.
# Neo4j and ChromaDB run as separate Railway services (not in this container).

FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and startup script
COPY src/ ./src/
COPY startup.sh ./src/startup.sh

# Move into src so all local imports (graph, nodes, tools…) resolve correctly
WORKDIR /app/src

RUN chmod +x startup.sh

# Expose the port Railway will route traffic to
EXPOSE 8000

# Run ingestion then start the API server
CMD ./startup.sh
