# Dockerfile — for Railway deployment (FastAPI app)
#
# Railway uses this to build and run the Graph RAG agent container.
# Neo4j and ChromaDB run as separate Railway services (not in this container).

FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/

# Expose the port Railway will route traffic to
EXPOSE 8000

# Start FastAPI with uvicorn
# Railway sets PORT env var automatically — we default to 8000
CMD ["sh", "-c", "uvicorn src.main:api --host 0.0.0.0 --port ${PORT:-8000}"]
