"""
sync_kb.py — Trigger a Bedrock Knowledge Base ingestion job.

Run this after uploading new documents to the S3 bucket so the KB index
stays up to date.

Usage:
    python scripts/sync_kb.py

Required environment variables (set in .env or shell):
    BEDROCK_KB_ID          Knowledge Base ID (from SAM outputs)
    BEDROCK_DATA_SOURCE_ID Data Source ID (from SAM outputs)
    AWS_REGION             AWS region (default: us-east-1)
"""

import os
import time
import logging
import boto3

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

KB_ID = os.environ["BEDROCK_KB_ID"]
DS_ID = os.environ["BEDROCK_DATA_SOURCE_ID"]
REGION = os.environ.get("AWS_REGION", "us-east-1")
POLL_INTERVAL = 10  # seconds

_client = boto3.client("bedrock-agent", region_name=REGION)


def start_ingestion() -> str:
    logger.info("Starting ingestion job for KB=%s DS=%s", KB_ID, DS_ID)
    response = _client.start_ingestion_job(
        knowledgeBaseId=KB_ID,
        dataSourceId=DS_ID,
    )
    job_id = response["ingestionJob"]["ingestionJobId"]
    logger.info("Ingestion job started: %s", job_id)
    return job_id


def wait_for_completion(job_id: str) -> None:
    while True:
        response = _client.get_ingestion_job(
            knowledgeBaseId=KB_ID,
            dataSourceId=DS_ID,
            ingestionJobId=job_id,
        )
        job = response["ingestionJob"]
        status = job["status"]
        stats = job.get("statistics", {})
        logger.info(
            "Status: %s | Scanned: %s | Indexed: %s | Failed: %s",
            status,
            stats.get("numberOfDocumentsScanned", "-"),
            stats.get("numberOfNewDocumentsIndexed", "-"),
            stats.get("numberOfDocumentsFailed", "-"),
        )

        if status == "COMPLETE":
            logger.info("Ingestion complete.")
            return
        if status == "FAILED":
            failures = job.get("failureReasons", [])
            raise RuntimeError(f"Ingestion failed: {failures}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    job_id = start_ingestion()
    wait_for_completion(job_id)
