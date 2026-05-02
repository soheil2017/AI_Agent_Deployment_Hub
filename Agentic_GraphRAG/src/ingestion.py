"""
ingestion.py — Load synthetic Conduit/dental-health data into Neo4j and ChromaDB.

Run once (or re-run to refresh) before starting the agent:
    python ingestion.py

What gets loaded:
  Neo4j  — patients, providers, organizations, conditions, procedures,
            referrals, prior-auth requests, and all relationships between them.
  Chroma — policy documents, FHIR spec excerpts, payer guidelines,
            CARIN Blue Button rules, TEFCA/OHIA specs.

The data is synthetic but realistic for a dental-interoperability company.
"""

import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from storage import get_neo4j, get_chroma

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Synthetic graph data ──────────────────────────────────────────────────────
#
# Think of this as a small but realistic slice of Conduit's world:
#   - 3 patients moving between dental and medical systems
#   - dental + medical providers at different organizations
#   - FHIR-based referrals (medical → dental and dental → medical)
#   - prior-auth requests at different stages
#   - conditions and procedures using real code systems
#

PATIENTS = [
    {"id": "P001", "name": "Alice Johnson",  "dob": "1985-03-12", "member_id": "UHC-88821"},
    {"id": "P002", "name": "Bob Martinez",   "dob": "1972-07-04", "member_id": "AET-44512"},
    {"id": "P003", "name": "Clara Lee",      "dob": "1990-11-30", "member_id": "BCB-99034"},
]

PROVIDERS = [
    {"npi": "1234567890", "name": "Dr. Sarah Chen",    "specialty": "General Dentistry",     "type": "dental"},
    {"npi": "2345678901", "name": "Dr. James Patel",   "specialty": "Periodontology",         "type": "dental"},
    {"npi": "3456789012", "name": "Dr. Maria Gomez",   "specialty": "Internal Medicine",      "type": "medical"},
    {"npi": "4567890123", "name": "Dr. Kevin Brown",   "specialty": "Cardiology",             "type": "medical"},
]

ORGANIZATIONS = [
    {"id": "ORG001", "name": "Bright Smiles Dental Group",  "type": "DSO"},
    {"id": "ORG002", "name": "Metro Health System",          "type": "health_system"},
    {"id": "ORG003", "name": "United Health Payer",          "type": "payer"},
    {"id": "ORG004", "name": "Aetna Dental",                 "type": "payer"},
]

CONDITIONS = [
    {"code": "K05.3",  "system": "ICD-10", "description": "Chronic periodontitis"},
    {"code": "I10",    "system": "ICD-10", "description": "Essential hypertension"},
    {"code": "K02.9",  "system": "ICD-10", "description": "Dental caries, unspecified"},
    {"code": "E11.9",  "system": "ICD-10", "description": "Type 2 diabetes mellitus"},
]

PROCEDURES = [
    {"code": "D4341", "system": "CDT", "description": "Periodontal scaling and root planing — 4+ teeth per quadrant"},
    {"code": "D0330", "system": "CDT", "description": "Panoramic radiographic image"},
    {"code": "99213", "system": "CPT", "description": "Office visit — established patient, moderate complexity"},
    {"code": "D2740", "system": "CDT", "description": "Crown — porcelain/ceramic substrate"},
]

REFERRALS = [
    {
        "id": "REF001",
        "type": "medical_to_dental",
        "status": "pending",
        "created_date": "2026-04-15",
        "patient_id": "P001",
        "from_npi": "3456789012",   # Dr. Gomez (medical)
        "to_npi":   "1234567890",   # Dr. Chen (dental)
        "reason": "Evaluate periodontal disease — patient has uncontrolled hypertension",
    },
    {
        "id": "REF002",
        "type": "dental_to_medical",
        "status": "completed",
        "created_date": "2026-03-20",
        "patient_id": "P002",
        "from_npi": "2345678901",   # Dr. Patel (dental)
        "to_npi":   "4567890123",   # Dr. Brown (cardiology)
        "reason": "Dental clearance before cardiac surgery",
    },
    {
        "id": "REF003",
        "type": "dental_to_dental",
        "status": "pending",
        "created_date": "2026-04-28",
        "patient_id": "P003",
        "from_npi": "1234567890",   # Dr. Chen
        "to_npi":   "2345678901",   # Dr. Patel (periodontist)
        "reason": "Specialist referral for advanced periodontal treatment",
    },
]

PRIOR_AUTHS = [
    {
        "id": "PA001",
        "status": "pending",
        "payer_org_id": "ORG003",
        "submitted_date": "2026-04-16",
        "decision_date": None,
        "referral_id": "REF001",
        "procedure_code": "D4341",
    },
    {
        "id": "PA002",
        "status": "approved",
        "payer_org_id": "ORG004",
        "submitted_date": "2026-03-21",
        "decision_date": "2026-03-25",
        "referral_id": "REF002",
        "procedure_code": "D2740",
    },
]

# Patient → condition and patient → procedure assignments
PATIENT_CONDITIONS = [
    ("P001", "K05.3"), ("P001", "I10"),
    ("P002", "K02.9"), ("P002", "E11.9"),
    ("P003", "K05.3"),
]

PATIENT_PROCEDURES = [
    ("P001", "D4341"), ("P001", "D0330"),
    ("P002", "D2740"), ("P002", "99213"),
    ("P003", "D0330"),
]

PROVIDER_ORGS = [
    ("1234567890", "ORG001"),
    ("2345678901", "ORG001"),
    ("3456789012", "ORG002"),
    ("4567890123", "ORG002"),
]


# ── Policy and specification documents (go into ChromaDB) ─────────────────────

POLICY_DOCUMENTS = [
    {
        "id": "doc_fhir_r4_overview",
        "text": (
            "FHIR R4 (Fast Healthcare Interoperability Resources Release 4) is the HL7 standard "
            "for exchanging healthcare information electronically. Conduit uses FHIR R4 as its "
            "single interchange format. Key resources: Patient, Practitioner, Organization, "
            "Condition, Procedure, ServiceRequest (referral), Claim, ClaimResponse, Coverage. "
            "All dental and medical data is normalized into these FHIR resources before transmission."
        ),
        "metadata": {"source": "fhir_spec", "topic": "overview"},
    },
    {
        "id": "doc_prior_auth_rules",
        "text": (
            "Prior authorization (prior auth) is required by most payers before dental procedures "
            "such as crowns (D2740), periodontal scaling (D4341), and orthodontics. "
            "Conduit routes prior-auth requests via FHIR ClaimResponse resources. "
            "The Da Vinci Prior Authorization Support (PAS) Implementation Guide defines the "
            "standard workflow: submit a Claim resource → payer returns ClaimResponse with "
            "status approved/denied/pending. Typical turnaround: 3–5 business days. "
            "Pending requests older than 7 days should trigger follow-up."
        ),
        "metadata": {"source": "payer_policy", "topic": "prior_auth"},
    },
    {
        "id": "doc_carin_blue_button",
        "text": (
            "CARIN Blue Button (Consumer-Directed Payer Data Exchange) requires payers to expose "
            "patient claims data via a FHIR R4 API. Patients can access their EOBs (Explanation "
            "of Benefits) through any certified app. Conduit implements the CARIN IG for Blue "
            "Button to enable patients to pull their dental claims into consumer health apps. "
            "Key resources: ExplanationOfBenefit, Coverage, Patient. "
            "The API must support OAuth 2.0 with SMART on FHIR scopes."
        ),
        "metadata": {"source": "carin_ig", "topic": "patient_access"},
    },
    {
        "id": "doc_tefca",
        "text": (
            "TEFCA (Trusted Exchange Framework and Common Agreement) establishes a universal "
            "floor for health information exchange across the US. Conduit participates in TEFCA "
            "as a QHIN-connected service, enabling dental records to flow to medical EHRs and "
            "vice versa. TEFCA supports four exchange purposes: Treatment, Payment, Healthcare "
            "Operations, and Individual Access. Dental referrals fall under Treatment."
        ),
        "metadata": {"source": "tefca_spec", "topic": "interoperability"},
    },
    {
        "id": "doc_medical_dental_referral",
        "text": (
            "Medical-to-dental referrals are a core Conduit use case. A physician (e.g., "
            "cardiologist or internist) sends a FHIR ServiceRequest to a dentist via Conduit. "
            "Common scenarios: cardiac patients needing dental clearance before surgery, "
            "diabetic patients requiring periodontal evaluation, oncology patients needing "
            "pre-chemo dental assessment. Conduit transforms the ServiceRequest into the "
            "target PMS format and delivers it electronically, replacing fax-based referrals."
        ),
        "metadata": {"source": "conduit_product", "topic": "referral_workflow"},
    },
    {
        "id": "doc_dental_data_isolation",
        "text": (
            "Dental data has historically been isolated from the rest of healthcare. "
            "Dental practice management systems (PMS) such as Dentrix, Eaglesoft, and Open Dental "
            "use proprietary formats incompatible with medical EHRs. "
            "Conduit solves this with a FHIR R4 bridge that connects any PMS to any EHR, "
            "payer, or consumer health app. Data harmonization resolves duplicate patient records "
            "across PMS platforms using deterministic matching on name, DOB, and member ID."
        ),
        "metadata": {"source": "conduit_product", "topic": "data_harmonization"},
    },
    {
        "id": "doc_ohia",
        "text": (
            "The Oral Health Interoperability Alliance (OHIA) is an industry coalition working "
            "to establish standards for dental data exchange. Conduit is an active contributor "
            "across all three OHIA implementation tracks: "
            "1) Dental-Medical Data Exchange, 2) Prior Authorization Automation, "
            "3) Patient Access and Consumer APIs. "
            "FHIR-based medical-dental referral transformation has been demonstrated in live "
            "standards testing environments by Conduit."
        ),
        "metadata": {"source": "ohia_spec", "topic": "standards"},
    },
    {
        "id": "doc_pms_vendors",
        "text": (
            "Conduit connects to major dental PMS vendors: Dentrix (Henry Schein), "
            "Eaglesoft (Patterson), Open Dental (open source), Curve Dental (cloud), "
            "Carestream Dental, and others. Each PMS has its own data model. "
            "Conduit maps all PMS fields to FHIR R4 resources through configurable "
            "transformation templates. DSOs (Dental Service Organizations) with multiple "
            "locations often run multiple PMS platforms — Conduit harmonizes patient records "
            "across all of them."
        ),
        "metadata": {"source": "conduit_product", "topic": "pms_integration"},
    },
]


# ── Neo4j ingestion ───────────────────────────────────────────────────────────

def ingest_to_neo4j() -> None:
    """Create all graph nodes and relationships in Neo4j."""
    db = get_neo4j()

    # Clear existing data (for re-runs during development)
    db.write("MATCH (n) DETACH DELETE n")
    logger.info("Neo4j: cleared existing data")

    # Patients
    for p in PATIENTS:
        db.write(
            "CREATE (:Patient {id: $id, name: $name, dob: $dob, member_id: $member_id})",
            p,
        )

    # Providers
    for prov in PROVIDERS:
        db.write(
            "CREATE (:Provider {npi: $npi, name: $name, specialty: $specialty, type: $type})",
            prov,
        )

    # Organizations
    for org in ORGANIZATIONS:
        db.write(
            "CREATE (:Organization {id: $id, name: $name, type: $type})",
            org,
        )

    # Conditions
    for c in CONDITIONS:
        db.write(
            "CREATE (:Condition {code: $code, system: $system, description: $description})",
            c,
        )

    # Procedures
    for proc in PROCEDURES:
        db.write(
            "CREATE (:Procedure {code: $code, system: $system, description: $description})",
            proc,
        )

    # Referrals
    for ref in REFERRALS:
        db.write(
            "CREATE (:Referral {id: $id, type: $type, status: $status, "
            "created_date: $created_date, reason: $reason})",
            ref,
        )

    # Prior Auths
    for pa in PRIOR_AUTHS:
        db.write(
            "CREATE (:PriorAuth {id: $id, status: $status, submitted_date: $submitted_date, "
            "decision_date: $decision_date, procedure_code: $procedure_code})",
            pa,
        )

    logger.info("Neo4j: created all nodes")

    # ── Relationships ──────────────────────────────────────────────────────────

    # Patient HAS_CONDITION
    for patient_id, cond_code in PATIENT_CONDITIONS:
        db.write(
            "MATCH (p:Patient {id: $pid}), (c:Condition {code: $code}) "
            "CREATE (p)-[:HAS_CONDITION]->(c)",
            {"pid": patient_id, "code": cond_code},
        )

    # Patient HAD_PROCEDURE
    for patient_id, proc_code in PATIENT_PROCEDURES:
        db.write(
            "MATCH (p:Patient {id: $pid}), (pr:Procedure {code: $code}) "
            "CREATE (p)-[:HAD_PROCEDURE]->(pr)",
            {"pid": patient_id, "code": proc_code},
        )

    # Provider WORKS_AT Organization
    for npi, org_id in PROVIDER_ORGS:
        db.write(
            "MATCH (prov:Provider {npi: $npi}), (org:Organization {id: $org_id}) "
            "CREATE (prov)-[:WORKS_AT]->(org)",
            {"npi": npi, "org_id": org_id},
        )

    # Referral relationships
    for ref in REFERRALS:
        db.write(
            "MATCH (pat:Patient {id: $pid}), (r:Referral {id: $rid}) "
            "CREATE (pat)-[:HAS_REFERRAL]->(r)",
            {"pid": ref["patient_id"], "rid": ref["id"]},
        )
        db.write(
            "MATCH (from_prov:Provider {npi: $from_npi}), (r:Referral {id: $rid}) "
            "CREATE (from_prov)-[:SENT_REFERRAL]->(r)",
            {"from_npi": ref["from_npi"], "rid": ref["id"]},
        )
        db.write(
            "MATCH (to_prov:Provider {npi: $to_npi}), (r:Referral {id: $rid}) "
            "CREATE (r)-[:ASSIGNED_TO]->(to_prov)",
            {"to_npi": ref["to_npi"], "rid": ref["id"]},
        )

    # PriorAuth → Referral and PriorAuth → Payer Organization
    for pa in PRIOR_AUTHS:
        db.write(
            "MATCH (r:Referral {id: $rid}), (auth:PriorAuth {id: $paid}) "
            "CREATE (r)-[:REQUIRES_AUTH]->(auth)",
            {"rid": pa["referral_id"], "paid": pa["id"]},
        )
        db.write(
            "MATCH (org:Organization {id: $org_id}), (auth:PriorAuth {id: $paid}) "
            "CREATE (auth)-[:SUBMITTED_TO]->(org)",
            {"org_id": pa["payer_org_id"], "paid": pa["id"]},
        )

    logger.info("Neo4j: created all relationships")


# ── ChromaDB ingestion ────────────────────────────────────────────────────────

def ingest_to_chroma() -> None:
    """Embed and store policy/spec documents into ChromaDB."""
    db = get_chroma()
    for doc in POLICY_DOCUMENTS:
        db.add(doc_id=doc["id"], text=doc["text"], metadata=doc["metadata"])
    logger.info("ChromaDB: ingested %d documents", len(POLICY_DOCUMENTS))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting data ingestion…")
    ingest_to_neo4j()
    ingest_to_chroma()
    logger.info("Ingestion complete.")
    logger.info("Neo4j nodes: patients=%d providers=%d referrals=%d prior_auths=%d",
                len(PATIENTS), len(PROVIDERS), len(REFERRALS), len(PRIOR_AUTHS))
    logger.info("ChromaDB docs: %d", get_chroma().count)
