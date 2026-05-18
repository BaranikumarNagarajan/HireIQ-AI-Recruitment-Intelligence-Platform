"""Legal Compliance RAG Agent."""
import logging
from typing import Dict, Any
from rag.retriever import retrieve_legal_context
from utils.llm_client import get_llm_response, parse_llm_json_response

logger = logging.getLogger(__name__)


def check_compliance(cv_data: Dict[str, Any], jd_data: Dict[str, Any]) -> Dict[str, Any]:
    """Check legal compliance for CV processing and hiring via RAG."""
    try:
        # Use a single focused query to keep context small for a 3B model
        context_chunks = retrieve_legal_context(
            "GDPR recruitment CV data processing automated screening compliance", n_results=3
        )
        legal_context = "\n\n".join(c["text"] for c in context_chunks)

        # Pass only key CV/JD fields — not the full dicts — to keep prompt size manageable
        candidate_name = cv_data.get("candidate_name", "Unknown")
        skills = cv_data.get("skills", [])
        job_title = jd_data.get("job_title", "Unknown role")
        domain = jd_data.get("domain_industry", "")

        prompt = f"""You are a legal compliance expert. Based on the GDPR/legal context below, evaluate the compliance of using AI to screen a CV.

Candidate: {candidate_name}, Skills: {skills}
Role: {job_title}, Domain: {domain}

Legal context:
{legal_context}

Reply with ONLY a JSON object (no extra text):
{{
  "compliance_flags": ["<issue1>", "<issue2>"],
  "gdpr_requirements": ["<req1>", "<req2>"],
  "recommendations": ["<action1>", "<action2>"],
  "risk_level": "LOW"
}}"""

        response = get_llm_response(prompt)

        try:
            return parse_llm_json_response(response)
        except Exception:
            logger.error("Failed to parse compliance response as JSON")
            # Return a safe default so the pipeline does not crash
            return {
                "compliance_flags": ["Manual review recommended — automated parsing may not capture all nuance"],
                "gdpr_requirements": ["Obtain candidate consent", "Provide data retention notice"],
                "recommendations": ["Ensure human review of AI output before decisions are made"],
                "risk_level": "MEDIUM",
            }

    except Exception as e:
        logger.error("Failed to check compliance: %s", e)
        return {"error": str(e)}
