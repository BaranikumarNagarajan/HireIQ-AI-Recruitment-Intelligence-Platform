"""Job Description Analyser Agent."""
import logging
from typing import Dict, Any
from utils.llm_client import get_llm_response, parse_llm_json_response

logger = logging.getLogger(__name__)

def analyse_jd(jd_text: str) -> Dict[str, Any]:
    """
    Analyse job description text.
    
    Args:
        jd_text: Raw job description text
        
    Returns:
        Dict with analysed JD data
    """
    try:
        prompt = f"""
        Extract the following information from this job description. Return as JSON format:
        - job_title: string
        - required_skills: list of strings (must-have)
        - preferred_skills: list of strings (nice-to-have)
        - minimum_years_experience: number
        - seniority_level: string (junior/mid/senior)
        - domain_industry: string
        - key_responsibilities: list of strings
        - culture_signals: list of strings (keywords about work style)
        
        Job Description:
        {jd_text}
        """
        
        response = get_llm_response(prompt)

        # Parse JSON response
        try:
            parsed_data = parse_llm_json_response(response)
        except Exception:
            logger.error("Failed to parse LLM response as JSON")
            return {"error": "Failed to parse JD data"}

        return parsed_data
        
    except Exception as e:
        logger.error(f"Failed to analyse JD: {e}")
        return {"error": str(e)}