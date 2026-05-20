"""Match Scoring Agent."""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Tuple
from config import SCORE_WEIGHTS
from utils.llm_client import get_llm_response, parse_llm_json_response

logger = logging.getLogger(__name__)

_RECOMMENDATION_THRESHOLDS = [(75, "STRONG FIT"), (50, "POTENTIAL FIT"), (25, "WEAK FIT")]


def score_candidate(cv_data: Dict[str, Any], jd_data: Dict[str, Any]) -> Dict[str, Any]:
    """Score candidate match against job description."""
    try:
        scorers = {
            "technical_skills": lambda: score_technical_skills(cv_data, jd_data),
            "experience_level": lambda: score_experience_level(cv_data, jd_data),
            "domain_relevance": lambda: score_domain_relevance(cv_data, jd_data),
            "employment_stability": lambda: score_employment_stability(cv_data),
            "education": lambda: score_education(cv_data, jd_data),
        }

        scores: Dict[str, float] = {}
        reasoning: Dict[str, str] = {}

        # Run all 5 LLM calls in parallel — they are fully independent
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fn): name for name, fn in scorers.items()}
            for future in as_completed(futures):
                name = futures[future]
                scores[name], reasoning[name] = future.result()

        total_score = sum(scores[dim] * SCORE_WEIGHTS[dim] for dim in scores)
        recommendation = next(
            (label for cutoff, label in _RECOMMENDATION_THRESHOLDS if total_score >= cutoff),
            "NOT RECOMMENDED",
        )

        return {
            "scores": scores,
            "reasoning": reasoning,
            "total_score": round(total_score, 1),
            "recommendation": recommendation,
            "weights": SCORE_WEIGHTS,
        }

    except Exception as e:
        logger.error("Failed to score candidate: %s", e)
        return {"error": str(e)}


def score_technical_skills(cv_data: Dict[str, Any], jd_data: Dict[str, Any]) -> Tuple[float, str]:
    """Score technical skills match using LLM semantic evaluation."""
    cv_skills = list(cv_data.get("skills") or [])
    required_skills = list(jd_data.get("required_skills") or [])
    preferred_skills = list(jd_data.get("preferred_skills") or [])

    if not required_skills:
        return 100.0, "No required skills specified"

    prompt = (
        f"You are evaluating a candidate's technical skills against a job description.\n\n"
        f"Candidate skills: {cv_skills}\n"
        f"Required skills: {required_skills}\n"
        f"Preferred/nice-to-have skills: {preferred_skills}\n\n"
        f"Instructions:\n"
        f"- Match semantically, not just by exact string. 'OpenAI' matches 'OpenAI API', 'langchain' matches 'LangChain', etc.\n"
        f"- Required skills are worth 70% of the score, preferred skills 30%.\n"
        f"- Give a score from 0-100 based on how well the candidate's skills cover the requirements.\n\n"
        'Reply with ONLY a JSON object: {"score": <number 0-100>, "reasoning": "<1-2 sentences>"}'
    )
    response = get_llm_response(prompt)
    try:
        parsed = parse_llm_json_response(response)
        return round(float(parsed.get("score", 50.0)), 1), parsed.get("reasoning", response)
    except Exception as exc:
        logger.warning("Failed to parse technical skills JSON, defaulting to 50: %s", exc)
        return 50.0, response


def score_experience_level(cv_data: Dict[str, Any], jd_data: Dict[str, Any]) -> Tuple[float, str]:
    """Score experience level match using LLM holistic evaluation."""
    cv_years = float(cv_data.get("total_years_experience") or 0)
    min_years = float(jd_data.get("minimum_years_experience") or 0)
    seniority = jd_data.get("seniority_level", "unknown")
    work_experience = cv_data.get("work_experience") or []

    prompt = (
        f"Evaluate the candidate's experience level for a {seniority}-level role requiring {min_years}+ years.\n\n"
        f"Candidate has {cv_years} years of work experience.\n"
        f"Work history: {[exp.get('role', '') + ' at ' + exp.get('company', '') + ' (' + str(exp.get('duration', '')) + ')' for exp in work_experience]}\n\n"
        f"Important: Internships, project portfolios, and hands-on AI/software work count as relevant experience. "
        f"A candidate with strong project work and internships may be well-suited for a mid-level role even with under 2 years of formal employment. "
        f"Be generous when the candidate shows strong practical skills through projects.\n\n"
        'Reply with ONLY a JSON object: {"score": <number 0-100>, "reasoning": "<1-2 sentences>"}'
    )
    response = get_llm_response(prompt)
    try:
        parsed = parse_llm_json_response(response)
        return round(float(parsed.get("score", 50.0)), 1), parsed.get("reasoning", response)
    except Exception as exc:
        logger.warning("Failed to parse experience level JSON, defaulting to 50: %s", exc)
        return 50.0, response


def score_domain_relevance(cv_data: Dict[str, Any], jd_data: Dict[str, Any]) -> Tuple[float, str]:
    """Score domain relevance."""
    cv_experience = cv_data.get("work_experience") or []
    jd_domain = jd_data.get("domain_industry") or ""

    prompt = (
        f"Evaluate how relevant the candidate's background is to the job domain.\n"
        f"Job Domain: {jd_domain}\n"
        f"Candidate Experience: {[exp.get('company', '') + ' - ' + exp.get('role', '') for exp in cv_experience]}\n\n"
        'Reply with ONLY a JSON object: {"score": <number 0-100>, "reasoning": "<brief explanation>"}'
    )

    response = get_llm_response(prompt)
    try:
        parsed = parse_llm_json_response(response)
        return round(float(parsed.get("score", 50.0)), 1), parsed.get("reasoning", response)
    except Exception as exc:
        logger.warning("Failed to parse domain relevance JSON, defaulting to 50: %s", exc)
        return 50.0, response


def score_employment_stability(cv_data: Dict[str, Any]) -> Tuple[float, str]:
    """Score employment stability."""
    experience = cv_data.get("work_experience") or []
    gaps = cv_data.get("employment_gaps") or []
    avg_tenure = float(cv_data.get("average_tenure_per_role") or 0)

    base_score = 100
    if gaps:
        base_score -= len(gaps) * 20
    if avg_tenure < 1:
        base_score -= 30
    elif avg_tenure < 2:
        base_score -= 10
    score = max(0, base_score)

    prompt = (
        f"Avg tenure: {avg_tenure} years, gaps: {gaps}, roles: {len(experience)}. "
        f"Stability score: {score}/100. Respond in 1-2 sentences."
    )
    return round(score, 1), get_llm_response(prompt)


def score_education(cv_data: Dict[str, Any], jd_data: Dict[str, Any]) -> Tuple[float, str]:
    """Score education match."""
    cv_education = cv_data.get("education") or []
    seniority = (jd_data.get("seniority_level") or "mid").lower()

    if seniority == "senior" and any("bachelor" in edu.lower() or "master" in edu.lower() for edu in cv_education):
        score = 100.0
    elif seniority == "mid" and any("bachelor" in edu.lower() for edu in cv_education):
        score = 100.0
    elif seniority == "junior":
        score = 80.0
    else:
        score = 50.0

    prompt = (
        f"Education: {cv_education}. Role seniority: {seniority}. "
        f"Education score: {score}/100. Respond in 1-2 sentences."
    )
    return round(score, 1), get_llm_response(prompt)
