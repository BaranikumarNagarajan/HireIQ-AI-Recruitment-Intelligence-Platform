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
    """Score technical skills match."""
    cv_skills = set(cv_data.get("skills") or [])
    required_skills = set(jd_data.get("required_skills") or [])
    preferred_skills = set(jd_data.get("preferred_skills") or [])

    if not required_skills:
        return 100.0, "No required skills specified"

    required_match = len(cv_skills & required_skills) / len(required_skills) * 100
    preferred_match = len(cv_skills & preferred_skills) / len(preferred_skills) * 100 if preferred_skills else 0
    score = (required_match * 0.7) + (preferred_match * 0.3)

    prompt = (
        f"Evaluate the technical skills match. CV Skills: {list(cv_skills)}. "
        f"Required: {list(required_skills)}. Preferred: {list(preferred_skills)}. Score: {score:.1f}/100. "
        "Respond in 1-2 sentences explaining the match."
    )
    return round(score, 1), get_llm_response(prompt)


def score_experience_level(cv_data: Dict[str, Any], jd_data: Dict[str, Any]) -> Tuple[float, str]:
    """Score experience level match."""
    cv_years = float(cv_data.get("total_years_experience") or 0)
    min_years = float(jd_data.get("minimum_years_experience") or 0)

    if cv_years >= min_years:
        score = 100.0
    else:
        score = max(0, (cv_years / min_years) * 100) if min_years > 0 else 50.0

    prompt = (
        f"Candidate has {cv_years} years experience; role requires {min_years}+ years "
        f"({jd_data.get('seniority_level', 'unknown')} level). Score: {score:.1f}/100. "
        "Respond in 1-2 sentences."
    )
    return round(score, 1), get_llm_response(prompt)


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
