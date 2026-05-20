"""Match Scoring Agent — single LLM call to avoid Groq TPM rate limits."""
import logging
from typing import Dict, Any, Tuple
from config import SCORE_WEIGHTS
from utils.llm_client import get_llm_response, parse_llm_json_response

logger = logging.getLogger(__name__)

_RECOMMENDATION_THRESHOLDS = [(75, "STRONG FIT"), (50, "POTENTIAL FIT"), (25, "WEAK FIT")]

_SCORE_PROMPT = """You are an expert recruiter evaluating a candidate against a job description.

Candidate:
- Skills: {cv_skills}
- Work history: {work_history}
- Years of experience: {cv_years}
- Education: {education}

Job Description:
- Required skills: {required_skills}
- Preferred skills: {preferred_skills}
- Minimum years: {min_years}
- Seniority: {seniority}
- Domain: {domain}

Score the candidate on these 3 dimensions (0-100 each):

1. technical_skills: How well do candidate skills match required+preferred skills?
   - Match SEMANTICALLY not literally. "OpenAI" matches "OpenAI API", "langchain" matches "LangChain", etc.
   - Required skills count 70%, preferred 30%.

2. experience_level: How well does the candidate's experience fit the role level?
   - Internships, projects, and hands-on work count as real experience.
   - Be fair to candidates with strong project portfolios even if formal employment is short.

3. domain_relevance: How relevant is the candidate's background to the job domain?

Reply with ONLY valid JSON, no explanation:
{{"technical_skills": {{"score": 85, "reasoning": "example"}}, "experience_level": {{"score": 70, "reasoning": "example"}}, "domain_relevance": {{"score": 75, "reasoning": "example"}}}}"""


def score_candidate(cv_data: Dict[str, Any], jd_data: Dict[str, Any]) -> Dict[str, Any]:
    """Score candidate — 1 LLM call for complex dims, rule-based for simple dims."""
    try:
        scores: Dict[str, float] = {}
        reasoning: Dict[str, str] = {}

        # Rule-based scores (no LLM needed)
        scores["employment_stability"], reasoning["employment_stability"] = _rule_employment_stability(cv_data)
        scores["education"], reasoning["education"] = _rule_education(cv_data, jd_data)

        # Single LLM call for the 3 complex dimensions
        llm_scores = _llm_score_three_dims(cv_data, jd_data)
        for dim in ("technical_skills", "experience_level", "domain_relevance"):
            scores[dim] = llm_scores[dim]["score"]
            reasoning[dim] = llm_scores[dim]["reasoning"]

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


def _llm_score_three_dims(cv_data: Dict[str, Any], jd_data: Dict[str, Any]) -> Dict[str, Any]:
    """Single LLM call scoring technical_skills, experience_level, domain_relevance."""
    cv_skills = (cv_data.get("skills") or [])[:20]
    required_skills = (jd_data.get("required_skills") or [])[:15]
    preferred_skills = (jd_data.get("preferred_skills") or [])[:10]
    work_experience = cv_data.get("work_experience") or []
    work_history = [
        f"{exp.get('role', '')} at {exp.get('company', '')} ({exp.get('duration', '')})"
        for exp in work_experience
    ][:5]

    prompt = _SCORE_PROMPT.format(
        cv_skills=cv_skills,
        work_history=work_history,
        cv_years=cv_data.get("total_years_experience", 0),
        education=(cv_data.get("education") or [])[:3],
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        min_years=jd_data.get("minimum_years_experience", 0),
        seniority=jd_data.get("seniority_level", "mid"),
        domain=jd_data.get("domain_industry", ""),
    )

    default = {
        "technical_skills": {"score": 50.0, "reasoning": "Could not evaluate"},
        "experience_level": {"score": 50.0, "reasoning": "Could not evaluate"},
        "domain_relevance": {"score": 50.0, "reasoning": "Could not evaluate"},
    }

    try:
        raw = get_llm_response(prompt)
        parsed = parse_llm_json_response(raw)
        result = {}
        for dim in ("technical_skills", "experience_level", "domain_relevance"):
            dim_data = parsed.get(dim, {})
            result[dim] = {
                "score": round(float(dim_data.get("score", 50.0)), 1),
                "reasoning": str(dim_data.get("reasoning", "No reasoning provided")),
            }
        return result
    except Exception as exc:
        logger.warning("LLM scoring failed: %s — using defaults", exc)
        return default


def _rule_employment_stability(cv_data: Dict[str, Any]) -> Tuple[float, str]:
    """Rule-based employment stability — no LLM call."""
    gaps = cv_data.get("employment_gaps") or []
    avg_tenure = float(cv_data.get("average_tenure_per_role") or 0)
    num_roles = len(cv_data.get("work_experience") or [])

    score = 100.0
    if gaps:
        score -= len(gaps) * 20
    if avg_tenure < 1:
        score -= 20
    elif avg_tenure < 2:
        score -= 10
    score = max(0.0, score)

    notes = []
    if gaps:
        notes.append(f"{len(gaps)} employment gap(s) detected")
    if avg_tenure < 1:
        notes.append("average tenure under 1 year")
    elif avg_tenure < 2:
        notes.append("average tenure 1-2 years")
    reasoning = "; ".join(notes) if notes else f"{num_roles} role(s), stable employment history"
    return round(score, 1), reasoning


def _rule_education(cv_data: Dict[str, Any], jd_data: Dict[str, Any]) -> Tuple[float, str]:
    """Rule-based education scoring — no LLM call."""
    cv_education = cv_data.get("education") or []
    seniority = (jd_data.get("seniority_level") or "mid").lower()
    edu_str = " ".join(cv_education).lower()

    has_masters = any(kw in edu_str for kw in ("master", "msc", "mba", "phd"))
    has_bachelors = any(kw in edu_str for kw in ("bachelor", "bsc", "b.sc", "degree"))

    if seniority == "senior":
        score = 100.0 if (has_bachelors or has_masters) else 60.0
    elif seniority == "mid":
        score = 100.0 if has_bachelors else 70.0
    else:
        score = 80.0

    label = "Master's" if has_masters else ("Bachelor's" if has_bachelors else "No degree found")
    return round(score, 1), f"{label} — {seniority}-level role"


# Keep individual functions available for direct use if needed
def score_technical_skills(cv_data: Dict[str, Any], jd_data: Dict[str, Any]) -> Tuple[float, str]:
    result = _llm_score_three_dims(cv_data, jd_data)
    return result["technical_skills"]["score"], result["technical_skills"]["reasoning"]


def score_experience_level(cv_data: Dict[str, Any], jd_data: Dict[str, Any]) -> Tuple[float, str]:
    result = _llm_score_three_dims(cv_data, jd_data)
    return result["experience_level"]["score"], result["experience_level"]["reasoning"]


def score_domain_relevance(cv_data: Dict[str, Any], jd_data: Dict[str, Any]) -> Tuple[float, str]:
    result = _llm_score_three_dims(cv_data, jd_data)
    return result["domain_relevance"]["score"], result["domain_relevance"]["reasoning"]


def score_employment_stability(cv_data: Dict[str, Any]) -> Tuple[float, str]:
    return _rule_employment_stability(cv_data)


def score_education(cv_data: Dict[str, Any], jd_data: Dict[str, Any]) -> Tuple[float, str]:
    return _rule_education(cv_data, jd_data)
