"""Interview Question Generator Agent."""
import logging
from typing import Dict, Any, List
from utils.llm_client import get_llm_response, parse_llm_json_response

logger = logging.getLogger(__name__)


def generate_questions(cv_data: Dict[str, Any], jd_data: Dict[str, Any], score_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate targeted interview questions based on CV-JD gaps."""
    try:
        gaps = identify_gaps(cv_data, jd_data, score_data)
        skills = [s.lower() for s in (cv_data.get("skills") or [])]
        required = jd_data.get("required_skills") or []
        missing = [s for s in required if s.lower() not in skills]
        job_title = jd_data.get("job_title", "the role")

        # Single focused prompt — 5 questions, simple format, easier for 3B model
        prompt = f"""Generate 5 interview questions for a {job_title} candidate.
Gaps identified: {gaps}
Missing skills: {missing}

Reply with ONLY a JSON array of 5 objects, each with:
- "question_text": the interview question
- "what_to_listen_for": ideal answer in one sentence
- "red_flag_indicators": warning sign in one sentence

Example format:
[{{"question_text": "...", "what_to_listen_for": "...", "red_flag_indicators": "..."}}]"""

        response = get_llm_response(prompt)

        try:
            questions = parse_llm_json_response(response)
            if not isinstance(questions, list):
                questions = questions.get("questions", []) if isinstance(questions, dict) else []
        except Exception:
            logger.warning("Could not parse questions JSON, using fallback")
            questions = _fallback_questions(job_title, missing)

        return {
            "questions": questions,
            "preparation_notes": "Questions target gaps identified in the candidate profile.",
        }

    except Exception as e:
        logger.error("Failed to generate questions: %s", e)
        return {"error": str(e)}


def _fallback_questions(job_title: str, missing_skills: List[str]) -> List[Dict[str, Any]]:
    """Return basic questions when LLM fails, so the pipeline never crashes."""
    base = [
        {
            "question_text": f"Walk me through your most relevant experience for this {job_title} role.",
            "what_to_listen_for": "Specific examples matching the job requirements.",
            "red_flag_indicators": "Vague or unrelated experience.",
        },
        {
            "question_text": "Describe a challenging technical problem you solved recently.",
            "what_to_listen_for": "Clear problem-solving process and measurable outcome.",
            "red_flag_indicators": "Cannot recall a specific example.",
        },
        {
            "question_text": "How do you stay up to date with new technologies in your field?",
            "what_to_listen_for": "Active learning habits, courses, projects.",
            "red_flag_indicators": "No evidence of continuous learning.",
        },
    ]
    for skill in missing_skills[:2]:
        base.append({
            "question_text": f"You don't have {skill} listed — how would you approach learning it?",
            "what_to_listen_for": "Willingness and a concrete learning plan.",
            "red_flag_indicators": "Dismissive or no concrete plan.",
        })
    return base


def identify_gaps(cv_data: Dict[str, Any], jd_data: Dict[str, Any], score_data: Dict[str, Any]) -> List[str]:
    """Identify gaps from scoring data."""
    gaps = []
    scores = score_data.get("scores", {})

    if scores.get("technical_skills", 100) < 70:
        gaps.append("Technical skills gaps")
    if scores.get("experience_level", 100) < 70:
        gaps.append("Insufficient experience level")
    if scores.get("domain_relevance", 100) < 70:
        gaps.append("Limited domain relevance")
    if scores.get("employment_stability", 100) < 70:
        gaps.append("Employment stability concerns")
    if scores.get("education", 100) < 70:
        gaps.append("Education level concerns")
    if cv_data.get("employment_gaps"):
        gaps.append("Employment gaps present")
    if float(cv_data.get("average_tenure_per_role") or 0) < 1.5:
        gaps.append("Short average tenure")

    return gaps
