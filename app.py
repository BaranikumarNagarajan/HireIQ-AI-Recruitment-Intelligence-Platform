"""HireIQ — AI Recruitment Intelligence Platform."""
import streamlit as st
import logging
from config import CHROMA_DB_PATH
from utils.chroma_client import get_chroma_client
from agents.parser import parse_cv
from agents.jd_analyser import analyse_jd
from agents.scorer import score_candidate
from agents.compliance import check_compliance
from agents.interviewer import generate_questions
from agents.reporter import generate_report
from rag.ingest import ingest_legal_documents
from eval.ragas_eval import run_evaluation, display_eval_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="HireIQ — AI Recruitment Intelligence",
    page_icon="🎯",
    layout="wide",
)

st.markdown("""
<style>
.score-card {
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    margin-bottom: 16px;
}
.dim-row {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
}
.skill-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.82rem;
    margin: 3px;
    font-weight: 500;
}
.skill-matched { background: #d4edda; color: #155724; }
.skill-missing  { background: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_chromadb():
    try:
        return get_chroma_client().get_collection(name="legal_documents")
    except Exception as e:
        logger.warning("ChromaDB load failed: %s", e)
        return None


def _skill_matched(jd_skill: str, cv_lower: list) -> bool:
    """Semantic skill matching — handles short abbreviations like RAG, SQL, LLM."""
    jd_lower = jd_skill.lower().strip()
    jd_words = [w for w in jd_lower.split() if len(w) >= 2]
    for cv_s in cv_lower:
        if jd_lower in cv_s or cv_s in jd_lower:
            return True
        if jd_words and any(w in cv_s for w in jd_words):
            return True
    return False


chroma_loaded = load_chromadb() is not None

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎯 HireIQ")
    st.caption("AI-Powered Recruitment Intelligence")
    st.divider()

    st.subheader("System Status")
    if chroma_loaded:
        st.success("✅ Legal Database Ready")
    else:
        st.warning("⚠️ Legal Database Not Loaded")
        if st.button("Ingest Legal Documents"):
            with st.spinner("Ingesting..."):
                ingest_legal_documents()
                st.rerun()

    from config import LLM_PROVIDER, CLAUDE_MODEL, GROQ_MODEL, XAI_MODEL
    from utils.llm_client import is_claude_available, is_groq_available, is_xai_available
    if LLM_PROVIDER == "groq":
        if is_groq_available():
            st.success(f"✅ Groq: {GROQ_MODEL}")
        else:
            st.error("❌ GROQ_API_KEY missing")
    elif LLM_PROVIDER == "claude":
        if is_claude_available():
            st.success(f"✅ Claude: {CLAUDE_MODEL}")
        else:
            st.error("❌ ANTHROPIC_API_KEY missing")
    elif LLM_PROVIDER == "xai":
        if is_xai_available():
            st.success(f"✅ xAI: {XAI_MODEL}")
        else:
            st.error("❌ XAI_API_KEY missing")

    st.divider()
    tab = st.radio("Navigation", ["CV Screening", "RAG Evaluation", "About"])

# ── CV Screening ──────────────────────────────────────────────────────────────
if tab == "CV Screening":
    st.title("🎯 CV Screening & Analysis")
    st.caption("Upload a CV and paste the job description to run an end-to-end AI recruitment analysis.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Candidate CV")
        cv_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
        if cv_file:
            st.success(f"✅ {cv_file.name}")
            if st.button("Preview Extracted Text", use_container_width=True):
                with st.spinner("Extracting..."):
                    cv_data = parse_cv(cv_file.getvalue())
                    if "error" in cv_data:
                        st.error(cv_data["error"])
                    else:
                        st.text_area("Extracted Text", cv_data.get("extracted_text", ""), height=200)

    with col2:
        st.subheader("💼 Job Description")
        jd_text = st.text_area("Paste job description", height=220, label_visibility="collapsed",
                               placeholder="Paste the full job description here…")

    st.divider()

    can_analyze = cv_file is not None and jd_text.strip() != ""
    if not can_analyze:
        st.info("Upload a CV and paste a job description to begin analysis.")

    if st.button("🚀 Run HireIQ Analysis", disabled=not can_analyze, type="primary", use_container_width=True):
        if not chroma_loaded:
            st.error("Legal database not loaded. Click 'Ingest Legal Documents' in the sidebar first.")
            st.stop()

        with st.status("Running HireIQ Analysis…", expanded=True) as status:
            try:
                status.update(label="📄 Parsing CV…")
                cv_data = parse_cv(cv_file.getvalue())
                if "error" in cv_data:
                    st.error(f"CV Parsing failed: {cv_data['error']}")
                    st.stop()
                st.write("✅ CV parsed")

                status.update(label="💼 Analysing job description…")
                jd_data = analyse_jd(jd_text)
                if "error" in jd_data:
                    st.error(f"JD Analysis failed: {jd_data['error']}")
                    st.stop()
                st.write("✅ JD analysed")

                status.update(label="📊 Scoring candidate…")
                score_data = score_candidate(cv_data, jd_data)
                if "error" in score_data:
                    st.error(f"Scoring failed: {score_data['error']}")
                    st.stop()
                st.write("✅ Candidate scored")

                status.update(label="⚖️ Checking legal compliance…")
                compliance_data = check_compliance(cv_data, jd_data)
                if "error" in compliance_data:
                    st.error(f"Compliance check failed: {compliance_data['error']}")
                    st.stop()
                st.write("✅ Compliance checked")

                status.update(label="💬 Generating interview questions…")
                questions_data = generate_questions(cv_data, jd_data, score_data)
                if "error" in questions_data:
                    st.error(f"Question generation failed: {questions_data['error']}")
                    st.stop()
                st.write("✅ Interview questions generated")

                status.update(label="📑 Creating PDF report…")
                pdf_bytes = generate_report(cv_data, jd_data, score_data, compliance_data, questions_data)
                st.write("✅ PDF report ready")

                st.session_state.results = {
                    "cv_data": cv_data,
                    "jd_data": jd_data,
                    "score_data": score_data,
                    "compliance_data": compliance_data,
                    "questions_data": questions_data,
                    "pdf_bytes": pdf_bytes,
                }
                status.update(label="Analysis complete!", state="complete")

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                status.update(label="Analysis failed", state="error")

    # ── Results ───────────────────────────────────────────────────────────────
    if "results" in st.session_state and st.session_state.results:
        results = st.session_state.results
        score_data = results["score_data"]
        cv_data = results["cv_data"]
        jd_data = results["jd_data"]
        total_score = score_data.get("total_score", 0)
        recommendation = score_data.get("recommendation", "")

        st.divider()

        # ── Score banner ──────────────────────────────────────────────────────
        if total_score >= 75:
            banner_color, banner_bg = "#155724", "#d4edda"
            banner_icon = "🟢"
        elif total_score >= 50:
            banner_color, banner_bg = "#856404", "#fff3cd"
            banner_icon = "🟡"
        elif total_score >= 25:
            banner_color, banner_bg = "#721c24", "#f8d7da"
            banner_icon = "🟠"
        else:
            banner_color, banner_bg = "#491217", "#f5c6cb"
            banner_icon = "🔴"

        st.markdown(f"""
        <div style="background:{banner_bg};border-radius:12px;padding:20px 28px;
                    border-left:6px solid {banner_color};margin-bottom:16px;">
          <div style="font-size:2.2rem;font-weight:700;color:{banner_color};">
            {banner_icon} {total_score}/100 &nbsp;·&nbsp; {recommendation}
          </div>
          <div style="color:{banner_color};margin-top:4px;font-size:0.95rem;">
            {cv_data.get('candidate_name','Candidate')} &nbsp;→&nbsp; {jd_data.get('job_title','Role')}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Download button
        st.download_button(
            "📄 Download PDF Report",
            data=results["pdf_bytes"],
            file_name="hireiq_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

        st.divider()

        # ── Score breakdown ───────────────────────────────────────────────────
        with st.expander("📊 Score Breakdown", expanded=True):
            scores = score_data.get("scores", {})
            weights = score_data.get("weights", {})
            reasoning_map = score_data.get("reasoning", {})

            for dim, score in scores.items():
                weight = weights.get(dim, 0)
                if score >= 75:
                    bar_color, label = "green", "Strong"
                elif score >= 50:
                    bar_color, label = "orange", "Moderate"
                else:
                    bar_color, label = "red", "Weak"

                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(f"**{dim.replace('_', ' ').title()}** · {weight*100:.0f}% weight")
                    st.progress(score / 100)
                with c2:
                    st.markdown(f"<div style='text-align:center;padding-top:8px;'>"
                                f"<b>{score:.0f}</b><br/><small>{label}</small></div>",
                                unsafe_allow_html=True)

                reason = reasoning_map.get(dim, "")
                if reason:
                    st.caption(reason)
                st.markdown("---")

        # ── Skills analysis ───────────────────────────────────────────────────
        with st.expander("🛠️ Skills Analysis", expanded=True):
            cv_skills = list(cv_data.get("skills", []))
            jd_required = list(jd_data.get("required_skills", []))
            jd_preferred = list(jd_data.get("preferred_skills", []))
            all_jd_skills = jd_required + jd_preferred
            cv_lower = [s.lower() for s in cv_skills]

            matched = [s for s in all_jd_skills if _skill_matched(s, cv_lower)]
            missing = [s for s in all_jd_skills if not _skill_matched(s, cv_lower)]

            match_pct = round(len(matched) / len(all_jd_skills) * 100) if all_jd_skills else 0
            st.markdown(f"**{len(matched)} / {len(all_jd_skills)} JD skills matched ({match_pct}%)**")
            st.progress(match_pct / 100)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**✅ Matched Skills**")
                if matched:
                    badges = "".join(
                        f'<span class="skill-badge skill-matched">{s}</span>' for s in matched
                    )
                    st.markdown(f'<div>{badges}</div>', unsafe_allow_html=True)
                else:
                    st.caption("No skills matched")

            with col2:
                st.markdown("**❌ Missing Skills**")
                if missing:
                    badges = "".join(
                        f'<span class="skill-badge skill-missing">{s}</span>' for s in missing
                    )
                    st.markdown(f'<div>{badges}</div>', unsafe_allow_html=True)
                else:
                    st.caption("All skills matched!")

        # ── Compliance ────────────────────────────────────────────────────────
        with st.expander("⚖️ Compliance & Legal"):
            compliance_data = results["compliance_data"]
            risk = compliance_data.get("risk_level", "UNKNOWN")
            risk_colors = {"LOW": ("✅", "#155724", "#d4edda"),
                           "MEDIUM": ("⚠️", "#856404", "#fff3cd"),
                           "HIGH": ("🚨", "#721c24", "#f8d7da")}
            icon, color, bg = risk_colors.get(risk, ("ℹ️", "#0c5460", "#d1ecf1"))
            st.markdown(f"""
            <div style="background:{bg};border-radius:8px;padding:12px 16px;
                        border-left:4px solid {color};margin-bottom:12px;">
              <b style="color:{color};">{icon} Risk Level: {risk}</b>
            </div>
            """, unsafe_allow_html=True)

            flags = compliance_data.get("compliance_flags", [])
            if flags:
                st.markdown("**Compliance Flags**")
                for f in flags:
                    st.warning(f"• {f}")

            gdpr = compliance_data.get("gdpr_requirements", [])
            if gdpr:
                st.markdown("**GDPR Requirements**")
                for r in gdpr:
                    st.info(f"• {r}")

            recs = compliance_data.get("recommendations", [])
            if recs:
                st.markdown("**Recommendations**")
                for r in recs:
                    st.success(f"• {r}")

        # ── Interview questions ───────────────────────────────────────────────
        with st.expander("💬 Interview Questions"):
            questions = results["questions_data"].get("questions", [])
            for i, q in enumerate(questions, 1):
                st.markdown(f"**Q{i}. {q.get('question_text', '')}**")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"🎯 *Listen for:* {q.get('what_to_listen_for', '')}")
                with col2:
                    st.markdown(f"🚩 *Red flags:* {q.get('red_flag_indicators', '')}")
                if i < len(questions):
                    st.divider()

# ── RAG Evaluation ────────────────────────────────────────────────────────────
elif tab == "RAG Evaluation":
    st.title("📊 RAG Evaluation Dashboard")
    st.caption("Evaluates the quality of the legal compliance RAG system across 5 test questions.")

    if st.button("🔬 Run RAGAS Evaluation", type="primary"):
        with st.spinner("Running evaluation… this takes ~1 minute (10 LLM calls)"):
            eval_results = run_evaluation()
            st.session_state.eval_results = eval_results

    if "eval_results" in st.session_state:
        display_eval_results(st.session_state.eval_results)

# ── About ─────────────────────────────────────────────────────────────────────
elif tab == "About":
    st.title("ℹ️ About HireIQ")
    st.markdown("""
## 🎯 HireIQ — AI Recruitment Intelligence Platform

HireIQ is an end-to-end AI-powered platform that automates CV screening, compliance checking, and interview preparation.

### 🏗️ Architecture — Multi-Agent Pipeline

| Agent | Role |
|-------|------|
| **Parser Agent** | Extracts structured data from PDF CVs (PyMuPDF + Tesseract OCR) |
| **JD Analyser Agent** | Extracts required skills, seniority level, and domain from job descriptions |
| **Scorer Agent** | 5-dimension weighted scoring with deterministic skill-overlap rule |
| **Compliance Agent** | RAG-powered GDPR and anti-discrimination checking via ChromaDB |
| **Interviewer Agent** | Generates 5 targeted interview questions based on identified gaps |
| **Reporter Agent** | Produces a professional PDF report with ReportLab |

### 📊 Scoring Dimensions

| Dimension | Weight | Method |
|-----------|--------|--------|
| Technical Skills | 35% | LLM + skill overlap rule |
| Experience Level | 25% | LLM |
| Domain Relevance | 20% | LLM + skill overlap rule |
| Employment Stability | 10% | Rule-based |
| Education | 10% | Rule-based |

### 🛠️ Tech Stack

- **LLM**: Groq API (`llama-3.1-8b-instant`)
- **Vector DB**: ChromaDB (embedded, persistent)
- **Embeddings**: sentence-transformers `all-MiniLM-L6-v2`
- **PDF Processing**: PyMuPDF + Tesseract OCR
- **Frontend**: Streamlit
- **Report Generation**: ReportLab
- **Deployment**: Google Cloud Run
- **CI/CD**: Google Cloud Build (auto-deploy on `git push`)
    """)

if __name__ == "__main__":
    if not chroma_loaded:
        ingest_legal_documents()
