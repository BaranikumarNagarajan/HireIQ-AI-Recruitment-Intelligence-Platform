"""HireIQ — AI Recruitment Intelligence Platform."""
import streamlit as st
import logging
from pathlib import Path
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="HireIQ — AI Recruitment Intelligence",
    page_icon="🎯",
    layout="wide"
)

# Initialize ChromaDB on startup
@st.cache_resource
def load_chromadb():
    try:
        return get_chroma_client().get_collection(name="legal_documents")
    except Exception as e:
        logger.warning("ChromaDB load failed: %s", e)
        return None

# Check if ChromaDB is loaded
chroma_loaded = load_chromadb() is not None

# Sidebar
with st.sidebar:
    st.title("🎯 HireIQ")
    st.markdown("AI-Powered Recruitment Intelligence")
    
    # Status indicators
    st.subheader("System Status")
    if chroma_loaded:
        st.success("✅ Legal Database Loaded")
    else:
        st.warning("⚠️ Legal Database Not Loaded")
        if st.button("Ingest Legal Documents"):
            with st.spinner("Ingesting legal documents..."):
                ingest_legal_documents()
                st.rerun()
    
    # Local Model check
    from config import LLM_PROVIDER, LLM_MODEL, CLAUDE_MODEL, GROQ_MODEL, XAI_MODEL
    from utils.llm_client import is_ollama_available, is_claude_available, is_groq_available, is_xai_available
    if LLM_PROVIDER == "xai":
        if is_xai_available():
            st.success(f"✅ xAI API ready: {XAI_MODEL}")
        else:
            st.error("❌ XAI_API_KEY missing in .env")
    elif LLM_PROVIDER == "groq":
        if is_groq_available():
            st.success(f"✅ Groq API ready: {GROQ_MODEL}")
        else:
            st.error("❌ GROQ_API_KEY missing in .env")
    elif LLM_PROVIDER == "claude":
        if is_claude_available():
            st.success(f"✅ Claude API ready: {CLAUDE_MODEL}")
        else:
            st.error("❌ ANTHROPIC_API_KEY missing in .env")
    elif LLM_PROVIDER == "ollama":
        if is_ollama_available():
            st.success(f"✅ Ollama ready: {LLM_MODEL}")
        else:
            st.error("❌ Ollama unavailable — run: ollama serve")
    else:
        st.warning(f"⚠️ Unknown LLM_PROVIDER: {LLM_PROVIDER}")
    
    # Navigation
    st.subheader("Navigation")
    tab = st.radio("Go to:", ["CV Screening", "Eval Dashboard", "About"])

# Main content
if tab == "CV Screening":
    st.title("🎯 CV Screening & Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Upload CV")
        cv_file = st.file_uploader("Upload PDF CV", type=["pdf"])
        
        if cv_file and st.button("Preview Extracted Text"):
            with st.spinner("Extracting text..."):
                cv_bytes = cv_file.getvalue()
                cv_data = parse_cv(cv_bytes)
                if "error" in cv_data:
                    st.error(f"Failed to parse CV: {cv_data['error']}")
                else:
                    st.text_area("Extracted CV Text", cv_data.get("extracted_text", ""), height=300)
    
    with col2:
        st.subheader("💼 Job Description")
        jd_text = st.text_area("Paste job description", height=300)
    
    # Analysis button
    can_analyze = cv_file is not None and jd_text.strip() != ""
    if st.button("🚀 Run HireIQ Analysis", disabled=not can_analyze, type="primary"):
        if not chroma_loaded:
            st.error("Please ingest legal documents first")
            st.stop()
        
        # Store results in session state
        if "results" not in st.session_state:
            st.session_state.results = {}
        
        with st.status("Running HireIQ Analysis...") as status:
            try:
                # Step 1: Parse CV
                status.update(label="Parsing CV...")
                if cv_file is None:
                    st.error("Please upload a CV before running analysis.")
                    st.stop()
                cv_bytes = cv_file.getvalue()
                cv_data = parse_cv(cv_bytes)
                if "error" in cv_data:
                    st.error(f"CV Parsing failed: {cv_data['error']}")
                    st.stop()
                
                # Step 2: Analyse JD
                status.update(label="Analysing job description...")
                jd_data = analyse_jd(jd_text)
                if "error" in jd_data:
                    st.error(f"JD Analysis failed: {jd_data['error']}")
                    st.stop()
                
                # Step 3: Score candidate
                status.update(label="Scoring candidate...")
                score_data = score_candidate(cv_data, jd_data)
                if "error" in score_data:
                    st.error(f"Scoring failed: {score_data['error']}")
                    st.stop()
                
                # Step 4: Check compliance
                status.update(label="Checking legal compliance...")
                compliance_data = check_compliance(cv_data, jd_data)
                if "error" in compliance_data:
                    st.error(f"Compliance check failed: {compliance_data['error']}")
                    st.stop()
                
                # Step 5: Generate questions
                status.update(label="Generating interview questions...")
                questions_data = generate_questions(cv_data, jd_data, score_data)
                if "error" in questions_data:
                    st.error(f"Question generation failed: {questions_data['error']}")
                    st.stop()
                
                # Step 6: Generate report
                status.update(label="Creating HR report...")
                pdf_bytes = generate_report(cv_data, jd_data, score_data, compliance_data, questions_data)
                
                # Store results
                st.session_state.results = {
                    "cv_data": cv_data,
                    "jd_data": jd_data,
                    "score_data": score_data,
                    "compliance_data": compliance_data,
                    "questions_data": questions_data,
                    "pdf_bytes": pdf_bytes
                }
                
                status.update(label="Analysis complete!", state="complete")
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                status.update(label="Analysis failed", state="error")
    
    # Display results
    if "results" in st.session_state and st.session_state.results:
        results = st.session_state.results
        
        # Overall score
        score_data = results["score_data"]
        total_score = score_data.get("total_score", 0)
        recommendation = score_data.get("recommendation", "")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if total_score >= 75:
                st.success(f"🎉 {total_score}/100")
            elif total_score >= 50:
                st.warning(f"⚠️ {total_score}/100")
            else:
                st.error(f"❌ {total_score}/100")
        
        with col2:
            st.subheader(f"Recommendation: {recommendation}")
        
        with col3:
            st.download_button(
                "📄 Download PDF Report",
                data=results["pdf_bytes"],
                file_name="hireiq_report.pdf",
                mime="application/pdf"
            )
        
        # Expandable sections
        with st.expander("📊 Score Breakdown", expanded=True):
            scores = score_data.get("scores", {})
            st.bar_chart(scores)
            
            st.subheader("Detailed Reasoning")
            reasoning = score_data.get("reasoning", {})
            for dim, reason in reasoning.items():
                st.write(f"**{dim.replace('_', ' ').title()}:** {reason}")
        
        with st.expander("🛠️ Skills Analysis"):
            cv_skills = list(results["cv_data"].get("skills", []))
            jd_required = list(results["jd_data"].get("required_skills", []))
            jd_preferred = list(results["jd_data"].get("preferred_skills", []))
            cv_lower = [s.lower() for s in cv_skills]

            def _skill_matched(jd_skill: str) -> bool:
                jd_words = [w for w in jd_skill.lower().split() if len(w) > 3]
                for cv_s in cv_lower:
                    if jd_skill.lower() in cv_s or cv_s in jd_skill.lower():
                        return True
                    if any(w in cv_s for w in jd_words):
                        return True
                return False

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("✅ Matched Skills")
                for skill in jd_required + jd_preferred:
                    if _skill_matched(skill):
                        st.success(skill)

            with col2:
                st.subheader("❌ Missing Skills")
                for skill in jd_required + jd_preferred:
                    if not _skill_matched(skill):
                        st.error(skill)
        
        with st.expander("⚖️ Compliance & Legal"):
            compliance_data = results["compliance_data"]
            risk_level = compliance_data.get("risk_level", "UNKNOWN")
            
            if risk_level == "HIGH":
                st.error(f"🚨 Risk Level: {risk_level}")
            elif risk_level == "MEDIUM":
                st.warning(f"⚠️ Risk Level: {risk_level}")
            else:
                st.success(f"✅ Risk Level: {risk_level}")
            
            flags = compliance_data.get("compliance_flags", [])
            if flags:
                st.subheader("Compliance Flags")
                for flag in flags:
                    st.warning(f"• {flag}")
            
            gdpr = compliance_data.get("gdpr_requirements", [])
            if gdpr:
                st.subheader("GDPR Requirements")
                for req in gdpr:
                    st.info(f"• {req}")
        
        with st.expander("💬 Interview Questions"):
            questions = results["questions_data"].get("questions", [])
            for i, q in enumerate(questions, 1):
                st.subheader(f"Question {i}")
                st.write(f"**{q.get('question_text', '')}**")
                st.write(f"*What to listen for:* {q.get('what_to_listen_for', '')}")
                st.write(f"*Red flags:* {q.get('red_flag_indicators', '')}")
                st.divider()

elif tab == "Eval Dashboard":
    st.title("📊 RAGAS Evaluation Dashboard")
    
    if st.button("🔬 Run RAGAS Evaluation"):
        with st.spinner("Running evaluation... This may take a few minutes."):
            eval_results = run_evaluation()
            st.session_state.eval_results = eval_results
    
    if "eval_results" in st.session_state:
        display_eval_results(st.session_state.eval_results)

elif tab == "About":
    st.title("ℹ️ About HireIQ")
    
    st.markdown("""
    ## 🎯 HireIQ — AI Recruitment Intelligence Platform
    
    HireIQ is an end-to-end AI-powered recruitment intelligence platform that automates CV screening and HR compliance checking.
    
    ### 🏗️ Architecture
    
    **Multi-Agent System:**
    - **Parser Agent**: Extracts structured data from CV PDFs using PyMuPDF + Tesseract OCR fallback
    - **JD Analyser Agent**: Extracts requirements and culture signals from job descriptions
    - **Scorer Agent**: Calculates match scores across 5 dimensions with configurable weights
    - **Compliance Agent**: RAG-powered legal compliance checking using ChromaDB
    - **Interviewer Agent**: Generates targeted interview questions based on gaps
    - **Reporter Agent**: Creates professional PDF reports with ReportLab
    
    ### 🛠️ Tech Stack
    
    - **LLM**: Local Ollama (`llama3.2:3b`)
    - **Agent Framework**: LangChain
    - **Vector DB**: ChromaDB (embedded)
    - **Embeddings**: sentence-transformers all-MiniLM-L6-v2
    - **PDF Processing**: PyMuPDF + Tesseract OCR
    - **Evaluation**: RAGAS
    - **Frontend**: Streamlit
    - **Report Generation**: ReportLab
    - **Deployment**: Google Cloud Run (Docker)
    
    ### 🚀 Features
    
    - Automated CV parsing with OCR fallback
    - Multi-dimensional candidate scoring
    - Legal compliance RAG system
    - Targeted interview question generation
    - Professional PDF report generation
    - RAGAS evaluation pipeline
    
    ### 📋 Usage
    
    1. Upload a candidate CV (PDF)
    2. Paste the job description
    3. Click "Run HireIQ Analysis"
    4. Review the comprehensive report
    5. Download the PDF for HR records
    
    ### 🔗 Links
    
    - [GitHub Repository](https://github.com/your-repo/hireiq)
    - [Documentation](https://docs.hireiq.com)
    """)

if __name__ == "__main__":
    # Ensure legal documents are ingested on first run
    if not chroma_loaded:
        logger.info("Ingesting legal documents on startup...")
        ingest_legal_documents()