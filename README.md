# HireIQ — AI Recruitment Intelligence Platform

> End-to-end AI-powered recruitment screening — upload a CV, paste a job description, get a full candidate analysis in under 60 seconds.

**Live demo**: [Deployed on Google Cloud Run](https://hireiq-838754609617.europe-west1.run.app) *(auto-deploys on every push via Cloud Build CI/CD)*

---

## What it does

HireIQ runs a candidate CV through a **6-agent AI pipeline** and produces:

- **Match score** (0–100) across 5 weighted dimensions
- **Skills gap analysis** — matched vs missing skills with semantic matching
- **GDPR compliance check** via RAG (ChromaDB vector search over legal documents)
- **5 targeted interview questions** based on identified gaps
- **Downloadable PDF report** with full reasoning

---

## Architecture

```
CV PDF + Job Description
        │
        ▼
┌─────────────────────┐
│   Parser Agent      │  PyMuPDF + Tesseract OCR → structured CV data
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   JD Analyser Agent │  Groq LLM → required skills, seniority, domain
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   Scorer Agent      │  LLM scoring + deterministic skill-overlap rule
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Compliance Agent   │  RAG (ChromaDB) → GDPR + bias compliance check
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│ Interviewer Agent   │  Groq LLM → 5 targeted interview questions
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Reporter Agent     │  ReportLab → professional PDF report
└─────────────────────┘
```

---

## Scoring Model

| Dimension | Weight | Method |
|-----------|--------|--------|
| Technical Skills | 35% | LLM + deterministic skill-overlap cap |
| Experience Level | 25% | LLM |
| Domain Relevance | 20% | LLM + deterministic skill-overlap cap |
| Employment Stability | 10% | Rule-based (gaps, tenure) |
| Education | 10% | Rule-based (degree level vs seniority) |

**Anti-inflation rule**: if CV–JD skill overlap < 10%, technical skills and domain relevance are hard-capped at 20 and 25 respectively — regardless of LLM output. This prevents mismatched candidates (e.g. a software engineer applying for a marketing role) from scoring high.

| Score | Recommendation |
|-------|---------------|
| 75–100 | STRONG FIT |
| 50–74 | POTENTIAL FIT |
| 25–49 | WEAK FIT |
| 0–24 | NOT RECOMMENDED |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq API (`llama-3.1-8b-instant`) |
| LLM Alternatives | Claude (Anthropic), xAI |
| Vector Database | ChromaDB (embedded, baked into Docker image) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| PDF Processing | PyMuPDF + Tesseract OCR |
| Evaluation | Custom RAG eval pipeline (faithfulness, relevancy, precision) |
| Frontend | Streamlit |
| Report Generation | ReportLab |
| Containerisation | Docker |
| Deployment | Google Cloud Run (serverless, scales to zero) |
| CI/CD | Google Cloud Build (auto-deploy on `git push` to `main`) |

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Groq API key](https://console.groq.com) (free)
- Tesseract OCR installed

### Local Setup

```bash
git clone https://github.com/BaranikumarNagarajan/HireIQ-AI-Recruitment-Intelligence-Platform.git
cd HireIQ-AI-Recruitment-Intelligence-Platform

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Environment

Create `.env` in the project root:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

### Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## Project Structure

```
├── app.py                     # Streamlit UI
├── config.py                  # Configuration + env vars
├── Dockerfile                 # Production container
├── cloudbuild.yaml            # Cloud Build CI/CD pipeline
├── requirements.txt
├── agents/
│   ├── parser.py              # CV extraction (PyMuPDF + OCR)
│   ├── jd_analyser.py         # Job description analysis
│   ├── scorer.py              # 5-dimension scoring with overlap rule
│   ├── compliance.py          # GDPR RAG compliance check
│   ├── interviewer.py         # Interview question generation
│   └── reporter.py            # PDF report (ReportLab)
├── rag/
│   ├── ingest.py              # Legal document ingestion → ChromaDB
│   └── retriever.py           # Semantic search
├── eval/
│   └── ragas_eval.py          # RAG quality evaluation pipeline
├── utils/
│   ├── llm_client.py          # Unified LLM client (Groq / Claude / xAI)
│   └── chroma_client.py       # ChromaDB singleton
└── data/
    └── (legal PDF documents for RAG)
```

---

## Deployment

Deployed on **Google Cloud Run** in `europe-west1`. Every push to `main` triggers a Cloud Build pipeline that:

1. Builds the Docker image (including pre-ingesting legal documents into ChromaDB)
2. Pushes to Artifact Registry
3. Deploys to Cloud Run with zero downtime

The service scales to zero when idle — no fixed cost.

---

## Disclaimer

HireIQ is designed to assist recruitment screening and must not be the sole decision-making tool. Always apply human review and comply with local employment laws. Automated CV screening using AI may be subject to GDPR Article 22 obligations.
