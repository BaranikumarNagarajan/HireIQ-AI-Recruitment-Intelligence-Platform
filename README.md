<div align="center">

# HireIQ — AI Recruitment Intelligence Platform

**End-to-end AI-powered CV screening. Upload a CV, paste a job description, get a full candidate analysis in under 60 seconds.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Google%20Cloud%20Run-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)](https://hireiq-hmdx5ylgna-ew.a.run.app)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20Llama%203.1-F55036?style=for-the-badge)](https://console.groq.com)
[![Cloud Run](https://img.shields.io/badge/Deploy-Cloud%20Run-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)](https://cloud.google.com/run)

*Auto-deploys on every push via Google Cloud Build CI/CD*

> First load may take ~5 seconds — Cloud Run scales to zero when idle.

</div>

---

## Overview

HireIQ is a production-grade multi-agent AI system that automates the first stage of recruitment. It replaces manual CV screening with a structured, explainable, and GDPR-aware AI pipeline — producing scores, skill gap analysis, compliance flags, and interview questions from a single CV upload.

**Problem it solves:** Recruiters spend 6–8 seconds per CV at scale. HireIQ screens a candidate end-to-end in under 60 seconds with full reasoning, not just a number.

---

## What It Produces

| Output | Detail |
|--------|--------|
| **Match Score** | 0–100 across 5 weighted dimensions |
| **Skills Gap Report** | Matched vs missing skills with semantic matching |
| **GDPR Compliance Check** | RAG-powered check against legal documents via ChromaDB |
| **Interview Questions** | 5 targeted questions based on identified skill gaps |
| **PDF Report** | Downloadable professional report with full reasoning |

---

## Multi-Agent Pipeline

```
CV PDF + Job Description
        │
        ▼
┌─────────────────────┐
│   Parser Agent      │  PyMuPDF + Tesseract OCR → structured CV data
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  JD Analyser Agent  │  Groq LLM → required skills, seniority, domain
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

Each agent has a single responsibility. Outputs are passed sequentially — no agent makes assumptions about another's internals.

---

## Scoring Model

| Dimension | Weight | Method |
|-----------|--------|--------|
| Technical Skills | 35% | LLM + deterministic skill-overlap cap |
| Experience Level | 25% | LLM (internships and projects credited) |
| Domain Relevance | 20% | LLM + deterministic skill-overlap cap |
| Employment Stability | 10% | Rule-based (gaps, average tenure) |
| Education | 10% | Rule-based (degree level vs seniority) |

### Anti-Inflation Rule

LLMs tend to find loose connections between unrelated domains. HireIQ adds a deterministic guard:

- **< 10% skill overlap** → technical skills capped at 20, domain relevance capped at 25
- **< 25% skill overlap** → both capped at 40

This prevents a software engineer applying for a marketing role from scoring 80%+ because the LLM found the word "communication" on the CV.

### Score Thresholds

| Score | Recommendation |
|-------|----------------|
| 75–100 | ✅ STRONG FIT |
| 50–74 | 🟡 POTENTIAL FIT |
| 25–49 | 🟠 WEAK FIT |
| 0–24 | ❌ NOT RECOMMENDED |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Groq API — `llama-3.1-8b-instant` |
| LLM Alternatives | Claude (Anthropic), xAI Grok |
| Vector Database | ChromaDB (embedded, pre-baked into Docker image) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| PDF Processing | PyMuPDF + Tesseract OCR |
| RAG Evaluation | Custom pipeline — faithfulness, answer relevancy, context precision |
| Frontend | Streamlit |
| Report Generation | ReportLab |
| Containerisation | Docker |
| Deployment | Google Cloud Run — `europe-west1`, scales to zero |
| CI/CD | Google Cloud Build — auto-deploy on `git push` to `main` |

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Groq API key](https://console.groq.com) (free tier available)
- Tesseract OCR installed locally

### Setup

```bash
git clone https://github.com/BaranikumarNagarajan/HireIQ-AI-Recruitment-Intelligence-Platform.git
cd HireIQ-AI-Recruitment-Intelligence-Platform

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
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
├── app.py                     # Streamlit UI — all tabs and result rendering
├── config.py                  # Environment variables and scoring weights
├── Dockerfile                 # Production container — pre-bakes ChromaDB
├── cloudbuild.yaml            # Cloud Build CI/CD pipeline
├── requirements.txt
├── agents/
│   ├── parser.py              # CV extraction — PyMuPDF with Tesseract OCR fallback
│   ├── jd_analyser.py         # Job description analysis → structured JSON
│   ├── scorer.py              # 5-dimension scoring with deterministic overlap rule
│   ├── compliance.py          # GDPR RAG compliance check
│   ├── interviewer.py         # Interview question generation from identified gaps
│   └── reporter.py            # PDF report generation — ReportLab
├── rag/
│   ├── ingest.py              # Legal document ingestion → ChromaDB
│   └── retriever.py           # Semantic search over legal document store
├── eval/
│   └── ragas_eval.py          # RAG quality evaluation — custom faithfulness/relevancy/precision
├── utils/
│   ├── llm_client.py          # Unified LLM client — Groq / Claude / xAI
│   └── chroma_client.py       # ChromaDB singleton
└── data/
    └── (GDPR and employment law PDFs for RAG)
```

---

## Deployment

Deployed on **Google Cloud Run** (`europe-west1`). Every `git push` to `main` triggers:

1. Docker build — ChromaDB is pre-ingested at build time (not at startup)
2. Push to Google Artifact Registry
3. Zero-downtime deploy to Cloud Run

The service **scales to zero** when idle — no fixed infrastructure cost.

---

## RAG Evaluation

HireIQ includes a built-in RAG evaluation dashboard. It tests the legal document retrieval system against 5 GDPR/compliance questions and scores:

- **Faithfulness** — is the answer grounded in retrieved context?
- **Answer Relevancy** — does the answer address the question?
- **Context Precision** — are the retrieved chunks relevant?

This is separate from candidate screening — it validates the compliance pipeline quality independently.

---

## Disclaimer

HireIQ is designed to assist recruitment screening and must not be the sole decision-making tool. Always apply human review and comply with applicable employment law. Automated CV screening using AI may be subject to GDPR Article 22 obligations in the EU.
