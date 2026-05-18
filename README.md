# HireIQ — AI Recruitment Intelligence Platform

HireIQ is an end-to-end AI-powered recruitment platform that automates CV screening and HR compliance checking. Upload a candidate CV (PDF) and a job description — the platform runs it through 6 specialised AI agents and produces match scores, compliance flags, interview questions, and a downloadable PDF report.

---

## Features

- **Automated CV Parsing** — extracts structured data from PDF CVs using PyMuPDF with Tesseract OCR fallback
- **Job Description Analysis** — identifies required skills, experience level, and culture signals
- **Multi-Dimensional Scoring** — evaluates candidates across 5 weighted dimensions
- **Legal Compliance RAG** — GDPR and anti-discrimination checking via ChromaDB vector search
- **Interview Question Generation** — targeted questions based on identified skill gaps
- **Professional PDF Reports** — downloadable HR reports generated with ReportLab
- **RAG Evaluation Dashboard** — sequential LLM-based evaluation pipeline (faithfulness, relevancy, precision)

---

## Architecture

### Multi-Agent Pipeline

```
CV PDF + Job Description
        │
        ▼
┌─────────────────┐
│  Parser Agent   │  → extracts name, skills, experience, education
└────────┬────────┘
         │
┌────────▼────────┐
│  JD Analyser   │  → extracts required skills, seniority, domain
└────────┬────────┘
         │
┌────────▼────────┐
│  Scorer Agent  │  → 5-dimension weighted score (parallel LLM calls)
└────────┬────────┘
         │
┌────────▼────────┐
│ Compliance Agent│  → RAG-powered GDPR + bias check
└────────┬────────┘
         │
┌────────▼────────┐
│Interviewer Agent│  → 5 targeted interview questions
└────────┬────────┘
         │
┌────────▼────────┐
│ Reporter Agent  │  → PDF report with scores, flags, questions
└─────────────────┘
```

### Scoring Dimensions

| Dimension | Weight |
|-----------|--------|
| Technical Skills | 35% |
| Experience Level | 25% |
| Domain Relevance | 20% |
| Employment Stability | 10% |
| Education | 10% |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq API (`llama-3.3-70b-versatile`) |
| LLM Fallbacks | xAI, Claude (Anthropic), Ollama (local) |
| Agent Framework | LangChain |
| Vector Database | ChromaDB (embedded, persistent) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| PDF Processing | PyMuPDF + Tesseract OCR |
| Evaluation | Custom sequential RAG eval pipeline |
| Frontend | Streamlit |
| Report Generation | ReportLab |
| Deployment | Google Cloud Run via GitHub Actions CI/CD |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Groq API key (free at [console.groq.com](https://console.groq.com))
- Tesseract OCR (for scanned PDFs)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/hireiq.git
cd hireiq

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:

```env
# LLM Provider — groq | xai | claude | ollama
LLM_PROVIDER=groq

# Groq API (free at console.groq.com — key starts with gsk_)
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Ollama (local fallback)
LLM_MODEL=llama3.2:3b
OLLAMA_API_URL=http://localhost:11434
```

### Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## Usage

1. Upload a candidate CV in PDF format
2. Paste the job description
3. Click **Run HireIQ Analysis**
4. Review scores, skills match, compliance flags, and interview questions
5. Download the PDF report for HR records

To evaluate RAG system quality, go to **Eval Dashboard** and click **Run RAGAS Evaluation**.

---

## Configuration

### LLM Provider

Switch provider in `.env`:

```env
# Groq (recommended — fast, free)
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile

# xAI / Grok
LLM_PROVIDER=xai
XAI_API_KEY=xai-...
XAI_MODEL=grok-2-mini-latest

# Claude (Anthropic)
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-...

# Ollama (local, no API key needed)
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
```

### Score Weights

Edit `config.py`:

```python
SCORE_WEIGHTS = {
    "technical_skills": 0.35,
    "experience_level": 0.25,
    "domain_relevance": 0.20,
    "employment_stability": 0.10,
    "education": 0.10,
}
```

---

## Project Structure

```
hireiq/
├── app.py                    # Streamlit UI
├── config.py                 # All configuration + env vars
├── requirements.txt
├── Dockerfile
├── docker-compose.yml        # Local Docker testing
├── .env                      # Secrets (never commit this)
├── .github/
│   └── workflows/
│       └── deploy.yml        # GitHub Actions CI/CD → Cloud Run
├── agents/
│   ├── parser.py             # CV extraction (PyMuPDF + OCR)
│   ├── jd_analyser.py        # Job description analysis
│   ├── scorer.py             # 5-dimension parallel scoring
│   ├── compliance.py         # GDPR RAG compliance check
│   ├── interviewer.py        # Interview question generation
│   └── reporter.py           # PDF report (ReportLab)
├── rag/
│   ├── ingest.py             # Legal document ingestion → ChromaDB
│   └── retriever.py          # Semantic search
├── eval/
│   └── ragas_eval.py         # Sequential RAG evaluation pipeline
├── utils/
│   ├── llm_client.py         # Unified LLM client (Groq/xAI/Claude/Ollama)
│   └── chroma_client.py      # Singleton ChromaDB client
└── data/
    └── (legal PDF documents for RAG)
```

---

## Deployment

### Local Docker

```bash
docker-compose up --build
# Open http://localhost:8501
```

### Google Cloud Run (CI/CD)

Every push to `main` automatically builds and deploys via GitHub Actions.

**One-time GCP setup:**

```bash
# Enable APIs
gcloud services enable run.googleapis.com artifactregistry.googleapis.com

# Create Artifact Registry repo
gcloud artifacts repositories create hireiq \
  --repository-format=docker --location=us-central1

# Create and configure service account
gcloud iam service-accounts create hireiq-deployer
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:hireiq-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.admin"
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:hireiq-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:hireiq-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

# Download key
gcloud iam service-accounts keys create key.json \
  --iam-account=hireiq-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

**GitHub Secrets required** (Settings → Secrets → Actions):

| Secret | Value |
|--------|-------|
| `GCP_PROJECT_ID` | Your GCP project ID |
| `GCP_SA_KEY` | Contents of `key.json` |
| `GROQ_API_KEY` | Your `gsk_...` Groq key |
| `GROQ_MODEL` | `llama-3.1-8b-instant` |

---

## Groq Free Tier Limits

| Model | Tokens/Day | Best for |
|-------|-----------|----------|
| `llama-3.3-70b-versatile` | 100k | CV analysis (more accurate) |
| `llama-3.1-8b-instant` | 500k | Evaluation pipeline (higher limit) |

If you hit the daily limit on `llama-3.3-70b-versatile`, switch to `llama-3.1-8b-instant` in `.env` until midnight UTC reset.

---

## Disclaimer

HireIQ is designed to assist with recruitment screening and should not be used as the sole decision-making tool. Always involve human review and comply with local employment laws and regulations.
