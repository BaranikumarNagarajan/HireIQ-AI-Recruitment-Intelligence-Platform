"""Configuration settings for HireIQ."""
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# LLM provider — "ollama" | "claude"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b").strip()
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434").strip()

# Claude / Anthropic
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5").strip()

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()

# xAI / Grok
XAI_API_KEY = os.getenv("XAI_API_KEY", "").strip()
XAI_MODEL = os.getenv("XAI_MODEL", "grok-3-mini").strip()

# Chunking and Retrieval Settings
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 5

# Embedding Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ChromaDB Path
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

# Score Weights
SCORE_WEIGHTS = {
    "technical_skills": 0.35,
    "experience_level": 0.25,
    "domain_relevance": 0.20,
    "employment_stability": 0.10,
    "education": 0.10
}

# OCR Settings
OCR_MIN_TEXT_LENGTH = 50