FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-ingest legal documents so chroma_db is baked into the image.
# This avoids cold-start ingestion on Cloud Run (which has ephemeral disk).
RUN python -c "from rag.ingest import ingest_legal_documents; ingest_legal_documents()"

EXPOSE 8080
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true"]
