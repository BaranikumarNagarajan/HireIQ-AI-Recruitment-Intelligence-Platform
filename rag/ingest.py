"""Legal Document Ingestion for RAG."""
import logging
from pathlib import Path
import numpy as np
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
from utils.chroma_client import get_chroma_client
logger = logging.getLogger(__name__)

def ingest_legal_documents() -> None:
    """
    Ingest PDF documents from data/ folder into ChromaDB.
    
    Extracts text, splits into chunks, generates embeddings, and stores in vector DB.
    """
    try:
        data_dir = Path("./data")
        if not data_dir.exists():
            logger.warning("data/ directory does not exist")
            return
        
        client = get_chroma_client()
        collection_name = "legal_documents"

        try:
            collection = client.get_collection(name=collection_name)
            if collection.count() > 0:
                logger.info("Legal documents already ingested (%d chunks), skipping", collection.count())
                return
            client.delete_collection(name=collection_name)
            collection = client.create_collection(name=collection_name)
        except ValueError:
            collection = client.create_collection(name=collection_name)
        
        # Load embedding model
        model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Process each PDF
        pdf_files = list(data_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files found in data/ directory")
            return
        
        all_chunks = []
        all_embeddings = []
        all_metadatas = []
        all_ids = []
        
        for pdf_path in pdf_files:
            logger.info(f"Processing {pdf_path.name}")
            
            # Extract text
            text = extract_text_from_pdf(str(pdf_path))
            if not text:
                logger.warning(f"No text extracted from {pdf_path.name}")
                continue
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_text(text)
            
            # Generate embeddings
            embeddings = model.encode(chunks, convert_to_numpy=True)
            
            # Convert numpy arrays to lists of floats for ChromaDB compatibility
            if isinstance(embeddings, np.ndarray):
                embeddings_list = embeddings.astype(np.float32).tolist()
            else:
                embeddings_list = [
                    emb.astype(np.float32).tolist() if isinstance(emb, np.ndarray) else emb
                    for emb in embeddings
                ]
            
            # Create metadata and IDs
            for i, chunk in enumerate(chunks):
                metadata = {
                    "source": pdf_path.name,
                    "page": "unknown",  # Could be improved with page tracking
                    "chunk_index": i
                }
                chunk_id = f"{pdf_path.stem}_chunk_{i}"
                
                all_chunks.append(chunk)
                all_embeddings.append(embeddings_list[i])
                all_metadatas.append(metadata)
                all_ids.append(chunk_id)
        
        # Store in ChromaDB
        if all_chunks:
            collection.add(
                embeddings=all_embeddings,
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids
            )
            logger.info(f"Successfully ingested {len(all_chunks)} chunks from {len(pdf_files)} documents")
        else:
            logger.warning("No chunks to ingest")
            
    except Exception as e:
        logger.error(f"Failed to ingest legal documents: {e}")
        raise

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file."""
    text_content = []
    doc = fitz.open(file_path)
    try:
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text()  # type: ignore[attr-defined]
            if text:
                text_content.append(f"--- Page {page_num + 1} ---\n{text}")
    finally:
        doc.close()
    return "\n\n".join(text_content)