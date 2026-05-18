"""CV Parser Agent."""
import logging
from typing import Dict, Any
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
from config import OCR_MIN_TEXT_LENGTH
from utils.llm_client import get_llm_response, parse_llm_json_response

logger = logging.getLogger(__name__)

def parse_cv(file_bytes: bytes) -> Dict[str, Any]:
    """
    Parse CV from PDF bytes.
    
    Args:
        file_bytes: PDF file content as bytes
        
    Returns:
        Dict with parsed CV data
    """
    try:
        # Try PyMuPDF first
        text = extract_text_with_pymupdf(file_bytes)
        extraction_method = "PyMuPDF"
        
        if len(text.strip()) < OCR_MIN_TEXT_LENGTH:
            # Fallback to OCR
            logger.info("Extracted text too short, falling back to OCR")
            text = extract_text_with_ocr(file_bytes)
            extraction_method = "Tesseract OCR"
        
        logger.info(f"Used {extraction_method} for text extraction")
        
        # Use the local LLM to extract structured data
        prompt = f"""
        Extract the following information from this CV text and return ONLY valid JSON. Do not include any explanation, markdown, or text outside the JSON object.
        - candidate_name: string
        - email: string
        - phone: string
        - skills: list of strings
        - work_experience: list of dicts with keys: company, role, duration, dates
        - education: list of strings
        - total_years_experience: number
        - employment_gaps: list of strings (any gap > 3 months between jobs)
        - average_tenure_per_role: number (in years)

        CV Text:
        {text}
        """
        
        response = get_llm_response(prompt)

        try:
            parsed_data = parse_llm_json_response(response)
        except Exception:
            logger.error("Failed to parse LLM response as JSON: %s", response)
            return {"error": "Failed to parse CV data"}

        parsed_data["extraction_method"] = extraction_method
        parsed_data["extracted_text"] = text  # needed by the Streamlit preview
        return parsed_data
        
    except Exception as e:
        logger.error(f"Failed to parse CV: {e}")
        return {"error": str(e)}

def extract_text_with_pymupdf(file_bytes: bytes) -> str:
    """Extract text using PyMuPDF."""
    text_content = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text:
                text_content.append(f"--- Page {page_num + 1} ---\n{text}")
    return "\n\n".join(text_content)

def extract_text_with_ocr(file_bytes: bytes) -> str:
    """Extract text using Tesseract OCR."""
    text_content = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc):
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            text = pytesseract.image_to_string(img)
            if text:
                text_content.append(f"--- Page {page_num + 1} ---\n{text}")
    return "\n\n".join(text_content)