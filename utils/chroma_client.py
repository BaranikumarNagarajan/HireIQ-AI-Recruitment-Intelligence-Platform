"""Single shared ChromaDB client for the whole process."""
import chromadb
from config import CHROMA_DB_PATH

chromadb.configure(anonymized_telemetry=False)

_client = None


def get_chroma_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _client
