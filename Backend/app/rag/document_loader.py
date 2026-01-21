"""Document loader for scanning and loading documents from local folder."""
import os
from pathlib import Path
from typing import List, Dict, Optional
from app.config import settings
from app.utils.logger import logger


class DocumentLoader:
    """Loads documents from a local folder."""
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.md', '.html', '.htm'}
    
    def __init__(self, folder_path: Optional[str] = None):
        """Initialize document loader with folder path."""
        self.folder_path = Path(folder_path or settings.documents_folder)
        if not self.folder_path.exists():
            logger.warning(f"Documents folder does not exist: {self.folder_path}")
            self.folder_path.mkdir(parents=True, exist_ok=True)
    
    def scan_documents(self) -> List[Dict[str, str]]:
        """Scan folder for supported documents and return metadata."""
        documents = []
        
        if not self.folder_path.exists():
            logger.warning(f"Documents folder not found: {self.folder_path}")
            return documents
        
        for file_path in self.folder_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    relative_path = file_path.relative_to(self.folder_path)
                    documents.append({
                        "path": str(file_path),
                        "filename": file_path.name,
                        "extension": file_path.suffix.lower(),
                        "relative_path": str(relative_path),
                        "size": file_path.stat().st_size
                    })
                    logger.info(f"Found document: {file_path.name}")
                except Exception as e:
                    logger.error(f"Error scanning file {file_path}: {e}")
        
        logger.info(f"Scanned {len(documents)} documents")
        return documents
    
    def load_document(self, file_path: str) -> Optional[bytes]:
        """Load document content as bytes."""
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"File not found: {file_path}")
                return None
            with open(path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {e}")
            return None
