"""Document processor using Unstructured library and pypdf for PDFs."""
from typing import List, Dict, Optional
from pathlib import Path
import hashlib
import os

# Use pypdf for PDFs to avoid onnxruntime dependency
from pypdf import PdfReader

from app.config import settings
from app.utils.logger import logger

# Import unstructured functions for non-PDF formats only
# Avoid importing PDF-related modules to prevent onnxruntime loading
try:
    from unstructured.partition.docx import partition_docx
    from unstructured.partition.text import partition_text
    from unstructured.partition.html import partition_html
    from unstructured.chunking.title import chunk_by_title #A library that chunks documents by title.
    UNSTRUCTURED_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Unstructured import failed: {e}")
    UNSTRUCTURED_AVAILABLE = False


class DocumentProcessor:
    """Processes documents using Unstructured library."""
    
    def __init__(self):
        """Initialize document processor."""
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
    
    def _extract_text_from_element(self, elem) -> str:
        """Extract text from various element types (dict, unstructured object, string)."""
        if isinstance(elem, str):
            return elem
        elif isinstance(elem, dict):
            return elem.get("text", "")
        else:
            # Handle unstructured library objects
            return getattr(elem, 'text', str(elem))
    
    def process_document(self, file_path: str, file_content: Optional[bytes] = None) -> List[Dict]:
        """Process a document and return chunks with metadata."""
        try:
            path = Path(file_path)
            file_extension = path.suffix.lower()
            
            # Determine file type
            if file_extension == '.pdf':
                # Use pypdf directly to avoid onnxruntime
                elements = self._process_pdf_with_pypdf(path)
            elif file_extension in ['.docx']:
                if not UNSTRUCTURED_AVAILABLE:
                    raise ImportError("Unstructured library not available")
                elements = partition_docx(filename=str(path))
            elif file_extension in ['.txt', '.md']:
                if file_content:
                    content = file_content.decode('utf-8', errors='ignore')
                else:
                    content = path.read_text(encoding='utf-8', errors='ignore')
                if UNSTRUCTURED_AVAILABLE:
                    try:
                        elements = partition_text(text=content)
                        if not elements or (isinstance(elements, list) and len(elements) == 0):
                            logger.warning(f"partition_text returned empty result for {file_path}, using fallback")
                            elements = [{"text": content, "type": "text"}]
                    except Exception as e:
                        logger.warning(f"partition_text failed for {file_path}: {e}, using fallback")
                        elements = [{"text": content, "type": "text"}]
                else:
                    # Fallback: simple text splitting
                    elements = [{"text": content, "type": "text"}]
            elif file_extension in ['.html', '.htm']:
                if file_content:
                    content = file_content.decode('utf-8', errors='ignore')
                else:
                    content = path.read_text(encoding='utf-8', errors='ignore')
                if UNSTRUCTURED_AVAILABLE:
                    try:
                        elements = partition_html(text=content)
                        if not elements or (isinstance(elements, list) and len(elements) == 0):
                            logger.warning(f"partition_html returned empty result for {file_path}, using fallback")
                            from bs4 import BeautifulSoup
                            soup = BeautifulSoup(content, 'html.parser')
                            elements = [{"text": soup.get_text(), "type": "html"}]
                    except Exception as e:
                        logger.warning(f"partition_html failed for {file_path}: {e}, using fallback")
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(content, 'html.parser')
                        elements = [{"text": soup.get_text(), "type": "html"}]
                else:
                    # Fallback: simple text extraction
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(content, 'html.parser')
                    elements = [{"text": soup.get_text(), "type": "html"}]
            else:
                logger.warning(f"Unsupported file type: {file_extension}")
                return []
            
            # Validate elements before chunking
            if not elements or (isinstance(elements, list) and len(elements) == 0):
                logger.warning(f"No elements extracted from {file_path}")
                return []
            
            # Chunk the document
            if UNSTRUCTURED_AVAILABLE and isinstance(elements, list) and len(elements) > 0 and not isinstance(elements[0], dict):
                # Use unstructured chunking if elements are unstructured objects (not dicts)
                try:
                    chunks = chunk_by_title(
                        elements,
                        max_characters=self.chunk_size,
                        combine_under_n_chars=self.chunk_size - self.chunk_overlap,
                        overlap=self.chunk_overlap
                    )
                except Exception as e:
                    logger.warning(f"Unstructured chunking failed, using simple chunking: {e}")
                    chunks = self._simple_chunk(elements)
            else:
                # Simple chunking for plain text elements (dicts or strings)
                chunks = self._simple_chunk(elements)
            
            # Prepare chunks with metadata
            processed_chunks = []
            doc_id = self._generate_doc_id(file_path)
            
            for i, chunk in enumerate(chunks):
                # Handle both unstructured objects and plain strings
                if isinstance(chunk, str):
                    chunk_text = chunk
                    chunk_type = 'text'
                else:
                    chunk_text = self._extract_text_from_element(chunk)
                    chunk_type = getattr(chunk, 'category', 'unknown')
                
                if not chunk_text.strip():
                    continue
                
                chunk_metadata = {
                    "doc_id": doc_id,
                    "filename": path.name,
                    "file_path": str(path),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_type": chunk_type,
                }
                
                # Add element metadata if available
                if hasattr(chunk, 'metadata'):
                    chunk_metadata.update({
                        "page_number": getattr(chunk.metadata, 'page_number', None),
                        "filename": getattr(chunk.metadata, 'filename', path.name),
                    })
                elif isinstance(chunk, dict) and 'page_number' in chunk:
                    chunk_metadata["page_number"] = chunk['page_number']
                
                processed_chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
            
            logger.info(f"Processed {path.name}: {len(processed_chunks)} chunks")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}", exc_info=True)
            return []
    
    def _process_pdf_with_pypdf(self, path: Path) -> List[Dict]:
        """Process PDF using pypdf (avoids onnxruntime dependency)."""
        try:
            reader = PdfReader(str(path))
            elements = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    elements.append({
                        "text": text,
                        "type": "text",
                        "page_number": page_num + 1
                    })
            return elements
        except Exception as e:
            logger.error(f"Error processing PDF with pypdf: {e}")
            return []
    
    def _simple_chunk(self, elements: List[Dict]) -> List[str]:
        """Simple chunking for plain text elements."""
        chunks = []
        full_text = ""
        
        for elem in elements:
            text = self._extract_text_from_element(elem)
            full_text += text + "\n\n"
        
        # Simple chunking by character count
        current_chunk = ""
        for char in full_text:
            current_chunk += char
            if len(current_chunk) >= self.chunk_size:
                chunks.append(current_chunk)
                # Overlap
                current_chunk = current_chunk[-self.chunk_overlap:]
        
        if current_chunk.strip():
            chunks.append(current_chunk)
        
        return chunks
    
    def _generate_doc_id(self, file_path: str) -> str:
        """Generate a unique document ID based on file path."""
        return hashlib.md5(str(file_path).encode()).hexdigest()