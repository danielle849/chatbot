"""File handling utilities."""
import tempfile
import os
from contextlib import contextmanager
from typing import Optional, Union


@contextmanager
def temp_document_file(content: Union[str, bytes], suffix: str = '.txt', mode: str = 'w', encoding: str = 'utf-8'):
    """Context manager for temporary document files.
    
    Args:
        content: Content to write to the file
        suffix: File suffix (e.g., '.txt', '.pdf')
        mode: File mode ('w' for text, 'wb' for binary)
        encoding: Text encoding (only used for text mode)
    
    Yields:
        Path to the temporary file
    """
    if mode == 'w':
        kwargs = {'mode': mode, 'delete': False, 'suffix': suffix, 'encoding': encoding}
    else:
        kwargs = {'mode': mode, 'delete': False, 'suffix': suffix}
    
    with tempfile.NamedTemporaryFile(**kwargs) as tmp_file:
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        yield tmp_path
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
