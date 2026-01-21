"""Zammad API client for fetching tickets and knowledge base entries."""
import requests
from typing import List, Dict, Optional, Any
from datetime import datetime
import io
from bs4 import BeautifulSoup
from app.config import settings
from app.utils.logger import logger


class ZammadClient:
    """Client for interacting with Zammad API."""
    
    def __init__(self, base_url: str, api_token: str):
        """Initialize Zammad client."""
        self.base_url = base_url.rstrip('/')
        self.api_token = api_token
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Token token={api_token}',
            'Content-Type': 'application/json'
        })
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """Make HTTP request to Zammad API."""
        url = f"{self.base_url}/api/v1/{endpoint}"
        logger.info(f"Zammad API Request: {method} {url}")
        try:
            response = self.session.request(method, url, **kwargs)
            logger.info(f"Zammad API Response: Status {response.status_code} for {endpoint}")
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Zammad API Response body type: {type(result)}, keys: {list(result.keys()) if isinstance(result, dict) else 'N/A (list)'}")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Zammad API request failed: {method} {url}")
            logger.error(f"Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response headers: {dict(e.response.headers)}")
                if hasattr(e.response, 'text'):
                    logger.error(f"Response body: {e.response.text[:500]}")  # Limit to 500 characters
            return None
    
    def _parse_response(self, result: Optional[Dict], key: str, default: List = None) -> List:
        """Parse API response, handling both list and dict formats.
        
        Args:
            result: API response (can be dict or list)
            key: Key to extract if result is a dict
            default: Default value if result is None
        
        Returns:
            List of items from the response
        """
        if result is None:
            return default or []
        if isinstance(result, list):
            return result
        return result.get(key, default or [])
    
    def _find_body_recursive(self, obj: Any) -> str:
        """Recursively search for a non-empty 'body' field in nested JSON structure."""
        if isinstance(obj, dict):
            # Check if this dict directly contains a non-empty body
            if "body" in obj and isinstance(obj["body"], str) and obj["body"].strip():
                return obj["body"]
            
            # Recursively search in all values
            for value in obj.values():
                result = self._find_body_recursive(value)
                if result:
                    return result
        
        elif isinstance(obj, list):
            # Recursively search in all list items
            for item in obj:
                result = self._find_body_recursive(item)
                if result:
                    return result
        
        return ""
    
    def _html_to_text(self, html: str) -> str:
        """Convert HTML string to plain text."""
        if not html:
            return ""
        try:
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text(separator=" ", strip=True)
        except Exception as e:
            logger.warning(f"Error converting HTML to text: {e}")
            return html  # Return original if conversion fails
    
    def get_tickets(self, limit: int = 100, page: int = 1) -> List[Dict]:
        """Fetch tickets from Zammad."""
        try:
            # Get tickets with articles
            params = {
                'per_page': limit,
                'page': page,
                'expand': 'true'  # Include articles
            }
            result = self._make_request('GET', 'tickets', params=params)
            tickets = self._parse_response(result, 'tickets')
            if tickets:
                logger.info(f"Fetched {len(tickets)} tickets from Zammad")
            return tickets
        except Exception as e:
            logger.error(f"Error fetching tickets: {e}")
            return []
    
    def get_ticket_articles(self, ticket_id: int) -> List[Dict]:
        """Fetch articles for a specific ticket."""
        try:
            # Try different endpoints for articles
            endpoints_to_try = [
                f'tickets/{ticket_id}/articles',
                f'tickets/{ticket_id}/articles.json',
            ]
            
            for endpoint in endpoints_to_try:
                result = self._make_request('GET', endpoint)
                if result is not None:
                    articles = self._parse_response(result, 'articles')
                    if articles:
                        return articles
            
            # If all endpoints fail, return empty list (404 is expected for some tickets)
            logger.debug(f"No articles found for ticket {ticket_id} (endpoint may not exist or ticket has no articles)")
            return []
        except Exception as e:
            logger.debug(f"Error fetching articles for ticket {ticket_id}: {e}")
            return []
    
    def get_knowledge_base_entries(self, kb_id: Optional[int] = None) -> List[Dict]:
        """Fetch knowledge base entries from Zammad using two-step API approach.
        
        Returns a list of dictionaries with structure compatible with zammad_loader:
        - id: answer_id
        - title: extracted from KnowledgeBaseAnswerTranslation
        - body: text content extracted from HTML body
        - answer_id: original answer ID
        - knowledge_base_id: KB ID
        """
        try:
            # Use configured KB ID if not provided
            if not kb_id:
                if hasattr(settings, 'zammad_kb_id') and settings.zammad_kb_id:
                    kb_id = settings.zammad_kb_id
                else:
                    logger.warning("No KB ID provided and no configured KB ID found")
                    return []
            
            logger.info(f"Fetching KB entries from knowledge base ID: {kb_id}")
            
            # Step 1: Get knowledge base to retrieve answer_ids
            kb_result = self._make_request('GET', f'knowledge_bases/{kb_id}')
            if kb_result is None:
                logger.error(f"Failed to fetch knowledge base {kb_id}")
                return []
            
            answer_ids = kb_result.get("answer_ids") or []
            if not isinstance(answer_ids, list):
                logger.error(f"Invalid answer_ids format in knowledge base {kb_id}")
                return []
            
            answer_ids = [int(x) for x in answer_ids if x]
            logger.info(f"Found {len(answer_ids)} answer IDs in knowledge base {kb_id}")
            
            if not answer_ids:
                logger.warning(f"No answer IDs found in knowledge base {kb_id}")
                return []
            
            # Step 2: Fetch each answer individually with include_contents
            entries = []
            for i, answer_id in enumerate(answer_ids, start=1):
                try:
                    # Fetch detailed answer with include_contents parameter
                    answer_result = self._make_request(
                        'GET',
                        f'knowledge_bases/{kb_id}/answers/{answer_id}',
                        params={"include_contents": answer_id}
                    )
                    
                    if answer_result is None:
                        logger.warning(f"[{i}/{len(answer_ids)}] Failed to fetch answer {answer_id}")
                        continue
                    
                    # Extract body HTML recursively
                    body_html = self._find_body_recursive(answer_result)
                    if not body_html:
                        logger.warning(f"[{i}/{len(answer_ids)}] No body found for answer {answer_id}")
                        continue
                    
                    # Convert HTML to text
                    body_text = self._html_to_text(body_html)
                    
                    # Extract title from KnowledgeBaseAnswerTranslation
                    title = f"KB Entry #{answer_id}"  # Default title
                    kb_translations = answer_result.get("KnowledgeBaseAnswerTranslation") or {}
                    if isinstance(kb_translations, dict) and kb_translations:
                        # Get first translation's title
                        first_translation = next(iter(kb_translations.values()))
                        if isinstance(first_translation, dict):
                            extracted_title = first_translation.get("title")
                            if extracted_title:
                                title = extracted_title
                    
                    # Build entry compatible with zammad_loader expectations
                    entry = {
                        'id': answer_id,  # For compatibility with loader
                        'answer_id': answer_id,  # Keep original ID
                        'title': title,
                        'body': body_text,  # Text content (not HTML)
                        'knowledge_base_id': kb_id,
                        # Include other metadata if available
                        'created_at': answer_result.get('created_at'),
                        'updated_at': answer_result.get('updated_at'),
                    }
                    
                    entries.append(entry)
                    logger.info(f"[{i}/{len(answer_ids)}] Successfully processed answer_id={answer_id}, title='{title}', text_len={len(body_text)}")
                    
                except Exception as e:
                    logger.error(f"[{i}/{len(answer_ids)}] Error processing answer {answer_id}: {e}", exc_info=True)
                    continue
            
            logger.info(f"Successfully fetched {len(entries)} KB entries from knowledge base {kb_id}")
            return entries
            
        except Exception as e:
            logger.error(f"Error fetching knowledge base entries: {e}", exc_info=True)
            return []
    
    def download_attachment(self, ticket_id: int, article_id: int, attachment_id: int) -> Optional[bytes]:
        """Download attachment from Zammad."""
        try:
            url = f"{self.base_url}/api/v1/tickets/{ticket_id}/articles/{article_id}/attachments/{attachment_id}"
            response = self.session.get(url, stream=True)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Error downloading attachment {attachment_id}: {e}")
            return None
    
    def get_attachment_info(self, ticket_id: int, article_id: int) -> List[Dict]:
        """Get attachment information for an article."""
        try:
            result = self._make_request('GET', f'tickets/{ticket_id}/articles/{article_id}/attachments')
            return self._parse_response(result, 'attachments')
        except Exception as e:
            logger.error(f"Error fetching attachment info: {e}")
            return []