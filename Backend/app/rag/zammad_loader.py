"""Zammad data loader for tickets and knowledge base entries."""
from typing import List, Dict, Optional
from datetime import datetime
import hashlib
from app.rag.zammad_client import ZammadClient
from app.config import settings
from app.utils.logger import logger


class ZammadLoader:
    """Loads tickets and knowledge base entries from Zammad."""
    
    def __init__(self, zammad_client: Optional[ZammadClient] = None):
        """Initialize Zammad loader."""  
        if zammad_client:
            self.client = zammad_client
        elif settings.zammad_base_url and settings.zammad_api_token:
            self.client = ZammadClient(settings.zammad_base_url, settings.zammad_api_token)
        else:
            raise ValueError("Zammad client or configuration required")
    
    def load_tickets(self, limit: int = 100) -> List[Dict]: # Récupère les tickets via l'API
        """Load tickets from Zammad and convert to document format."""
        documents = []
        
        try:
            tickets = self.client.get_tickets(limit=limit)
            
            # Pour chaque ticket
            for ticket in tickets:
                try:
                    ticket_id = ticket.get('id')
                    ticket_number = ticket.get('number')
                    title = ticket.get('title', f'Ticket #{ticket_number}')
                    state = ticket.get('state')
                    priority = ticket.get('priority')
                    created_at = ticket.get('created_at')
                    
                    # Check if articles are already in the ticket response (with expand=true)
                    articles = ticket.get('articles', [])
                    if not articles:
                        # Try to get articles separately
                        articles = self.client.get_ticket_articles(ticket_id)
                        if not articles:
                            logger.debug(f"Ticket #{ticket_number} has no articles (404 or empty)")
                    else:
                        logger.debug(f"Ticket #{ticket_number} has {len(articles)} articles from expand")
                    
                    # Combine ticket info and articles into document
                    ticket_text_parts = [
                        f"Ticket #{ticket_number}",
                        f"Title: {title}",
                        f"State: {state}",
                        f"Priority: {priority}",
                        f"Created: {created_at}",
                        ""
                    ]
                    
                    # Add articles
                    attachments_info = []
                    for article in articles:
                        article_id = article.get('id')
                        body = article.get('body', '')
                        from_field = article.get('from', '')
                        subject = article.get('subject', '')
                        created = article.get('created_at', '')
                        
                        ticket_text_parts.append(f"Article #{article_id}")
                        ticket_text_parts.append(f"From: {from_field}")
                        ticket_text_parts.append(f"Subject: {subject}")
                        ticket_text_parts.append(f"Created: {created}")
                        ticket_text_parts.append(f"Content:\n{body}")
                        ticket_text_parts.append("")
                        
                        # Get attachments for this article
                        attachments = self.client.get_attachment_info(ticket_id, article_id)
                        for attachment in attachments:
                            attachment_id = attachment.get('id')
                            filename = attachment.get('filename', '')
                            content_type = attachment.get('content_type', '')
                            size = attachment.get('size', 0)
                            
                            attachments_info.append({
                                'ticket_id': ticket_id,
                                'article_id': article_id,
                                'attachment_id': attachment_id,
                                'filename': filename,
                                'content_type': content_type,
                                'size': size
                            })
                    
                    ticket_text = "\n".join(ticket_text_parts)
                    
                    # Skip tickets with no content (no articles and minimal info)
                    if not ticket_text.strip() or (len(articles) == 0 and not title.strip()):
                        logger.warning(f"Skipping ticket #{ticket_number} - no content available")
                        continue
                    
                    # Generate document ID
                    doc_id = self._generate_doc_id(f"zammad_ticket_{ticket_id}")
                    
                    documents.append({
                        'id': doc_id,
                        'source': 'zammad_ticket',
                        'source_id': str(ticket_id),
                        'title': title,
                        'text': ticket_text,
                        'metadata': {
                            'ticket_id': ticket_id,
                            'ticket_number': ticket_number,
                            'state': state,
                            'priority': priority,
                            'created_at': created_at,
                            'articles_count': len(articles),
                            'attachments_count': len(attachments_info)
                        },
                        'attachments': attachments_info
                    })
                    
                    logger.info(f"Loaded ticket #{ticket_number}: {len(articles)} articles, {len(attachments_info)} attachments")
                    
                except Exception as e:
                    logger.error(f"Error processing ticket {ticket.get('id')}: {e}")
                    continue
            
            logger.info(f"Loaded {len(documents)} tickets from Zammad")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading tickets from Zammad: {e}")
            return []
    
    def load_knowledge_base_entries(self, kb_id: Optional[int] = None) -> List[Dict]:
        """Load knowledge base entries from Zammad."""
        documents = []
        
        try:
            # Use configured KB ID if not specified
            if kb_id is None and hasattr(settings, 'zammad_kb_id') and settings.zammad_kb_id:
                kb_id = settings.zammad_kb_id
                logger.info(f"Using configured KB ID: {kb_id}")
            
            entries = self.client.get_knowledge_base_entries(kb_id)
            logger.info(f"Retrieved {len(entries)} entries from Zammad API")
            
            if not entries:
                logger.warning("No KB entries retrieved from Zammad API")
                return []
            
            for entry in entries:
                try:
                    entry_id = entry.get('id')
                    title = entry.get('title', f'KB Entry #{entry_id}')
                    body = entry.get('body', '')
                    kb_id = entry.get('knowledge_base_id')
                    category_id = entry.get('category_id')
                    created_at = entry.get('created_at')
                    updated_at = entry.get('updated_at')
                    
                    # Combine entry info into document
                    entry_text = f"Knowledge Base Entry\n"
                    entry_text += f"Title: {title}\n"
                    entry_text += f"Knowledge Base ID: {kb_id}\n"
                    entry_text += f"Category ID: {category_id}\n"
                    entry_text += f"Created: {created_at}\n"
                    entry_text += f"Updated: {updated_at}\n\n"
                    entry_text += f"Content:\n{body}"
                    
                    # Generate document ID
                    doc_id = self._generate_doc_id(f"zammad_kb_{entry_id}")
                    
                    documents.append({
                        'id': doc_id,
                        'source': 'zammad_kb',
                        'source_id': str(entry_id),
                        'title': title,
                        'text': entry_text,
                        'metadata': {
                            'kb_entry_id': entry_id,
                            'knowledge_base_id': kb_id,
                            'category_id': category_id,
                            'created_at': created_at,
                            'updated_at': updated_at
                        },
                        'attachments': []  # KB entries might have attachments too
                    })
                    
                    logger.info(f"Loaded KB entry: {title}")
                    
                except Exception as e:
                    logger.error(f"Error processing KB entry {entry.get('id')}: {e}", exc_info=True)
                    continue
            
            logger.info(f"Loaded {len(documents)} knowledge base entries from Zammad")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading knowledge base entries from Zammad: {e}", exc_info=True)
            return []
    
    def load_attachment(self, ticket_id: int, article_id: int, attachment_id: int, filename: str) -> Optional[Dict]:
        """Load an attachment from Zammad."""
        try:
            content = self.client.download_attachment(ticket_id, article_id, attachment_id)
            if content:
                return {
                    'filename': filename,
                    'content': content,
                    'size': len(content)
                }
            return None
        except Exception as e:
            logger.error(f"Error loading attachment {attachment_id}: {e}")
            return None
    
    def _generate_doc_id(self, source: str) -> str:
        """Generate a unique document ID."""
        return hashlib.md5(source.encode()).hexdigest()
