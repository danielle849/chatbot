# Backend - RAG Chatbot API

FastAPI backend application for the RAG chatbot system with Zammad integration.

## Features

- FastAPI REST API
- Qdrant vector database integration
- HuggingFace LLM (Mistral 7B) and embeddings
- LangChain RAG pipeline
- Unstructured document processing
- **Zammad integration** - Sync tickets and knowledge base entries
- API key authentication (OAuth/SSO planned for future)
- German language support

## Setup

### Prerequisites

- Docker and Docker Compose
- HuggingFace API token
- Zammad API token (for Zammad sync)

### Configuration

1. Copy `env.example` to `.env`:
   ```bash
   cp env.example .env
   ```

2. Edit `.env` and set:
   - `HF_API_TOKEN`: Your HuggingFace API token
   - `API_KEY`: A secure API key for authentication
   - `DOCUMENTS_FOLDER`: Path to your documents folder (default: `../data`)
   - `ZAMMAD_BASE_URL`: Your Zammad instance URL (e.g., `https://helpdesk.company.com`)
   - `ZAMMAD_API_TOKEN`: Your Zammad API token
   - `ZAMMAD_SYNC_TICKETS`: Enable ticket sync (default: `true`)
   - `ZAMMAD_SYNC_KB`: Enable knowledge base sync (default: `true`)
   - `ZAMMAD_SYNC_ATTACHMENTS`: Enable attachment processing (default: `true`)

### Running with Docker

```bash
docker-compose up -d
```

This will start:
- Qdrant vector database (port 6333)
- FastAPI backend (port 8000)

### Running Locally

1. Install Python 3.11+
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start Qdrant (using Docker):
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

4. Run the backend:
   ```bash
   uvicorn app.main:app --reload
   ```

## API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints

### Chat
- `POST /api/chat` - Send a chat message
- `POST /api/chat/stream` - Stream chat response

### Documents
- `POST /api/documents/ingest?source=local` - Ingest documents from local folder
- `POST /api/documents/ingest?source=zammad` - Ingest documents from Zammad
- `POST /api/documents/sync/zammad` - Sync documents from Zammad (tickets and KB)
- `GET /api/documents` - List ingested documents
- `DELETE /api/documents/{doc_id}` - Delete a document

### Health
- `GET /api/health` - Health check
- `GET /api/health/qdrant` - Qdrant connection check

## Zammad Integration

The system can sync data from Zammad:

1. **Tickets**: Fetches tickets with their articles and metadata
2. **Knowledge Base**: Fetches knowledge base entries
3. **Attachments**: Processes attachments (PDF, DOCX, HTML) from tickets

### Syncing from Zammad

```bash
curl -X POST "http://localhost:8000/api/documents/sync/zammad" \
  -H "X-API-Key: your-api-key-here"
```

Or use the Swagger UI at `http://localhost:8000/docs`

### Zammad API Token

To get a Zammad API token:
1. Log into your Zammad instance
2. Go to Settings → API Tokens
3. Create a new token with appropriate permissions
4. Copy the token to your `.env` file

## Usage

### Ingest Local Documents

Place your documents (PDF, DOCX, TXT, MD, HTML) in the configured folder, then:

```bash
curl -X POST "http://localhost:8000/api/documents/ingest?source=local" \
  -H "X-API-Key: your-api-key-here"
```

### Sync from Zammad

```bash
curl -X POST "http://localhost:8000/api/documents/sync/zammad" \
  -H "X-API-Key: your-api-key-here"
```

### Chat

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "X-API-Key: your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"message": "Was ist die Firmenpolitik?"}'
```

## Project Structure

```
Backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration
│   ├── models.py            # Pydantic models
│   ├── auth.py              # Authentication (API key, OAuth/SSO planned)
│   ├── rag/                 # RAG components
│   │   ├── document_loader.py
│   │   ├── zammad_client.py    # Zammad API client
│   │   ├── zammad_loader.py    # Zammad data loader
│   │   ├── document_processor.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   └── chain.py
│   ├── api/                 # API endpoints
│   │   ├── chat.py
│   │   ├── documents.py
│   │   └── health.py
│   └── utils/               # Utilities
│       └── logger.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Authentication

Currently uses API key authentication. OAuth/SSO integration is planned for future releases.

## Troubleshooting

- Check logs: `docker-compose logs -f backend`
- Verify Qdrant: `curl http://localhost:6333/health`
- Verify Zammad connection: Check ZAMMAD_BASE_URL and ZAMMAD_API_TOKEN in `.env`
- Check environment variables are set correctly
