# Multi-Client RAG Backend System

## Overview
Enterprise-grade RAG (Retrieval-Augmented Generation) system supporting multiple clients with complete data isolation and advanced features.

## Features
- **Multi-tenant architecture** with 4 isolated clients
- **Advanced RAG modes**: Pure, Creative, and Hybrid
- **Synthetic personas system** for market research
- **Document management** with multimodal support
- **Image generation** with DALL-E 3
- **Complete data isolation** per client via separate Azure indexes

## Supported Clients
- Tigo Honduras (Telecommunications)
- Unilever (Consumer Goods)
- Nestlé (Food & Beverages)
- Alpina (Dairy & Food)

## Tech Stack
- **FastAPI** - High-performance Python web framework
- **Azure OpenAI** - GPT-4 and embeddings
- **Azure AI Search** - Vector database with separate indexes per client
- **Railway** - Production deployment platform

## API Endpoints

### Authentication
- `POST /api/auth/login` - Multi-client authentication

### RAG Endpoints (per client)
- `POST /api/{client_id}/rag-pure` - Pure RAG mode
- `POST /api/{client_id}/rag-creative` - Creative RAG mode
- `POST /api/{client_id}/rag-hybrid` - Hybrid RAG mode
- `POST /api/{client_id}/chat` - Chat interface

### Synthetic Personas
- `POST /api/{client_id}/persona-chat` - 1:1 persona conversations
- `POST /api/{client_id}/persona-survey` - Survey responses
- `POST /api/{client_id}/focus-group` - Focus group simulations
- `POST /api/{client_id}/persona-validation` - Validation endpoints
- `POST /api/{client_id}/enhanced-generate` - Enhanced generation
- `POST /api/{client_id}/synthetic-chat` - Advanced synthetic chat

### Document Management
- `POST /api/{client_id}/add-document` - Add document to index
- `POST /api/{client_id}/upload-file` - Upload files

### Additional Features
- `POST /api/{client_id}/generate-image` - DALL-E image generation
- `GET /api/{client_id}/stats` - Client statistics
- `GET /api/health` - Health check

## Security Features
- **Complete data isolation** between clients
- **Auto-detection** of client by email domain
- **No visible client selector** for confidentiality
- **Separate Azure indexes** per client
- **Enterprise-grade authentication**

## Deployment
Configured for Railway deployment with:
- `railway.json` - Railway configuration
- `app.py` - Production entry point
- `requirements.txt` - Python dependencies

## Environment Variables Required
```bash
# Azure OpenAI
AZURE_OPENAI_API_KEY
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_VERSION
AZURE_OPENAI_DEPLOYMENT_NAME
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
AZURE_OPENAI_DALLE_DEPLOYMENT_NAME

# Azure Search
AZURE_SEARCH_SERVICE_NAME
AZURE_SEARCH_ADMIN_KEY

# Client-specific indexes
TIGO_HONDURAS_INDEX_NAME
UNILEVER_INDEX_NAME
NESTLE_INDEX_NAME
ALPINA_INDEX_NAME
```

## Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
python -m uvicorn main:app --reload --port 8000
```

## Production
```bash
# Railway automatically runs
python app.py
```

## API Documentation
Once running, visit `/docs` for interactive Swagger documentation.

## License
Proprietary - All rights reserved