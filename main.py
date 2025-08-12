# main.py
"""
Multi-Client RAG System - FastAPI Application
Supports multiple clients: Tigo Honduras, Unilever, Nestlé, Alpina
Each client has dedicated Azure Search indexes and personalized configurations
"""

import os
import json
import uvicorn
import requests
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import base64

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[SUCCESS] Environment variables loaded from .env file")
except ImportError:
    print("[WARNING] python-dotenv not installed, using system environment variables")

from core.multi_client_vector_store import MultiClientAzureSearchVectorStore
from core.multimodal_processor import MultimodalInputProcessor
from core.multimodal_output import MultimodalOutputGenerator
from core.intelligent_suggestions import IntelligentSuggestionEngine
from core.data_exporter import RAGDataExporter
from core.client_configuration_manager import ClientConfigurationManager

# Pydantic models for API
class MultimodalQuery(BaseModel):
    """Multimodal query input"""
    text: Optional[str] = Field(None, description="Text query")
    images: Optional[List[str]] = Field(None, description="Base64 encoded images")
    audio: Optional[List[str]] = Field(None, description="Base64 encoded audio")
    metadata_filter: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    output_types: Optional[List[str]] = Field(["text"], description="Output types: text, table, chart, image")
    rag_percentage: Optional[int] = Field(None, description="RAG vs LLM percentage (hybrid mode)")

class RAGResponse(BaseModel):
    """RAG response output"""
    answer: str
    visualizations: Dict[str, List[Dict[str, Any]]]
    citations: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    has_visualizations: bool
    suggestions: Optional[Dict[str, Any]] = None
    timestamp: str

class LoginRequest(BaseModel):
    """Login request model"""
    username: str
    password: str
    client: str  # New field for client selection

class LoginResponse(BaseModel):
    """Login response model"""
    success: bool
    token: Optional[str] = None
    message: str
    user: Optional[Dict[str, Any]] = None
    client_info: Optional[Dict[str, Any]] = None

class ExportQuery(BaseModel):
    """Export RAG response query"""
    rag_response: Dict[str, Any] = Field(..., description="RAG response data to export")
    format_type: str = Field("excel", description="Export format: excel, csv, json, html")
    include_metadata: bool = Field(True, description="Include metadata in export")

class MultiClientRAGSystem:
    """Multi-Client RAG System supporting multiple clients"""
    
    def __init__(self):
        """Initialize multi-client system"""
        # Load client configurations
        self.config_manager = ClientConfigurationManager()
        self.clients_config = self.config_manager.load_all_client_configs()
        
        # Initialize core components
        self.vector_store = MultiClientAzureSearchVectorStore(self.clients_config)
        self.multimodal_processor = MultimodalInputProcessor(self.clients_config)
        self.output_generator = MultimodalOutputGenerator(self.clients_config)
        self.suggestion_engine = IntelligentSuggestionEngine()
        self.data_exporter = RAGDataExporter()
        
        print("🚀 Multi-Client RAG System initialized")
        print(f"   📊 Supported clients: {', '.join(self.clients_config.keys())}")
        print(f"   🎯 Vector stores configured for all clients")
        print(f"   🔍 Azure AI Search indexes: {len(self.clients_config)} indexes")

    def get_client_config(self, client_id: str) -> Dict[str, Any]:
        """Get configuration for specific client"""
        if client_id not in self.clients_config:
            raise ValueError(f"Unknown client: {client_id}")
        return self.clients_config[client_id]

    def process_multimodal_query(self, 
                                query_data: MultimodalQuery, 
                                mode: str, 
                                client_id: str) -> Dict[str, Any]:
        """Process multimodal query for specific client"""
        try:
            start_time = datetime.now()
            
            # Get client configuration
            client_config = self.get_client_config(client_id)
            
            # 1. Process multimodal input with client context
            input_data = {
                "text": query_data.text,
                "images": query_data.images or [],
                "audio": query_data.audio or [],
                "metadata": {
                    "mode": mode, 
                    "endpoint": f"rag_{mode}",
                    "client": client_id
                }
            }
            
            processed_input = self.multimodal_processor.process_input(input_data, client_config)
            
            if "error" in processed_input:
                raise HTTPException(status_code=400, detail=processed_input["error"])
            
            # 2. Extract query intent
            intent_data = self.multimodal_processor.extract_query_intent(processed_input)
            
            # 3. Prepare search with client-specific filters
            search_query = processed_input["combined_content"] or query_data.text or ""
            if not search_query.strip():
                raise HTTPException(status_code=400, detail="No searchable content provided")
            
            # 4. Perform vector search in client-specific index
            endpoint_config = client_config["endpoints"][f"rag_{mode}"]
            
            search_results = self.vector_store.similarity_search(
                client_id=client_id,
                query=search_query,
                k=endpoint_config.get("max_context_chunks", 5),
                metadata_filter=query_data.metadata_filter,
                min_similarity=0.01
            )
            
            # 5. Build context and citations
            context_parts = []
            citations = []
            
            for doc, similarity in search_results:
                context_parts.append(f"[Documento: {doc.metadata.get('document_name', 'Unknown')}]\n{doc.content}")
                citations.append({
                    "document": doc.metadata.get("document_name", "Unknown"),
                    "study_type": doc.metadata.get("study_type", "Unknown"),
                    "year": doc.metadata.get("year", "Unknown"),
                    "similarity": round(similarity, 3),
                    "section": doc.metadata.get("section_type", "Unknown"),
                    "client": client_id
                })
            
            context = "\n\n".join(context_parts)
            
            # 6. Generate response with client branding
            response_data = self.output_generator.generate_response(
                query=search_query,
                context=context,
                mode=mode,
                output_types=query_data.output_types or ["text"],
                client_config=client_config
            )
            
            # 7. Generate intelligent suggestions
            response_metadata = {
                "mode": mode,
                "client": client_id,
                "chunks_retrieved": len(search_results),
                "has_visualizations": len(response_data.get("tables", [])) + len(response_data.get("charts", [])) + len(response_data.get("images", [])) > 0
            }
            
            suggestions = self.suggestion_engine.analyze_response(
                answer=response_data.get("text_response", ""),
                citations=citations,
                metadata=response_metadata
            )
            
            # 8. Build final response
            processing_time = (datetime.now() - start_time).total_seconds()
            
            final_response = {
                "answer": response_data.get("text_response", ""),
                "visualizations": {
                    "tables": response_data.get("tables", []),
                    "charts": response_data.get("charts", []),
                    "images": response_data.get("images", [])
                },
                "citations": citations,
                "metadata": {
                    "client": client_id,
                    "client_name": client_config["client_info"]["name"],
                    "mode": mode,
                    "processing_time_seconds": round(processing_time, 2),
                    "chunks_retrieved": len(search_results),
                    "index_name": client_config["azure_search"]["index_name"],
                    "endpoint_config": endpoint_config
                },
                "has_visualizations": len(response_data.get("tables", [])) + len(response_data.get("charts", [])) + len(response_data.get("images", [])) > 0,
                "suggestions": suggestions,
                "timestamp": datetime.now().isoformat()
            }
            
            return final_response
            
        except Exception as e:
            print(f"[ERROR] Error processing query for client {client_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")

    def get_system_stats(self, client_id: str = None) -> Dict[str, Any]:
        """Get system statistics for specific client or all clients"""
        if client_id:
            # Single client stats
            client_config = self.get_client_config(client_id)
            vector_stats = self.vector_store.get_document_stats(client_id)
            
            return {
                "client_info": client_config["client_info"],
                "vector_store": vector_stats,
                "endpoints": {
                    name: {
                        "rag_percentage": config["rag_percentage"],
                        "creativity_level": config["creativity_level"],
                        "enable_visualization": config.get("enable_visualization", False)
                    }
                    for name, config in client_config["endpoints"].items()
                },
                "capabilities": {
                    "multimodal_input": ["text", "images", "audio"],
                    "multimodal_output": ["text", "tables", "charts", "images"],
                    "metadata_filtering": True,
                    "client_specific_branding": True
                }
            }
        else:
            # All clients stats
            all_stats = {}
            for cid in self.clients_config.keys():
                try:
                    all_stats[cid] = self.get_system_stats(cid)
                except Exception as e:
                    all_stats[cid] = {"error": str(e)}
            
            return {
                "system_info": {
                    "name": "Multi-Client RAG System",
                    "version": "1.0.0",
                    "supported_clients": list(self.clients_config.keys())
                },
                "clients": all_stats,
                "total_clients": len(self.clients_config)
            }

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Client RAG System",
    description="Advanced RAG system supporting multiple clients with dedicated indexes",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5183",
        "http://127.0.0.1:3000", 
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5183",
        "https://multi-client-rag-frontend.vercel.app",
        "*"  # Remove in production
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for deployment"""
    return {"status": "healthy", "service": "multi-client-rag-backend"}

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Multi-Client RAG System", "status": "running", "clients": ["tigo_honduras", "unilever", "nestle", "alpina"]}

# Initialize multi-client RAG system
def initialize_multi_client_rag_system():
    """Initialize multi-client RAG system with fallback"""
    try:
        required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_SEARCH_SERVICE", "AZURE_SEARCH_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            print(f"[WARNING] Missing environment variables: {missing_vars}")
            print("[INFO] Running in minimal mode without full RAG functionality")
            return None
            
        rag_system = MultiClientRAGSystem()
        print("[SUCCESS] Multi-client RAG system initialization successful")
        return rag_system
        
    except Exception as e:
        print(f"[ERROR] Multi-client RAG system initialization failed: {e}")
        print("[INFO] Running in minimal mode")
        return None

rag_system = initialize_multi_client_rag_system()

# Multi-client authentication
CLIENT_USERS = {
    "tigo_honduras": {
        "ejecutivo@tigo.com.hn": "TigoHN2024!",
        "marketing@tigo.com.hn": "Marketing2024!",
        "insights@tigo.com.hn": "Insights2024!",
        "juan@genius-labs.com.co": "GeniusLabs2024!"
    },
    "unilever": {
        "research@unilever.com": "Unilever2024!",
        "marketing@unilever.com": "ULMarketing2024!",
        "insights@unilever.com": "ULInsights2024!"
    },
    "nestle": {
        "research@nestle.com": "Nestle2024!",
        "marketing@nestle.com": "NEMarketing2024!",
        "insights@nestle.com": "NEInsights2024!"
    },
    "alpina": {
        "research@alpina.com": "Alpina2024!",
        "marketing@alpina.com": "ALMarketing2024!",
        "insights@alpina.com": "ALInsights2024!"
    }
}

@app.post("/api/auth/login", response_model=LoginResponse)
async def login_endpoint(request: LoginRequest):
    """Multi-client authentication endpoint"""
    try:
        client_users = CLIENT_USERS.get(request.client, {})
        
        if request.username in client_users and client_users[request.username] == request.password:
            import hashlib
            import time
            token_data = f"{request.username}:{request.client}:{time.time()}"
            token = hashlib.sha256(token_data.encode()).hexdigest()
            
            # Get client info
            if rag_system:
                try:
                    client_config = rag_system.get_client_config(request.client)
                    client_info = client_config["client_info"]
                except:
                    client_info = {"name": request.client.title(), "industry": "Unknown"}
            else:
                client_info = {"name": request.client.title(), "industry": "Unknown"}
            
            return LoginResponse(
                success=True,
                token=token,
                message=f"Login successful for {client_info['name']}",
                user={
                    "username": request.username,
                    "client": request.client,
                    "role": "admin" if "ejecutivo" in request.username or "research" in request.username else "user",
                    "permissions": ["chat", "export", "view_stats"]
                },
                client_info=client_info
            )
        else:
            return LoginResponse(
                success=False,
                message="Invalid credentials"
            )
    except Exception as e:
        print(f"Login error: {e}")
        return LoginResponse(
            success=False,
            message="Authentication error"
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "multi_client_system_initialized": rag_system is not None,
        "supported_clients": list(CLIENT_USERS.keys()) if rag_system else [],
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint with system information"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="Multi-client RAG system not initialized")
    
    return {
        "message": "Multi-Client RAG System",
        "status": "online",
        "supported_clients": list(rag_system.clients_config.keys()),
        "endpoints": {
            "/api/{client}/rag-pure": "100% retrieval-based responses",
            "/api/{client}/rag-creative": "Creative responses with visualizations",
            "/api/{client}/rag-hybrid": "Balanced RAG/LLM approach",
            "/api/{client}/stats": "Client-specific statistics"
        },
        "authentication_required": True,
        "version": "1.0.0"
    }

@app.get("/api/{client_id}/stats")
async def get_client_stats(client_id: str):
    """Get statistics for specific client"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        return rag_system.get_system_stats(client_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/api/stats")
async def get_all_stats():
    """Get statistics for all clients"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    return rag_system.get_system_stats()

# Client-specific RAG endpoints
@app.post("/api/{client_id}/rag-pure", response_model=RAGResponse)
async def client_rag_pure_endpoint(client_id: str, query: MultimodalQuery):
    """Pure RAG mode for specific client"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    return rag_system.process_multimodal_query(query, "pure", client_id)

@app.post("/api/{client_id}/rag-creative", response_model=RAGResponse)
async def client_rag_creative_endpoint(client_id: str, query: MultimodalQuery):
    """Creative RAG mode for specific client"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    return rag_system.process_multimodal_query(query, "creative", client_id)

@app.post("/api/{client_id}/rag-hybrid", response_model=RAGResponse)
async def client_rag_hybrid_endpoint(client_id: str, query: MultimodalQuery):
    """Hybrid RAG mode for specific client"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    return rag_system.process_multimodal_query(query, "hybrid", client_id)

# Frontend compatibility endpoint
@app.post("/api/{client_id}/chat")
async def client_chat_endpoint(client_id: str, request: Dict[str, Any]):
    """Client-specific chat endpoint compatible with frontend"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        messages = request.get("messages", [])
        mode = request.get("mode", "general")
        
        if not messages or not messages[-1].get("content"):
            raise HTTPException(status_code=400, detail="No message content provided")
        
        user_message = messages[-1]["content"]
        
        # Map frontend modes to backend modes
        backend_mode_mapping = {
            "general": "pure",
            "creative": "creative"
        }
        backend_mode = backend_mode_mapping.get(mode, "hybrid")
        
        # Create query for RAG system
        multimodal_query = MultimodalQuery(
            text=user_message,
            output_types=["text", "table", "chart"] if mode == "creative" else ["text"]
        )
        
        # Process with RAG system
        response = rag_system.process_multimodal_query(multimodal_query, backend_mode, client_id)
        
        # Format response for frontend
        return {
            "answer": response["answer"],
            "content": response["answer"],
            "citations": response.get("citations", []),
            "visualization": response.get("visualizations", {}),
            "suggestions": response.get("suggestions", {}).get("suggestions", []) if response.get("suggestions") else [],
            "metadata": response.get("metadata", {}),
            "timestamp": response.get("timestamp"),
            "mode": mode,
            "backend_mode_used": backend_mode,
            "client": client_id
        }
        
    except Exception as e:
        print(f"Error in client chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat for {client_id}: {str(e)}")

@app.post("/api/{client_id}/export-data")
async def client_export_data_endpoint(client_id: str, query: ExportQuery):
    """Export data for specific client"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Add client context to export
        query.rag_response["client_context"] = {
            "client_id": client_id,
            "client_name": rag_system.get_client_config(client_id)["client_info"]["name"]
        }
        
        export_result = rag_system.data_exporter.export_rag_response(
            rag_response=query.rag_response,
            format_type=query.format_type,
            include_metadata=query.include_metadata
        )
        
        if "error" in export_result:
            raise HTTPException(status_code=400, detail=export_result["error"])
        
        return export_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed for {client_id}: {str(e)}")

if __name__ == "__main__":
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Get port from environment or default
    port = int(os.getenv("PORT", 8000))
    
    # Run application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )