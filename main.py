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

# Import Personas System
from personas.persona_system import ComprehensivePersonaSystem

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

# Persona System Models
class PersonaChatQuery(BaseModel):
    """Persona chat query"""
    message: str
    session_id: Optional[str] = None
    persona_id: Optional[str] = None

class PersonaSurveyQuery(BaseModel):
    """Mass survey query"""
    questions: List[str]
    persona_ids: Optional[List[str]] = None
    max_personas: Optional[int] = 20

class PersonaFocusGroupQuery(BaseModel):
    """Focus group query"""
    topic: str
    persona_ids: Optional[List[str]] = None
    group_size: Optional[int] = 8

class PersonaValidationQuery(BaseModel):
    """Persona validation query"""
    persona_count: Optional[int] = 50
    diversity_target: Optional[float] = 0.8
    quality_threshold: Optional[float] = 0.7

class EnhancedPersonaGenerationQuery(BaseModel):
    """Enhanced persona generation with advanced methodologies"""
    persona_count: Optional[int] = 50
    study_level: Optional[str] = "exploratory_study"  # pilot_study, exploratory_study, sensitivity_analysis
    use_implicit_demographics: Optional[bool] = True
    include_temporal_context: Optional[bool] = True
    generate_interview_transcripts: Optional[bool] = False

class EnhancedChatQuery(BaseModel):
    """Enhanced chat with advanced methodologies"""
    message: str
    session_id: str
    use_temperature_optimization: Optional[bool] = True
    use_temporal_context: Optional[bool] = True

class StudyValidationQuery(BaseModel):
    """Study readiness validation query"""
    study_level: str  # pilot_study, exploratory_study, sensitivity_analysis
    persona_ids: Optional[List[str]] = None

class SyntheticChatQuery(BaseModel):
    """Synthetic archetype chat query"""
    message: str
    archetype: str  # consumer, business, expert, etc.
    context: Optional[Dict[str, Any]] = None

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
        
        # Initialize Personas System
        try:
            self.persona_system = ComprehensivePersonaSystem()
            print("[SUCCESS] Personas System initialized")
        except Exception as e:
            print(f"[WARNING] Personas System initialization failed: {e}")
            self.persona_system = None
        
        print("[STARTUP] Multi-Client RAG System initialized")
        print(f"[INFO] Supported clients: {', '.join(self.clients_config.keys())}")
        print(f"[INFO] Vector stores configured for all clients")
        print(f"[INFO] Azure AI Search indexes: {len(self.clients_config)} indexes")

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

# ===== PERSONA SYSTEM ENDPOINTS =====
# Multi-client synthetic persona generation and interaction endpoints
# Each client has isolated persona environments

@app.post("/api/{client_id}/persona-chat")
async def client_persona_chat_endpoint(client_id: str, query: PersonaChatQuery):
    """
    1:1 Conversation with Synthetic Persona (Multi-Client)
    Start or continue conversation with client-specific persona
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not rag_system.persona_system:
        raise HTTPException(status_code=503, detail="Persona system not available")
    
    try:
        # Validate client
        client_config = rag_system.get_client_config(client_id)
        
        # If no session_id provided, start new conversation
        if not query.session_id:
            if not query.persona_id:
                # Generate a new persona if none specified
                persona_result = rag_system.persona_system.generate_validated_personas(
                    count=1, diversity_target=0.8, quality_threshold=0.7
                )
                
                if not persona_result["success"]:
                    raise HTTPException(status_code=400, detail="Failed to generate persona")
                
                persona_id = persona_result["personas"][0]["id"]
            else:
                persona_id = query.persona_id
            
            # Start new conversation with client context
            session_id = await rag_system.persona_system.start_persona_conversation(
                persona_id, "chat", {
                    "user_initiated": True, 
                    "client_id": client_id,
                    "client_context": client_config
                }
            )
            
            return {
                "message": "Conversation started",
                "session_id": session_id,
                "persona_id": persona_id,
                "client_id": client_id,
                "persona_profile": {
                    "age": rag_system.persona_system.generated_personas[persona_id]["characteristics"]["age"],
                    "gender": rag_system.persona_system.generated_personas[persona_id]["characteristics"]["gender"],
                    "location": rag_system.persona_system.generated_personas[persona_id]["characteristics"]["geographic_region"],
                    "service_type": rag_system.persona_system.generated_personas[persona_id]["characteristics"]["service_type"]
                },
                "instructions": "Send your message using this session_id to continue the conversation"
            }
        
        else:
            # Continue existing conversation
            response = await rag_system.persona_system.send_message_to_persona(
                query.session_id, query.message
            )
            
            return {
                "client_id": client_id,
                "session_id": query.session_id,
                "persona_response": response["response"],
                "message_count": response["message_count"],
                "validation": response["validation"],
                "rag_context_used": response["rag_context_used"],
                "persona_consistency": response["persona_consistency"],
                "timestamp": datetime.now().isoformat()
            }
            
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in persona chat for {client_id}: {str(e)}")

@app.post("/api/{client_id}/persona-survey")
async def client_persona_survey_endpoint(client_id: str, query: PersonaSurveyQuery):
    """
    Mass Survey with Multiple Personas (Multi-Client)
    Conduct survey with multiple synthetic personas for specific client
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
        
    if not rag_system.persona_system:
        raise HTTPException(status_code=503, detail="Persona system not available")
    
    try:
        # Validate client
        client_config = rag_system.get_client_config(client_id)
        
        # Generate or use existing personas
        if not query.persona_ids:
            # Generate new personas for this survey
            persona_result = rag_system.persona_system.generate_validated_personas(
                count=query.max_personas or 20,
                diversity_target=0.8,
                quality_threshold=0.7
            )
            
            if not persona_result["success"]:
                raise HTTPException(status_code=400, detail="Failed to generate personas for survey")
            
            persona_ids = [p["id"] for p in persona_result["personas"]]
        else:
            persona_ids = query.persona_ids
        
        # Conduct mass survey
        survey_result = await rag_system.persona_system.conduct_mass_survey(
            persona_ids, query.questions, {
                "client_id": client_id,
                "client_context": client_config
            }
        )
        
        return {
            "client_id": client_id,
            "survey_id": survey_result["survey_id"],
            "total_personas": len(persona_ids),
            "questions_asked": len(query.questions),
            "responses": survey_result["responses"],
            "summary_insights": survey_result["insights"],
            "completion_rate": survey_result["completion_rate"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Survey failed for {client_id}: {str(e)}")

@app.post("/api/{client_id}/persona-focus-group")
async def client_persona_focus_group_endpoint(client_id: str, query: PersonaFocusGroupQuery):
    """
    Simulated Focus Group Discussion (Multi-Client)
    Simulate focus group with diverse synthetic personas for specific client
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
        
    if not rag_system.persona_system:
        raise HTTPException(status_code=503, detail="Persona system not available")
    
    try:
        # Validate client
        client_config = rag_system.get_client_config(client_id)
        
        # Generate or use existing personas
        if not query.persona_ids:
            # Generate diverse personas for focus group
            persona_result = rag_system.persona_system.generate_validated_personas(
                count=query.group_size or 8,
                diversity_target=0.9,  # Higher diversity for focus groups
                quality_threshold=0.8
            )
            
            if not persona_result["success"]:
                raise HTTPException(status_code=400, detail="Failed to generate personas for focus group")
            
            persona_ids = [p["id"] for p in persona_result["personas"]]
        else:
            persona_ids = query.persona_ids[:query.group_size or 8]
        
        # Conduct focus group
        focus_group_result = await rag_system.persona_system.conduct_focus_group(
            persona_ids, query.topic, {
                "client_id": client_id,
                "client_context": client_config,
                "session_type": "focus_group"
            }
        )
        
        return {
            "client_id": client_id,
            "focus_group_id": focus_group_result["session_id"],
            "topic": query.topic,
            "participants": len(persona_ids),
            "discussion_rounds": focus_group_result["rounds"],
            "key_insights": focus_group_result["insights"],
            "consensus_points": focus_group_result["consensus"],
            "diverse_opinions": focus_group_result["diversity_metrics"],
            "transcript": focus_group_result["transcript"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Focus group failed for {client_id}: {str(e)}")

@app.post("/api/{client_id}/persona-validate")
async def client_persona_validate_endpoint(client_id: str, query: PersonaValidationQuery):
    """
    Generate and Validate Persona Batches (Multi-Client)
    Generate new personas with comprehensive bias detection and validation for specific client
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
        
    if not rag_system.persona_system:
        raise HTTPException(status_code=503, detail="Persona system not available")
    
    try:
        # Validate client
        client_config = rag_system.get_client_config(client_id)
        
        # Generate and validate personas
        validation_result = rag_system.persona_system.generate_validated_personas(
            count=query.persona_count or 50,
            diversity_target=query.diversity_target or 0.8,
            quality_threshold=query.quality_threshold or 0.7,
            client_context={
                "client_id": client_id,
                "client_config": client_config
            }
        )
        
        return {
            "client_id": client_id,
            "validation_successful": validation_result["success"],
            "total_generated": validation_result["total_generated"],
            "total_validated": validation_result["total_validated"],
            "validation_rate": validation_result["validation_rate"],
            "diversity_achieved": validation_result["diversity_metrics"]["overall_diversity"],
            "quality_metrics": validation_result["quality_metrics"],
            "bias_detection": validation_result["bias_analysis"],
            "personas": validation_result["personas"],
            "recommendations": validation_result["recommendations"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Persona validation failed for {client_id}: {str(e)}")

@app.get("/api/{client_id}/persona-export")
async def client_persona_export_endpoint(
    client_id: str,
    format: str = "json",
    persona_ids: Optional[str] = None
):
    """
    Export Generated Personas (Multi-Client)
    Export personas in various formats for specific client
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
        
    if not rag_system.persona_system:
        raise HTTPException(status_code=503, detail="Persona system not available")
    
    try:
        # Validate client
        client_config = rag_system.get_client_config(client_id)
        
        # Parse persona IDs if provided
        if persona_ids:
            persona_id_list = persona_ids.split(",")
        else:
            # Export all personas for this client (if we implement client-specific storage)
            persona_id_list = list(rag_system.persona_system.generated_personas.keys())
        
        # Export personas
        export_result = rag_system.persona_system.export_personas(
            persona_id_list, 
            format=format,
            client_context={
                "client_id": client_id,
                "client_config": client_config
            }
        )
        
        return {
            "client_id": client_id,
            "export_format": format,
            "total_exported": len(persona_id_list),
            "export_data": export_result["data"],
            "metadata": export_result["metadata"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Persona export failed for {client_id}: {str(e)}")

@app.post("/api/{client_id}/persona-enhanced-generate")
async def client_enhanced_persona_generation_endpoint(client_id: str, query: EnhancedPersonaGenerationQuery):
    """
    Enhanced Persona Generation with Advanced Methodologies (Multi-Client)
    Uses context-rich prompting, temperature optimization, implicit demographics,
    temporal context, and staged validation for specific client
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
        
    if not rag_system.persona_system:
        raise HTTPException(status_code=503, detail="Persona system not available")
    
    try:
        # Validate client
        client_config = rag_system.get_client_config(client_id)
        
        # Enhanced persona generation with all methodologies
        generation_result = await rag_system.persona_system.enhanced_persona_generation(
            count=query.persona_count or 50,
            study_level=query.study_level or "exploratory_study",
            use_implicit_demographics=query.use_implicit_demographics,
            include_temporal_context=query.include_temporal_context,
            generate_interview_transcripts=query.generate_interview_transcripts,
            client_context={
                "client_id": client_id,
                "client_config": client_config
            }
        )
        
        return {
            "client_id": client_id,
            "generation_successful": generation_result["success"],
            "study_level": query.study_level,
            "methodology_status": {
                "context_rich_prompting": generation_result["methodology_status"]["context_rich_prompting"],
                "temperature_optimization": generation_result["methodology_status"]["temperature_optimization"],
                "implicit_demographics": generation_result["methodology_status"]["implicit_demographics"],
                "temporal_context": generation_result["methodology_status"]["temporal_context"],
                "staged_validation": generation_result["methodology_status"]["staged_validation"]
            },
            "personas_generated": generation_result["total_generated"],
            "quality_metrics": generation_result["quality_metrics"],
            "diversity_analysis": generation_result["diversity_analysis"],
            "validation_summary": generation_result["validation_summary"],
            "personas": generation_result["personas"],
            "interview_transcripts": generation_result.get("interview_transcripts", []),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced persona generation failed for {client_id}: {str(e)}")

@app.post("/api/{client_id}/synthetic-chat")
async def client_synthetic_archetype_chat(client_id: str, query: SyntheticChatQuery):
    """
    Synthetic Archetype Chat (Multi-Client)
    Chat with pre-defined synthetic archetypes (consumer, business, expert) for specific client
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
        
    if not rag_system.persona_system:
        raise HTTPException(status_code=503, detail="Persona system not available")
    
    try:
        # Validate client
        client_config = rag_system.get_client_config(client_id)
        
        # Chat with synthetic archetype
        chat_result = await rag_system.persona_system.synthetic_archetype_chat(
            archetype=query.archetype,
            message=query.message,
            context={
                "client_id": client_id,
                "client_config": client_config,
                **(query.context or {})
            }
        )
        
        return {
            "client_id": client_id,
            "archetype": query.archetype,
            "user_message": query.message,
            "archetype_response": chat_result["response"],
            "archetype_profile": chat_result["profile"],
            "confidence_score": chat_result["confidence"],
            "reasoning": chat_result["reasoning"],
            "rag_context_used": chat_result["rag_context_used"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthetic chat failed for {client_id}: {str(e)}")

# ===== DOCUMENT MANAGEMENT ENDPOINTS =====

@app.post("/api/{client_id}/add-document")
async def client_add_document_endpoint(
    client_id: str,
    document_text: str = Form(...),
    title: str = Form(...),
    source: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None)
):
    """
    Add Document to Client-Specific Index
    Upload and index document content for specific client
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Validate client
        client_config = rag_system.get_client_config(client_id)
        
        # Parse metadata if provided
        doc_metadata = {}
        if metadata:
            try:
                doc_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                doc_metadata = {"raw_metadata": metadata}
        
        # Add client context to metadata
        doc_metadata.update({
            "client_id": client_id,
            "client_name": client_config["client_info"]["name"],
            "upload_timestamp": datetime.now().isoformat(),
            "title": title,
            "source": source or "manual_upload"
        })
        
        # Add document to client-specific vector store
        result = rag_system.vector_store.add_document(
            client_id=client_id,
            content=document_text,
            metadata=doc_metadata
        )
        
        return {
            "client_id": client_id,
            "success": True,
            "document_id": result["document_id"],
            "title": title,
            "content_length": len(document_text),
            "metadata": doc_metadata,
            "indexed": result["indexed"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document upload failed for {client_id}: {str(e)}")

@app.post("/api/{client_id}/upload-file")
async def client_upload_file_endpoint(
    client_id: str,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    source: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None)
):
    """
    Upload File to Client-Specific Index
    Upload and process file (PDF, DOCX, TXT) for specific client
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Validate client
        client_config = rag_system.get_client_config(client_id)
        
        # Validate file type
        allowed_types = {".txt", ".pdf", ".docx", ".md"}
        file_extension = os.path.splitext(file.filename.lower())[1] if file.filename else ""
        
        if file_extension not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file_extension} not supported. Allowed: {allowed_types}"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Process file based on type
        if file_extension == ".txt" or file_extension == ".md":
            document_text = file_content.decode('utf-8')
        elif file_extension == ".pdf":
            # Would need PDF processing library
            raise HTTPException(status_code=501, detail="PDF processing not implemented yet")
        elif file_extension == ".docx":
            # Would need DOCX processing library
            raise HTTPException(status_code=501, detail="DOCX processing not implemented yet")
        
        # Parse metadata
        doc_metadata = {}
        if metadata:
            try:
                doc_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                doc_metadata = {"raw_metadata": metadata}
        
        # Add file and client context
        doc_metadata.update({
            "client_id": client_id,
            "client_name": client_config["client_info"]["name"],
            "filename": file.filename,
            "file_size": len(file_content),
            "file_type": file_extension,
            "upload_timestamp": datetime.now().isoformat(),
            "title": title or file.filename,
            "source": source or "file_upload"
        })
        
        # Add document to client-specific vector store
        result = rag_system.vector_store.add_document(
            client_id=client_id,
            content=document_text,
            metadata=doc_metadata
        )
        
        return {
            "client_id": client_id,
            "success": True,
            "document_id": result["document_id"],
            "filename": file.filename,
            "file_size": len(file_content),
            "content_length": len(document_text),
            "metadata": doc_metadata,
            "indexed": result["indexed"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed for {client_id}: {str(e)}")

# ===== IMAGE GENERATION ENDPOINT =====

@app.post("/api/{client_id}/generate-image")
async def client_generate_image_endpoint(client_id: str, request: dict):
    """
    Generate Image for Client
    Generate images using DALL-E with client-specific context
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Validate client
        client_config = rag_system.get_client_config(client_id)
        
        prompt = request.get("prompt", "")
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        # Add client context to prompt if needed
        client_context = f"For {client_config['client_info']['name']} ({client_config['client_info']['industry']}): "
        contextual_prompt = client_context + prompt
        
        # Generate image using multimodal output generator
        if rag_system.output_generator:
            image_result = await rag_system.output_generator.generate_images(
                prompt=contextual_prompt,
                client_id=client_id,
                count=request.get("count", 1),
                size=request.get("size", "1024x1024"),
                style=request.get("style", "natural")
            )
            
            return {
                "client_id": client_id,
                "success": True,
                "prompt": prompt,
                "contextual_prompt": contextual_prompt,
                "images": image_result["images"],
                "count": len(image_result["images"]),
                "metadata": {
                    "client_name": client_config["client_info"]["name"],
                    "industry": client_config["client_info"]["industry"],
                    "size": request.get("size", "1024x1024"),
                    "style": request.get("style", "natural")
                },
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=503, detail="Image generation not available")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed for {client_id}: {str(e)}")

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