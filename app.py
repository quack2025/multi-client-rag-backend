# app.py - Railway entry point for Multi-Client RAG System
"""
Simplified Railway entry point for Multi-Client RAG System
Supports: Tigo Honduras, Unilever, Nestlé, Alpina
"""

import os
import sys

# Set environment variables for Railway
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

try:
    # Import the FastAPI app
    print("[STARTUP] Attempting to import multi-client main app...")
    from main import app
    print("[SUCCESS] Successfully imported multi-client app - FULL RAG SYSTEM ACTIVE")
    print("[INFO] Supported clients: Tigo Honduras, Unilever, Nestle, Alpina")
except Exception as e:
    print(f"[ERROR] Error importing main: {e}")
    import traceback
    traceback.print_exc()
    # Create a minimal FastAPI app as fallback
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    
    app = FastAPI(title="Multi-Client RAG Backend - Minimal Mode")
    
    # Add CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def health_check():
        return {
            "status": "ok", 
            "message": "Multi-Client RAG Backend is running (minimal mode)",
            "clients": ["tigo_honduras", "unilever", "nestle", "alpina"],
            "mode": "minimal"
        }
    
    @app.get("/health")
    async def railway_health():
        return {"status": "healthy", "service": "multi-client-rag-backend"}
    
    @app.get("/debug")
    async def debug_info():
        return {
            "python_version": sys.version,
            "environment_vars": list(os.environ.keys()),
            "working_directory": os.getcwd(),
            "mode": "minimal_fallback"
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0"
    
    print(f"[STARTUP] Starting Multi-Client RAG System on {host}:{port}")
    print("[INFO] Clients: Tigo Honduras, Unilever, Nestle, Alpina")
    
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")