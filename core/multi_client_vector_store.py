"""
Multi-Client Azure Search Vector Store
Handles multiple Azure AI Search indexes for different clients
"""

import os
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import *
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
import hashlib

@dataclass
class Document:
    """Document class for vector store"""
    content: str
    metadata: Dict[str, Any]
    
class MultiClientAzureSearchVectorStore:
    """Multi-client Azure Search vector store manager"""
    
    def __init__(self, clients_config: Dict[str, Dict[str, Any]]):
        """Initialize multi-client vector store"""
        self.clients_config = clients_config
        self.search_clients = {}
        self.index_clients = {}
        
        # Initialize Azure Search clients for each client
        for client_id, config in clients_config.items():
            try:
                search_config = config["azure_search"]
                credential = AzureKeyCredential(search_config["api_key"])
                
                # Create search client
                self.search_clients[client_id] = SearchClient(
                    endpoint=search_config["endpoint"],
                    index_name=search_config["index_name"],
                    credential=credential
                )
                
                # Create index client
                self.index_clients[client_id] = SearchIndexClient(
                    endpoint=search_config["endpoint"],
                    credential=credential
                )
                
                print(f"[SUCCESS] {client_id}: Azure Search client initialized")
                
            except Exception as e:
                print(f"[ERROR] {client_id}: Failed to initialize Azure Search client: {e}")
                self.search_clients[client_id] = None
                self.index_clients[client_id] = None
    
    def get_client_search_client(self, client_id: str) -> SearchClient:
        """Get search client for specific client"""
        if client_id not in self.search_clients:
            raise ValueError(f"Unknown client: {client_id}")
        
        client = self.search_clients[client_id]
        if client is None:
            raise RuntimeError(f"Search client not initialized for {client_id}")
        
        return client
    
    def similarity_search(self, 
                         client_id: str,
                         query: str, 
                         k: int = 5, 
                         metadata_filter: Optional[Dict[str, Any]] = None,
                         min_similarity: float = 0.0) -> List[Tuple[Document, float]]:
        """Perform similarity search for specific client"""
        try:
            search_client = self.get_client_search_client(client_id)
            
            # Build search parameters
            search_params = {
                "search_text": query,
                "top": k,
                "include_total_count": True
            }
            
            # Add metadata filters if provided
            if metadata_filter:
                filter_parts = []
                for key, value in metadata_filter.items():
                    if isinstance(value, str):
                        filter_parts.append(f"{key} eq '{value}'")
                    elif isinstance(value, (int, float)):
                        filter_parts.append(f"{key} eq {value}")
                    elif isinstance(value, list):
                        # Handle list filters (OR condition)
                        list_filters = []
                        for v in value:
                            if isinstance(v, str):
                                list_filters.append(f"{key} eq '{v}'")
                            else:
                                list_filters.append(f"{key} eq {v}")
                        if list_filters:
                            filter_parts.append(f"({' or '.join(list_filters)})")
                
                if filter_parts:
                    search_params["filter"] = " and ".join(filter_parts)
            
            # Perform search
            results = search_client.search(**search_params)
            
            # Process results
            documents = []
            for result in results:
                try:
                    # Extract content and metadata
                    content = result.get("content", "")
                    
                    metadata = {
                        "document_name": result.get("document_name", "Unknown"),
                        "study_type": result.get("study_type", "Unknown"),
                        "year": result.get("year", "Unknown"),
                        "section_type": result.get("section_type", "Unknown"),
                        "client_id": client_id
                    }
                    
                    # Add additional metadata fields if present
                    for field in ["brand", "category", "product", "region"]:
                        if field in result:
                            metadata[field] = result[field]
                    
                    # Create document
                    doc = Document(content=content, metadata=metadata)
                    
                    # Calculate similarity score (Azure Search returns relevance score)
                    similarity = result.get("@search.score", 0.0)
                    
                    if similarity >= min_similarity:
                        documents.append((doc, similarity))
                        
                except Exception as e:
                    print(f"Error processing search result: {e}")
                    continue
            
            # Sort by similarity (descending)
            documents.sort(key=lambda x: x[1], reverse=True)
            
            print(f"🔍 {client_id}: Found {len(documents)} documents for query: '{query[:50]}...'")
            return documents[:k]
            
        except Exception as e:
            print(f"[ERROR] Error in similarity search for {client_id}: {e}")
            return []
    
    def add_document(self, client_id: str, content: str, metadata: Dict[str, Any]) -> str:
        """Add document to client's vector store"""
        try:
            search_client = self.get_client_search_client(client_id)
            
            # Generate document ID
            doc_id = hashlib.md5(f"{client_id}_{content[:100]}".encode()).hexdigest()
            
            # Prepare document for indexing
            document = {
                "id": doc_id,
                "content": content,
                "document_name": metadata.get("document_name", "Unknown"),
                "study_type": metadata.get("study_type", "Unknown"),
                "year": metadata.get("year", 2024),
                "section_type": metadata.get("section_type", "content"),
                "client_id": client_id
            }
            
            # Add optional metadata fields
            for field in ["brand", "category", "product", "region"]:
                if field in metadata:
                    document[field] = metadata[field]
            
            # Upload to Azure Search
            result = search_client.upload_documents([document])
            
            if result and len(result) > 0 and result[0].succeeded:
                print(f"[SUCCESS] {client_id}: Document added successfully: {doc_id}")
                return doc_id
            else:
                raise Exception("Failed to upload document to Azure Search")
                
        except Exception as e:
            print(f"[ERROR] Error adding document to {client_id}: {e}")
            raise
    
    def get_document_stats(self, client_id: str) -> Dict[str, Any]:
        """Get document statistics for client"""
        try:
            search_client = self.get_client_search_client(client_id)
            client_config = self.clients_config[client_id]
            
            # Get total document count
            results = search_client.search("*", include_total_count=True, top=1)
            total_docs = results.get_count() or 0
            
            # Get sample documents to analyze metadata
            sample_results = list(search_client.search("*", top=100))
            
            # Analyze document types and years
            study_types = {}
            years = {}
            brands = set()
            
            for doc in sample_results:
                # Study types
                study_type = doc.get("study_type", "Unknown")
                study_types[study_type] = study_types.get(study_type, 0) + 1
                
                # Years
                year = doc.get("year", "Unknown")
                years[year] = years.get(year, 0) + 1
                
                # Brands
                if "brand" in doc and doc["brand"]:
                    brands.add(doc["brand"])
            
            return {
                "total_documents": total_docs,
                "index_name": client_config["azure_search"]["index_name"],
                "client_name": client_config["client_info"]["name"],
                "study_types": study_types,
                "years_distribution": years,
                "unique_brands": len(brands),
                "sample_brands": list(brands)[:10],
                "last_updated": "Real-time Azure Search"
            }
            
        except Exception as e:
            print(f"[ERROR] Error getting stats for {client_id}: {e}")
            return {
                "total_documents": 0,
                "error": str(e)
            }
    
    def search_with_filters(self, 
                           client_id: str,
                           query: str = "*",
                           filters: Optional[Dict[str, Any]] = None,
                           top: int = 10) -> List[Dict[str, Any]]:
        """Advanced search with custom filters"""
        try:
            search_client = self.get_client_search_client(client_id)
            
            search_params = {
                "search_text": query,
                "top": top,
                "include_total_count": True
            }
            
            # Apply filters
            if filters:
                filter_parts = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        filter_parts.append(f"{key} eq '{value}'")
                    elif isinstance(value, (int, float)):
                        filter_parts.append(f"{key} eq {value}")
                    elif isinstance(value, dict):
                        # Handle range filters
                        if "gte" in value:
                            filter_parts.append(f"{key} ge {value['gte']}")
                        if "lte" in value:
                            filter_parts.append(f"{key} le {value['lte']}")
                
                if filter_parts:
                    search_params["filter"] = " and ".join(filter_parts)
            
            results = search_client.search(**search_params)
            
            documents = []
            for result in results:
                documents.append({
                    "content": result.get("content", "")[:200] + "...",
                    "document_name": result.get("document_name", "Unknown"),
                    "study_type": result.get("study_type", "Unknown"),
                    "year": result.get("year", "Unknown"),
                    "score": result.get("@search.score", 0.0)
                })
            
            return documents
            
        except Exception as e:
            print(f"[ERROR] Error in advanced search for {client_id}: {e}")
            return []
    
    def health_check(self, client_id: str) -> Dict[str, Any]:
        """Check health of client's search service"""
        try:
            search_client = self.get_client_search_client(client_id)
            
            # Simple search to test connectivity
            results = search_client.search("*", top=1)
            list(results)  # Force execution
            
            return {
                "status": "healthy",
                "client_id": client_id,
                "index_name": self.clients_config[client_id]["azure_search"]["index_name"],
                "message": "Search service is accessible"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "client_id": client_id,
                "error": str(e)
            }