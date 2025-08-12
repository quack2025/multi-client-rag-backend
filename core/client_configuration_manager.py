"""
Client Configuration Manager
Manages configurations for multiple clients (Tigo, Unilever, Nestlé, Alpina)
"""

import os
import json
from typing import Dict, Any
from pathlib import Path

class ClientConfigurationManager:
    """Manages client-specific configurations"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    
    def load_all_client_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load configurations for all supported clients"""
        clients_config = {}
        
        # Client configurations
        client_configs = {
            "tigo_honduras": self._get_tigo_config(),
            "unilever": self._get_unilever_config(),
            "nestle": self._get_nestle_config(),
            "alpina": self._get_alpina_config()
        }
        
        for client_id, config in client_configs.items():
            try:
                # Try to load from file first
                config_file = self.config_dir / f"{client_id}_config.json"
                if config_file.exists():
                    with open(config_file, 'r', encoding='utf-8') as f:
                        clients_config[client_id] = json.load(f)
                    print(f"[SUCCESS] {client_id} config loaded from file")
                else:
                    # Use default configuration
                    clients_config[client_id] = config
                    # Save default config to file
                    self._save_client_config(client_id, config)
                    print(f"[SUCCESS] {client_id} config created with defaults")
                    
            except Exception as e:
                print(f"[WARNING] Error loading {client_id} config: {e}, using defaults")
                clients_config[client_id] = config
        
        return clients_config
    
    def _save_client_config(self, client_id: str, config: Dict[str, Any]):
        """Save client configuration to file"""
        config_file = self.config_dir / f"{client_id}_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def _get_tigo_config(self) -> Dict[str, Any]:
        """Get Tigo Honduras configuration"""
        return {
            "client_info": {
                "name": "Tigo Honduras",
                "industry": "telecommunications",
                "market": "honduras",
                "language": "spanish_honduras",
                "brand_colors": ["#0066CC", "#FFFFFF"],
                "logo_url": "https://tigo.com.hn/logo.png"
            },
            "azure_openai": {
                "endpoint": "https://insightgenius-rag-v1-resource.cognitiveservices.azure.com/",
                "api_key": os.getenv("AZURE_OPENAI_API_KEY", ""),
                "api_version": "2024-12-01-preview",
                "embedding_deployment": "text-embedding-3-large",
                "chat_deployment": "gpt-4.1",
                "dalle_deployment": "dall-e-3",
                "max_tokens": 2000,
                "timeout": 60
            },
            "azure_search": {
                "endpoint": "https://insightgenius-search.search.windows.net",
                "api_key": os.getenv("AZURE_SEARCH_KEY", ""),
                "index_name": "tigo-insights",
                "semantic_config": "semantic-config"
            },
            "endpoints": {
                "rag_pure": {
                    "name": "Pure RAG Mode",
                    "rag_percentage": 100,
                    "creativity_level": 0.0,
                    "enable_visualization": False,
                    "max_context_chunks": 5
                },
                "rag_creative": {
                    "name": "Creative RAG Mode", 
                    "rag_percentage": 60,
                    "creativity_level": 0.7,
                    "enable_visualization": True,
                    "max_context_chunks": 3
                },
                "rag_hybrid": {
                    "name": "Hybrid RAG Mode",
                    "rag_percentage": 80,
                    "creativity_level": 0.3,
                    "enable_visualization": True,
                    "max_context_chunks": 4
                }
            }
        }
    
    def _get_unilever_config(self) -> Dict[str, Any]:
        """Get Unilever configuration"""
        return {
            "client_info": {
                "name": "Unilever",
                "industry": "consumer_goods",
                "market": "colombia",
                "language": "spanish_colombia",
                "brand_colors": ["#003366", "#0099CC"],
                "logo_url": "https://unilever.com/logo.png"
            },
            "azure_openai": {
                "endpoint": "https://insightgenius-rag-v1-resource.cognitiveservices.azure.com/",
                "api_key": os.getenv("AZURE_OPENAI_API_KEY", ""),
                "api_version": "2024-12-01-preview",
                "embedding_deployment": "text-embedding-3-large",
                "chat_deployment": "gpt-4.1",
                "dalle_deployment": "dall-e-3",
                "max_tokens": 2000,
                "timeout": 60
            },
            "azure_search": {
                "endpoint": "https://insightgenius-search.search.windows.net",
                "api_key": os.getenv("AZURE_SEARCH_KEY", ""),
                "index_name": "unilever-documents-complete",
                "semantic_config": "semantic-config"
            },
            "endpoints": {
                "rag_pure": {
                    "name": "Pure RAG Mode",
                    "rag_percentage": 100,
                    "creativity_level": 0.0,
                    "enable_visualization": False,
                    "max_context_chunks": 5
                },
                "rag_creative": {
                    "name": "Creative RAG Mode",
                    "rag_percentage": 65,
                    "creativity_level": 0.6,
                    "enable_visualization": True,
                    "max_context_chunks": 4
                },
                "rag_hybrid": {
                    "name": "Hybrid RAG Mode", 
                    "rag_percentage": 75,
                    "creativity_level": 0.4,
                    "enable_visualization": True,
                    "max_context_chunks": 4
                }
            }
        }
    
    def _get_nestle_config(self) -> Dict[str, Any]:
        """Get Nestlé configuration"""
        return {
            "client_info": {
                "name": "Nestlé",
                "industry": "food_beverages",
                "market": "colombia",
                "language": "spanish_colombia",
                "brand_colors": ["#DC143C", "#FFFFFF"],
                "logo_url": "https://nestle.com/logo.png"
            },
            "azure_openai": {
                "endpoint": "https://insightgenius-rag-v1-resource.cognitiveservices.azure.com/",
                "api_key": os.getenv("AZURE_OPENAI_API_KEY", ""),
                "api_version": "2024-12-01-preview",
                "embedding_deployment": "text-embedding-3-large",
                "chat_deployment": "gpt-4.1",
                "dalle_deployment": "dall-e-3",
                "max_tokens": 2000,
                "timeout": 60
            },
            "azure_search": {
                "endpoint": "https://insightgenius-search.search.windows.net",
                "api_key": os.getenv("AZURE_SEARCH_KEY", ""),
                "index_name": "nestle-documents-complete",
                "semantic_config": "semantic-config"
            },
            "endpoints": {
                "rag_pure": {
                    "name": "Pure RAG Mode",
                    "rag_percentage": 100,
                    "creativity_level": 0.0,
                    "enable_visualization": False,
                    "max_context_chunks": 5
                },
                "rag_creative": {
                    "name": "Creative RAG Mode",
                    "rag_percentage": 70,
                    "creativity_level": 0.5,
                    "enable_visualization": True,
                    "max_context_chunks": 3
                },
                "rag_hybrid": {
                    "name": "Hybrid RAG Mode",
                    "rag_percentage": 85,
                    "creativity_level": 0.3,
                    "enable_visualization": True,
                    "max_context_chunks": 4
                }
            }
        }
    
    def _get_alpina_config(self) -> Dict[str, Any]:
        """Get Alpina configuration"""
        return {
            "client_info": {
                "name": "Alpina",
                "industry": "dairy_food",
                "market": "colombia",
                "language": "spanish_colombia",
                "brand_colors": ["#FF6600", "#FFFFFF"],
                "logo_url": "https://alpina.com/logo.png"
            },
            "azure_openai": {
                "endpoint": "https://insightgenius-rag-v1-resource.cognitiveservices.azure.com/",
                "api_key": os.getenv("AZURE_OPENAI_API_KEY", ""),
                "api_version": "2024-12-01-preview",
                "embedding_deployment": "text-embedding-3-large",
                "chat_deployment": "gpt-4.1",
                "dalle_deployment": "dall-e-3",
                "max_tokens": 2000,
                "timeout": 60
            },
            "azure_search": {
                "endpoint": "https://insightgenius-search.search.windows.net",
                "api_key": os.getenv("AZURE_SEARCH_KEY", ""),
                "index_name": "alpina-documents-complete",
                "semantic_config": "semantic-config"
            },
            "endpoints": {
                "rag_pure": {
                    "name": "Pure RAG Mode",
                    "rag_percentage": 100,
                    "creativity_level": 0.0,
                    "enable_visualization": False,
                    "max_context_chunks": 5
                },
                "rag_creative": {
                    "name": "Creative RAG Mode",
                    "rag_percentage": 65,
                    "creativity_level": 0.6,
                    "enable_visualization": True,
                    "max_context_chunks": 3
                },
                "rag_hybrid": {
                    "name": "Hybrid RAG Mode",
                    "rag_percentage": 80,
                    "creativity_level": 0.35,
                    "enable_visualization": True,
                    "max_context_chunks": 4
                }
            }
        }
    
    def update_client_config(self, client_id: str, updates: Dict[str, Any]) -> bool:
        """Update client configuration"""
        try:
            config_file = self.config_dir / f"{client_id}_config.json"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Apply updates recursively
                self._deep_update(config, updates)
                
                # Save updated config
                self._save_client_config(client_id, config)
                return True
            return False
        except Exception as e:
            print(f"Error updating {client_id} config: {e}")
            return False
    
    def _deep_update(self, base_dict: Dict[str, Any], updates: Dict[str, Any]):
        """Recursively update nested dictionary"""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value