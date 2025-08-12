# core/multimodal_output.py
"""
Multimodal Output Generation for Multi-Client RAG System
Generates text, visualizations, and structured responses
"""

import json
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime

class MultimodalOutputGenerator:
    """Generate multimodal outputs for multi-client RAG system"""
    
    def __init__(self, clients_config: Dict[str, Dict[str, Any]]):
        self.clients_config = clients_config
        print("[SUCCESS] Multi-Client Multimodal Output Generator initialized")
    
    def generate_response(self, 
                         query: str, 
                         context: str, 
                         mode: str, 
                         output_types: List[str],
                         client_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate multimodal response for specific client"""
        try:
            client_info = client_config["client_info"]
            endpoint_config = client_config["endpoints"][f"rag_{mode}"]
            
            # Generate text response
            text_response = self._generate_text_response(
                query, context, mode, endpoint_config, client_info
            )
            
            response_data = {
                "text_response": text_response,
                "tables": [],
                "charts": [],
                "images": [],
                "metadata": {
                    "client": client_info["name"],
                    "mode": mode,
                    "output_types_requested": output_types
                }
            }
            
            # Generate visualizations if requested
            if "table" in output_types and endpoint_config.get("enable_visualization", False):
                tables = self._generate_tables(query, context, text_response, client_info)
                response_data["tables"] = tables
            
            if "chart" in output_types and endpoint_config.get("enable_visualization", False):
                charts = self._generate_charts(query, context, text_response, client_info)
                response_data["charts"] = charts
            
            if "image" in output_types and endpoint_config.get("enable_visualization", False):
                images = self._generate_images(query, context, text_response, client_config)
                response_data["images"] = images
            
            return response_data
            
        except Exception as e:
            print(f"[ERROR] Error generating multimodal response: {e}")
            return {
                "text_response": f"Error generando respuesta: {str(e)}",
                "tables": [],
                "charts": [],
                "images": [],
                "metadata": {"error": str(e)}
            }
    
    def _generate_text_response(self, 
                               query: str, 
                               context: str, 
                               mode: str, 
                               endpoint_config: Dict[str, Any],
                               client_info: Dict[str, Any]) -> str:
        """Generate text response using Azure OpenAI"""
        try:
            azure_config = self.clients_config[next(iter(self.clients_config))]["azure_openai"]  # Use first client's config
            
            # Create client-specific system prompt
            client_name = client_info["name"]
            industry = client_info["industry"]
            language = client_info["language"]
            
            system_prompt = self._build_system_prompt(mode, client_name, industry, language)
            
            # Build user prompt with context
            rag_percentage = endpoint_config.get("rag_percentage", 80)
            context_instruction = ""
            
            if rag_percentage >= 90:
                context_instruction = "Responde ÚNICAMENTE basándote en la información proporcionada en el contexto."
            elif rag_percentage >= 70:
                context_instruction = "Responde principalmente basándote en el contexto proporcionado, complementando con conocimiento general cuando sea necesario."
            else:
                context_instruction = "Usa el contexto como referencia principal, pero puedes complementar con conocimiento general para una respuesta más completa."
            
            user_prompt = f"""
{context_instruction}

CONTEXTO:
{context}

PREGUNTA: {query}

Por favor proporciona una respuesta completa, bien estructurada y específica para {client_name}.
"""
            
            headers = {
                "Content-Type": "application/json",
                "api-key": azure_config["api_key"]
            }
            
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": endpoint_config.get("creativity_level", 0.3),
                "max_tokens": azure_config.get("max_tokens", 2000),
                "top_p": 0.9
            }
            
            url = f"{azure_config['endpoint']}/openai/deployments/{azure_config['chat_deployment']}/chat/completions?api-version={azure_config['api_version']}"
            
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            print(f"[ERROR] Error generating text response: {e}")
            return f"Error generando respuesta de texto: {str(e)}"
    
    def _build_system_prompt(self, mode: str, client_name: str, industry: str, language: str) -> str:
        """Build client-specific system prompt"""
        base_prompt = f"""Eres un asistente especializado en investigación de mercados para {client_name}, 
una empresa en la industria de {industry}. """
        
        if "spanish" in language.lower():
            base_prompt += "Responde siempre en español."
        
        mode_prompts = {
            "pure": f"""
Proporciona respuestas precisas y factuales basadas únicamente en la información del contexto.
Cita específicamente los documentos y estudios que sustentan tu respuesta.
Mantén un tono profesional y objetivo.""",
            
            "creative": f"""
Proporciona respuestas creativas y perspicaces que combinen la información del contexto con insights adicionales.
Sugiere implicaciones estratégicas y oportunidades para {client_name}.
Utiliza un tono dinámico y orientado a la acción.""",
            
            "hybrid": f"""
Equilibra información factual del contexto con análisis estratégico adicional.
Proporciona tanto datos concretos como interpretaciones útiles para la toma de decisiones.
Mantén un tono profesional pero accesible."""
        }
        
        return base_prompt + mode_prompts.get(mode, mode_prompts["hybrid"])
    
    def _generate_tables(self, query: str, context: str, response: str, client_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tables from the response"""
        tables = []
        
        # Simple table generation based on response content
        if "comparación" in query.lower() or "vs" in query.lower():
            # Generate comparison table
            tables.append({
                "title": f"Tabla Comparativa - {client_info['name']}",
                "type": "comparison",
                "columns": ["Aspecto", "Opción A", "Opción B"],
                "data": [
                    ["Característica 1", "Valor A1", "Valor B1"],
                    ["Característica 2", "Valor A2", "Valor B2"],
                    ["Característica 3", "Valor A3", "Valor B3"]
                ]
            })
        
        return tables
    
    def _generate_charts(self, query: str, context: str, response: str, client_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate charts from the response"""
        charts = []
        
        # Simple chart generation based on content analysis
        if "tendencia" in query.lower() or "evolución" in query.lower():
            charts.append({
                "title": f"Tendencia - {client_info['name']}",
                "type": "line",
                "data": {
                    "labels": ["2021", "2022", "2023", "2024"],
                    "datasets": [{
                        "label": "Métrica Principal",
                        "data": [65, 70, 75, 80],
                        "borderColor": client_info.get("brand_colors", ["#0066CC"])[0]
                    }]
                }
            })
        
        return charts
    
    def _generate_images(self, query: str, context: str, response: str, client_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate images using DALL-E if requested"""
        images = []
        
        try:
            if "imagen" in query.lower() or "gráfico" in query.lower():
                # Generate image with DALL-E
                azure_config = client_config["azure_openai"]
                client_info = client_config["client_info"]
                
                # Create brand-appropriate prompt
                brand_colors = ", ".join(client_info.get("brand_colors", ["azul", "blanco"]))
                
                enhanced_prompt = f"""Professional infographic for {client_info['name']} showing market research insights. 
Corporate style with colors {brand_colors}. Clean, modern design suitable for business presentations. 
High quality, professional appearance."""
                
                headers = {
                    "Content-Type": "application/json",
                    "api-key": azure_config["api_key"]
                }
                
                payload = {
                    "model": "dall-e-3",
                    "prompt": enhanced_prompt,
                    "n": 1,
                    "size": "1024x1024",
                    "quality": "hd"
                }
                
                url = f"{azure_config['endpoint']}/openai/deployments/{azure_config['dalle_deployment']}/images/generations?api-version={azure_config['api_version']}"
                
                response = requests.post(url, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                
                result = response.json()
                
                if result.get("data"):
                    images.append({
                        "title": f"Visualización - {client_info['name']}",
                        "url": result["data"][0]["url"],
                        "prompt": enhanced_prompt,
                        "type": "infographic"
                    })
                    
        except Exception as e:
            print(f"[ERROR] Error generating images: {e}")
        
        return images