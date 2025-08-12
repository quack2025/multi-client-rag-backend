# core/multimodal_processor.py
"""
Multimodal Input Processing for Multi-Client RAG System
Handles text, images, and audio inputs using Azure OpenAI
"""

import os
import base64
import json
import tempfile
from typing import Dict, List, Any, Optional, Union
import requests

class MultimodalInputProcessor:
    """Process multimodal inputs for multi-client RAG system"""
    
    def __init__(self, clients_config: Dict[str, Dict[str, Any]]):
        self.clients_config = clients_config
        
        # Supported formats
        self.supported_image_formats = ['.jpg', '.jpeg', '.png', '.webp', '.gif']
        self.supported_audio_formats = ['.mp3', '.wav', '.m4a', '.ogg']
        
        print("[SUCCESS] Multi-Client Multimodal Input Processor initialized")
    
    def process_input(self, input_data: Dict[str, Any], client_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process multimodal input for specific client"""
        try:
            processed_data = {
                "text_content": "",
                "image_analyses": [],
                "audio_transcriptions": [],
                "combined_content": "",
                "metadata": input_data.get("metadata", {}),
                "processing_info": {
                    "has_text": False,
                    "has_images": False,
                    "has_audio": False,
                    "total_content_length": 0
                }
            }
            
            # Process text input
            if "text" in input_data and input_data["text"]:
                processed_data["text_content"] = input_data["text"]
                processed_data["processing_info"]["has_text"] = True
            
            # Process image inputs
            if "images" in input_data and input_data["images"]:
                for i, image_input in enumerate(input_data["images"]):
                    try:
                        image_analysis = self._process_image(image_input, f"image_{i}", client_config)
                        if image_analysis:
                            processed_data["image_analyses"].append(image_analysis)
                            processed_data["processing_info"]["has_images"] = True
                    except Exception as e:
                        print(f"[WARNING] Error processing image {i}: {e}")
            
            # Process audio inputs
            if "audio" in input_data and input_data["audio"]:
                for i, audio_input in enumerate(input_data["audio"]):
                    try:
                        audio_transcription = self._process_audio(audio_input, f"audio_{i}", client_config)
                        if audio_transcription:
                            processed_data["audio_transcriptions"].append(audio_transcription)
                            processed_data["processing_info"]["has_audio"] = True
                    except Exception as e:
                        print(f"[WARNING] Error processing audio {i}: {e}")
            
            # Combine all content
            processed_data["combined_content"] = self._combine_content(processed_data)
            processed_data["processing_info"]["total_content_length"] = len(processed_data["combined_content"])
            
            return processed_data
            
        except Exception as e:
            print(f"[ERROR] Error in multimodal processing: {e}")
            return {"error": str(e)}
    
    def _process_image(self, image_input: Union[str, bytes], image_id: str, client_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process single image using Azure OpenAI Vision"""
        try:
            # Handle base64 input
            if isinstance(image_input, str):
                if image_input.startswith('data:image'):
                    # Extract base64 data from data URI
                    image_data = image_input.split(',')[1]
                else:
                    # Assume it's already base64
                    image_data = image_input
            else:
                # Convert bytes to base64
                image_data = base64.b64encode(image_input).decode()
            
            # Prepare Azure OpenAI request
            azure_config = client_config["azure_openai"]
            headers = {
                "Content-Type": "application/json",
                "api-key": azure_config["api_key"]
            }
            
            # Create prompt for image analysis
            client_name = client_config["client_info"]["name"]
            industry = client_config["client_info"]["industry"]
            
            prompt = f"""Analiza esta imagen desde la perspectiva de {client_name} en la industria {industry}.

Describe:
1. Qué elementos visuales son relevantes para el negocio
2. Texto o información que pueda ser útil
3. Contexto de marca o producto si es aplicable
4. Insights para investigación de mercado

Responde en español de manera concisa pero informativa."""
            
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.1
            }
            
            url = f"{azure_config['endpoint']}/openai/deployments/{azure_config['chat_deployment']}/chat/completions?api-version={azure_config['api_version']}"
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            analysis_text = result["choices"][0]["message"]["content"]
            
            return {
                "image_id": image_id,
                "analysis": analysis_text,
                "status": "success"
            }
            
        except Exception as e:
            print(f"[ERROR] Error processing image {image_id}: {e}")
            return {
                "image_id": image_id,
                "analysis": f"Error analyzing image: {str(e)}",
                "status": "error"
            }
    
    def _process_audio(self, audio_input: Union[str, bytes], audio_id: str, client_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process single audio file using Azure OpenAI Whisper"""
        try:
            # Convert base64 to bytes if needed
            if isinstance(audio_input, str):
                audio_bytes = base64.b64decode(audio_input)
            else:
                audio_bytes = audio_input
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name
            
            try:
                # Prepare Azure OpenAI Whisper request
                azure_config = client_config["azure_openai"]
                
                headers = {
                    "api-key": azure_config["api_key"]
                }
                
                files = {
                    'file': (temp_path, open(temp_path, 'rb'), 'audio/wav')
                }
                
                data = {
                    'model': 'whisper-1',
                    'language': 'es' if 'spanish' in client_config['client_info']['language'] else 'auto'
                }
                
                url = f"{azure_config['endpoint']}/openai/deployments/{azure_config.get('whisper_deployment', 'whisper-1')}/audio/transcriptions?api-version={azure_config['api_version']}"
                
                response = requests.post(url, headers=headers, files=files, data=data, timeout=60)
                response.raise_for_status()
                
                result = response.json()
                transcription = result.get("text", "")
                
                return {
                    "audio_id": audio_id,
                    "transcription": transcription,
                    "status": "success"
                }
                
            finally:
                # Clean up temporary file
                os.unlink(temp_path)
            
        except Exception as e:
            print(f"[ERROR] Error processing audio {audio_id}: {e}")
            return {
                "audio_id": audio_id,
                "transcription": f"Error transcribing audio: {str(e)}",
                "status": "error"
            }
    
    def _combine_content(self, processed_data: Dict[str, Any]) -> str:
        """Combine all processed content into unified text"""
        content_parts = []
        
        # Add text content
        if processed_data["text_content"]:
            content_parts.append(f"TEXTO: {processed_data['text_content']}")
        
        # Add image analyses
        for image_analysis in processed_data["image_analyses"]:
            if image_analysis["status"] == "success":
                content_parts.append(f"IMAGEN ({image_analysis['image_id']}): {image_analysis['analysis']}")
        
        # Add audio transcriptions
        for audio_transcription in processed_data["audio_transcriptions"]:
            if audio_transcription["status"] == "success":
                content_parts.append(f"AUDIO ({audio_transcription['audio_id']}): {audio_transcription['transcription']}")
        
        return "\n\n".join(content_parts)
    
    def extract_query_intent(self, processed_input: Dict[str, Any]) -> Dict[str, Any]:
        """Extract query intent and suggest filters"""
        try:
            combined_content = processed_input.get("combined_content", "")
            
            # Basic intent analysis
            intent_keywords = {
                "brand_query": ["marca", "brand", "producto", "product"],
                "temporal_query": ["último", "reciente", "año", "mes", "2023", "2024", "2025"],
                "study_query": ["estudio", "research", "investigación", "análisis"],
                "comparison_query": ["comparar", "versus", "diferencia", "mejor"]
            }
            
            detected_intents = []
            for intent, keywords in intent_keywords.items():
                if any(keyword.lower() in combined_content.lower() for keyword in keywords):
                    detected_intents.append(intent)
            
            # Suggest filters based on content
            suggested_filters = {}
            
            # Year detection
            import re
            years = re.findall(r'\b(20\d{2})\b', combined_content)
            if years:
                suggested_filters["year"] = int(years[0])
            
            return {
                "intent": detected_intents[0] if detected_intents else "general_query",
                "confidence": 0.8 if detected_intents else 0.5,
                "detected_intents": detected_intents,
                "suggested_filters": suggested_filters
            }
            
        except Exception as e:
            print(f"[ERROR] Error extracting query intent: {e}")
            return {
                "intent": "general_query",
                "confidence": 0.5,
                "error": str(e)
            }